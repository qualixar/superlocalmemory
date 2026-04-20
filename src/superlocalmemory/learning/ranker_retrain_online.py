# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — F4.A Stage-8 H-01 + LLD-10

"""LLD-10 online retrain: hyperparam-capped LightGBM training + candidate
persist + atomic lineage flip.

All seams (``_train_booster``, ``_fetch_training_rows``,
``_measure_serialized_size``, ``_persist_candidate``,
``_promote_candidate``, ``_feature_names``) are module-level functions so
tests can monkey-patch them via the shim (``consolidation_worker``)
which re-exports them with the shim's own module bindings.

The orchestrator ``_run_shadow_cycle`` lives in the shim
(``consolidation_worker``) so patches on ``consolidation_worker`` keep
working without any test churn.

Contract refs:
  - LLD-10 §2 (triggers), §3.2 (caps), §5 (lineage flip).
  - IMPLEMENTATION-MANIFEST v3.4.22 FINAL A.3.
  - Stage 8 H-01 (architect).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from superlocalmemory.learning.ranker_common import (
    _build_training_matrix,
    _compute_eval_metrics,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLD-10 retrain constants
# ---------------------------------------------------------------------------

#: LightGBM hyperparameter caps. Contractual — violating these is a
#: Stage-8 audit failure. Manifest A.3 names: num_leaves ≤ 31,
#: max_depth ≤ 7, feature_fraction ≤ 0.8.
RETRAIN_HYPERPARAM_CAPS: Final[dict] = {
    "num_leaves": 31,
    "max_depth": 7,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "min_data_in_leaf": 20,
    "num_boost_round": 50,
    "learning_rate": 0.05,
    "metric": "ndcg",
    "ndcg_eval_at": [1, 3, 5, 10],
    "verbosity": -1,
}

#: Wall-time ceiling (seconds) for a single retrain cycle.
RETRAIN_WALL_TIME_BUDGET_SEC: Final[float] = 30.0

#: Model-size ceiling for the serialised booster blob (10 MB).
RETRAIN_MODEL_SIZE_BYTES_CAP: Final[int] = 10 * 1024 * 1024

#: Trigger: new outcomes since last retrain ≥ this → retrain.
RETRAIN_NEW_OUTCOMES_THRESHOLD: Final[int] = 50

#: Trigger: hours since last retrain ≥ this → retrain.
RETRAIN_HOURS_THRESHOLD: Final[float] = 24.0


class RetrainWallTimeExceeded(Exception):
    """Raised by ``_train_booster`` when the 30 s budget is blown."""

    def __init__(self, *, elapsed_sec: float) -> None:
        super().__init__(
            f"retrain wall-time exceeded: {elapsed_sec:.1f}s > "
            f"{RETRAIN_WALL_TIME_BUDGET_SEC:.1f}s",
        )
        self.elapsed_sec = elapsed_sec


__all__ = (
    "RetrainWallTimeExceeded",
    "RETRAIN_HYPERPARAM_CAPS",
    "RETRAIN_WALL_TIME_BUDGET_SEC",
    "RETRAIN_MODEL_SIZE_BYTES_CAP",
    "RETRAIN_NEW_OUTCOMES_THRESHOLD",
    "RETRAIN_HOURS_THRESHOLD",
    "_feature_names",
    "_fetch_training_rows",
    "_measure_serialized_size",
    "_train_booster",
    "_persist_candidate",
    "_promote_candidate",
    "_check_rollback",
)


# ---------------------------------------------------------------------------
# Seams: module-level functions so tests can monkey-patch via the shim.
# ---------------------------------------------------------------------------


def _feature_names() -> list[str]:
    """Indirection for tests — returns the live ranker FEATURE_NAMES."""
    from superlocalmemory.learning.features import FEATURE_NAMES
    return list(FEATURE_NAMES)


def _fetch_training_rows(
    learning_db_path: str, profile_id: str,
) -> tuple[list[dict], list[str]]:
    """Fetch real training rows + queue of candidate query_ids.

    Returns ``(rows, candidate_ids)`` — ``rows`` matches the shape that
    ``_build_training_matrix`` expects (``query_id``, ``fact_id``,
    ``position``, ``features`` dict, ``outcome_reward``).
    Tests monkey-patch this seam to inject deterministic fixtures.
    """
    from superlocalmemory.learning.database import LearningDatabase

    db = LearningDatabase(learning_db_path)
    rows = db.fetch_training_examples(
        profile_id=profile_id,
        limit=5000,
        min_outcome_age_sec=60,
        include_synthetic=False,
    )
    return rows, [r.get("query_id", "") for r in rows if r.get("query_id")]


def _measure_serialized_size(booster) -> int:
    """Return the serialised booster size in bytes. Seam for tests."""
    try:
        return len(booster.model_to_string().encode("utf-8"))
    except Exception:
        return 0


def _train_booster(
    learning_db_path: str,
    profile_id: str,
    *,
    training_rows: list[dict],
    feature_names: list[str],
    prior_row: dict | None,
):
    """Train a LightGBM booster with the HARD hyperparam caps + wall-time
    guard. Raises :class:`RetrainWallTimeExceeded` on budget breach.

    Returns ``(booster, metrics_dict)``. Tests monkey-patch this seam;
    production invocation uses the real path.
    """
    try:
        import numpy as np
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover — platform guard
        raise RuntimeError(f"lightgbm unavailable: {exc}") from exc

    from superlocalmemory.learning.labeler import label_gain

    X, y_int, groups = _build_training_matrix(training_rows, feature_names)
    if groups is None or len(groups) < 2:
        raise ValueError(
            f"insufficient query groups for retrain "
            f"(got {None if groups is None else len(groups)})",
        )
    assert sum(groups) == X.shape[0], (
        f"group sum mismatch: {sum(groups)} != {X.shape[0]}")

    gain = label_gain()
    y_int = np.clip(y_int, 0, len(gain) - 1)

    ds_train = lgb.Dataset(
        X,
        label=y_int,
        group=groups,
        feature_name=list(feature_names),
        free_raw_data=False,
    )

    _allowed_objectives = {"lambdarank", "rank_xendcg"}
    objective = os.environ.get("SLM_LGBM_OBJECTIVE", "lambdarank").strip()
    if objective not in _allowed_objectives:
        objective = "lambdarank"

    # CAPS — values are enforced both in the params dict (trainer side)
    # and in RETRAIN_HYPERPARAM_CAPS (surface constant for tests + ops).
    params = dict(RETRAIN_HYPERPARAM_CAPS)
    params["objective"] = objective
    params["label_gain"] = gain
    params["num_threads"] = max(1, (os.cpu_count() or 2) - 1)
    num_boost_round = int(params.pop("num_boost_round"))

    start = time.monotonic()
    # S9-defer H-P-08: before v3.4.22 the wall-time check fired only
    # AFTER ``lgb.train`` returned, which could take minutes on a
    # pathological dataset before we noticed. LightGBM's ``callbacks``
    # parameter accepts a per-iteration hook that can stop training
    # early. We install a callback that raises once the cumulative
    # wall-time exceeds ``RETRAIN_WALL_TIME_BUDGET_SEC`` so overrun
    # aborts mid-training, not post-training.
    class _WallTimeAbort(Exception):
        pass

    def _timeout_cb(_env):
        if time.monotonic() - start >= RETRAIN_WALL_TIME_BUDGET_SEC:
            raise _WallTimeAbort

    try:
        booster = lgb.train(
            params, ds_train, num_boost_round=num_boost_round,
            callbacks=[_timeout_cb],
        )
    except _WallTimeAbort:
        elapsed = time.monotonic() - start
        raise RetrainWallTimeExceeded(elapsed_sec=elapsed)
    except TypeError:
        # Older lightgbm binaries don't accept callbacks= in lgb.train.
        # Fall back to the original post-hoc check (still bounded by
        # num_boost_round * per-round work, which params cap tightly).
        booster = lgb.train(
            params, ds_train, num_boost_round=num_boost_round,
        )
    except lgb.basic.LightGBMError as exc:  # pragma: no cover
        raise RuntimeError(f"lgb.train failed: {exc}") from exc
    elapsed = time.monotonic() - start
    if elapsed >= RETRAIN_WALL_TIME_BUDGET_SEC:
        raise RetrainWallTimeExceeded(elapsed_sec=elapsed)

    metrics = _compute_eval_metrics(booster, training_rows, feature_names)
    metrics["wall_time_sec"] = elapsed
    return booster, metrics


def _persist_candidate(
    learning_db_path: str,
    *,
    profile_id: str,
    state_bytes: bytes,
    feature_names: list[str],
    trained_on_count: int,
    metrics: dict,
    shadow_results: dict | None,
) -> int:
    """Insert a fresh candidate row with is_candidate=1 + is_active=0.

    Honours the partial unique index ``idx_model_candidate_one`` —
    callers must discard/reject any prior candidate before insert.
    """
    now = datetime.now(timezone.utc).isoformat()
    sha = hashlib.sha256(state_bytes).hexdigest()
    metrics_json = json.dumps(metrics or {}, separators=(",", ":"))
    fn_json = json.dumps(list(feature_names), separators=(",", ":"))
    shadow_json = json.dumps(shadow_results or {}, separators=(",", ":"))

    with sqlite3.connect(learning_db_path, timeout=10) as conn:
        try:
            conn.execute("BEGIN IMMEDIATE")
            # Wipe any stale candidate first (one-at-a-time contract).
            conn.execute(
                "DELETE FROM learning_model_state "
                "WHERE profile_id = ? AND is_candidate = 1",
                (profile_id,),
            )
            cur = conn.execute(
                "INSERT INTO learning_model_state "
                "(profile_id, model_version, state_bytes, bytes_sha256, "
                " trained_on_count, feature_names, metrics_json, "
                " is_active, is_candidate, shadow_results_json, "
                " trained_at, updated_at) "
                "VALUES (?, '3.4.22', ?, ?, ?, ?, ?, 0, 1, ?, ?, ?)",
                (
                    profile_id, state_bytes, sha, int(trained_on_count),
                    fn_json, metrics_json, shadow_json, now, now,
                ),
            )
            conn.commit()
            candidate_id = int(cur.lastrowid or 0)
        except sqlite3.Error:
            conn.rollback()
            raise

    # S9-W1 C1: attach the new candidate to the shadow router so live
    # recall-settled signals start accumulating into its ShadowTest.
    # Without this call, ``ShadowTest.decide()`` would sit at
    # ``insufficient_data`` forever and the candidate would never be
    # promoted or rejected — LLD-10 would be shipped-but-cold (exactly
    # the failure mode Stage 8 SB-1 claimed to close but only wired the
    # consumer side for). Fail-soft so persist itself never regresses.
    try:
        from superlocalmemory.core import shadow_router as _sr
        # memory_db path is not available here; the router accepts empty
        # string — attach_candidate only touches learning_db state.
        router = _sr.get_shadow_router(
            memory_db="",
            learning_db=learning_db_path,
            profile_id=profile_id,
        )
        router.attach_candidate(candidate_id)
    except Exception as exc:  # noqa: BLE001 — defence in depth
        logger.debug("attach_candidate failed (non-fatal): %s", exc)

    return candidate_id


def _promote_candidate(
    learning_db_path: str, *, profile_id: str, candidate_id: int,
) -> bool:
    """Atomic lineage flip (LLD-10 §5 / §6.1).

    Invariants (enforced by M009 partial unique indexes):
      * Exactly one ``is_active=1`` per profile at any instant.
      * Exactly one ``is_candidate=1`` per profile at any instant.

    Flip order inside one BEGIN IMMEDIATE:
      1. Clear existing is_previous (it becomes is_rollback).
      2. Current active → is_active=0, is_previous=1.
      3. Candidate → is_active=1, is_candidate=0, promoted_at=now.
    """
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(learning_db_path, timeout=10) as conn:
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("BEGIN IMMEDIATE")

            # Step 1 — demote the existing previous to rollback, if any.
            existing_prev = conn.execute(
                "SELECT id FROM learning_model_state "
                "WHERE profile_id = ? AND is_previous = 1",
                (profile_id,),
            ).fetchone()
            if existing_prev is not None:
                conn.execute(
                    "UPDATE learning_model_state "
                    "SET is_previous = 0, is_rollback = 1 "
                    "WHERE id = ?",
                    (existing_prev["id"],),
                )

            # Step 2 — demote current active. Clear is_active first so
            # the partial unique index on is_active=1 never sees two
            # rows simultaneously.
            conn.execute(
                "UPDATE learning_model_state "
                "SET is_active = 0, is_previous = 1 "
                "WHERE profile_id = ? AND is_active = 1",
                (profile_id,),
            )

            # Step 3 — promote candidate.
            conn.execute(
                "UPDATE learning_model_state "
                "SET is_active = 1, is_candidate = 0, promoted_at = ? "
                "WHERE id = ?",
                (now, candidate_id),
            )

            # Reset the outcome counter on the new active.
            row = conn.execute(
                "SELECT metadata_json FROM learning_model_state "
                "WHERE id = ?", (candidate_id,),
            ).fetchone()
            try:
                meta = json.loads(row["metadata_json"] or "{}")
            except (TypeError, ValueError):
                meta = {}
            meta["new_outcomes_since_last_retrain"] = 0
            meta["last_retrain_at"] = now
            conn.execute(
                "UPDATE learning_model_state SET metadata_json = ? "
                "WHERE id = ?",
                (json.dumps(meta), candidate_id),
            )
            conn.commit()
            return True
        except sqlite3.Error as exc:
            conn.rollback()
            logger.error("_promote_candidate sqlite error: %s", exc)
            return False


def _check_rollback(
    *,
    learning_db_path: str,
    profile_id: str,
    observations: list[float],
    baseline_ndcg: float,
) -> bool:
    """Evaluate the 200-recall watch window and fire rollback if needed.

    Returns True iff rollback was executed.
    """
    from superlocalmemory.learning.model_rollback import ModelRollback

    rb = ModelRollback(
        learning_db_path=learning_db_path,
        profile_id=profile_id,
        baseline_ndcg=baseline_ndcg,
    )
    for i, val in enumerate(observations):
        rb.record_post_promotion(query_id=f"watch-{i}", ndcg_at_10=val)
    if rb.should_rollback():
        return rb.execute_rollback(reason="watch_window_regression")
    return False
