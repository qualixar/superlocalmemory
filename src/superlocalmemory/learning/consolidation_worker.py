# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Sleep-Time Consolidation Worker — **back-compat shim**.

As of v3.4.22 (F4.A Stage-8 H-01 fix), the 1344-LOC god-module was
split into five cohesive files:

  - ``consolidation_cycle.py``     — :class:`ConsolidationWorker`.
  - ``pattern_miner.py``           — :func:`generate_patterns`.
  - ``ranker_retrain_legacy.py``   — deprecated cold-start path.
  - ``ranker_retrain_online.py``   — LLD-10 candidate seams.
  - ``ranker_common.py``           — training-matrix + NDCG helpers.

This shim exists so that the 3830 live tests + dashboard + MCP tools +
managed server routes keep importing from
``superlocalmemory.learning.consolidation_worker`` with zero churn.

Tests that ``monkeypatch.setattr(cw_mod, "_train_booster", fake)`` work
unchanged because :func:`_run_shadow_cycle` is defined here and resolves
its helper names through *this* module's globals.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Re-exports — anything tests or production code reached for on the old
# module stays reachable here under the same name.
# ---------------------------------------------------------------------------

from superlocalmemory.learning.consolidation_cycle import (  # noqa: E402
    ConsolidationWorker,
)
from superlocalmemory.learning.ranker_common import (  # noqa: E402
    _build_training_matrix,
    _compute_eval_metrics,
    _shadow_test_improved,
    holdout_metrics,
    split_by_query,
)
from superlocalmemory.learning.ranker_retrain_legacy import (  # noqa: E402
    _LEGACY_RETRAIN_DEPRECATED,
    _retrain_ranker_impl,
)
from superlocalmemory.learning.ranker_retrain_online import (  # noqa: E402
    RETRAIN_HOURS_THRESHOLD,
    RETRAIN_HYPERPARAM_CAPS,
    RETRAIN_MODEL_SIZE_BYTES_CAP,
    RETRAIN_NEW_OUTCOMES_THRESHOLD,
    RETRAIN_WALL_TIME_BUDGET_SEC,
    RetrainWallTimeExceeded,
    _check_rollback,
    _feature_names,
    _fetch_training_rows,
    _measure_serialized_size,
    _persist_candidate,
    _promote_candidate,
    _train_booster,
)

__all__ = (
    "ConsolidationWorker",
    "RetrainWallTimeExceeded",
    "RETRAIN_HYPERPARAM_CAPS",
    "RETRAIN_WALL_TIME_BUDGET_SEC",
    "RETRAIN_MODEL_SIZE_BYTES_CAP",
    "RETRAIN_NEW_OUTCOMES_THRESHOLD",
    "RETRAIN_HOURS_THRESHOLD",
    "_retrain_ranker_impl",
    "_LEGACY_RETRAIN_DEPRECATED",
    "_build_training_matrix",
    "_compute_eval_metrics",
    "_shadow_test_improved",
    "_feature_names",
    "_fetch_training_rows",
    "_measure_serialized_size",
    "_train_booster",
    "_persist_candidate",
    "_promote_candidate",
    "_check_rollback",
    "_run_shadow_cycle",
)


# ---------------------------------------------------------------------------
# Orchestrator — defined HERE so tests patching ``cw_mod._train_booster``
# actually intercept the helper call.
# ---------------------------------------------------------------------------


#: Distinct informative labels required before a retrain is worth running.
#:
#: Measured on a live production store: ``learning_features`` holds 5,352 rows, of
#: which **5,350 carry label 0.0** and exactly two are non-zero (0.6 and 1.0) —
#: the two real feedback events in the whole database. The active model was
#: trained on 972 such rows and reorders results at random; the heuristic it
#: displaced was better.
#:
#: An earlier design proposed gating on ``len(set(labels)) <= 1``. That check PASSES
#: here — three distinct values exist — and would train on a 2,675:1 imbalance
#: while reporting success, which is worse than not training at all because it
#: looks like progress. A count of informative labels is the honest gate.
MIN_INFORMATIVE_LABELS = 50


def _count_informative_labels(rows: list[dict]) -> int:
    """Rows whose outcome says something other than "nothing happened".

    A reward of exactly 0.5 is the neutral default the settlement path applies
    when no outcome was reported, and ``None`` means the outcome column is
    absent. Neither is evidence. Everything else is.
    """
    seen = 0
    for row in rows:
        reward = row.get("outcome_reward")
        if reward is None:
            continue
        try:
            value = float(reward)
        except (TypeError, ValueError):
            continue
        if abs(value - 0.5) > 1e-9:
            seen += 1
    return seen

def _run_shadow_cycle(
    *,
    memory_db_path: str,
    learning_db_path: str,
    profile_id: str,
) -> dict:
    """Top-level online retrain cycle — runs inside the consolidation
    worker.

    Orchestrates: fetch rows → train → size-check → persist candidate
    (NOT auto-promote). Promotion happens separately once the live
    shadow-router accumulates enough observations (see
    :mod:`superlocalmemory.core.shadow_router`).

    Helper functions are looked up via this module's namespace so test
    monkey-patches on ``consolidation_worker`` take effect without any
    test churn.

    Returns a dict with keys:
      * ``aborted``: reason string if aborted (``'insufficient_data'``,
        ``'model_too_large'``, ``'wall_time_exceeded'``, ``'train_error'``).
      * ``candidate_persisted``: True if a candidate row was written.
      * ``promoted``: False (always — promotion is a separate step).
      * ``metrics``: training metrics dict on success.
    """
    out: dict = {
        "aborted": None, "candidate_persisted": False,
        "promoted": False, "metrics": None,
    }

    try:
        rows, _qids = _fetch_training_rows(learning_db_path, profile_id)
    except Exception as exc:
        logger.debug("fetch_training_rows failed: %s", exc)
        out["aborted"] = "fetch_error"
        return out

    if len(rows) < 20:
        out["aborted"] = "insufficient_data"
        return out

    informative = _count_informative_labels(rows)
    if informative < MIN_INFORMATIVE_LABELS:
        # Honest refusal. See _count_informative_labels for why a row count is
        # not a substitute for this, and why the answer must not be to lower
        # the threshold until training starts.
        logger.info(
            "retrain skipped: %d informative label(s) of %d rows "
            "(need %d). Rewards accumulate from reported outcomes; there is "
            "nothing to learn from a corpus of one label.",
            informative, len(rows), MIN_INFORMATIVE_LABELS,
        )
        out["aborted"] = "insufficient_label_signal"
        out["metrics"] = {
            "rows": len(rows),
            "informative_labels": informative,
            "required": MIN_INFORMATIVE_LABELS,
        }
        return out

    # Load prior active for in-sample shadow.
    try:
        from superlocalmemory.learning.database import LearningDatabase
        db = LearningDatabase(learning_db_path)
        prior_row = db.load_active_model(profile_id)
    except Exception:
        prior_row = None

    feature_names = _feature_names()
    # Query-disjoint holdout: metrics and the gate below are measured on
    # queries the booster never saw (they were computed on its own training
    # rows before 4.1.25). Too few queries for a holdout -> train on all and
    # let the live shadow test be the only judge.
    train_rows, holdout_rows = split_by_query(rows)
    if len({r.get("query_id") for r in train_rows}) < 2 or not holdout_rows:
        train_rows, holdout_rows = rows, []

    try:
        booster, metrics = _train_booster(
            learning_db_path, profile_id,
            training_rows=train_rows, feature_names=feature_names,
            prior_row=prior_row,
        )
    except RetrainWallTimeExceeded as exc:
        out["aborted"] = "wall_time_exceeded"
        out["metrics"] = {"wall_time_sec": exc.elapsed_sec}
        return out
    except Exception as exc:
        logger.debug("train_booster failed: %s", exc)
        out["aborted"] = "train_error"
        return out

    # Model-size guardrail (LLD-10 §3.2 post-train check).
    size_bytes = _measure_serialized_size(booster)
    if size_bytes > RETRAIN_MODEL_SIZE_BYTES_CAP:
        logger.warning(
            "retrain: candidate %.2f MB exceeds %.1f MB cap — rejecting",
            size_bytes / 1e6, RETRAIN_MODEL_SIZE_BYTES_CAP / 1e6,
        )
        out["aborted"] = "model_too_large"
        out["metrics"] = metrics
        return out

    if holdout_rows:
        try:
            metrics = {**(metrics or {}),
                       **holdout_metrics(booster, holdout_rows, feature_names)}
        except Exception as exc:  # pragma: no cover — metrics are advisory
            logger.debug("holdout metrics skipped: %s", exc)

    # Offline paired gate on the SAME held-out labelled queries (both models
    # rank identical queries, so a paired comparison is valid here) — a cheap
    # filter before spending live recalls on the independent-groups test.
    if prior_row is not None and holdout_rows:
        if not _shadow_test_improved(
            prior_row, booster, holdout_rows, feature_names,
        ):
            out["aborted"] = "insample_shadow_fail"
            out["metrics"] = metrics
            return out

    try:
        state_bytes = booster.model_to_string().encode("utf-8")
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("model serialise failed: %s", exc)
        out["aborted"] = "serialise_error"
        return out

    try:
        _persist_candidate(
            learning_db_path, profile_id=profile_id,
            state_bytes=state_bytes, feature_names=feature_names,
            trained_on_count=len(rows), metrics=metrics,
            shadow_results={"holdout_gate_pass": bool(prior_row is not None
                                                       and holdout_rows)},
        )
    except sqlite3.Error as exc:
        logger.warning("persist_candidate failed: %s", exc)
        out["aborted"] = "persist_error"
        return out

    out["candidate_persisted"] = True
    out["promoted"] = False  # Promotion is a separate live-shadow step.
    out["metrics"] = metrics
    return out
