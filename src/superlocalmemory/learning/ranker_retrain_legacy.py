# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — F4.A Stage-8 H-07 deprecation

"""Legacy ranker retrain path (signal_count >= 200 cold-start).

**DEPRECATED** as of v3.4.22: superseded by the LLD-10 online retrain
cycle in ``ranker_retrain_online.py``. Retained for two reasons:

  1. Back-compat — ``tests/test_learning/test_ranker_v2.py`` and
     ``tests/test_api/test_dashboard_phase_truth.py`` still import
     ``_retrain_ranker_impl`` directly.
  2. Cold-start — profiles with signals but no active model yet
     (first-ever training) still need a path to bootstrap lineage.

Gating (Stage 8 H-07 fix): the `ConsolidationWorker.run` call site
invokes this path ONLY when ``_should_retrain`` returns False (no
active model) AND raw signal_count ≥ 200. Once a profile has outcomes
and an active model, the online retrain wins unconditionally.

Every public entry point here emits a one-shot DeprecationWarning the
first time it runs per process so operators see the signal without log
spam.
"""

from __future__ import annotations

import hashlib
import logging
import os
import warnings
from pathlib import Path

from superlocalmemory.learning.ranker_common import (
    _build_training_matrix,
    _compute_eval_metrics,
    _shadow_test_improved,
)

logger = logging.getLogger(__name__)

_LEGACY_RETRAIN_DEPRECATED = True

# One-shot warning flag per process.
_warned = False


def _emit_deprecation_once() -> None:
    """Fire the DeprecationWarning exactly once per process."""
    global _warned
    if _warned:
        return
    _warned = True
    # M-P-10: HARD removal target — v3.5.0. The legacy path must not
    # survive into v3.5.x tech debt. Remove with the dashboard lineage
    # UI ship (Stage 7+) per MANIFEST-DEVIATION A.3 #3.
    warnings.warn(
        "ranker_retrain_legacy is deprecated as of SLM v3.4.22 and will "
        "be REMOVED in v3.5.0 — use ranker_retrain_online._run_shadow_cycle "
        "instead. The legacy path is kept only for cold-start profiles "
        "with no active model.",
        DeprecationWarning,
        stacklevel=3,
    )


__all__ = (
    "_retrain_ranker_impl",
    "_LEGACY_RETRAIN_DEPRECATED",
    "_build_training_matrix",  # re-export for callers that imported here
    "_compute_eval_metrics",
    "_shadow_test_improved",
)


def _retrain_ranker_impl(
    learning_db: str | Path,
    profile_id: str,
    *,
    include_synthetic: bool = False,
) -> bool:
    """Legacy cold-start training path — pure function.

    DEPRECATED: prefer ``ranker_retrain_online._run_shadow_cycle``.

    ``include_synthetic`` forwards to
    :meth:`LearningDatabase.fetch_training_examples` so migrated legacy
    rows (``is_synthetic=1``) participate in training when the user opts
    in via the dashboard "Migrate legacy data" flow.
    """
    _emit_deprecation_once()

    try:
        import numpy as np
        import lightgbm as lgb  # noqa: PLC0415
    except ImportError:
        logger.info("lightgbm or numpy missing; skipping retrain")
        return False

    from superlocalmemory.learning.database import LearningDatabase
    from superlocalmemory.learning.features import FEATURE_NAMES
    from superlocalmemory.learning.labeler import label_gain

    db = LearningDatabase(learning_db)
    rows = db.fetch_training_examples(
        profile_id=profile_id,
        limit=2000,
        min_outcome_age_sec=60,
        include_synthetic=include_synthetic,
    )
    if len(rows) < 200:
        logger.info(
            "retrain: need ≥200 rows, have %d — deferring", len(rows),
        )
        return False

    X, y_int, groups = _build_training_matrix(rows, FEATURE_NAMES)
    if groups is None or len(groups) < 2:
        logger.info(
            "retrain: insufficient query groups (%s) — deferring",
            None if groups is None else len(groups),
        )
        return False
    assert sum(groups) == X.shape[0], (
        f"group sum mismatch: {sum(groups)} != {X.shape[0]}"
    )

    gain = label_gain()
    # Defensive: clamp any out-of-range label.
    y_int = np.clip(y_int, 0, len(gain) - 1)

    ds_train = lgb.Dataset(
        X,
        label=y_int,
        group=groups,
        feature_name=list(FEATURE_NAMES),
        free_raw_data=False,
    )

    # MKT-v2-M-01: allow switching between ``lambdarank`` (default,
    # LLD-02 CR1) and ``rank_xendcg`` via ``SLM_LGBM_OBJECTIVE``.
    _allowed_objectives = {"lambdarank", "rank_xendcg"}
    objective = os.environ.get("SLM_LGBM_OBJECTIVE", "lambdarank").strip()
    if objective not in _allowed_objectives:
        logger.warning(
            "SLM_LGBM_OBJECTIVE=%r not in %s; defaulting to lambdarank",
            objective, sorted(_allowed_objectives),
        )
        objective = "lambdarank"
    params = {
        "objective": objective,
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3, 5, 10],
        "label_gain": gain,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "verbosity": -1,
        "num_threads": max(1, (os.cpu_count() or 2) - 1),
    }
    try:
        booster_new = lgb.train(params, ds_train, num_boost_round=50)
    except lgb.basic.LightGBMError as exc:
        logger.warning("retrain: lightgbm train failed: %s", exc)
        return False

    # Shadow test: only promote if better than prior active model.
    prior = db.load_active_model(profile_id)
    if prior is not None:
        if not _shadow_test_improved(prior, booster_new, rows, FEATURE_NAMES):
            logger.info("Shadow test: new model did not beat prior; keeping")
            return False

    model_str = booster_new.model_to_string()
    state_bytes = model_str.encode("utf-8")
    sha = hashlib.sha256(state_bytes).hexdigest()
    try:
        db.persist_model(
            profile_id=profile_id,
            state_bytes=state_bytes,
            bytes_sha256=sha,
            feature_names=list(FEATURE_NAMES),
            trained_on_count=len(rows),
            metrics=_compute_eval_metrics(booster_new, rows, FEATURE_NAMES),
        )
    except Exception as exc:
        logger.warning("persist_model failed: %s", exc)
        return False

    # Invalidate in-process cache so new model is picked up.
    try:
        from superlocalmemory.learning.model_cache import invalidate
        invalidate(profile_id)
    except Exception:  # pragma: no cover — defensive
        pass

    logger.info(
        "Ranker retrained (legacy, lambdarank): %d rows, %d groups, "
        "promoted=True",
        len(rows), len(groups),
    )
    return True
