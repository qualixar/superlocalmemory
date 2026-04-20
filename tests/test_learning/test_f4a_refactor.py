# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — F4.A Stage-8 refactor regression suite

"""Regression + architecture tests for the F4.A Stage-8 learning-track split.

Covers:
  * Import-path compatibility — ``ConsolidationWorker`` /
    ``HnswDeduplicator`` remain reachable through the shim.
  * H-07 DeprecationWarning fires exactly once per process for the
    legacy retrain path.
  * H-07 legacy retrain is skipped when the profile has outcomes
    (i.e. ``ConsolidationWorker.run`` routes to the online cycle).
  * H-17 hnswlib fallback emits a warning + increments the degradation
    counter.
  * H-06 substring-leak regression is covered in
    ``test_fact_outcome_joins.py``.
"""

from __future__ import annotations

import importlib
import json
import logging
import sqlite3
import sys
import warnings
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Architecture assertions — import paths unchanged
# ---------------------------------------------------------------------------


def test_consolidation_worker_re_exports_ConsolidationWorker() -> None:
    """Back-compat: ConsolidationWorker imports unchanged after split."""
    from superlocalmemory.learning.consolidation_worker import (
        ConsolidationWorker,
    )
    from superlocalmemory.learning.consolidation_cycle import (
        ConsolidationWorker as CanonCW,
    )
    # They are the same class — shim re-export, not a copy.
    assert ConsolidationWorker is CanonCW


def test_hnsw_dedup_re_exports_HnswDeduplicator() -> None:
    """Back-compat: HnswDeduplicator imports unchanged after split."""
    from superlocalmemory.learning.hnsw_dedup import HnswDeduplicator
    from superlocalmemory.learning.dedup_hnsw import (
        HnswDeduplicator as CanonHD,
    )
    assert HnswDeduplicator is CanonHD


def test_shim_re_exports_all_retrain_seams() -> None:
    """Tests that monkey-patch ``cw_mod._train_booster`` etc. must find
    these names as module-level attributes of the shim."""
    from superlocalmemory.learning import consolidation_worker as cw
    for name in (
        "_train_booster", "_fetch_training_rows", "_measure_serialized_size",
        "_persist_candidate", "_promote_candidate", "_feature_names",
        "_run_shadow_cycle", "_retrain_ranker_impl", "_build_training_matrix",
        "_shadow_test_improved", "_compute_eval_metrics",
        "RETRAIN_HYPERPARAM_CAPS", "RetrainWallTimeExceeded",
    ):
        assert hasattr(cw, name), f"consolidation_worker missing {name}"


def test_hnsw_shim_re_exports_helpers() -> None:
    from superlocalmemory.learning import hnsw_dedup as hd
    for name in (
        "HnswDeduplicator", "run_reward_gated_archive",
        "apply_strong_memory_boost", "select_high_reward_fact_ids",
        "ram_reservation", "get_hnsw_degraded_count",
        "REWARD_WINDOW_DAYS", "ARCHIVE_REWARD_THRESHOLD",
        "STRONG_BOOST_INCREMENT", "STRONG_BOOST_CAP",
    ):
        assert hasattr(hd, name), f"hnsw_dedup missing {name}"


# ---------------------------------------------------------------------------
# H-07 — DeprecationWarning fires + legacy gated when outcomes present
# ---------------------------------------------------------------------------


def _reset_deprecation_state() -> None:
    """Reset the one-shot flag so each test gets a fresh firing."""
    import superlocalmemory.learning.ranker_retrain_legacy as leg
    leg._warned = False


def test_ranker_retrain_legacy_deprecation_warning_fires_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The legacy ``_retrain_ranker_impl`` MUST raise a DeprecationWarning
    the first time it runs in a process. Subsequent calls in the same
    process emit nothing (log hygiene).
    """
    _reset_deprecation_state()
    # Shadow lightgbm+numpy as missing so the impl bails immediately
    # after emitting the warning. We DON'T care about the return value.
    from superlocalmemory.learning import ranker_retrain_legacy as leg

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Force import error path to avoid actual training.
        monkeypatch.setattr(
            "superlocalmemory.learning.database.LearningDatabase.__init__",
            lambda self, *a, **kw: (_ for _ in ()).throw(
                ImportError("shim — skip real training")
            ),
            raising=False,
        )
        try:
            leg._retrain_ranker_impl(str(tmp_path / "learn.db"), "p1")
        except Exception:
            pass
        # Second invocation — should NOT re-emit.
        try:
            leg._retrain_ranker_impl(str(tmp_path / "learn.db"), "p1")
        except Exception:
            pass

    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep_warnings) == 1, (
        f"expected exactly one DeprecationWarning, got {len(dep_warnings)}: "
        f"{[str(w.message) for w in dep_warnings]}"
    )


def test_ranker_retrain_legacy_gated_when_outcomes_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage-8 H-07: once a profile has an active model (with outcomes),
    the legacy cold-start path MUST be skipped. The online retrain path
    owns the profile after that point.
    """
    from superlocalmemory.learning.consolidation_worker import (
        ConsolidationWorker,
    )
    from superlocalmemory.learning import consolidation_worker as cw_mod

    # Build minimal schemas so _should_retrain can read the active row.
    memory_db = tmp_path / "memory.db"
    learning_db = tmp_path / "learning.db"

    with sqlite3.connect(memory_db) as conn:
        conn.execute(
            "CREATE TABLE atomic_facts ("
            "fact_id TEXT PRIMARY KEY, profile_id TEXT, content TEXT, "
            "lifecycle TEXT DEFAULT 'active', created_at TEXT, "
            "session_id TEXT, confidence REAL DEFAULT 1.0, "
            "canonical_entities_json TEXT, fact_type TEXT)"
        )
    with sqlite3.connect(learning_db) as conn:
        conn.execute(
            "CREATE TABLE learning_model_state ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, profile_id TEXT, "
            "model_version TEXT, state_bytes BLOB, bytes_sha256 TEXT, "
            "trained_on_count INTEGER, feature_names TEXT, "
            "metrics_json TEXT, is_active INTEGER DEFAULT 0, "
            "is_candidate INTEGER DEFAULT 0, is_previous INTEGER DEFAULT 0, "
            "is_rollback INTEGER DEFAULT 0, shadow_results_json TEXT, "
            "trained_at TEXT, updated_at TEXT, promoted_at TEXT, "
            "metadata_json TEXT)"
        )
        # Active row with outcomes — forces _should_retrain=True.
        conn.execute(
            "INSERT INTO learning_model_state "
            "(profile_id, is_active, metadata_json) VALUES (?, 1, ?)",
            ("p1", json.dumps({
                "new_outcomes_since_last_retrain": 100,
                "last_retrain_at": "2026-01-01T00:00:00+00:00",
            })),
        )
        conn.commit()

    # Observe which path fires by stubbing both.
    calls = {"online": 0, "legacy": 0}

    def _fake_online(*, memory_db_path, learning_db_path, profile_id):
        calls["online"] += 1
        return {"aborted": None, "candidate_persisted": True,
                "promoted": False, "metrics": {}}

    def _fake_legacy_impl(*a, **kw):
        calls["legacy"] += 1
        return False

    monkeypatch.setattr(cw_mod, "_run_shadow_cycle", _fake_online)
    monkeypatch.setattr(cw_mod, "_retrain_ranker_impl", _fake_legacy_impl)

    # Feedback collector must not error — stub signal_count high.
    class _FakeCollector:
        def __init__(self, *a, **kw):
            pass
        def get_feedback_count(self, pid):
            return 500  # well above 200 legacy trigger

    monkeypatch.setattr(
        "superlocalmemory.learning.feedback.FeedbackCollector",
        _FakeCollector,
    )

    worker = ConsolidationWorker(memory_db, learning_db)
    worker.run("p1", dry_run=False)

    assert calls["online"] == 1, (
        f"online retrain should run when active model exists; got {calls}"
    )
    assert calls["legacy"] == 0, (
        f"legacy retrain MUST be gated when outcomes present; got {calls}"
    )


# ---------------------------------------------------------------------------
# H-17 — hnswlib fallback emits warning + counter
# ---------------------------------------------------------------------------


def test_hnswlib_fallback_emits_warning_and_counter_increments(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Stage-8 H-17: fallback to prefix path must be OBSERVABLE.

    Old code used ``logger.debug`` — silent in production. New code
    uses ``logger.warning`` AND increments a module-level counter that
    operators can surface on the dashboard.
    """
    from superlocalmemory.learning import dedup_hnsw as mod

    db = tmp_path / "memory.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE atomic_facts (
                fact_id TEXT PRIMARY KEY, profile_id TEXT,
                content TEXT, canonical_entities_json TEXT,
                embedding TEXT, importance REAL DEFAULT 0.5,
                confidence REAL DEFAULT 1.0,
                created_at TEXT DEFAULT (datetime('now')),
                archive_status TEXT DEFAULT 'live'
            );
            INSERT INTO atomic_facts (fact_id, profile_id, content,
                canonical_entities_json, embedding)
            VALUES ('a', 'p1', 'aaa', '[]', '[1,0,0]'),
                   ('b', 'p1', 'bbb', '[]', '[0,1,0]');
            """,
        )

    mod.reset_hnsw_degraded_count()
    before = mod.get_hnsw_degraded_count()
    assert before == 0

    with caplog.at_level(logging.WARNING, logger=mod.__name__):
        dedup = mod.HnswDeduplicator(memory_db_path=db)
        # _force_unavailable triggers the hnswlib-missing path.
        dedup.find_merge_candidates("p1", _force_unavailable=True)

    after = mod.get_hnsw_degraded_count()
    assert after == before + 1, (
        f"degradation counter must increment: {before} -> {after}"
    )
    # A WARNING-level log line mentions the degradation.
    assert any(
        "degraded" in rec.message.lower() and rec.levelno == logging.WARNING
        for rec in caplog.records
    ), (
        f"expected a WARNING-level degradation log; got "
        f"{[(r.levelname, r.message) for r in caplog.records]}"
    )
