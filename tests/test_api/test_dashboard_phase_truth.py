# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-02 §6.5

"""Dashboard phase-truth tests (LLD-02 §4.10)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

try:
    import lightgbm  # noqa: F401
except (ImportError, OSError):
    pytest.skip("lightgbm not available (missing libomp on macOS CI)", allow_module_level=True)

from superlocalmemory.learning import model_cache
from superlocalmemory.learning.consolidation_worker import _retrain_ranker_impl
from superlocalmemory.learning.signals import record_signal_batch
from superlocalmemory.server.routes.learning import _compute_ranker_phase
from tests.test_learning._signal_fixtures import (
    make_batch,
    make_db_with_migrations,
    open_conn,
)


def _seed_signals(db, n_queries: int, per_query: int, *,
                   profile_id: str = "p1") -> None:
    """Seed the phase counter with feedback, and the trainer with exposures.

    As of 4.1.0 the ranking phase counts FEEDBACK only: ``candidate`` rows are
    exposures, one per fact displayed, and counting them reported 5,352 signals
    on a store holding two real feedback events — a 2,670x inflation that held
    the ranker in Phase 3 on a model trained entirely on zero labels.

    ``record_signal_batch`` writes exposures, which is right for the trainer and
    no longer moves the phase. So the feedback rows are inserted FIRST, with the
    exposures after them. The order is load-bearing:
    ``test_active_model_but_below_200_signals_stays_phase2`` scrubs
    ``WHERE id > 199`` to land on exactly 199 signals, and that only means 199
    COUNTED signals if the low ids are the feedback rows.
    """
    conn = open_conn(db)
    for i in range(n_queries * per_query):
        conn.execute(
            "INSERT INTO learning_signals (profile_id, query, fact_id, "
            " signal_type, value, created_at) "
            "VALUES (?, 'q', ?, 'legacy_feedback', 1.0, '2026-01-01')",
            (profile_id, f"fb-{i:05d}"),
        )
    conn.commit()
    for q in range(n_queries):
        record_signal_batch(
            conn,
            make_batch(profile_id=profile_id,
                       query_id=f"q-{q:04d}",
                       query_text=f"q{q}",
                       n_candidates=per_query),
        )
    conn.close()


def test_phase1_on_clean_db(tmp_path):
    db = make_db_with_migrations(tmp_path)
    phase = _compute_ranker_phase(
        "p1", learning_db_path=Path(db._db_path),
    )
    assert phase["phase"] == 1
    assert phase["model_active"] is False
    assert phase["signals"] == 0
    assert phase["key"] == "baseline"
    assert phase["gates"] == {
        "rule_based_min_signals": 50,
        "ml_model_min_signals": 200,
        "ml_model_requires_verified_active_model": True,
    }


def test_phase_read_does_not_migrate_a_legacy_empty_database(tmp_path):
    """Status reads must leave an upgraded user's database untouched."""
    db_path = tmp_path / "legacy-learning.db"
    sqlite3.connect(db_path).close()

    phase = _compute_ranker_phase("p1", learning_db_path=db_path)

    assert phase["phase"] == 1
    assert phase["status"] == "missing_table"
    connection = sqlite3.connect(db_path)
    try:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'",
        ).fetchall()
    finally:
        connection.close()
    assert tables == []


def test_phase2_with_signals_no_model(tmp_path):
    db = make_db_with_migrations(tmp_path)
    # 100 signals — no model yet.
    _seed_signals(db, n_queries=10, per_query=10)
    phase = _compute_ranker_phase(
        "p1", learning_db_path=Path(db._db_path),
    )
    assert phase["phase"] == 2
    assert phase["model_active"] is False
    assert phase["signals"] == 100


def test_phase3_requires_active_and_verified_model(tmp_path, monkeypatch):
    from superlocalmemory.learning import ranker_retrain_legacy

    db = make_db_with_migrations(tmp_path)
    _seed_signals(db, n_queries=40, per_query=10)  # 400 signals
    # The warning fires once per process; another test may have fired it first.
    monkeypatch.setattr(ranker_retrain_legacy, "_warned", False)
    with pytest.warns(DeprecationWarning, match="ranker_retrain_legacy is deprecated"):
        assert _retrain_ranker_impl(db._db_path, "p1")
    phase = _compute_ranker_phase(
        "p1", learning_db_path=Path(db._db_path),
    )
    assert phase["phase"] == 3
    assert phase["model_active"] is True
    assert phase["signals"] == 400

    # Tamper state_bytes → SHA mismatch → phase drops to 2.
    conn = sqlite3.connect(db._db_path)
    conn.execute(
        "UPDATE learning_model_state "
        "SET state_bytes = ? WHERE is_active = 1",
        (b"tampered", ),
    )
    conn.commit()
    conn.close()
    model_cache.invalidate("p1")
    phase2 = _compute_ranker_phase(
        "p1", learning_db_path=Path(db._db_path),
    )
    assert phase2["phase"] == 2
    assert phase2["model_active"] is False


def test_active_model_but_below_200_signals_stays_phase2(tmp_path):
    """Phase 3 requires BOTH active model AND >=200 signals."""
    db = make_db_with_migrations(tmp_path)
    _seed_signals(db, n_queries=40, per_query=10)
    assert _retrain_ranker_impl(db._db_path, "p1")

    # Scrub some signals to drop below 200 threshold.
    conn = sqlite3.connect(db._db_path)
    conn.execute(
        "DELETE FROM learning_signals WHERE id > ?",
        (199,),  # keep ids 1..199 — exactly 199 rows
    )
    conn.commit()
    conn.close()

    phase = _compute_ranker_phase(
        "p1", learning_db_path=Path(db._db_path),
    )
    assert phase["phase"] == 2
    assert phase["signals"] == 199
