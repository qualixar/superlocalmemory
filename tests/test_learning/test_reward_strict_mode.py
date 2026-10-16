# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The daemon's hook event writer needs the reward model to say when a write
failed (so the event stays spooled) and to wait as long as every other writer.
Existing callers keep the old fail-soft behaviour."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.learning.reward import EngagementRewardModel


def test_default_model_stays_fail_soft(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    sqlite3.connect(db).close()
    model = EngagementRewardModel(db)
    try:
        assert model.register_signal(outcome_id="o", signal_name="edit", signal_value=True) is False
        assert model.match_pending_for_fact_ids(session_id="s", fact_ids=["f"]) == []
        assert model.finalize_outcome(outcome_id="o") == model.FALLBACK_REWARD
    finally:
        model.close()


def test_strict_model_raises_database_errors(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    sqlite3.connect(db).close()
    model = EngagementRewardModel(db, raise_on_db_error=True)
    try:
        with pytest.raises(sqlite3.OperationalError):
            model.register_signal(outcome_id="o", signal_name="edit", signal_value=True)
        with pytest.raises(sqlite3.OperationalError):
            model.match_pending_for_fact_ids(session_id="s", fact_ids=["f"])
        with pytest.raises(sqlite3.OperationalError):
            model.finalize_outcome(outcome_id="o")
    finally:
        model.close()


def test_busy_timeout_is_configurable(tmp_path: Path) -> None:
    model = EngagementRewardModel(tmp_path / "m.db", busy_timeout_ms=10_000)
    try:
        with model._lock:
            assert model._get_conn().execute("PRAGMA busy_timeout").fetchone()[0] == 10_000
    finally:
        model.close()
