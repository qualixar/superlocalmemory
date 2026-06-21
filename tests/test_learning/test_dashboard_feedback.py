# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Regression tests for FeedbackCollector.record_dashboard_feedback.

Issues #53 / #59: the HTTP routes in ``server/routes/learning.py`` (the
dashboard thumbs-up/thumbs-down/pin and dwell handlers) called
``feedback.record_dashboard_feedback(...)`` — a method that did not exist on
``FeedbackCollector``. Every dashboard feedback write therefore raised
``AttributeError`` (caught by the route's ``except``, so no lock leak — but
the feature was entirely dead). These tests lock the method's contract:

  1. The method exists and persists a row, returning its id.
  2. Each dashboard vocabulary term maps to the documented (signal_type,
     value) pair.
  3. The raw query is hashed, never stored verbatim.
  4. Missing memory_id is rejected (returns None, no row).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.learning.feedback import (
    _DASHBOARD_SIGNAL_MAP,
    FeedbackCollector,
)


@pytest.fixture()
def collector(tmp_path: Path) -> FeedbackCollector:
    return FeedbackCollector(tmp_path / "learning.db")


def _rows(collector: FeedbackCollector) -> list[sqlite3.Row]:
    conn = sqlite3.connect(str(collector._db_path))
    conn.row_factory = sqlite3.Row
    try:
        return list(conn.execute("SELECT * FROM learning_feedback"))
    finally:
        conn.close()


def test_method_exists() -> None:
    # The exact regression: the attribute must be present and callable.
    assert callable(getattr(FeedbackCollector, "record_dashboard_feedback", None))


def test_records_and_returns_rowid(collector: FeedbackCollector) -> None:
    row_id = collector.record_dashboard_feedback(
        memory_id="fact-123", query="why is X", feedback_type="thumbs_up",
        profile_id="p1",
    )
    assert isinstance(row_id, int)
    rows = _rows(collector)
    assert len(rows) == 1
    assert rows[0]["fact_id"] == "fact-123"
    assert rows[0]["profile_id"] == "p1"


@pytest.mark.parametrize("feedback_type", sorted(_DASHBOARD_SIGNAL_MAP))
def test_each_dashboard_type_maps_correctly(
    collector: FeedbackCollector, feedback_type: str,
) -> None:
    expected_signal, expected_value = _DASHBOARD_SIGNAL_MAP[feedback_type]
    collector.record_dashboard_feedback(
        memory_id="m1", feedback_type=feedback_type,
    )
    row = _rows(collector)[0]
    assert row["signal_type"] == expected_signal
    assert row["signal_value"] == pytest.approx(expected_value)


def test_unknown_type_falls_back_to_neutral(collector: FeedbackCollector) -> None:
    collector.record_dashboard_feedback(memory_id="m1", feedback_type="banana")
    row = _rows(collector)[0]
    assert row["signal_type"] == "user_correction"
    assert row["signal_value"] == pytest.approx(0.5)


def test_query_is_hashed_not_stored_raw(collector: FeedbackCollector) -> None:
    secret = "this is the user's private query text"
    collector.record_dashboard_feedback(
        memory_id="m1", query=secret, feedback_type="pin",
    )
    row = _rows(collector)[0]
    assert row["query_hash"] is not None
    assert secret not in (row["query_hash"] or "")
    assert len(row["query_hash"]) == 16  # SHA-256[:16]


def test_empty_query_yields_null_hash(collector: FeedbackCollector) -> None:
    collector.record_dashboard_feedback(memory_id="m1", feedback_type="pin")
    assert _rows(collector)[0]["query_hash"] is None


def test_missing_memory_id_rejected(collector: FeedbackCollector) -> None:
    assert collector.record_dashboard_feedback(memory_id="", feedback_type="pin") is None
    assert _rows(collector) == []


def test_blank_profile_defaults(collector: FeedbackCollector) -> None:
    collector.record_dashboard_feedback(
        memory_id="m1", feedback_type="thumbs_down", profile_id="",
    )
    assert _rows(collector)[0]["profile_id"] == "default"
