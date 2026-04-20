# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — S9-DASH-02

"""Tests for the outcome-queue producer wiring + perf regression guard.

Contract (from the LLD and from Varun's spoken directive):

1. ``enqueue_recall`` is non-blocking — microseconds per call.
2. Adding ``enqueue_recall`` to the recall hot path must NOT regress
   recall wall time. Any developer running ``slm recall`` should see
   the same latency before and after the producer is wired.
3. Queue is bounded; overflow drops a row, not a crash.
4. ``record_recall`` is eventually called by the background worker
   for every enqueued event (given time + an open DB).
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from superlocalmemory.learning import outcome_queue
from superlocalmemory.learning.outcome_queue import (
    RecallEvent, enqueue_recall, get_counters, queue_size,
    start_worker, stop_worker, _reset_for_testing,
)


# ---------------------------------------------------------------------------
# Enqueue is microseconds — perf regression guard
# ---------------------------------------------------------------------------

def test_enqueue_recall_is_microseconds() -> None:
    """1000 enqueue calls must complete in <20 ms total (p95 per call
    ~2 µs). This is the gate that protects the recall hot path — if
    anyone adds I/O or a DB call here in a future change, this test
    fails and the regression is caught before Varun's laptop feels it.
    """
    _reset_for_testing()
    evt = RecallEvent(
        session_id="perf-test",
        profile_id="p",
        query="q",
        fact_ids=("f1", "f2", "f3"),
        query_id="qid-123",
    )
    t0 = time.perf_counter()
    for _ in range(1000):
        enqueue_recall(evt)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _reset_for_testing()

    # 20 ms for 1000 calls → 20 µs per call budget. Very generous.
    assert elapsed_ms < 20.0, (
        f"enqueue_recall too slow: {elapsed_ms:.2f}ms for 1000 calls "
        f"(budget 20ms total). Did someone add I/O to the hot path?"
    )


def test_enqueue_accepts_valid_event() -> None:
    _reset_for_testing()
    enqueue_recall(RecallEvent(
        session_id="s1", profile_id="p", query="q",
        fact_ids=("f1",),
    ))
    assert queue_size() == 1
    assert get_counters()["recall_enqueued"] == 1
    _reset_for_testing()


def test_enqueue_rejects_missing_session_id() -> None:
    """Events with empty session_id cannot match to hook signals later,
    so we drop them at enqueue time rather than creating an orphan
    pending_outcome."""
    _reset_for_testing()
    enqueue_recall(RecallEvent(
        session_id="", profile_id="p", query="q", fact_ids=("f",),
    ))
    assert queue_size() == 0
    _reset_for_testing()


def test_enqueue_rejects_missing_profile_id() -> None:
    _reset_for_testing()
    enqueue_recall(RecallEvent(
        session_id="s1", profile_id="", query="q", fact_ids=("f",),
    ))
    assert queue_size() == 0
    _reset_for_testing()


def test_enqueue_drops_oldest_on_overflow() -> None:
    """Queue has a cap — overflow drops oldest, never raises."""
    _reset_for_testing()
    original = outcome_queue._MAX_QUEUE
    # Swap in a tiny queue for this test.
    import queue as _q
    outcome_queue._queue = _q.Queue(maxsize=3)
    try:
        for i in range(5):
            enqueue_recall(RecallEvent(
                session_id=f"s{i}", profile_id="p",
                query="q", fact_ids=("f",),
            ))
        # qsize() should be capped.
        assert queue_size() <= 3
        # Drop counter bumped for the overflow events.
        assert get_counters()["recall_dropped_queue_full"] >= 2
    finally:
        outcome_queue._queue = _q.Queue(maxsize=original)
        _reset_for_testing()


# ---------------------------------------------------------------------------
# Worker drains to pending_outcomes
# ---------------------------------------------------------------------------

def _mk_schema(db_path: Path) -> None:
    """Apply minimal memory.db schema needed by EngagementRewardModel."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            profile_id TEXT PRIMARY KEY DEFAULT 'default'
        )
    """)
    conn.execute("INSERT OR IGNORE INTO profiles VALUES ('default')")
    conn.execute("INSERT OR IGNORE INTO profiles VALUES ('p')")
    # Minimal pending_outcomes DDL — mirrors M007.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pending_outcomes (
            outcome_id TEXT PRIMARY KEY,
            profile_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            recall_query_id TEXT NOT NULL DEFAULT '',
            fact_ids_json TEXT NOT NULL DEFAULT '[]',
            query_text_hash TEXT NOT NULL DEFAULT '',
            created_at_ms INTEGER NOT NULL,
            expires_at_ms INTEGER NOT NULL,
            signals_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'pending'
        )
    """)
    conn.commit()
    conn.close()


def test_worker_drains_enqueued_events(tmp_path: Path) -> None:
    """Full producer → drain → pending_outcomes contract."""
    _reset_for_testing()
    db_path = tmp_path / "memory.db"
    _mk_schema(db_path)

    enqueue_recall(RecallEvent(
        session_id="sess-x", profile_id="p",
        query="turboquant", fact_ids=("f1", "f2"),
        query_id="qid-1",
    ))
    assert queue_size() == 1

    # Drain manually (no worker thread needed for this test).
    persisted = outcome_queue._drain_once(db_path)
    assert persisted == 1

    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT session_id, profile_id, status FROM pending_outcomes"
    ).fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0][0] == "sess-x"
    assert rows[0][1] == "p"
    assert rows[0][2] == "pending"
    _reset_for_testing()


def test_worker_start_stop_is_idempotent(tmp_path: Path) -> None:
    """start_worker twice is safe; stop_worker without start is safe."""
    _reset_for_testing()
    db_path = tmp_path / "memory.db"
    _mk_schema(db_path)
    start_worker(db_path, interval_s=0.05)
    start_worker(db_path, interval_s=0.05)  # second call no-op
    remaining = stop_worker(timeout_s=1.0)
    assert remaining == 0
    # Calling stop again is safe.
    stop_worker(timeout_s=0.5)
    _reset_for_testing()
