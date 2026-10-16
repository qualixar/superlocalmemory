# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The daemon's recall -> pending_outcomes producer does not lose rows to a busy store.

Before 4.2.8 ``record_recall`` waited 500 ms for memory.db, swallowed the
"database is locked" and returned a sentinel; ``_drain_once`` then counted the
event as persisted. Under a long memory write the row was gone and the
counters said it was saved.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest

from superlocalmemory.learning import outcome_queue as oq
from tests.helpers.hook_event_db import bootstrap, query


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    return bootstrap(tmp_path / "memory.db")


@pytest.fixture(autouse=True)
def _empty_queue():
    while not oq._queue.empty():  # noqa: SLF001
        oq._queue.get_nowait()  # noqa: SLF001
    yield
    while not oq._queue.empty():  # noqa: SLF001
        oq._queue.get_nowait()  # noqa: SLF001


def _event(n: int = 1) -> oq.RecallEvent:
    return oq.RecallEvent(session_id="sess-1", profile_id="default", query=f"q{n}",
                          fact_ids=("f1",), query_id=f"qid-{n}")


def _hold_writer(db: Path, seconds: float) -> threading.Event:
    holder = sqlite3.connect(db, isolation_level=None, check_same_thread=False)
    holder.execute("BEGIN IMMEDIATE")
    done = threading.Event()

    def release() -> None:
        time.sleep(seconds)
        holder.execute("COMMIT")
        holder.close()
        done.set()

    threading.Thread(target=release, daemon=True).start()
    return done


def test_a_recall_waits_out_a_long_write_instead_of_vanishing(db: Path) -> None:
    oq._queue.put_nowait(_event())  # noqa: SLF001
    released = _hold_writer(db, 1.2)
    try:
        persisted = oq._drain_once(db)  # noqa: SLF001
    finally:
        released.wait(5)
    rows = query(db, "SELECT recall_query_id FROM pending_outcomes")
    assert persisted == 1 and [r["recall_query_id"] for r in rows] == ["qid-1"]


def test_a_failed_write_is_counted_as_failed_never_as_persisted(tmp_path: Path) -> None:
    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).close()  # no pending_outcomes table
    before = oq.get_counters()
    oq._queue.put_nowait(_event())  # noqa: SLF001
    assert oq._drain_once(empty) == 0  # noqa: SLF001
    after = oq.get_counters()
    assert after["recall_persisted"] == before["recall_persisted"]
    assert after["recall_persist_failed"] == before["recall_persist_failed"] + 1


def test_a_lock_that_outlasts_the_wait_keeps_the_event_queued(db: Path, monkeypatch) -> None:
    monkeypatch.setenv("SLM_DB_BUSY_TIMEOUT_MS", "100")
    oq._queue.put_nowait(_event())  # noqa: SLF001
    released = _hold_writer(db, 0.6)
    try:
        assert oq._drain_once(db) == 0  # noqa: SLF001
        assert oq.queue_size() == 1, "a locked store must not consume the event"
    finally:
        released.wait(5)
    assert oq._drain_once(db) == 1  # noqa: SLF001
    assert len(query(db, "SELECT * FROM pending_outcomes")) == 1
