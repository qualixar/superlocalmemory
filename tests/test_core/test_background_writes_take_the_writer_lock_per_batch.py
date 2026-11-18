# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""WS-7 writer-lock audit: every 4.2.30 background writer takes the writer lock around ONE short
write, never around a loop or a slow call.

``storage/write_lock.py``: every writer of a store file takes the file's lock before it opens a
write transaction, so in-process writers never meet SQLite's BUSY. Three 4.2.30 passes wrote
through their own raw connection and skipped it (receipt and need-turn retention, exposure
retention, the workflow shadow's results), so a save could meet their transaction and retry; the
signing pass took the lock once for up to five batches. Each is now one lock hold per write.

How these tests see it: a trace on the pass's own connection records, for every write statement
(BEGIN, INSERT, UPDATE, DELETE, COMMIT), whether this thread then holds the file's lock, and for
every id-selecting read, that it does not.
"""

from __future__ import annotations

import sqlite3
import types
from datetime import UTC, datetime, timedelta

from superlocalmemory.storage.write_lock import get_write_lock
from tests.test_learning.test_exposure_retention import stores as exposure_stores  # noqa: F401

WRITES = ("BEGIN", "INSERT", "UPDATE", "DELETE", "COMMIT")


class Trace:
    """What the connections of one module did, and whether the file's lock was held."""

    def __init__(self, monkeypatch, module, db_path) -> None:
        self.lock = get_write_lock(db_path)
        self.writes: list[tuple[str, bool]] = []
        self.reads: list[tuple[str, bool]] = []
        real_connect = sqlite3.connect

        def connect(*args, **kwargs):
            conn = real_connect(*args, **kwargs)
            conn.set_trace_callback(self._seen)
            return conn

        shim = types.SimpleNamespace(connect=connect, Error=sqlite3.Error,
                                     OperationalError=sqlite3.OperationalError)
        monkeypatch.setattr(module, "sqlite3", shim)

    def _seen(self, statement: str) -> None:
        head = statement.lstrip().split(None, 1)[0].upper() if statement.strip() else ""
        held = self.lock._is_owned()
        if head in WRITES:
            self.writes.append((head, held))
        elif head == "SELECT" and "LIMIT" in statement.upper():
            self.reads.append((statement[:40], held))

    def assert_writes_held_and_reads_free(self) -> None:
        assert self.writes, "the pass wrote nothing: the test proves nothing"
        assert all(held for _, held in self.writes), [w for w in self.writes if not w[1]]
        assert not any(held for _, held in self.reads), "an id read ran inside the lock"


# -- receipt and need-turn retention ------------------------------------------------------------

NOW = datetime.now(UTC)


def test_receipt_retention_locks_each_batch_not_the_loop(tmp_path, monkeypatch) -> None:
    from superlocalmemory.outcomes import receipt_retention as rr
    from tests.test_core.test_coverage_backfill import _db
    from tests.test_outcomes import test_receipt_retention as base

    store = _db(tmp_path / "memory.db")
    fact = base._fact(store, "the cache key includes the profile")
    monkeypatch.setattr(rr, "BATCH", 2)
    old = NOW - timedelta(days=rr.KEEP_PULL_DAYS + 5)
    for i in range(5):
        base._pull(store, tmp_path, old - timedelta(minutes=i), fact=fact)
    trace = Trace(monkeypatch, rr, store.db_path)
    assert rr.prune(store.db_path, "default")["pull_deliveries"] == 5
    trace.assert_writes_held_and_reads_free()
    commits = [h for h, _ in trace.writes if h == "COMMIT"]
    assert len(commits) >= 3  # 2 + 2 + 1: one hold per batch, the lock let go between them


def test_need_turn_retention_locks_each_batch(tmp_path, monkeypatch) -> None:
    from superlocalmemory.core import need_gate as ng
    from superlocalmemory.outcomes import receipt_retention as rr
    from tests.test_core.test_coverage_backfill import _db

    store = _db(tmp_path / "memory.db")
    monkeypatch.setattr(rr, "BATCH", 2)
    old = NOW - timedelta(days=rr.KEEP_PULL_DAYS + 5)
    for i in range(5):
        decision = ng.Decision(f"turn_{i}", True, None, 0.5, None, False, 1.0, old)
        assert ng.log(store.db_path, decision, profile_id="default", session_id="c",
                      prompt_digest=None)
    trace = Trace(monkeypatch, rr, store.db_path)
    assert rr.prune(store.db_path, "default")["need_turns"] == 5
    trace.assert_writes_held_and_reads_free()


# -- exposure retention (learning.db) -----------------------------------------------------------

def test_exposure_retention_locks_each_batch(exposure_stores, monkeypatch) -> None:  # noqa: F811
    from superlocalmemory.learning import exposure_retention as er
    from tests.test_learning import test_exposure_retention as base

    learning, memory = exposure_stores
    monkeypatch.setattr(er, "BATCH", 2)
    for i in range(5):
        base._exposure(learning, f"q{i}", hours=100)
    trace = Trace(monkeypatch, er, learning)
    assert er.prune(learning, memory, "default")["unlabellable"] == 5
    trace.assert_writes_held_and_reads_free()
    assert len([h for h, _ in trace.writes if h == "COMMIT"]) >= 3


# -- workflow shadow (learning.db) --------------------------------------------------------------

def test_the_workflow_shadow_computes_outside_the_lock_and_writes_inside(tmp_path,
                                                                         monkeypatch) -> None:
    from superlocalmemory.learning import workflows as wf

    miner = wf.WorkflowMiner(tmp_path / "learning.db")
    held_while_mining: list[bool] = []
    lock = get_write_lock(tmp_path / "learning.db")
    original = miner.mine_from_memories

    def mining(*args, **kwargs):
        held_while_mining.append(lock._is_owned())
        return original(*args, **kwargs)

    monkeypatch.setattr(miner, "mine_from_memories", mining)
    trace = Trace(monkeypatch, wf, tmp_path / "learning.db")
    memories = [{"content": f"Ran the tests and then pushed commit {i}",
                 "created_at": f"2026-11-0{1 + i % 5}T10:00:00+00:00"} for i in range(12)]
    miner.learn_from_memories("default", memories)
    assert held_while_mining == [False]
    trace.assert_writes_held_and_reads_free()
