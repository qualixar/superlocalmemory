# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A-14 (Fable audit): an answer the client abandoned is never recorded as delivered (D-47).

The hook daemon stops waiting at its budget, but the queued recall used to run on and write a
``hook_recall`` receipt with ``delivered = 1``: the agent got nothing and the ledger said it did,
and the already-shown check then hid those memories for the rest of the conversation. The client's
absolute deadline now travels through the queue; a request past it is cancelled before it runs, and
a recall that finishes past it writes no receipt.
"""

from __future__ import annotations

import sqlite3
import time
from types import SimpleNamespace

import pytest

from superlocalmemory.core import recall_deadline
from superlocalmemory.core.queue_consumer import QueueConsumer
from superlocalmemory.core.recall_queue import QueueCancelledError, QueueTimeoutError, RecallQueue
from superlocalmemory.outcomes import receipts


@pytest.fixture
def queue(tmp_path):
    q = RecallQueue(tmp_path / "q.db")
    yield q
    q.close()


def _enqueue(q, **kw):
    kw = {"query": "what did we decide", "limit_n": 5, "mode": "A", "agent_id": "hook_daemon",
          "session_id": "conv-1", "priority": "high", **kw}
    return q.enqueue(**kw)


# -- the queue carries the deadline ------------------------------------------------------------

def test_the_request_row_holds_the_clients_absolute_deadline(queue) -> None:
    before = time.time()
    rid = _enqueue(queue, deadline_s=4.0)
    row = queue._get_row(rid)
    assert before + 3.9 <= row["deadline_at"] <= time.time() + 4.1
    assert queue._get_row(_enqueue(queue, query="no deadline"))["deadline_at"] is None


def test_a_second_subscriber_extends_the_deadline_never_shortens_it(queue) -> None:
    rid = _enqueue(queue, deadline_s=1.0)
    first = queue._get_row(rid)["deadline_at"]
    assert _enqueue(queue, deadline_s=5.0) == rid
    assert queue._get_row(rid)["deadline_at"] > first + 3.0
    longest = queue._get_row(rid)["deadline_at"]
    _enqueue(queue, deadline_s=0.1)
    assert queue._get_row(rid)["deadline_at"] == longest


def test_a_queue_file_from_before_the_deadline_column_is_upgraded(tmp_path) -> None:
    path = tmp_path / "old.db"
    q = RecallQueue(path)
    q.close()
    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE recall_requests DROP COLUMN deadline_at")
    conn.commit()
    conn.close()
    upgraded = RecallQueue(path)
    try:
        assert upgraded._get_row(_enqueue(upgraded, deadline_s=2.0))["deadline_at"] is not None
    finally:
        upgraded.close()


# -- the consumer honours it -------------------------------------------------------------------

class _Pool:
    def __init__(self, seconds=0.0):
        self.seconds, self.calls, self.expired_at_end = seconds, [], None

    def recall(self, query, limit=10, session_id=""):
        self.calls.append(query)
        time.sleep(self.seconds)
        self.expired_at_end = recall_deadline.expired()
        return {"ok": True, "results": [{"fact_id": "f1"}], "query_id": "q1"}


def test_a_request_already_past_its_deadline_is_cancelled_and_never_run(queue) -> None:
    rid = _enqueue(queue, deadline_s=0.01)
    time.sleep(0.05)
    pool = _Pool()
    consumer = QueueConsumer(queue, pool)
    assert consumer._try_claim_and_process("high") is True
    assert pool.calls == []
    assert queue._get_row(rid)["cancelled"] == 1
    with pytest.raises(QueueCancelledError):
        queue.poll_result(rid, timeout_s=0.1)


def test_the_recall_sees_the_deadline_while_it_runs(queue) -> None:
    rid = _enqueue(queue, deadline_s=0.2)
    pool = _Pool(seconds=0.4)
    QueueConsumer(queue, pool)._try_claim_and_process("high")
    assert pool.expired_at_end is True and queue._get_row(rid)["completed"] == 1


def test_no_deadline_means_none_applies(queue) -> None:
    rid = _enqueue(queue)
    pool = _Pool(seconds=0.05)
    QueueConsumer(queue, pool)._try_claim_and_process("high")
    assert pool.expired_at_end is False and queue._get_row(rid)["completed"] == 1


def test_the_marker_does_not_leak_outside_the_recall() -> None:
    assert recall_deadline.expired() is False
    with recall_deadline.scope(time.time() - 1):
        assert recall_deadline.expired() is True
    assert recall_deadline.expired() is False
    with recall_deadline.scope(None):
        assert recall_deadline.expired() is False


# -- the adapter writes no receipt for an abandoned answer ---------------------------------------

def _adapter(tmp_path, monkeypatch, seconds):
    from superlocalmemory.server.unified_daemon import EngineRecallAdapter
    from tests.test_core.test_coverage_backfill import _db

    sent: list = []
    monkeypatch.setattr(receipts, "submit_pull", sent.append)
    adapter = EngineRecallAdapter(SimpleNamespace(
        config=SimpleNamespace(base_dir=tmp_path), db=_db(tmp_path / "memory.db"),
        profile_id="default"))

    def slow(query, limit, session_id):
        time.sleep(seconds)
        return {"ok": True, "results": [{"fact_id": "a", "score": 0.4}], "query_id": "q"}

    monkeypatch.setattr(adapter, "_recall", slow)
    monkeypatch.setattr(adapter, "_not_yet_shown", lambda out, sid: out)
    return adapter, sent


def test_an_abandoned_answer_writes_no_receipt_and_is_marked(tmp_path, monkeypatch) -> None:
    adapter, sent = _adapter(tmp_path, monkeypatch, 0.2)
    with recall_deadline.scope(time.time() + 0.05):
        out = adapter.recall("q", 3, "conv-9")
    assert sent == [] and out["ok"] is False and out["error"] == "deadline_passed"


def test_an_answer_in_time_still_writes_its_receipt(tmp_path, monkeypatch) -> None:
    adapter, sent = _adapter(tmp_path, monkeypatch, 0.0)
    with recall_deadline.scope(time.time() + 5):
        out = adapter.recall("q", 3, "conv-9")
    assert len(sent) == 1 and out["ok"] is True


def test_with_no_deadline_the_receipt_is_written_as_before(tmp_path, monkeypatch) -> None:
    adapter, sent = _adapter(tmp_path, monkeypatch, 0.0)
    adapter.recall("q", 3, "conv-9")
    assert len(sent) == 1


def test_abandoned_memories_are_not_hidden_from_the_next_prompt(tmp_path, monkeypatch) -> None:
    """The harm: no receipt means ``shown_in_session`` never lists them, so the next prompt can
    still deliver them."""
    adapter, sent = _adapter(tmp_path, monkeypatch, 0.1)
    with recall_deadline.scope(time.time() + 0.01):
        adapter.recall("q", 3, "conv-9")
    assert sent == []


# -- the whole path: hook daemon budget -> queue -> consumer -> adapter -------------------------------

def test_the_probe_a_slow_recall_the_client_gave_up_on_records_nothing(
        tmp_path, monkeypatch, queue) -> None:
    adapter, sent = _adapter(tmp_path, monkeypatch, 0.6)
    consumer = QueueConsumer(queue, adapter)
    rid = _enqueue(queue, deadline_s=0.2, stall_timeout_s=0.2)
    consumer.start()
    try:
        with pytest.raises(QueueTimeoutError):
            queue.poll_result(rid, timeout_s=0.2)  # the hook daemon gives up: the hook gets {}
        time.sleep(1.0)  # the consumer finishes the recall anyway
    finally:
        consumer.stop()
    assert sent == []  # the old code reached the receipt point here


def test_the_hook_daemon_passes_its_wait_as_the_deadline(tmp_path, monkeypatch) -> None:
    from superlocalmemory.hooks import hook_daemon

    seen: dict = {}

    class FakeQueue:
        def enqueue(self, **kw):
            seen.update(kw)
            return "r-1"

        def poll_result(self, request_id, *, timeout_s):
            return {"ok": True, "results": []}

    daemon = hook_daemon.HookDaemon.__new__(hook_daemon.HookDaemon)
    daemon._queue, daemon._adapter = FakeQueue(), None
    monkeypatch.setattr("superlocalmemory.hooks.auto_recall_hook._switched_off", lambda: False)
    daemon._process_request({"prompt": "what did we decide about the schema", "session_id": "s"})
    assert seen["deadline_s"] == seen["stall_timeout_s"] > 0
