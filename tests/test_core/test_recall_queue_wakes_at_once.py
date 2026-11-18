# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""WS-7: the per-prompt hook path does not sleep through its own work.

Profile (4.2.30, store copy, a prompt every 4 s): of 1.70 s p50 per prompt, 0.24 s went to the
queue consumer finishing its idle sleep before it noticed the request (its sleep grows to
1 s while idle) and 0.27 s to the client's poll finishing its own sleep after the answer was
ready. Both are polling in the same process; an in-process bell wakes them. Other processes
still find work by polling, so nothing is lost: the bell only shortens the wait.
"""

from __future__ import annotations

import time

import pytest

from superlocalmemory.core import queue_consumer as qc
from superlocalmemory.core.queue_consumer import QueueConsumer
from superlocalmemory.core.recall_queue import QueueTimeoutError, RecallQueue


class Pool:
    def __init__(self, work_s: float = 0.0) -> None:
        self.work_s = work_s

    def recall(self, query: str, limit: int = 10, session_id: str = "") -> dict:
        time.sleep(self.work_s)
        return {"ok": True, "results": [{"fact_id": "f1"}]}


def _enqueue(queue: RecallQueue, text: str = "what is the release branch") -> str:
    return queue.enqueue(query=text, limit_n=10, mode="a", agent_id="t", session_id="s",
                         priority="high", stall_timeout_s=5.0, deadline_s=5.0)


@pytest.fixture
def slow_idle(monkeypatch):
    """An idle consumer that, left to polling, would not look for 5 s."""
    monkeypatch.setattr(qc, "_POLL_BACKOFF_START_S", 5.0)
    monkeypatch.setattr(qc, "_POLL_BACKOFF_MAX_S", 5.0)


def test_a_request_wakes_an_idle_consumer_in_the_same_process(tmp_path, slow_idle) -> None:
    path = tmp_path / "q.db"
    consumer = QueueConsumer(queue=RecallQueue(path), pool=Pool())
    consumer.start()
    client = RecallQueue(path)  # the hook daemon holds its own instance of the same file
    try:
        time.sleep(0.3)  # the consumer is now inside its idle wait
        started = time.perf_counter()
        result = client.poll_result(_enqueue(client), timeout_s=4.0)
        elapsed = time.perf_counter() - started
        assert result["ok"] is True
        assert elapsed < 1.0, f"the answer took {elapsed:.2f}s; an idle sleep was not cut short"
    finally:
        consumer.stop()


def test_an_answer_reaches_a_waiting_client_when_it_is_ready(tmp_path) -> None:
    path = tmp_path / "q.db"
    consumer = QueueConsumer(queue=RecallQueue(path), pool=Pool(work_s=0.8))
    consumer.start()
    client = RecallQueue(path)
    try:
        time.sleep(0.2)
        started = time.perf_counter()
        client.poll_result(_enqueue(client), timeout_s=4.0)
        elapsed = time.perf_counter() - started
        # polling alone noticed it at 1.15 s (its schedule: .05 .1 .2 .3 .5 after the start)
        assert 0.8 <= elapsed < 1.1, elapsed
    finally:
        consumer.stop()


def test_stopping_does_not_wait_out_an_idle_sleep(tmp_path, slow_idle) -> None:
    consumer = QueueConsumer(queue=RecallQueue(tmp_path / "q.db"), pool=Pool())
    consumer.start()
    time.sleep(0.3)
    started = time.perf_counter()
    consumer.stop()
    assert time.perf_counter() - started < 1.0


def test_a_client_still_times_out_with_no_consumer(tmp_path) -> None:
    client = RecallQueue(tmp_path / "q.db")
    started = time.perf_counter()
    with pytest.raises(QueueTimeoutError):
        client.poll_result(_enqueue(client), timeout_s=0.3)
    assert 0.3 <= time.perf_counter() - started < 0.8


def test_another_process_is_still_found_by_polling(tmp_path, monkeypatch) -> None:
    """A request written through a path the bell never saw (another process, same file)."""
    path = tmp_path / "q.db"
    monkeypatch.setattr(qc, "_POLL_BACKOFF_START_S", 0.05)
    monkeypatch.setattr(qc, "_POLL_BACKOFF_MAX_S", 0.1)
    consumer = QueueConsumer(queue=RecallQueue(path), pool=Pool())
    consumer.start()
    client = RecallQueue(path)
    try:
        client._ring = lambda: None  # a process that shares no memory with the consumer
        rid = _enqueue(client)
        assert client.poll_result(rid, timeout_s=3.0)["ok"] is True
    finally:
        consumer.stop()


def test_a_wakeup_that_comes_before_the_wait_is_not_lost(tmp_path) -> None:
    queue = RecallQueue(tmp_path / "q.db")
    seen = queue.work_generation()
    _enqueue(queue)  # arrives between the consumer's empty claim and its wait
    started = time.perf_counter()
    assert queue.wait_for_work(seen, timeout_s=5.0) is True
    assert time.perf_counter() - started < 0.2


def test_two_queue_files_do_not_wake_each_other(tmp_path) -> None:
    a, b = RecallQueue(tmp_path / "a.db"), RecallQueue(tmp_path / "b.db")
    seen = b.work_generation()
    _enqueue(a)
    assert b.work_generation() == seen
    assert b.wait_for_work(seen, timeout_s=0.1) is False


def test_the_same_file_by_two_spellings_shares_one_bell(tmp_path) -> None:
    (tmp_path / "sub").mkdir()
    a = RecallQueue(tmp_path / "q.db")
    b = RecallQueue(tmp_path / "sub" / ".." / "q.db")
    seen = a.work_generation()
    _enqueue(b)
    assert a.work_generation() != seen


def test_a_waiting_client_sleeps_between_looks(tmp_path) -> None:
    """The bell shortens a sleep; it must not turn the wait into a spin."""
    client = RecallQueue(tmp_path / "q.db")
    looks = []
    real = client._get_row
    client._get_row = lambda rid: (looks.append(1), real(rid))[1]
    with pytest.raises(QueueTimeoutError):
        client.poll_result(_enqueue(client), timeout_s=0.5)
    assert len(looks) <= 8, f"{len(looks)} looks in 0.5 s: the wait is spinning"


def test_an_idle_consumer_sleeps_between_looks(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(qc, "_POLL_BACKOFF_START_S", 0.2)
    monkeypatch.setattr(qc, "_POLL_BACKOFF_MAX_S", 0.2)
    queue = RecallQueue(tmp_path / "q.db")
    claims = []
    real = queue.claim_pending
    queue.claim_pending = lambda **kw: (claims.append(1), real(**kw))[1]
    consumer = QueueConsumer(queue=queue, pool=Pool())
    consumer.start()
    time.sleep(1.0)
    consumer.stop()
    assert len(claims) <= 16, f"{len(claims)} claims in 1 s: the idle wait is spinning"
