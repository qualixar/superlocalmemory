# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""WS-7: concurrent sessions are not served one at a time.

Profile (4.2.30, four agent sessions prompting at once): one consumer thread ran every hook
recall in turn, so a session's wait was the sum of the recalls queued before it; at four
sessions most prompts reached the daemon's 4 s budget and injected nothing. The engine already
serves concurrent recalls (the dashboard search runs them on a thread pool), so the consumer
runs a small fixed number of workers. Each request is still claimed by exactly one.
"""

from __future__ import annotations

import threading
import time

import pytest

from superlocalmemory.core.queue_consumer import MAX_WORKERS, QueueConsumer
from superlocalmemory.core.recall_queue import RecallQueue


class Pool:
    def __init__(self, work_s: float) -> None:
        self.work_s = work_s
        self.calls: list[str] = []
        self.threads: set[str] = set()
        self.active = 0
        self.peak = 0
        self._lock = threading.Lock()

    def recall(self, query: str, limit: int = 10, session_id: str = "") -> dict:
        with self._lock:
            self.calls.append(query)
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.threads.add(threading.current_thread().name)
        time.sleep(self.work_s)
        with self._lock:
            self.active -= 1
        return {"ok": True, "results": [{"fact_id": query}]}


def _enqueue(queue: RecallQueue, text: str) -> str:
    return queue.enqueue(query=text, limit_n=10, mode="a", agent_id="t", session_id=text,
                         priority="high", stall_timeout_s=5.0, deadline_s=5.0)


def _serve(tmp_path, workers: int, work_s: float, n: int):
    queue = RecallQueue(tmp_path / f"q{workers}.db")
    pool = Pool(work_s)
    consumer = QueueConsumer(queue=queue, pool=pool, workers=workers)
    consumer.start()
    client = RecallQueue(tmp_path / f"q{workers}.db")
    try:
        time.sleep(0.2)
        started = time.perf_counter()
        ids = [_enqueue(client, f"query number {i}") for i in range(n)]
        answers = [client.poll_result(rid, timeout_s=10.0) for rid in ids]
        return time.perf_counter() - started, answers, pool
    finally:
        consumer.stop()


def test_two_workers_serve_two_sessions_at_once(tmp_path) -> None:
    elapsed, answers, pool = _serve(tmp_path, workers=2, work_s=0.6, n=2)
    assert elapsed < 1.1, f"{elapsed:.2f}s: the two recalls ran one after the other"
    assert pool.peak == 2
    assert [a["results"][0]["fact_id"] for a in answers] == ["query number 0", "query number 1"]


def test_one_worker_is_the_old_behaviour(tmp_path) -> None:
    elapsed, _answers, pool = _serve(tmp_path, workers=1, work_s=0.4, n=2)
    assert elapsed >= 0.8 and pool.peak == 1


def test_every_request_is_served_exactly_once(tmp_path) -> None:
    _elapsed, answers, pool = _serve(tmp_path, workers=MAX_WORKERS, work_s=0.05, n=12)
    assert len(answers) == 12 and all(a["ok"] for a in answers)
    assert sorted(pool.calls) == sorted(f"query number {i}" for i in range(12))
    assert len(pool.threads) > 1


def test_the_worker_count_is_bounded(tmp_path) -> None:
    queue = RecallQueue(tmp_path / "q.db")
    assert QueueConsumer(queue=queue, pool=Pool(0), workers=0).workers == 1
    assert QueueConsumer(queue=queue, pool=Pool(0), workers=10_000).workers == MAX_WORKERS
    assert QueueConsumer(queue=queue, pool=Pool(0), workers=3).workers == 3
    assert QueueConsumer(queue=queue, pool=Pool(0)).workers == 1  # a caller that says nothing


def test_stop_ends_every_worker(tmp_path) -> None:
    before = {t.name for t in threading.enumerate()}
    consumer = QueueConsumer(queue=RecallQueue(tmp_path / "q.db"), pool=Pool(0), workers=3)
    consumer.start()
    time.sleep(0.2)
    assert len([t for t in threading.enumerate() if t.name.startswith("slm-queue-consumer")
                and t.name not in before]) == 3
    started = time.perf_counter()
    consumer.stop()
    assert time.perf_counter() - started < 2.0
    assert not [t for t in threading.enumerate() if t.name.startswith("slm-queue-consumer")
                and t.name not in before and t.is_alive()]
    assert not consumer.running


def test_start_twice_does_not_double_the_workers(tmp_path) -> None:
    before = {t.ident for t in threading.enumerate()}
    consumer = QueueConsumer(queue=RecallQueue(tmp_path / "q.db"), pool=Pool(0), workers=2)
    consumer.start()
    consumer.start()
    try:
        mine = [t for t in threading.enumerate()
                if t.name.startswith("slm-queue-consumer") and t.ident not in before]
        assert len(consumer._threads) == len(mine) == 2
    finally:
        consumer.stop()


@pytest.mark.parametrize("workers", [2, 3])
def test_a_failing_recall_does_not_stop_the_other_workers(tmp_path, workers) -> None:
    class Flaky(Pool):
        def recall(self, query, limit=10, session_id=""):
            if query.endswith("0"):
                raise RuntimeError("boom")
            return super().recall(query, limit, session_id)

    queue = RecallQueue(tmp_path / "q.db")
    consumer = QueueConsumer(queue=queue, pool=Flaky(0.05), workers=workers)
    consumer.start()
    client = RecallQueue(tmp_path / "q.db")
    try:
        bad, good = _enqueue(client, "query number 0"), _enqueue(client, "query number 1")
        assert client.poll_result(bad, timeout_s=5.0)["ok"] is False
        assert client.poll_result(good, timeout_s=5.0)["ok"] is True
    finally:
        consumer.stop()


def test_the_daemon_runs_two_workers_unless_told_otherwise(monkeypatch) -> None:
    from superlocalmemory.core import queue_consumer as qc

    monkeypatch.delenv(qc.WORKERS_ENV, raising=False)
    assert qc.configured_workers() == qc.DEFAULT_DAEMON_WORKERS == 2


@pytest.mark.parametrize("value,expected", [("1", 1), ("3", 3), ("0", 1), ("-4", 1),
                                            ("99", MAX_WORKERS), ("many", 2), ("", 2)])
def test_the_worker_count_can_be_set_and_is_clamped(monkeypatch, value, expected) -> None:
    from superlocalmemory.core import queue_consumer as qc

    monkeypatch.setenv(qc.WORKERS_ENV, value)
    assert qc.configured_workers() == expected


def test_the_daemon_builds_its_consumer_with_the_configured_workers() -> None:
    import inspect

    from superlocalmemory.server import unified_daemon

    source = inspect.getsource(unified_daemon)
    assert "workers=configured_workers()" in source
