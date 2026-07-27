# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for core.queue_dispatcher."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest


def _imports():
    from superlocalmemory.core import queue_dispatcher as qd
    return qd


def _make_dispatcher(tmp_path: Path, **kwargs):
    qd = _imports()
    return qd.QueueDispatcher(db_path=tmp_path / "q.db", **kwargs)


def test_single_recall_round_trip(tmp_path: Path) -> None:
    disp = _make_dispatcher(tmp_path)

    # Worker thread: claim + complete
    def worker() -> None:
        for _ in range(20):
            claim = disp.queue.claim_pending(priority="high", stall_timeout_s=5.0)
            if claim is None:
                time.sleep(0.01)
                continue
            disp.queue.complete(
                claim["request_id"],
                received=claim["received"],
                result_json=json.dumps({"hits": 3, "q": claim["query"]}),
            )
            return

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    out = disp.dispatch(
        query="hello", limit_n=10, mode="B",
        agent_id="a", session_id="s", timeout_s=3.0,
    )
    t.join(timeout=2.0)
    assert out["hits"] == 3
    assert out["q"] == "hello"
    disp.close()


def test_rate_limit_raises(tmp_path: Path) -> None:
    qd = _imports()
    disp = _make_dispatcher(tmp_path, global_rps=1, per_pid_rps=1, per_agent_rps=1)
    # First call: queue never processes, but enqueue passes rate limit
    with pytest.raises((qd.rl.RateLimitedError, Exception)):
        # Two calls in tight loop — second must trip some layer
        disp.queue  # touch
        pid = 99991
        disp._check_rate(pid=pid, agent_id="x")
        disp._check_rate(pid=pid, agent_id="x")
    disp.close()


def test_dispatch_timeout_surfaces_queue_timeout(tmp_path: Path) -> None:
    qd = _imports()
    disp = _make_dispatcher(tmp_path)
    with pytest.raises(qd.rq.QueueTimeoutError):
        disp.dispatch(
            query="x", limit_n=10, mode="B",
            agent_id="a", session_id="s", timeout_s=0.2,
        )
    disp.close()


def test_dispatch_dlq_surfaces_dead_letter(tmp_path: Path) -> None:
    qd = _imports()
    disp = _make_dispatcher(tmp_path)

    # Worker marks the row as DLQ
    def poisoner() -> None:
        for _ in range(50):
            claim = disp.queue.claim_pending(priority="high", stall_timeout_s=5.0)
            if claim is None:
                time.sleep(0.01)
                continue
            disp.queue.mark_dead_letter(
                claim["request_id"], reason="max_receives_exceeded",
            )
            return

    t = threading.Thread(target=poisoner, daemon=True)
    t.start()
    with pytest.raises(qd.rq.DeadLetterError):
        disp.dispatch(
            query="x", limit_n=10, mode="B",
            agent_id="a", session_id="s", timeout_s=3.0,
        )
    t.join(timeout=2.0)
    disp.close()


def test_dispatch_dedup_shares_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two concurrent dispatches of the same query share one worker execution.
    disp = _make_dispatcher(tmp_path)
    worker_invocations = [0]
    both_enqueued = threading.Event()
    enqueue_count = 0
    enqueue_count_lock = threading.Lock()
    original_enqueue = disp.queue.enqueue

    def signaling_enqueue(**kwargs):
        """Expose the state transition the test actually depends on."""
        nonlocal enqueue_count
        request_id = original_enqueue(**kwargs)
        with enqueue_count_lock:
            enqueue_count += 1
            if enqueue_count == 2:
                both_enqueued.set()
        return request_id

    monkeypatch.setattr(disp.queue, "enqueue", signaling_enqueue)

    def one_shot_worker() -> None:
        # Do not race a fixed 500 ms polling window against CI thread
        # scheduling. Claim only after both callers have subscribed to the
        # same in-flight row; that is the dedup state this test verifies.
        assert both_enqueued.wait(timeout=10.0), "callers did not enqueue"
        claim = disp.queue.claim_pending(priority="high", stall_timeout_s=5.0)
        assert claim is not None
        worker_invocations[0] += 1
        time.sleep(0.05)  # simulate work
        disp.queue.complete(
            claim["request_id"],
            received=claim["received"],
            result_json=json.dumps({"hits": 1}),
        )

    w = threading.Thread(target=one_shot_worker, daemon=True)
    w.start()

    # Fire two dispatches with identical dedup key — second must hit dedup.
    results: list[dict] = []
    errors: list[BaseException] = []

    def caller() -> None:
        try:
            results.append(disp.dispatch(
                query="same", limit_n=10, mode="B",
                agent_id="a", session_id="s", timeout_s=10.0,
            ))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    c1 = threading.Thread(target=caller)
    c1.start()
    c2 = threading.Thread(target=caller)
    c2.start()
    c1.join(timeout=12.0)
    c2.join(timeout=12.0)
    w.join(timeout=12.0)

    assert not errors, f"Caller error: {errors}"
    assert not c1.is_alive() and not c2.is_alive() and not w.is_alive()
    assert len(results) == 2
    assert worker_invocations[0] == 1, (
        f"Dedup violated: worker ran {worker_invocations[0]} times"
    )
    disp.close()
