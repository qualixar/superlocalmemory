# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Queue consumer — background loop that drains recall_queue.db.

Polls the recall queue for pending jobs, claims them atomically,
routes through pool.recall() (NEVER engine directly), and writes
results back. Runs as a daemon thread inside the SLM daemon process.

MEMORY SAFETY: This module must NEVER import MemoryEngine. All recall
goes through WorkerPool which manages the recall_worker subprocess.
The engine lives only in that subprocess — not in this process.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Protocol

logger = logging.getLogger(__name__)

_MAX_RECEIVES = 3
_HIGH_PRIORITY = "high"
_LOW_PRIORITY = "low"

_POLL_BACKOFF_START_S = 0.02
_POLL_BACKOFF_MAX_S = 1.0
_POLL_BACKOFF_FACTOR = 1.5
_CLEANUP_INTERVAL_ITERATIONS = 500

#: Workers one consumer may run (routing only: a recall is CPU-bound, so more than a few
#: concurrent ones only queue inside the machine).
MAX_WORKERS = 4
#: What the daemon runs when nothing says otherwise: two sessions served at once (WS-7).
DEFAULT_DAEMON_WORKERS = 2
WORKERS_ENV = "SLM_HOOK_RECALL_WORKERS"


def configured_workers() -> int:
    """The daemon's worker count: ``SLM_HOOK_RECALL_WORKERS``, else two; 1 to ``MAX_WORKERS``."""
    try:
        wanted = int(os.environ.get(WORKERS_ENV, DEFAULT_DAEMON_WORKERS))
    except ValueError:
        wanted = DEFAULT_DAEMON_WORKERS
    return max(1, min(wanted, MAX_WORKERS))


class RecallPoolProtocol(Protocol):
    def recall(self, query: str, limit: int = 10, session_id: str = "") -> dict: ...


class QueueConsumer:
    """Drains recall_queue.db by routing jobs through WorkerPool.

    Lifecycle: start() begins ``workers`` daemon threads that poll the queue.
    stop() signals them to exit and joins them.

    Each worker claims one job at a time, and a job is claimed by exactly one
    worker (the claim is one atomic write). With ``workers=1`` (the default for a
    caller that says nothing) jobs run strictly in turn; the daemon runs two, so
    two sessions are not served one after the other (WS-7). Identical requests are
    still one execution through the queue dedup: 5 identical requests become 1
    execution, 5 results.
    """

    def __init__(
        self,
        queue: Any,
        pool: RecallPoolProtocol,
        max_receives: int = _MAX_RECEIVES,
        workers: int = 1,
    ) -> None:
        self._queue = queue
        self._pool = pool
        self._max_receives = max_receives
        self.workers = max(1, min(int(workers), MAX_WORKERS))
        self._running = False
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        self._stop_event.clear()
        self._running = True
        self._threads = [
            threading.Thread(target=self._poll_loop, daemon=True,
                             name=f"slm-queue-consumer-{i}")
            for i in range(self.workers)
        ]
        for thread in self._threads:
            thread.start()
        logger.info("QueueConsumer started (%d worker%s)", self.workers,
                    "" if self.workers == 1 else "s")

    def stop(self) -> None:
        if not self._running:
            return
        self._stop_event.set()
        self._running = False
        ring = getattr(self._queue, "_ring", None)
        if callable(ring):
            ring()  # an idle consumer is asleep in the bell: wake it to see the stop
        for thread in self._threads:
            thread.join(timeout=5.0)
        self._threads = []
        logger.info("QueueConsumer stopped")

    def _poll_loop(self) -> None:
        backoff = _POLL_BACKOFF_START_S
        iteration = 0

        while not self._stop_event.is_set():
            seen = self._queue_generation()  # before the look: a request after it is not lost
            processed = self._try_claim_and_process(_HIGH_PRIORITY)
            if not processed:
                processed = self._try_claim_and_process(_LOW_PRIORITY)

            if processed:
                backoff = _POLL_BACKOFF_START_S
            else:
                self._idle_wait(seen, backoff)
                backoff = min(backoff * _POLL_BACKOFF_FACTOR, _POLL_BACKOFF_MAX_S)

            iteration += 1
            if iteration % _CLEANUP_INTERVAL_ITERATIONS == 0:
                self._cleanup_completed()

    def _queue_generation(self) -> int | None:
        """The queue's in-process doorbell reading, or None for a queue without one."""
        reader = getattr(self._queue, "work_generation", None)
        return reader() if callable(reader) else None

    def _idle_wait(self, seen: int | None, backoff: float) -> None:
        """Sleep ``backoff`` seconds; a request written in this process, or stop(), ends it early."""
        waiter = getattr(self._queue, "wait_for_work", None)
        if seen is None or not callable(waiter):
            self._stop_event.wait(timeout=backoff)
            return
        waiter(seen, backoff)

    def _try_claim_and_process(self, priority: str) -> bool:
        try:
            claimed = self._queue.claim_pending(
                priority=priority,
                stall_timeout_s=25.0,
            )
        except Exception as exc:
            logger.warning("Queue claim failed: %s", exc)
            return False

        if claimed is None:
            return False

        request_id = claimed["request_id"]
        received = claimed["received"]
        query = claimed.get("query", "")
        limit_n = claimed.get("limit_n", 10)
        session_id = claimed.get("session_id", "")

        if received >= self._max_receives:
            try:
                self._queue.mark_dead_letter(
                    request_id, reason="max_receives_exceeded",
                )
            except Exception as exc:
                logger.warning("DLQ mark failed for %s: %s", request_id, exc)
            return True

        deadline_at = claimed.get("deadline_at")
        if deadline_at is not None and time.time() >= deadline_at:
            # The client stopped waiting before we started: nobody will read the answer, and
            # running it would record a delivery that never happened (A-14).
            self._cancel_abandoned(request_id)
            return True

        from superlocalmemory.core import recall_deadline

        with recall_deadline.scope(deadline_at):
            result_json = self._execute_recall(query, limit_n, session_id)

        try:
            n = self._queue.complete(
                request_id, received=received, result_json=result_json,
            )
            if n == 0:
                logger.info("Fenced out on complete: %s (received=%d)", request_id, received)
        except Exception as exc:
            logger.warning("Queue complete failed for %s: %s", request_id, exc)

        return True

    def _cancel_abandoned(self, request_id: str) -> None:
        try:
            self._queue.cancel(request_id)
        except Exception as exc:
            logger.warning("Queue cancel failed for %s: %s", request_id, exc)

    def _cleanup_completed(self) -> None:
        try:
            self._queue._conn.execute(
                "DELETE FROM recall_requests "
                "WHERE (completed = 1 OR cancelled = 1 OR dead_letter = 1) "
                "AND created_at < ?",
                (time.time() - 3600,),
            )
        except Exception as exc:
            logger.debug("Queue cleanup failed: %s", exc)

    def _execute_recall(self, query: str, limit: int, session_id: str) -> str:
        """Run one queued recall. Every queued recall is the per-prompt hook
        loading context — not a question — so the answer check is skipped:
        nothing is judged, sent or billed. The daemon's pool
        (``EngineRecallAdapter``) runs the recall on this same thread, so the
        marker reaches it; a pool that crossed a process would have to carry
        the request on its message instead."""
        from superlocalmemory.core.answer_check_scope import skip_answer_check

        try:
            with skip_answer_check():
                result = self._pool.recall(query, limit=limit, session_id=session_id)
            return json.dumps(result, default=str)
        except Exception as exc:
            logger.warning("pool.recall failed: %s", exc)
            return json.dumps({"ok": False, "error": "recall_failed"})
