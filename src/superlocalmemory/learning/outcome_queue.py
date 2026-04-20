# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.21 — Stage 9 DASH-02

"""Background queue for producer-side engagement-reward recording.

**Why this module exists.** Before v3.4.21, ``EngagementRewardModel.record_recall``
had zero production callers. Every recall path (CLI, MCP, dashboard, daemon
HTTP) returned results without ever creating a ``pending_outcomes`` row. The
closed-loop learning pipeline therefore had no producer — its consumers
(PostToolUse hook, Stop hook, finalize_outcome, action_outcomes, retrain,
shadow, rollback) were all consuming an empty stream.

This module wires the producer as a non-blocking enqueue + background drain.
The I1 invariant is absolute — recall wall time must NOT regress. So we:

1. ``enqueue_recall`` is ``queue.put_nowait`` — microseconds.
2. A single daemon thread drains the queue and calls
   ``EngagementRewardModel.record_recall`` (one SQLite INSERT per recall).
3. Queue is bounded; a full queue drops the oldest entry and bumps a
   counter. Signal quality is NEVER load-bearing on recall correctness.

The worker lives for the daemon's lifetime and is stopped on shutdown
by ``unified_daemon.py``'s lifespan hook.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclass — the enqueue payload
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RecallEvent:
    """One recall event awaiting pending_outcomes persistence.

    Fields mirror ``EngagementRewardModel.record_recall`` arguments.
    ``session_id`` is REQUIRED for hook-based signal matching; if the
    caller can't produce one, pass a stable synthetic like
    ``f"cli:{os.getpid()}"`` or ``"dashboard:<profile>"``.
    """
    session_id: str
    profile_id: str
    query: str
    fact_ids: tuple[str, ...]
    query_id: str = ""


# ---------------------------------------------------------------------------
# Module state — bounded queue + counters
# ---------------------------------------------------------------------------

#: S9-DASH-02: cap at 1000 pending recalls. At ~50 ms per row drain, even
#: a full queue drains in 50 s. Bigger than realistic daemon burst (p99
#: recalls/sec < 20). Fuller queue → drop oldest, bump counter.
_MAX_QUEUE = 1000

_queue: "queue.Queue[RecallEvent]" = queue.Queue(maxsize=_MAX_QUEUE)
_worker_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()
_counters: dict[str, int] = {
    "recall_enqueued": 0,
    "recall_dropped_queue_full": 0,
    "recall_persisted": 0,
    "recall_persist_failed": 0,
    "recall_reaped": 0,
}
_counters_lock = threading.Lock()


def _bump(name: str, n: int = 1) -> None:
    with _counters_lock:
        _counters[name] = _counters.get(name, 0) + n


def get_counters() -> dict[str, int]:
    """Snapshot of queue counters for dashboards / tests."""
    with _counters_lock:
        return dict(_counters)


def queue_size() -> int:
    """Current queue depth (approximate, no lock)."""
    return _queue.qsize()


# ---------------------------------------------------------------------------
# Public API — called from the recall hot path
# ---------------------------------------------------------------------------

def enqueue_recall(event: RecallEvent) -> None:
    """Non-blocking enqueue for later ``record_recall`` persistence.

    Hot-path cost: one ``put_nowait`` — ~1 µs. Drops the event if the
    queue is full; never raises. Signal quality is not load-bearing on
    recall correctness (S9-DASH-02 contract).
    """
    if not isinstance(event, RecallEvent):
        return
    if not event.session_id or not event.profile_id:
        # S9-DASH-02: session_id is mandatory — hooks key by it.
        # If the caller can't name a session, we silently drop: this
        # is a recall whose outcome cannot match to a signal anyway.
        return
    try:
        _queue.put_nowait(event)
        _bump("recall_enqueued")
    except queue.Full:
        # Drop oldest-first so newer recalls always make it in.
        try:
            _queue.get_nowait()
        except queue.Empty:
            pass
        try:
            _queue.put_nowait(event)
            _bump("recall_dropped_queue_full")
        except queue.Full:  # pragma: no cover — defensive
            _bump("recall_dropped_queue_full")


# ---------------------------------------------------------------------------
# Worker — persists to pending_outcomes
# ---------------------------------------------------------------------------

def _drain_once(memory_db_path: Path, max_batch: int = 50) -> int:
    """Drain up to ``max_batch`` events, persisting each via
    ``EngagementRewardModel.record_recall``. Returns count persisted.
    """
    from superlocalmemory.learning.reward import EngagementRewardModel

    model = EngagementRewardModel(memory_db_path)
    persisted = 0
    try:
        for _ in range(max_batch):
            try:
                event = _queue.get_nowait()
            except queue.Empty:
                break
            try:
                model.record_recall(
                    profile_id=event.profile_id,
                    session_id=event.session_id,
                    recall_query_id=event.query_id or "",
                    fact_ids=list(event.fact_ids),
                    query_text=event.query,
                )
                _bump("recall_persisted")
                persisted += 1
            except Exception as exc:
                _bump("recall_persist_failed")
                logger.debug(
                    "outcome_queue: record_recall failed: %s (session=%s)",
                    exc, event.session_id,
                )
    finally:
        try:
            model.close()
        except Exception:
            pass
    return persisted


#: S9-DASH-02: reaper cadence — force-finalize pending_outcomes older
#: than this so CLI/dashboard recalls (no Stop hook) still land in
#: action_outcomes with a neutral reward. One hour keeps slow-moving
#: interactive sessions alive while still draining abandoned recalls.
_REAP_INTERVAL_S = 300.0  # check every 5 minutes
_REAP_AGE_MS = 3_600_000  # 1 hour


def _reap_stale(memory_db_path: Path) -> int:
    """Force-finalize pending_outcomes older than ``_REAP_AGE_MS``.

    Uses ``EngagementRewardModel.reap_stale`` which computes the label
    from whatever signals accumulated (``0.5`` if none, which is the
    intended neutral for CLI/dashboard recalls without hook coverage).
    """
    try:
        from superlocalmemory.learning.reward import EngagementRewardModel
        model = EngagementRewardModel(memory_db_path)
        try:
            return int(model.reap_stale(older_than_ms=_REAP_AGE_MS))
        finally:
            try:
                model.close()
            except Exception:
                pass
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("reap_stale failed: %s", exc)
        return 0


def _worker_loop(memory_db_path: Path, interval_s: float) -> None:
    logger.info(
        "outcome_queue worker started (db=%s interval=%.2fs)",
        memory_db_path, interval_s,
    )
    import time as _time
    next_reap = _time.monotonic() + _REAP_INTERVAL_S
    while not _stop_event.wait(interval_s):
        try:
            _drain_once(memory_db_path)
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("outcome_queue drain crashed: %s", exc)
        # Periodic reaper for CLI/dashboard outcomes that no Stop hook
        # will ever finalize. Runs OFF the drain path so a busy queue
        # doesn't starve the reaper.
        now = _time.monotonic()
        if now >= next_reap:
            try:
                reaped = _reap_stale(memory_db_path)
                if reaped:
                    logger.info(
                        "outcome_queue reaper: finalized %d stale rows",
                        reaped,
                    )
                _bump("recall_reaped", reaped)
            except Exception:  # pragma: no cover
                pass
            next_reap = now + _REAP_INTERVAL_S
    # Final drain on graceful shutdown.
    try:
        _drain_once(memory_db_path, max_batch=1000)
    except Exception:
        pass
    logger.info("outcome_queue worker stopped")


def start_worker(memory_db_path: Path, *, interval_s: float = 0.25) -> None:
    """Start the drain thread (idempotent)."""
    global _worker_thread
    if _worker_thread is not None and _worker_thread.is_alive():
        return
    _stop_event.clear()
    _worker_thread = threading.Thread(
        target=_worker_loop,
        args=(Path(memory_db_path), interval_s),
        name="slm-outcome-queue",
        daemon=True,
    )
    _worker_thread.start()


def stop_worker(*, timeout_s: float = 2.0) -> int:
    """Signal the worker to stop and wait up to ``timeout_s`` for flush."""
    global _worker_thread
    _stop_event.set()
    if _worker_thread is not None and _worker_thread.is_alive():
        _worker_thread.join(timeout=timeout_s)
    remaining = _queue.qsize()
    _worker_thread = None
    return remaining


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _reset_for_testing() -> None:
    """TEST-ONLY: drain queue and zero counters. Never called in prod."""
    global _worker_thread
    _stop_event.set()
    if _worker_thread is not None and _worker_thread.is_alive():
        _worker_thread.join(timeout=1.0)
    _worker_thread = None
    while True:
        try:
            _queue.get_nowait()
        except queue.Empty:
            break
    with _counters_lock:
        for k in _counters:
            _counters[k] = 0
    _stop_event.clear()
