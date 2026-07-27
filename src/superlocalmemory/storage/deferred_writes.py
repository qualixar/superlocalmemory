# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com

"""Best-effort, coalesced deferred writes that must NOT block a read path.

Recall must be read-only on its hot path.  Historically the entity resolver
stamped ``canonical_entities.last_seen`` inline during recall (an ``UPDATE``
that takes the write lock), so recall waited behind writers — the root of the
"recall is 8 s" regression.  ``last_seen`` is consumed ONLY by the dashboard
(entities / graph "last seen" columns); it never feeds recall ranking, so it
can be written a moment later with zero quality loss.

This module records such touches in memory (instant, lock-free for the caller)
and flushes them from a single background thread in small coalesced batches.
Bursts for the same entity collapse to one UPDATE per flush.  Failures are
swallowed — bookkeeping must never raise into a recall/ingest caller.

This is deliberately small and self-contained; it is the seed of the wider
single-writer queue (see WRITE-QUEUE-PLAN.md).
"""
from __future__ import annotations

import atexit
import queue
import threading

_FLUSH_INTERVAL_S = 2.0


# ---------------------------------------------------------------------------
# General best-effort background writer (seed of the single-writer queue).
# For NON-ESSENTIAL bookkeeping writes that must never block a read/recall
# path (access logging, activation-cache warming, etc.). Fire-and-forget:
# jobs are dropped under extreme backpressure rather than blocking a caller.
# Substantive/durable writes (remember, materialize) do NOT use this — they
# get the durable single-writer queue (see WRITE-QUEUE-PLAN.md Stage 3).
# ---------------------------------------------------------------------------

_BG_MAXSIZE = 20000
_bg_queue: "queue.Queue" = queue.Queue(maxsize=_BG_MAXSIZE)
_bg_started = False
_bg_start_lock = threading.Lock()
_bg_stop = threading.Event()
_bg_thread: threading.Thread | None = None


def _bg_run(work_queue: "queue.Queue", stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            fn = work_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        try:
            fn()
        except Exception:
            # Best-effort: bookkeeping must never crash the writer thread.
            pass
        finally:
            _bg_queue.task_done()


def _ensure_bg_thread() -> None:
    global _bg_queue, _bg_started, _bg_stop, _bg_thread
    if _bg_thread is not None and _bg_thread.is_alive():
        return
    with _bg_start_lock:
        if _bg_thread is not None and _bg_thread.is_alive():
            return
        _bg_queue = queue.Queue(maxsize=_BG_MAXSIZE)
        _bg_stop = threading.Event()
        _bg_thread = threading.Thread(
            target=_bg_run,
            args=(_bg_queue, _bg_stop),
            name="slm-bg-writer",
            daemon=True,
        )
        _bg_thread.start()
        _bg_started = True


def submit_background(fn) -> None:
    """Run *fn* on the shared background writer. Fire-and-forget, best-effort.

    Recall/read paths call this instead of writing inline, so they never wait
    on the write lock.  Under extreme backpressure the job is dropped (the
    write was non-essential bookkeeping).
    """
    _ensure_bg_thread()
    try:
        _bg_queue.put_nowait(fn)
    except queue.Full:
        pass


def _shutdown_background_writer(timeout: float) -> None:
    """Stop the shared bookkeeping worker and wait for its thread to exit."""
    global _bg_started, _bg_thread
    with _bg_start_lock:
        thread = _bg_thread
        if thread is None:
            return
        _bg_stop.set()
    if thread is not threading.current_thread():
        thread.join(timeout=timeout)
    if not thread.is_alive():
        with _bg_start_lock:
            if _bg_thread is thread:
                _bg_thread = None
                _bg_started = False


class DeferredLastSeen:
    """Coalescing background flusher for canonical_entities.last_seen."""

    def __init__(self, db, interval_s: float = _FLUSH_INTERVAL_S) -> None:
        self._db = db
        self._interval = interval_s
        self._pending: dict[tuple[str, str], str] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._stopped = False
        self._thread = threading.Thread(
            target=self._run, name="slm-deferred-lastseen", daemon=True
        )
        self._thread.start()

    def touch(self, entity_id: str, profile_id: str, ts: str) -> None:
        """Record a last_seen update. Instant, coalesced, never blocks."""
        with self._lock:
            self._pending[(entity_id, profile_id)] = ts

    def flush(self) -> int:
        """Write all pending updates now. Returns count. Best-effort."""
        with self._lock:
            if not self._pending:
                return 0
            batch = self._pending
            self._pending = {}
        try:
            # One transaction => one write-lock acquisition for the whole
            # batch, off the recall/ingest thread.
            with self._db.transaction():
                for (entity_id, profile_id), ts in batch.items():
                    self._db.execute(
                        "UPDATE canonical_entities SET last_seen = ? "
                        "WHERE entity_id = ? AND profile_id = ?",
                        (ts, entity_id, profile_id),
                    )
            return len(batch)
        except Exception:
            # Best-effort: drop this batch rather than raise into a caller.
            return 0

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self.flush()

    @property
    def is_stopped(self) -> bool:
        return self._stopped

    def stop(self, timeout: float = 3.0) -> None:
        """Flush then join the owner thread so it cannot outlive its database."""
        self._stop.set()
        self.flush()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=timeout)
        self._stopped = not self._thread.is_alive()


_registry: dict[int, DeferredLastSeen] = {}
_registry_lock = threading.Lock()


def get_deferred_last_seen(db) -> DeferredLastSeen:
    """Return the process-wide DeferredLastSeen flusher for *db* (lazy singleton)."""
    key = id(db)
    with _registry_lock:
        writer = _registry.get(key)
        if writer is None or writer.is_stopped:
            writer = DeferredLastSeen(db)
            _registry[key] = writer
        return writer


def shutdown_deferred_writes(timeout: float = 3.0) -> None:
    """Stop every deferred-write worker owned by this process.

    This is intentionally idempotent: service shutdown and pytest cleanup can
    both call it, and a later caller can lazily create fresh workers.
    """
    with _registry_lock:
        writers = list(_registry.values())
        _registry.clear()
    for writer in writers:
        writer.stop(timeout=timeout)
    _shutdown_background_writer(timeout)


atexit.register(shutdown_deferred_writes)


__all__ = [
    "DeferredLastSeen",
    "get_deferred_last_seen",
    "shutdown_deferred_writes",
    "submit_background",
]
