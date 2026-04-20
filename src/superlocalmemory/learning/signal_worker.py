# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-02 §4.2

"""Background signal drain worker.

LLD reference: ``.backup/active-brain/lld/LLD-02-signal-pipeline-and-lightgbm.md``
Section 4.2 — moves signal writes off the recall hot path.

Contract (hard rules, enforced by tests):
    SW1 — Hot path never waits for disk.
    SW2 — Drop + counter on full queue, never raise.
    SW3 — Graceful flush ≤3 s on ``stop()``.
    SW4 — Connection is thread-local; never shared across threads.
"""

from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from superlocalmemory.learning import signals as _signals_mod
from superlocalmemory.learning.signals import (
    SignalBatch,
    bump_counter as _bump_counter,
    get_queue as _signals_get_queue,
    record_signal_batch,
)


def _current_queue() -> "queue.Queue[SignalBatch]":
    """Resolve the queue through the public ``signals.get_queue()``
    contract. Tests monkeypatching ``signals._Q`` still win because
    ``get_queue`` reads the attribute dynamically via ``sys.modules``.
    S8-ARC-03 (v3.4.22): no more private ``_Q`` reach-through.
    """
    return _signals_get_queue()

logger = logging.getLogger(__name__)


_DRAIN_BATCH_DEFAULT = 50
_DRAIN_INTERVAL_MS_DEFAULT = 250
_FLUSH_TIMEOUT_S_DEFAULT = 3.0


class SignalWorker:
    """Background drainer for the module-level signal queue.

    One instance per daemon. Creates its own thread and sqlite3 connection
    on ``start()``; the connection is thread-local (SW4).
    """

    def __init__(
        self,
        learning_db: str | Path,
        *,
        batch_size: int = _DRAIN_BATCH_DEFAULT,
        interval_ms: int = _DRAIN_INTERVAL_MS_DEFAULT,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if interval_ms < 0:
            raise ValueError("interval_ms must be >= 0")
        self._db_path = str(learning_db)
        self._batch_size = batch_size
        self._interval_s = interval_ms / 1000.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._conn_thread_id: int | None = None

    # --- public API ------------------------------------------------------

    def start(self) -> None:
        """Start the background drain thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        t = threading.Thread(
            target=self._run,
            name="slm-signal-worker",
            daemon=True,
        )
        self._thread = t
        t.start()

    def stop(self, *, timeout: float = _FLUSH_TIMEOUT_S_DEFAULT) -> int:
        """Stop the worker, flushing pending batches up to ``timeout`` seconds.

        Returns the number of batches dropped because they couldn't be flushed
        before the timeout (SW3). Never raises.
        """
        if self._thread is None:
            remaining = _drain_and_drop(log_prefix="no-thread")
            return remaining

        self._stop_event.set()
        self._thread.join(timeout=max(0.0, timeout))

        # After join, drain anything left and count it as drop-on-flush.
        remaining = _drain_and_drop(log_prefix="post-join")
        self._thread = None
        return remaining

    # --- internals -------------------------------------------------------

    def _open_threadlocal_conn(self) -> sqlite3.Connection:
        """Open the drain connection — SW4 threadlocal. Called once inside
        the thread's run loop. The caller retains ownership and closes it
        at shutdown.
        """
        conn = sqlite3.connect(
            self._db_path,
            isolation_level=None,
            timeout=10,
            check_same_thread=True,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA foreign_keys=ON")
        self._conn_thread_id = threading.get_ident()
        return conn

    def _run(self) -> None:
        """Main loop: drain up to ``batch_size`` every ``interval_s``."""
        try:
            conn = self._open_threadlocal_conn()
        except sqlite3.Error as exc:  # pragma: no cover — DB unavailable
            logger.error("signal_worker: cannot open DB: %s", exc)
            return

        try:
            while not self._stop_event.is_set():
                self._drain_once(conn)
                if self._stop_event.wait(self._interval_s):
                    break
            # Final drain on graceful stop.
            self._drain_once(conn)
        finally:
            try:
                conn.close()
            except sqlite3.Error:  # pragma: no cover
                pass

    def _drain_once(self, conn: sqlite3.Connection) -> int:
        """Drain up to ``batch_size`` batches from the queue.

        Returns the number of batches written. On OperationalError we retry
        each batch up to 3 times with backoff; persistent failures are
        dropped and counted (see LLD-02 §8 error matrix).
        """
        written = 0
        for _ in range(self._batch_size):
            try:
                batch = _current_queue().get_nowait()
            except queue.Empty:
                break

            ok = _write_with_retry(conn, batch)
            if ok:
                written += 1
            else:
                _bump_counter("signal_dropped_total")
        return written


def _write_with_retry(
    conn: sqlite3.Connection,
    batch: SignalBatch,
    *,
    attempts: int = 3,
) -> bool:
    """Try to write a batch; retry on operational errors with backoff.

    Returns True on success, False if dropped.
    """
    backoff_ms = 50
    for attempt in range(1, attempts + 1):
        try:
            record_signal_batch(conn, batch)
            return True
        except sqlite3.OperationalError as exc:
            logger.warning(
                "signal_worker: write attempt %d failed: %s", attempt, exc,
            )
            if attempt == attempts:
                return False
            time.sleep(backoff_ms / 1000.0)
            backoff_ms *= 2
        except sqlite3.Error as exc:  # pragma: no cover — defensive
            logger.error("signal_worker: non-retriable error: %s", exc)
            return False
        except Exception as exc:  # pragma: no cover — never propagate
            logger.error("signal_worker: unexpected: %s", exc)
            return False
    return False  # pragma: no cover — defensive


def _drain_and_drop(*, log_prefix: str = "") -> int:
    """Drain remaining batches from the queue; count as drop-on-flush.

    Used during shutdown when the worker could not flush in time (SW3).
    """
    remaining = 0
    while True:
        try:
            _current_queue().get_nowait()
        except queue.Empty:
            break
        remaining += 1
    if remaining:
        _bump_counter("signal_drop_on_flush_total", remaining)
        logger.info(
            "signal_worker: %s dropped %d unflushed batches",
            log_prefix, remaining,
        )
    return remaining


# ---------------------------------------------------------------------------
# Module-level singleton helpers (S8-SK-01 integration): let the daemon
# start/stop one SignalWorker without knowing the class internals. Callers
# in ``unified_daemon.lifespan`` use ``start(learning_db)`` and ``stop()``.
# ---------------------------------------------------------------------------

_WORKER_SINGLETON: SignalWorker | None = None
_WORKER_LOCK = threading.Lock()


def start(learning_db: "str | Path", **kwargs) -> SignalWorker:
    """Create-or-return the module-level SignalWorker; start its thread.

    Idempotent. Safe to call twice — the existing instance is returned
    and its ``start()`` is a no-op if the thread is already alive. The
    daemon calls this from its lifespan once per process.
    """
    global _WORKER_SINGLETON
    with _WORKER_LOCK:
        if _WORKER_SINGLETON is None:
            _WORKER_SINGLETON = SignalWorker(learning_db, **kwargs)
        _WORKER_SINGLETON.start()
        return _WORKER_SINGLETON


def stop(*, timeout: float = _FLUSH_TIMEOUT_S_DEFAULT) -> int:
    """Stop the module-level SignalWorker (if any); returns drop count."""
    global _WORKER_SINGLETON
    with _WORKER_LOCK:
        if _WORKER_SINGLETON is None:
            return 0
        worker = _WORKER_SINGLETON
    dropped = worker.stop(timeout=timeout)
    with _WORKER_LOCK:
        _WORKER_SINGLETON = None
    return dropped


def current() -> SignalWorker | None:
    """Return the current singleton worker (or ``None``). TEST helper."""
    with _WORKER_LOCK:
        return _WORKER_SINGLETON


__all__ = ("SignalWorker", "start", "stop", "current")
