# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-02 §6.4

"""TDD tests for ``learning/signal_worker.py``."""

from __future__ import annotations

import queue
import sqlite3
import threading
import time

import pytest

from superlocalmemory.learning import signals as signals_mod
from superlocalmemory.learning.signal_worker import (
    SignalWorker,
    _write_with_retry,
)
from superlocalmemory.learning.signals import (
    enqueue,
    get_counters,
    reset_counters,
    _drain_queue_for_tests,
    _Q as _SIGNAL_QUEUE,
)
from tests.test_learning._signal_fixtures import (
    make_db_with_migrations,
    make_batch,
)


@pytest.fixture(autouse=True)
def _clean_state():
    _drain_queue_for_tests()
    reset_counters()
    yield
    _drain_queue_for_tests()
    reset_counters()


# ---------------------------------------------------------------------------
# §6.4 test_flush_on_shutdown_persists_pending (SW3)
# ---------------------------------------------------------------------------


def test_flush_on_shutdown_persists_pending(tmp_path):
    db = make_db_with_migrations(tmp_path)
    worker = SignalWorker(db._db_path, batch_size=50, interval_ms=20)
    worker.start()
    try:
        for i in range(100):
            enqueue(make_batch(query_id=f"q-{i:04d}", n_candidates=2))
        # Wait for the queue to drain before asking worker to stop.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and _SIGNAL_QUEUE.qsize() > 0:
            time.sleep(0.02)
    finally:
        dropped = worker.stop(timeout=5.0)
    assert dropped == 0

    conn = sqlite3.connect(db._db_path)
    n = conn.execute(
        "SELECT COUNT(*) FROM learning_signals"
    ).fetchone()[0]
    conn.close()
    # 100 batches × 2 candidates = 200 rows expected.
    assert n == 200


# ---------------------------------------------------------------------------
# §6.4 test_shutdown_timeout_drops_rest_counted
# ---------------------------------------------------------------------------


def test_shutdown_timeout_drops_rest_counted(tmp_path, monkeypatch):
    # Keep queue large enough for the 10k we enqueue.
    big_q: "queue.Queue" = queue.Queue(maxsize=20000)
    monkeypatch.setattr(signals_mod, "_Q", big_q)
    reset_counters()

    db = make_db_with_migrations(tmp_path)
    # Tiny batch size so the worker can't keep up within 0.1 s.
    worker = SignalWorker(db._db_path, batch_size=1, interval_ms=10)

    # Pre-load without starting so nothing drains.
    for i in range(10_000):
        enqueue(make_batch(query_id=f"q-{i}", n_candidates=0))

    worker.start()
    # Immediately stop — most of the queue will remain.
    dropped = worker.stop(timeout=0.1)
    counters = get_counters()
    assert dropped >= 1
    assert counters["signal_drop_on_flush_total"] == dropped


# ---------------------------------------------------------------------------
# §6.4 test_threadlocal_connection_per_worker (SW4)
# ---------------------------------------------------------------------------


def test_threadlocal_connection_per_worker(tmp_path):
    db = make_db_with_migrations(tmp_path)
    worker = SignalWorker(db._db_path, batch_size=5, interval_ms=50)
    worker.start()
    try:
        for i in range(5):
            enqueue(make_batch(query_id=f"q-{i}", n_candidates=1))
        # Give the worker at least one tick to open its connection.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and worker._conn_thread_id is None:
            time.sleep(0.01)
    finally:
        worker.stop(timeout=3.0)

    # Confirm the thread id that opened the conn was the worker thread,
    # NOT the main thread.
    assert worker._conn_thread_id is not None
    assert worker._conn_thread_id != threading.get_ident()


# ---------------------------------------------------------------------------
# Retry path — operational error is retried, then succeeds.
# ---------------------------------------------------------------------------


class _RetryableConn:
    """Fails N times on record_signal_batch, then delegates."""

    def __init__(self, inner: sqlite3.Connection, n_failures: int) -> None:
        self._inner = inner
        self._n = n_failures

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __enter__(self):
        if self._n > 0:
            self._n -= 1
            raise sqlite3.OperationalError("transient lock")
        return self._inner.__enter__()

    def __exit__(self, *args):
        return self._inner.__exit__(*args)


def test_write_with_retry_eventually_succeeds(tmp_path):
    db = make_db_with_migrations(tmp_path)
    real = sqlite3.connect(db._db_path, isolation_level=None)
    wrapper = _RetryableConn(real, n_failures=2)
    batch = make_batch(n_candidates=1)
    ok = _write_with_retry(wrapper, batch, attempts=3)
    assert ok is True
    real.close()


def test_write_with_retry_gives_up(tmp_path):
    db = make_db_with_migrations(tmp_path)
    real = sqlite3.connect(db._db_path, isolation_level=None)
    wrapper = _RetryableConn(real, n_failures=5)
    batch = make_batch(n_candidates=1)
    ok = _write_with_retry(wrapper, batch, attempts=3)
    assert ok is False
    real.close()


# ---------------------------------------------------------------------------
# Double start is idempotent.
# ---------------------------------------------------------------------------


def test_double_start_is_idempotent(tmp_path):
    db = make_db_with_migrations(tmp_path)
    worker = SignalWorker(db._db_path, batch_size=5, interval_ms=50)
    worker.start()
    worker.start()  # second call should be a no-op.
    assert worker._thread is not None
    worker.stop(timeout=2.0)


def test_invalid_batch_size():
    with pytest.raises(ValueError):
        SignalWorker("x.db", batch_size=0)
    with pytest.raises(ValueError):
        SignalWorker("x.db", interval_ms=-1)


def test_stop_without_start_returns_zero(tmp_path):
    worker = SignalWorker(str(tmp_path / "x.db"))
    # No thread — drain-and-count path.
    assert worker.stop(timeout=0.1) == 0


def test_drain_once_drops_when_retry_exhausted(tmp_path, monkeypatch):
    """_drain_once bumps signal_dropped_total when _write_with_retry gives up."""
    from superlocalmemory.learning import signal_worker as worker_mod

    db = make_db_with_migrations(tmp_path)
    worker = SignalWorker(db._db_path, batch_size=2, interval_ms=10)
    # Preload one batch.
    enqueue(make_batch(query_id="q-drop", n_candidates=1))

    # Force _write_with_retry to always fail.
    monkeypatch.setattr(worker_mod, "_write_with_retry",
                         lambda conn, batch, attempts=3: False)
    # Open threadlocal conn manually (synchronous drain).
    conn = sqlite3.connect(db._db_path, isolation_level=None)
    reset_counters()
    written = worker._drain_once(conn)
    conn.close()
    assert written == 0
    assert get_counters()["signal_dropped_total"] >= 1


def test_stop_without_start_drains_pending(tmp_path, monkeypatch):
    big_q: "queue.Queue" = queue.Queue(maxsize=100)
    monkeypatch.setattr(signals_mod, "_Q", big_q)
    reset_counters()

    worker = SignalWorker(str(tmp_path / "x.db"))
    for i in range(5):
        enqueue(make_batch(query_id=f"q-{i}", n_candidates=0))
    dropped = worker.stop(timeout=0.1)
    assert dropped == 5
