# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""Tests for MetricsPersistence (optimize/metrics/persistence.py)."""

import threading
import time
from pathlib import Path

import pytest

from superlocalmemory.optimize.metrics.counters import MetricsCollector
from superlocalmemory.optimize.metrics.persistence import MetricsPersistence
from superlocalmemory.optimize.storage.db import CacheDB
import superlocalmemory.optimize.metrics.persistence as _persistence_mod


@pytest.fixture(autouse=True)
def _reset_singleton():
    MetricsCollector._instance = None
    yield
    MetricsCollector._instance = None


@pytest.fixture
def _db(tmp_path):
    return CacheDB(tmp_path / "llmcache.db")


def test_flush_then_load_roundtrip(_db):
    """Flush to tmp db, reset collector, load, verify fields match."""
    mc = MetricsCollector.get_instance()
    mc.on_hit(100, 50)
    mc.on_hit(200, 80)
    mc.on_miss()
    mc.on_compress(1000, 700)
    mc.on_eviction()
    mc.record_latency(5.0)

    mp = MetricsPersistence()
    mp.flush(mc, _db)

    # Reset and reload
    mc.reset()
    snap_before = mc.snapshot()
    assert snap_before.hits == 0

    mp.load(mc, _db)
    snap_after = mc.snapshot()
    assert snap_after.hits == 2  # 2 on_hit calls
    assert snap_after.misses == 1
    assert snap_after.tokens_saved_input == 300
    assert snap_after.tokens_saved_output == 130
    assert snap_after.evictions == 1
    assert snap_after.compress_runs == 1


def test_load_empty_db(_db):
    """Loading from empty db doesn't crash."""
    mc = MetricsCollector.get_instance()
    mp = MetricsPersistence()
    mp.load(mc, _db)
    snap = mc.snapshot()
    assert snap.hits == 0


def test_flush_preserves_across_resets(_db):
    """Multiple flush+load cycles preserve accumulated state."""
    mc = MetricsCollector.get_instance()
    mp = MetricsPersistence()

    # First session
    mc.on_hit(100, 0)
    mp.flush(mc, _db)

    # Simulate restart
    MetricsCollector._instance = None
    mc2 = MetricsCollector.get_instance()
    mp.load(mc2, _db)
    assert mc2.snapshot().hits == 1

    # Second session adds more
    mc2.on_hit(50, 0)
    mp.flush(mc2, _db)

    # Simulate another restart
    MetricsCollector._instance = None
    mc3 = MetricsCollector.get_instance()
    mp.load(mc3, _db)
    assert mc3.snapshot().hits == 2
    assert mc3.snapshot().tokens_saved_input == 150


# ---------------------------------------------------------------------------
# F-002: Background flush thread coverage
# ---------------------------------------------------------------------------

def test_start_background_flush_actually_flushes(_db):
    """Background thread flushes to DB at least once within 3 seconds."""
    original_interval = _persistence_mod._FLUSH_INTERVAL_SECONDS
    _persistence_mod._FLUSH_INTERVAL_SECONDS = 0.1  # flush every 100ms
    try:
        mc = MetricsCollector.get_instance()
        mc.on_hit(100, 50)
        mp = MetricsPersistence()
        mp.start_background_flush(mc, _db)
        assert mp._thread is not None
        assert mp._thread.is_alive()
        time.sleep(0.5)  # allow at least one flush cycle
        mp.stop_background_flush()
        # Verify data landed in DB
        MetricsCollector._instance = None
        mc2 = MetricsCollector.get_instance()
        mp2 = MetricsPersistence()
        mp2.load(mc2, _db)
        assert mc2.snapshot().hits == 1
    finally:
        _persistence_mod._FLUSH_INTERVAL_SECONDS = original_interval


def test_start_background_flush_idempotent(_db):
    """Calling start_background_flush twice must not spawn a second thread."""
    original_interval = _persistence_mod._FLUSH_INTERVAL_SECONDS
    _persistence_mod._FLUSH_INTERVAL_SECONDS = 0.1
    try:
        mc = MetricsCollector.get_instance()
        mp = MetricsPersistence()
        mp.start_background_flush(mc, _db)
        thread_1 = mp._thread
        mp.start_background_flush(mc, _db)  # second call — must be no-op
        assert mp._thread is thread_1, "Second call spawned a new thread"
        mp.stop_background_flush()
    finally:
        _persistence_mod._FLUSH_INTERVAL_SECONDS = original_interval


def test_stop_background_flush_with_no_thread_does_not_raise():
    """stop_background_flush on a fresh instance (no thread) must not raise."""
    mp = MetricsPersistence()
    assert mp._thread is None
    mp.stop_background_flush()  # must not raise
    assert mp._thread is None


def test_background_flush_error_does_not_crash_thread(_db, monkeypatch):
    """Exception inside flush() must be caught; thread keeps running."""
    original_interval = _persistence_mod._FLUSH_INTERVAL_SECONDS
    _persistence_mod._FLUSH_INTERVAL_SECONDS = 0.1
    mc = MetricsCollector.get_instance()
    mp = MetricsPersistence()
    flush_count = [0]
    original_flush = mp.flush

    def _flaky_flush(collector, db):
        flush_count[0] += 1
        if flush_count[0] == 1:
            raise RuntimeError("simulated DB failure")
        original_flush(collector, db)

    mp.flush = _flaky_flush
    try:
        mp.start_background_flush(mc, _db)
        time.sleep(0.5)
        assert mp._thread is not None and mp._thread.is_alive(), "Thread died after flush error"
    finally:
        mp.stop_background_flush()
        _persistence_mod._FLUSH_INTERVAL_SECONDS = original_interval
