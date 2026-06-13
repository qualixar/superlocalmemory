# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""Tests for MetricsCollector (optimize/metrics/counters.py)."""

import threading

import pytest

from superlocalmemory.optimize.metrics.counters import MetricsCollector, get_metrics


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the singleton between tests."""
    MetricsCollector._instance = None
    yield
    MetricsCollector._instance = None


def test_collector_singleton():
    """Two calls to get_instance() return same object."""
    a = MetricsCollector.get_instance()
    b = MetricsCollector.get_instance()
    assert a is b


def test_on_hit_increments():
    mc = MetricsCollector.get_instance()
    mc.on_hit(100, 50)
    mc.on_hit(200, 80)
    snap = mc.snapshot()
    assert snap.hits == 2
    assert snap.tokens_saved_input == 300
    assert snap.tokens_saved_output == 130


def test_on_miss_increments():
    mc = MetricsCollector.get_instance()
    mc.on_miss()
    mc.on_miss()
    mc.on_miss()
    snap = mc.snapshot()
    assert snap.misses == 3


def test_on_compress_increments():
    mc = MetricsCollector.get_instance()
    mc.on_compress(1000, 700)
    mc.on_compress(500, 400)
    snap = mc.snapshot()
    assert snap.compress_runs == 2
    assert snap.compress_bytes_original == 1500
    assert snap.compress_bytes_after == 1100


def test_on_eviction_increments():
    mc = MetricsCollector.get_instance()
    mc.on_eviction()
    mc.on_eviction()
    snap = mc.snapshot()
    assert snap.evictions == 2


def test_record_latency():
    mc = MetricsCollector.get_instance()
    mc.record_latency(5.0)
    mc.record_latency(10.0)
    snap = mc.snapshot()
    assert snap.latency_overhead_ms_sum == 15.0
    assert snap.latency_samples == 2


def test_record_latency_negative_ignored():
    mc = MetricsCollector.get_instance()
    mc.record_latency(-1.0)
    snap = mc.snapshot()
    assert snap.latency_samples == 0


def test_snapshot_fields_complete():
    """Snapshot has all 16 fields."""
    mc = MetricsCollector.get_instance()
    snap = mc.snapshot()
    from superlocalmemory.optimize.storage.db import MetricsSnapshot
    expected_fields = {f for f in MetricsSnapshot.__dataclass_fields__}
    import dataclasses
    snap_dict = dataclasses.asdict(snap)
    for field_name in expected_fields:
        assert field_name in snap_dict, f"Missing field: {field_name}"


def test_reset():
    mc = MetricsCollector.get_instance()
    mc.on_hit(100, 50)
    mc.on_miss()
    mc.reset()
    snap = mc.snapshot()
    assert snap.hits == 0
    assert snap.misses == 0
    assert snap.tokens_saved_input == 0


def test_thread_safety():
    """Concurrent increments don't lose updates."""
    mc = MetricsCollector.get_instance()
    n_threads = 10
    n_increments = 100
    barrier = threading.Barrier(n_threads)

    def _worker():
        barrier.wait()
        for _ in range(n_increments):
            mc.on_hit(1, 0)

    threads = [threading.Thread(target=_worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = mc.snapshot()
    assert snap.hits == n_threads * n_increments
    assert snap.tokens_saved_input == n_threads * n_increments


def test_get_metrics_returns_singleton():
    a = get_metrics()
    b = get_metrics()
    assert a is b


def test_m03_on_compress_uses_token_params() -> None:
    """M-03: on_compress accepts tokens_before/tokens_after (not bytes)."""
    mc = MetricsCollector.get_instance()
    mc.reset()
    mc.on_compress(tokens_before=100, tokens_after=70)
    snap = mc.snapshot()
    assert snap.compress_runs == 1
    assert snap.compress_bytes_original == 100
    assert snap.compress_bytes_after == 70
    assert snap.tokens_saved_compress == 30


def test_m03_on_compress_positional_args_still_work() -> None:
    """M-03: positional call (router pattern) still works after rename."""
    mc = MetricsCollector.get_instance()
    mc.reset()
    mc.on_compress(200, 150)
    snap = mc.snapshot()
    assert snap.compress_bytes_original == 200
    assert snap.tokens_saved_compress == 50
