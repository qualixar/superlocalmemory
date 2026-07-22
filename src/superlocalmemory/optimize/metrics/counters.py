# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Thread-safe atomic counters for all SLM Optimize operations."""

from __future__ import annotations

import threading
from typing import ClassVar

from superlocalmemory.optimize.storage.db import MetricsSnapshot


class MetricsCollector:
    """Thread-safe atomic in-memory counters. Daemon-scoped singleton.

    All increment methods acquire a threading.Lock. Each increment is
    O(1) and must add < 1 µs of latency on the hot path.

    Hook binding (from INTERFACE-CONTRACT §3):
        on_hit(tokens_saved_input, tokens_saved_output):  cache hit
        on_miss():                                         cache miss
        on_compress(tokens_before, tokens_after):          compress ran (word-count proxy)
        on_eviction():                                     entry expired/evicted
    """

    _instance: ClassVar[MetricsCollector | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self) -> None:
        self._data_lock = threading.Lock()
        self._hits: int = 0
        self._misses: int = 0
        self._calls_skipped: int = 0
        self._tokens_saved_input: int = 0
        self._tokens_saved_output: int = 0
        self._tokens_saved_compress: int = 0
        self._evictions: int = 0
        self._latency_overhead_ms_sum: float = 0.0
        self._latency_samples: int = 0
        self._compress_runs: int = 0
        self._lossy_compress_runs: int = 0
        self._compress_bytes_original: int = 0
        self._compress_bytes_after: int = 0
        self._cache_size_bytes: int = 0
        self._cache_entry_count: int = 0

    @classmethod
    def get_instance(cls) -> MetricsCollector:
        """Return the process-global singleton. Thread-safe."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def on_hit(self, tokens_saved_input: int = 0, tokens_saved_output: int = 0) -> None:
        """Record a cache hit with tokens saved."""
        with self._data_lock:
            self._hits += 1
            self._tokens_saved_input += max(0, tokens_saved_input)
            self._tokens_saved_output += max(0, tokens_saved_output)

    def on_miss(self) -> None:
        """Record a cache miss."""
        with self._data_lock:
            self._misses += 1

    def on_compress(self, tokens_before: int, tokens_after: int, lossy: bool = False) -> None:
        """Record a compression run. Arguments are word-count proxy estimates from _token_estimate().

        M-03: consistent naming — these are token estimates, not byte counts.
        Stored in compress_bytes_original/after fields for DB schema compat; unit is word-count.

        ``lossy`` distinguishes a Layer-2 lossy (LLMLingua prose) run from a
        lossless Layer-1/JSON-minify run; tracked in memory (not yet persisted).
        """
        with self._data_lock:
            self._compress_runs += 1
            if lossy:
                self._lossy_compress_runs += 1
            self._compress_bytes_original += max(0, tokens_before)
            self._compress_bytes_after += max(0, tokens_after)
            self._tokens_saved_compress += max(0, tokens_before - tokens_after)

    def lossy_compress_runs(self) -> int:
        """Session-scoped count of lossy (Layer-2) compression runs."""
        with self._data_lock:
            return self._lossy_compress_runs

    def on_eviction(self) -> None:
        """Record an eviction."""
        with self._data_lock:
            self._evictions += 1

    def increment_skipped_temperature(self) -> None:
        """C-04: record a cache skip due to non-zero temperature."""
        with self._data_lock:
            self._calls_skipped += 1

    def record_latency(self, ms: float) -> None:
        """Record latency overhead in milliseconds."""
        if ms < 0:
            return
        with self._data_lock:
            self._latency_overhead_ms_sum += ms
            self._latency_samples += 1

    def snapshot(self, cache_size_bytes: int | None = None, cache_entry_count: int | None = None) -> MetricsSnapshot:
        """Return a consistent read of all counters as a MetricsSnapshot."""
        with self._data_lock:
            import time
            return MetricsSnapshot(
                id=1,
                hits=self._hits,
                misses=self._misses,
                calls_skipped=self._calls_skipped,
                tokens_saved_input=self._tokens_saved_input,
                tokens_saved_output=self._tokens_saved_output,
                tokens_saved_compress=self._tokens_saved_compress,
                evictions=self._evictions,
                latency_overhead_ms_sum=self._latency_overhead_ms_sum,
                latency_samples=self._latency_samples,
                compress_runs=self._compress_runs,
                compress_bytes_original=self._compress_bytes_original,
                compress_bytes_after=self._compress_bytes_after,
                cache_size_bytes=cache_size_bytes if cache_size_bytes is not None else self._cache_size_bytes,
                cache_entry_count=cache_entry_count if cache_entry_count is not None else self._cache_entry_count,
                updated_at=time.time(),
            )

    def reset(self) -> None:
        """Reset all counters. FOR TESTS ONLY."""
        with self._data_lock:
            self._hits = 0
            self._misses = 0
            self._calls_skipped = 0
            self._tokens_saved_input = 0
            self._tokens_saved_output = 0
            self._tokens_saved_compress = 0
            self._evictions = 0
            self._latency_overhead_ms_sum = 0.0
            self._latency_samples = 0
            self._compress_runs = 0
            self._lossy_compress_runs = 0
            self._compress_bytes_original = 0
            self._compress_bytes_after = 0
            self._cache_size_bytes = 0
            self._cache_entry_count = 0


def get_metrics() -> MetricsCollector:
    """Return the process-global MetricsCollector singleton."""
    return MetricsCollector.get_instance()
