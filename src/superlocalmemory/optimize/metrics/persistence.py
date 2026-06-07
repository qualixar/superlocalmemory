# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Flush in-memory counters to llmcache.db. Load on startup to restore counters."""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

from superlocalmemory.optimize.metrics.counters import MetricsCollector
from superlocalmemory.optimize.storage.db import CacheDB, MetricsSnapshot

if TYPE_CHECKING:
    pass

logger = logging.getLogger("superlocalmemory.optimize.metrics.persistence")

_FLUSH_INTERVAL_SECONDS: float = 60.0


class MetricsPersistence:
    """Flush in-memory counters -> llmcache.db. Load on startup -> restore counters.

    Table: llmcache_metrics (16 columns — INTERFACE-CONTRACT §6).
    Single aggregate row (id=1). INSERT OR REPLACE pattern.
    NEVER reads or writes memory.db.
    """

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def flush(self, collector: MetricsCollector, db: CacheDB) -> None:
        """Flush current collector state to llmcache_metrics via CacheDB."""
        try:
            try:
                cache_size_bytes = db.db_size_bytes()
            except Exception:
                cache_size_bytes = 0
            try:
                cache_entry_count = db.entry_count()
            except Exception:
                cache_entry_count = 0
            snap = collector.snapshot(
                cache_size_bytes=cache_size_bytes,
                cache_entry_count=cache_entry_count,
            )
            db.metrics_flush(snap)
        except Exception as exc:
            logger.warning("MetricsPersistence.flush failed: %s", exc)

    def load(self, collector: MetricsCollector, db: CacheDB) -> None:
        """Load persisted counters from llmcache_metrics into the live collector.

        Runs at daemon startup to restore counters after restart.
        """
        try:
            snap = db.metrics_load()
            if snap.hits > 0 or snap.misses > 0:
                # Restore counters from persisted snapshot
                with collector._data_lock:
                    collector._hits = snap.hits
                    collector._misses = snap.misses
                    collector._calls_skipped = snap.calls_skipped
                    collector._tokens_saved_input = snap.tokens_saved_input
                    collector._tokens_saved_output = snap.tokens_saved_output
                    collector._tokens_saved_compress = snap.tokens_saved_compress
                    collector._evictions = snap.evictions
                    collector._latency_overhead_ms_sum = snap.latency_overhead_ms_sum
                    collector._latency_samples = snap.latency_samples
                    collector._compress_runs = snap.compress_runs
                    collector._compress_bytes_original = snap.compress_bytes_original
                    collector._compress_bytes_after = snap.compress_bytes_after
                    collector._cache_size_bytes = snap.cache_size_bytes
                    collector._cache_entry_count = snap.cache_entry_count
                logger.info(
                    "MetricsPersistence: restored %d hits, %d misses from llmcache.db",
                    snap.hits, snap.misses,
                )
        except Exception as exc:
            logger.warning("MetricsPersistence.load failed: %s", exc)

    def start_background_flush(
        self, collector: MetricsCollector, db: CacheDB
    ) -> None:
        """Start a background thread that flushes every 60s."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()

        def _loop() -> None:
            while not self._stop_event.is_set():
                if self._stop_event.wait(timeout=_FLUSH_INTERVAL_SECONDS):
                    return
                try:
                    self.flush(collector, db)
                except Exception as exc:
                    logger.warning("MetricsPersistence background flush error: %s", exc)

        self._thread = threading.Thread(
            target=_loop, name="slm-metrics-flush", daemon=True,
        )
        self._thread.start()

    def stop_background_flush(self) -> None:
        """Signal the background flush thread to stop."""
        self._stop_event.set()
        t = self._thread
        if t is not None:
            t.join(timeout=5.0)
        self._thread = None
