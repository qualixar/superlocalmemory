# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Optional Prometheus/OpenTelemetry export. Guarded by try/except — no hard dependency."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("superlocalmemory.optimize.metrics.exporters")


def get_prometheus_exporter() -> Any | None:
    """Return a Prometheus exporter if prometheus_client is installed, else None.

    Prometheus port: 9091 (NOT 8765, NOT 8766).
    """
    try:
        from prometheus_client import Counter, Gauge, start_http_server  # noqa: F401
        return _PrometheusExporter()
    except ImportError:
        return None


class _PrometheusExporter:
    """Thin wrapper around prometheus_client for SLM Optimize metrics."""

    def __init__(self) -> None:
        from prometheus_client import Counter, Gauge, start_http_server

        self._cache_hits = Counter(
            "slm_cache_hits_total", "Total cache hits"
        )
        self._cache_misses = Counter(
            "slm_cache_misses_total", "Total cache misses"
        )
        self._tokens_saved = Counter(
            "slm_tokens_saved_total", "Total tokens saved (cache + compress)"
        )
        self._compress_runs = Counter(
            "slm_compress_runs_total", "Total compression runs"
        )
        self._cache_size = Gauge(
            "slm_cache_size_bytes", "Cache size in bytes"
        )
        self._started = False

    def start(self, port: int = 9091) -> None:
        """Start Prometheus HTTP server on the given port."""
        if self._started:
            return
        try:
            from prometheus_client import start_http_server
            start_http_server(port)
            self._started = True
            logger.info("Prometheus exporter started on port %d", port)
        except Exception as exc:
            logger.warning("Failed to start Prometheus exporter: %s", exc)

    def update(self, snapshot: dict[str, Any]) -> None:
        """Update Prometheus gauges/counters from a metrics snapshot dict."""
        if not self._started:
            return
        try:
            self._cache_hits._value.set(snapshot.get("hits", 0))
            self._cache_misses._value.set(snapshot.get("misses", 0))
            self._tokens_saved._value.set(
                snapshot.get("tokens_saved_input", 0)
                + snapshot.get("tokens_saved_output", 0)
                + snapshot.get("tokens_saved_compress", 0)
            )
            self._compress_runs._value.set(snapshot.get("compress_runs", 0))
            self._cache_size.set(snapshot.get("cache_size_bytes", 0))
        except Exception as exc:
            logger.debug("Prometheus update failed: %s", exc)
