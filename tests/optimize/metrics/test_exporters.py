"""Coverage for metrics/exporters.py — mocks prometheus_client (optional dep)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


def _make_prometheus_mock() -> MagicMock:
    """Build a mock prometheus_client module with realistic attribute structure."""
    prom = MagicMock()
    counter_instance = MagicMock()
    counter_instance._value = MagicMock()
    prom.Counter.return_value = counter_instance

    gauge_instance = MagicMock()
    gauge_instance.set = MagicMock()
    prom.Gauge.return_value = gauge_instance
    return prom


# ---- get_prometheus_exporter ------------------------------------------------

def test_get_prometheus_exporter_import_error_returns_none() -> None:
    """prometheus_client not installed → returns None (default test env)."""
    from superlocalmemory.optimize.metrics.exporters import get_prometheus_exporter

    result = get_prometheus_exporter()
    assert result is None


def test_get_prometheus_exporter_returns_exporter_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    """With prometheus_client mocked → returns a _PrometheusExporter."""
    prom = _make_prometheus_mock()
    monkeypatch.setitem(sys.modules, "prometheus_client", prom)

    from superlocalmemory.optimize.metrics.exporters import get_prometheus_exporter

    result = get_prometheus_exporter()
    assert result is not None
    # Four Counters: cache_hits, cache_misses, tokens_saved, compress_runs
    assert prom.Counter.call_count == 4
    # One Gauge: cache_size_bytes
    assert prom.Gauge.call_count == 1


# ---- start() ----------------------------------------------------------------

def test_prometheus_exporter_start_calls_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """start() calls start_http_server and marks _started=True."""
    prom = _make_prometheus_mock()
    monkeypatch.setitem(sys.modules, "prometheus_client", prom)

    from superlocalmemory.optimize.metrics.exporters import get_prometheus_exporter

    exporter = get_prometheus_exporter()
    exporter.start(port=9999)

    assert exporter._started is True
    prom.start_http_server.assert_called_once_with(9999)


def test_prometheus_exporter_start_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling start() twice → start_http_server invoked exactly once."""
    prom = _make_prometheus_mock()
    monkeypatch.setitem(sys.modules, "prometheus_client", prom)

    from superlocalmemory.optimize.metrics.exporters import get_prometheus_exporter

    exporter = get_prometheus_exporter()
    exporter.start()
    exporter.start()

    assert prom.start_http_server.call_count == 1


def test_prometheus_exporter_start_exception_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """OSError from start_http_server is swallowed — _started stays False."""
    prom = _make_prometheus_mock()
    prom.start_http_server.side_effect = OSError("port already in use")
    monkeypatch.setitem(sys.modules, "prometheus_client", prom)

    from superlocalmemory.optimize.metrics.exporters import get_prometheus_exporter

    exporter = get_prometheus_exporter()
    exporter.start()  # must not raise

    assert exporter._started is False


# ---- update() ---------------------------------------------------------------

def test_prometheus_exporter_update_noop_when_not_started(monkeypatch: pytest.MonkeyPatch) -> None:
    """update() before start() is a no-op — no _value.set calls."""
    prom = _make_prometheus_mock()
    monkeypatch.setitem(sys.modules, "prometheus_client", prom)

    from superlocalmemory.optimize.metrics.exporters import get_prometheus_exporter

    exporter = get_prometheus_exporter()
    # not started
    exporter.update({"hits": 10, "misses": 5, "cache_size_bytes": 1024})

    prom.Counter.return_value._value.set.assert_not_called()
    prom.Gauge.return_value.set.assert_not_called()


def test_prometheus_exporter_update_sets_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """update() after start() writes all snapshot fields to prometheus objects."""
    prom = _make_prometheus_mock()
    monkeypatch.setitem(sys.modules, "prometheus_client", prom)

    from superlocalmemory.optimize.metrics.exporters import get_prometheus_exporter

    exporter = get_prometheus_exporter()
    exporter.start()
    exporter.update({
        "hits": 100,
        "misses": 20,
        "tokens_saved_input": 50,
        "tokens_saved_output": 30,
        "tokens_saved_compress": 10,
        "compress_runs": 5,
        "cache_size_bytes": 2048,
    })

    # _value.set called for: hits, misses, tokens_saved, compress_runs (4 calls,
    # all on the same mock Counter instance because MagicMock returns_value is shared)
    assert prom.Counter.return_value._value.set.call_count == 4
    # Gauge.set called once for cache_size_bytes
    prom.Gauge.return_value.set.assert_called_once_with(2048)


def test_prometheus_exporter_update_exception_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exception inside update() is swallowed — no propagation."""
    prom = _make_prometheus_mock()
    counter_instance = MagicMock()
    counter_instance._value.set.side_effect = RuntimeError("internal metric error")
    prom.Counter.return_value = counter_instance
    monkeypatch.setitem(sys.modules, "prometheus_client", prom)

    from superlocalmemory.optimize.metrics.exporters import get_prometheus_exporter

    exporter = get_prometheus_exporter()
    exporter.start()
    exporter.update({"hits": 1})  # must not raise
