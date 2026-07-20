# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Integration tests for Health Monitor (Phase B)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.slow


class TestHealthCheckRegistry:

    def test_register_and_run(self):
        from superlocalmemory.core.health_monitor import (
            _HEALTH_CHECKS,
            register_health_check,
            run_all_health_checks,
        )
        # Clear registry for test isolation
        original = list(_HEALTH_CHECKS)
        _HEALTH_CHECKS.clear()

        def check_a():
            return {"name": "a", "status": "ok", "detail": "good"}

        def check_b():
            return {"name": "b", "status": "warning", "detail": "hmm"}

        register_health_check(check_a)
        register_health_check(check_b)
        results = run_all_health_checks()

        assert len(results) == 2
        assert results[0]["name"] == "a"
        assert results[1]["status"] == "warning"

        # Restore
        _HEALTH_CHECKS.clear()
        _HEALTH_CHECKS.extend(original)

    def test_check_exception_handled(self):
        from superlocalmemory.core.health_monitor import (
            _HEALTH_CHECKS,
            register_health_check,
            run_all_health_checks,
        )
        original = list(_HEALTH_CHECKS)
        _HEALTH_CHECKS.clear()

        def bad_check():
            raise RuntimeError("boom")

        register_health_check(bad_check)
        results = run_all_health_checks()

        assert len(results) == 1
        assert results[0]["status"] == "error"
        assert "boom" in results[0]["detail"]

        _HEALTH_CHECKS.clear()
        _HEALTH_CHECKS.extend(original)


class TestStructuredLogging:

    def test_setup_creates_log_file(self, tmp_path):
        from superlocalmemory.core.health_monitor import log_structured, setup_structured_logging
        log_dir = tmp_path / "logs"
        setup_structured_logging(log_dir)
        log_structured(level="info", operation="test", message="hello structured")

        log_file = log_dir / "daemon.json.log"
        assert log_file.exists()
        content = log_file.read_text()
        assert "hello structured" in content
        assert '"operation": "test"' in content


class TestHealthMonitorInit:

    def test_monitor_without_psutil(self):
        """Health monitor should not crash if psutil unavailable."""
        from superlocalmemory.core.health_monitor import HealthMonitor

        with patch("superlocalmemory.core.health_monitor.PSUTIL_AVAILABLE", False):
            monitor = HealthMonitor(enable_structured_logging=False)
        assert monitor._budget_mb == 8000
        assert monitor._heartbeat_timeout == 60

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("psutil"),
        reason="psutil not installed",
    )
    def test_monitor_starts_and_stops(self):
        from superlocalmemory.core.health_monitor import HealthMonitor
        monitor = HealthMonitor(
            check_interval_sec=1,
            enable_structured_logging=False,
        )
        monitor.start()
        assert monitor._thread is not None
        assert monitor._thread.is_alive()
        monitor.stop()
        monitor._thread.join(timeout=3)
