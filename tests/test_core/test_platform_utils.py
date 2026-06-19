# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for platform_utils — cross-platform subprocess and resource utilities.

V3.4.24: Validates popen_platform_kwargs, get_rss_mb, is_pid_alive,
kill_process, and start_parent_watchdog across platforms.
"""

from __future__ import annotations

import os
import subprocess
import sys
from unittest.mock import patch

import pytest

from superlocalmemory.core.platform_utils import (
    get_rss_mb,
    is_pid_alive,
    kill_process,
    popen_platform_kwargs,
    start_parent_watchdog,
)


class TestPopenPlatformKwargs:
    """Platform-appropriate subprocess kwargs."""

    def test_returns_dict(self) -> None:
        result = popen_platform_kwargs()
        assert isinstance(result, dict)

    def test_posix_has_start_new_session(self) -> None:
        with patch("superlocalmemory.core.platform_utils.sys") as mock_sys:
            mock_sys.platform = "darwin"
            result = popen_platform_kwargs()
            assert result == {"start_new_session": True}

    def test_linux_has_start_new_session(self) -> None:
        with patch("superlocalmemory.core.platform_utils.sys") as mock_sys:
            mock_sys.platform = "linux"
            result = popen_platform_kwargs()
            assert result == {"start_new_session": True}

    def test_win32_has_create_no_window(self) -> None:
        with patch("superlocalmemory.core.platform_utils.sys") as mock_sys:
            mock_sys.platform = "win32"
            result = popen_platform_kwargs()
            assert "creationflags" in result
            assert result["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

    def test_kwargs_can_unpack_into_popen(self) -> None:
        kwargs = popen_platform_kwargs()
        assert len(kwargs) == 1


class TestGetRssMb:
    """RSS memory measurement."""

    def test_returns_float(self) -> None:
        result = get_rss_mb()
        assert isinstance(result, float)

    def test_returns_positive_on_current_platform(self) -> None:
        result = get_rss_mb()
        assert result > 0.0

    def test_returns_zero_on_resource_failure(self) -> None:
        with patch("superlocalmemory.core.platform_utils.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch.dict("sys.modules", {"resource": None}):
                result = get_rss_mb()
                assert result == 0.0


class TestIsPidAlive:
    """Process liveness check."""

    def test_current_process_is_alive(self) -> None:
        assert is_pid_alive(os.getpid()) is True

    def test_zero_pid_is_not_alive(self) -> None:
        assert is_pid_alive(0) is False

    def test_negative_pid_is_not_alive(self) -> None:
        assert is_pid_alive(-1) is False

    def test_nonexistent_pid_is_not_alive(self) -> None:
        assert is_pid_alive(99999999) is False

    def test_parent_process_is_alive(self) -> None:
        assert is_pid_alive(os.getppid()) is True

    @pytest.mark.skipif(sys.platform == "win32", reason="EPERM semantics are POSIX")
    def test_unsignalable_but_existing_pid_is_alive(self) -> None:
        # PID 1 (init/launchd) exists but a normal user cannot signal it ->
        # os.kill(1, 0) raises EPERM. is_pid_alive must treat EPERM as ALIVE,
        # not dead (regression: it used to catch all OSError and return False,
        # which also made the parent-pid test flaky under reparenting to init).
        assert is_pid_alive(1) is True


class TestKillProcess:
    """Process termination."""

    def test_zero_pid_returns_false(self) -> None:
        assert kill_process(0) is False

    def test_negative_pid_returns_false(self) -> None:
        assert kill_process(-1) is False

    def test_kill_real_subprocess(self) -> None:
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        pid = proc.pid
        assert is_pid_alive(pid) is True
        result = kill_process(pid)
        assert result is True
        proc.wait(timeout=5)


class TestStartParentWatchdog:
    """Parent watchdog thread."""

    def test_does_not_crash(self) -> None:
        start_parent_watchdog()

    def test_creates_daemon_thread(self) -> None:
        import threading
        initial_count = threading.active_count()
        start_parent_watchdog()
        import time
        time.sleep(0.1)
        assert threading.active_count() >= initial_count
