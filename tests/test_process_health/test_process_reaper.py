# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for ProcessReaper — orphan SLM process detection and cleanup.

TDD order 3-4: depends on PidManager.
Tests: T1, T2, T3, T4, T9, T10, T11, T12, T13.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# T1: test_detect_orphan_process
# ---------------------------------------------------------------------------
class TestFindOrphans:
    """T1 + T2: Detect orphans, skip active processes."""

    def test_detect_orphan_process(self, default_reaper_config, monkeypatch) -> None:
        """Detect a process whose parent PID is dead."""
        from superlocalmemory.infra.process_reaper import (
            ReaperConfig,
            SlmProcessInfo,
            find_orphans,
            find_slm_processes,
        )

        config = ReaperConfig(orphan_age_threshold_hours=0.0)

        fake_orphan = SlmProcessInfo(
            pid=12345,
            ppid=99999,
            start_time=time.time() - 7200,  # 2 hours ago
            command="python -m superlocalmemory.mcp.server",
            is_orphan=True,
            parent_name="",
            age_hours=2.0,
        )

        monkeypatch.setattr(
            "superlocalmemory.infra.process_reaper.find_slm_processes",
            lambda: [fake_orphan],
        )

        orphans = find_orphans(config)
        assert len(orphans) == 1
        assert orphans[0].pid == 12345
        assert orphans[0].is_orphan is True

    def test_skip_active_process(self, default_reaper_config, monkeypatch) -> None:
        """Do NOT flag a process whose parent is alive."""
        from superlocalmemory.infra.process_reaper import (
            SlmProcessInfo,
            find_orphans,
        )

        active_proc = SlmProcessInfo(
            pid=12345,
            ppid=os.getppid(),
            start_time=time.time() - 3600,
            command="python -m superlocalmemory.mcp.server",
            is_orphan=False,
            parent_name="node",
            age_hours=1.0,
        )

        monkeypatch.setattr(
            "superlocalmemory.infra.process_reaper.find_slm_processes",
            lambda: [active_proc],
        )

        orphans = find_orphans(default_reaper_config)
        assert len(orphans) == 0

    def test_skip_young_orphan(self, monkeypatch) -> None:
        """Skip orphans younger than the age threshold."""
        from superlocalmemory.infra.process_reaper import (
            ReaperConfig,
            SlmProcessInfo,
            find_orphans,
        )

        config = ReaperConfig(orphan_age_threshold_hours=4.0)

        young_orphan = SlmProcessInfo(
            pid=12345,
            ppid=99999,
            start_time=time.time() - 1800,  # 30 min ago
            command="python -m superlocalmemory.mcp.server",
            is_orphan=True,
            parent_name="",
            age_hours=0.5,
        )

        monkeypatch.setattr(
            "superlocalmemory.infra.process_reaper.find_slm_processes",
            lambda: [young_orphan],
        )

        orphans = find_orphans(config)
        assert len(orphans) == 0

    def test_never_returns_self(self, monkeypatch) -> None:
        """find_orphans never includes the current process."""
        from superlocalmemory.infra.process_reaper import (
            ReaperConfig,
            SlmProcessInfo,
            find_orphans,
        )

        config = ReaperConfig(orphan_age_threshold_hours=0.0)

        self_proc = SlmProcessInfo(
            pid=os.getpid(),
            ppid=1,
            start_time=time.time() - 7200,
            command="python -m superlocalmemory.mcp.server",
            is_orphan=True,
            parent_name="",
            age_hours=2.0,
        )

        monkeypatch.setattr(
            "superlocalmemory.infra.process_reaper.find_slm_processes",
            lambda: [self_proc],
        )

        orphans = find_orphans(config)
        assert len(orphans) == 0


# ---------------------------------------------------------------------------
# T3: test_kill_orphan_graceful_sigterm
# ---------------------------------------------------------------------------
class TestKillOrphan:
    """T3 + T4: Kill with SIGTERM, escalate to SIGKILL."""

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals only")
    def test_kill_orphan_graceful_sigterm(self, tmp_path: Path) -> None:
        """SIGTERM kills a cooperative process."""
        from superlocalmemory.infra.process_reaper import kill_orphan

        # Use a Python subprocess that responds to SIGTERM cleanly.
        # Start the process in a new session (os.setsid) so it is
        # NOT our child -- this avoids the zombie problem where
        # os.kill(pid, 0) returns success for unreaped children.
        ready_file = tmp_path / "ready"
        pid_file = tmp_path / "child_pid"
        launcher = tmp_path / "launcher.py"
        launcher.write_text(
            "import os, pathlib, subprocess, sys\n"
            f"proc = subprocess.Popen([sys.executable, '-c', "
            "'import time, pathlib; "
            f"pathlib.Path(\\\"{ready_file}\\\").touch(); "
            "time.sleep(300)'"
            f"], start_new_session=True)\n"
            f"pathlib.Path('{pid_file}').write_text(str(proc.pid))\n"
        )
        # Run launcher, which starts child in new session then exits
        subprocess.run(
            [sys.executable, str(launcher)], timeout=5, check=True
        )

        target_pid = int(pid_file.read_text().strip())

        # Wait for the process to be fully running
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if ready_file.exists():
                break
            time.sleep(0.1)

        result = kill_orphan(target_pid, graceful_timeout_seconds=10.0)

        assert result["killed"] is True
        assert result["method"] == "sigterm"
        assert result["error"] is None

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals only")
    def test_kill_orphan_force_sigkill(self, tmp_path: Path) -> None:
        """SIGKILL used when process ignores SIGTERM."""
        from superlocalmemory.infra.process_reaper import kill_orphan

        # Create a SIGTERM-ignoring subprocess
        script = tmp_path / "stubborn.py"
        script.write_text(
            "import signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "time.sleep(300)\n"
        )
        proc = subprocess.Popen([sys.executable, str(script)])
        time.sleep(0.5)  # Let it start and register handler

        try:
            result = kill_orphan(proc.pid, graceful_timeout_seconds=1.0)
        finally:
            # This test created the child, so it must reap it. kill_orphan only
            # owns a PID and production targets are normally unrelated orphans.
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)

        assert result["killed"] is True
        assert result["method"] == "sigkill"
        assert proc.returncode == -signal.SIGKILL

    def test_kill_already_dead_process(self) -> None:
        """Killing a dead process returns already_dead."""
        from superlocalmemory.infra.process_reaper import kill_orphan

        proc = subprocess.Popen(["true"])
        proc.wait()
        time.sleep(0.1)

        result = kill_orphan(proc.pid)
        assert result["killed"] is False
        assert result["method"] == "already_dead"


# ---------------------------------------------------------------------------
# T12: test_kill_refuses_pid_1
# ---------------------------------------------------------------------------
class TestKillSafety:
    """T12 + T13: Refuse to kill PID 0, 1, self, parent."""

    def test_kill_refuses_pid_0(self) -> None:
        from superlocalmemory.infra.process_reaper import kill_orphan

        result = kill_orphan(0)
        assert result["killed"] is False
        assert result["method"] == "refused"
        assert "Safety check" in result["error"]

    def test_kill_refuses_pid_1(self) -> None:
        """Refuse to kill PID 1 (init/launchd)."""
        from superlocalmemory.infra.process_reaper import kill_orphan

        result = kill_orphan(1)
        assert result["killed"] is False
        assert result["method"] == "refused"
        assert "Safety check" in result["error"]

    def test_kill_refuses_self(self) -> None:
        """Refuse to kill own process."""
        from superlocalmemory.infra.process_reaper import kill_orphan

        result = kill_orphan(os.getpid())
        assert result["killed"] is False
        assert result["method"] == "refused"
        assert "Safety check" in result["error"]

    def test_kill_refuses_parent(self) -> None:
        """Refuse to kill own parent."""
        from superlocalmemory.infra.process_reaper import kill_orphan

        result = kill_orphan(os.getppid())
        assert result["killed"] is False
        assert result["method"] == "refused"
        assert "Safety check" in result["error"]


# ---------------------------------------------------------------------------
# T9: test_startup_reaps_stale_from_pid_file
# ---------------------------------------------------------------------------
class TestReapStaleOnStartup:
    """T9: Dead PIDs in file are cleaned up on startup."""

    def test_startup_reaps_stale_from_pid_file(
        self, tmp_pid_file: Path, default_reaper_config
    ) -> None:
        from superlocalmemory.infra.pid_manager import PidManager
        from superlocalmemory.infra.process_reaper import reap_stale_on_startup

        mgr = PidManager(tmp_pid_file)
        mgr.register(99999, 88888)  # Dead PID

        result = reap_stale_on_startup(default_reaper_config, mgr)
        records = mgr.read_all()

        # Dead PID should be gone, current PID should be registered
        pids = [r.pid for r in records]
        assert 99999 not in pids
        assert os.getpid() in pids
        assert result["registered_pid"] == os.getpid()

    def test_startup_disabled_still_registers(
        self, tmp_pid_file: Path
    ) -> None:
        """When reaper is disabled, still register current PID."""
        from superlocalmemory.infra.pid_manager import PidManager
        from superlocalmemory.infra.process_reaper import ReaperConfig, reap_stale_on_startup

        config = ReaperConfig(enabled=False)
        mgr = PidManager(tmp_pid_file)

        result = reap_stale_on_startup(config, mgr)
        records = mgr.read_all()

        pids = [r.pid for r in records]
        assert os.getpid() in pids
        assert result["registered_pid"] == os.getpid()


# ---------------------------------------------------------------------------
# T10: test_dry_run_no_kill
# ---------------------------------------------------------------------------
class TestCleanupAllOrphans:
    """T10 + T11: cleanup_all_orphans dry_run and safety."""

    def test_dry_run_no_kill(self, default_reaper_config, monkeypatch) -> None:
        """--dry-run reports orphans but does not kill them."""
        from superlocalmemory.infra.process_reaper import (
            SlmProcessInfo,
            cleanup_all_orphans,
        )

        fake_orphan = SlmProcessInfo(
            pid=12345,
            ppid=99999,
            start_time=time.time() - 18000,  # 5 hours ago
            command="python -m superlocalmemory.mcp.server",
            is_orphan=True,
            parent_name="",
            age_hours=5.0,
        )

        monkeypatch.setattr(
            "superlocalmemory.infra.process_reaper.find_slm_processes",
            lambda: [fake_orphan],
        )

        kill_mock = MagicMock()
        monkeypatch.setattr(
            "superlocalmemory.infra.process_reaper.kill_orphan", kill_mock
        )

        result = cleanup_all_orphans(default_reaper_config, dry_run=True)
        assert result["orphans_found"] >= 1
        assert result["killed"] == 0
        kill_mock.assert_not_called()

    def test_no_false_kills_safety(
        self, default_reaper_config, monkeypatch
    ) -> None:
        """Active processes with living parents are NEVER killed."""
        from superlocalmemory.infra.process_reaper import (
            ReaperConfig,
            SlmProcessInfo,
            cleanup_all_orphans,
        )

        config = ReaperConfig(orphan_age_threshold_hours=0.0)

        orphan = SlmProcessInfo(
            pid=11111,
            ppid=99999,
            start_time=time.time() - 7200,
            command="python -m superlocalmemory.mcp.server",
            is_orphan=True,
            parent_name="",
            age_hours=2.0,
        )
        active = SlmProcessInfo(
            pid=22222,
            ppid=os.getppid(),
            start_time=time.time() - 3600,
            command="python -m superlocalmemory.mcp.server",
            is_orphan=False,
            parent_name="node",
            age_hours=1.0,
        )

        monkeypatch.setattr(
            "superlocalmemory.infra.process_reaper.find_slm_processes",
            lambda: [orphan, active],
        )

        killed_pids: list[int] = []

        def mock_kill(pid: int, **kwargs) -> dict:
            killed_pids.append(pid)
            return {"pid": pid, "killed": True, "method": "sigterm", "error": None}

        monkeypatch.setattr(
            "superlocalmemory.infra.process_reaper.kill_orphan", mock_kill
        )

        result = cleanup_all_orphans(config)

        # Only orphan should have been killed
        assert 11111 in killed_pids
        assert 22222 not in killed_pids
        assert result["skipped"] >= 1

    def test_force_kills_all_except_self(
        self, default_reaper_config, monkeypatch
    ) -> None:
        """--force kills ALL SLM processes except current."""
        from superlocalmemory.infra.process_reaper import (
            SlmProcessInfo,
            cleanup_all_orphans,
        )

        proc1 = SlmProcessInfo(
            pid=11111, ppid=22222, start_time=time.time() - 3600,
            command="python -m superlocalmemory.mcp.server",
            is_orphan=False, parent_name="node", age_hours=1.0,
        )

        monkeypatch.setattr(
            "superlocalmemory.infra.process_reaper.find_slm_processes",
            lambda: [proc1],
        )

        killed_pids: list[int] = []

        def mock_kill(pid: int, **kwargs) -> dict:
            killed_pids.append(pid)
            return {"pid": pid, "killed": True, "method": "sigterm", "error": None}

        monkeypatch.setattr(
            "superlocalmemory.infra.process_reaper.kill_orphan", mock_kill
        )

        result = cleanup_all_orphans(default_reaper_config, force=True)
        assert result["killed"] >= 1
        assert 11111 in killed_pids


# ---------------------------------------------------------------------------
# Additional coverage tests for _check_parent and find_slm_processes
# ---------------------------------------------------------------------------
class TestCheckParent:
    """Coverage tests for _check_parent edge cases."""

    def test_check_parent_pid_1(self) -> None:
        """PID 1 (init/launchd) is always orphan."""
        from superlocalmemory.infra.process_reaper import _check_parent

        is_orphan, name = _check_parent(1)
        assert is_orphan is True
        assert name == "init"

    def test_check_parent_pid_0(self) -> None:
        """PID 0 is always orphan."""
        from superlocalmemory.infra.process_reaper import _check_parent

        is_orphan, name = _check_parent(0)
        assert is_orphan is True
        assert name == "init"

    def test_check_parent_dead_pid(self) -> None:
        """Dead parent PID returns orphan."""
        from superlocalmemory.infra.process_reaper import _check_parent

        # PID 99999 is almost certainly dead
        is_orphan, name = _check_parent(99999)
        assert is_orphan is True
        assert name == ""

    def test_check_parent_alive_pid(self) -> None:
        """Own PID (alive) returns non-orphan."""
        from superlocalmemory.infra.process_reaper import _check_parent

        is_orphan, name = _check_parent(os.getpid())
        assert is_orphan is False
        assert isinstance(name, str)

    def test_check_parent_permission_error(self, monkeypatch) -> None:
        """PermissionError returns non-orphan (conservative)."""
        from superlocalmemory.infra.process_reaper import _check_parent

        def mock_kill(pid, sig):
            raise PermissionError("Operation not permitted")

        monkeypatch.setattr("os.kill", mock_kill)
        is_orphan, name = _check_parent(12345)
        assert is_orphan is False
        assert name == "unknown"


class TestFindSlmProcesses:
    """Coverage tests for find_slm_processes edge cases."""

    def test_ps_command_failure(self, monkeypatch) -> None:
        """Returns empty list when ps fails."""
        from superlocalmemory.infra.process_reaper import find_slm_processes

        def mock_run(*args, **kwargs):
            raise subprocess.SubprocessError("ps not found")

        monkeypatch.setattr("subprocess.run", mock_run)
        result = find_slm_processes()
        assert result == []

    def test_ps_nonzero_returncode(self, monkeypatch) -> None:
        """Returns empty list when ps returns non-zero."""
        from superlocalmemory.infra.process_reaper import find_slm_processes

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock_result)

        result = find_slm_processes()
        assert result == []

    def test_ps_with_slm_output(self, monkeypatch) -> None:
        """Parses ps output with superlocalmemory lines."""
        from superlocalmemory.infra.process_reaper import find_slm_processes

        # Fabricate ps output with a fake SLM process line
        fake_output = (
            "  PID  PPID                     STARTED COMMAND\n"
            "99998     1 Mon Mar 30 14:25:01 2026 python -m superlocalmemory.mcp.server\n"
        )

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = fake_output

        monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock_result)
        # _check_parent will try os.kill(1, 0) for ppid=1 -> orphan
        result = find_slm_processes()

        # Should find the fabricated process (PID 99998)
        pids = [p.pid for p in result]
        assert 99998 in pids
        matched = [p for p in result if p.pid == 99998]
        assert matched[0].is_orphan is True

    def test_ps_skips_non_slm_lines(self, monkeypatch) -> None:
        """Lines without 'superlocalmemory' are skipped."""
        from superlocalmemory.infra.process_reaper import find_slm_processes

        fake_output = (
            "  PID  PPID                     STARTED COMMAND\n"
            " 1234     1 Mon Mar 30 14:25:01 2026 python -m some.other.module\n"
        )

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = fake_output
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock_result)

        result = find_slm_processes()
        assert result == []

    def test_ps_skips_bad_date(self, monkeypatch) -> None:
        """Lines with unparseable dates are skipped."""
        from superlocalmemory.infra.process_reaper import find_slm_processes

        fake_output = (
            "  PID  PPID                     STARTED COMMAND\n"
            "99997     1 BADDATE BADDATE BADDATE BADDATE BADDATE python -m superlocalmemory.mcp.server\n"
        )

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = fake_output
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock_result)

        result = find_slm_processes()
        assert result == []


class TestCleanupErrors:
    """Test error handling paths in cleanup functions."""

    def test_force_dry_run(self, default_reaper_config, monkeypatch) -> None:
        """Force + dry_run shows would_kill without killing."""
        from superlocalmemory.infra.process_reaper import (
            SlmProcessInfo,
            cleanup_all_orphans,
        )

        proc = SlmProcessInfo(
            pid=11111, ppid=22222, start_time=time.time() - 3600,
            command="python -m superlocalmemory.mcp.server",
            is_orphan=False, parent_name="node", age_hours=1.0,
        )

        monkeypatch.setattr(
            "superlocalmemory.infra.process_reaper.find_slm_processes",
            lambda: [proc],
        )

        result = cleanup_all_orphans(default_reaper_config, dry_run=True, force=True)
        assert result["killed"] == 0
        processes = result["processes"]
        assert len(processes) == 1
        assert processes[0]["status"] == "would_kill"

    def test_force_kill_error(self, default_reaper_config, monkeypatch) -> None:
        """Force mode handles kill errors gracefully."""
        from superlocalmemory.infra.process_reaper import (
            SlmProcessInfo,
            cleanup_all_orphans,
        )

        proc = SlmProcessInfo(
            pid=11111, ppid=22222, start_time=time.time() - 3600,
            command="python -m superlocalmemory.mcp.server",
            is_orphan=False, parent_name="node", age_hours=1.0,
        )

        monkeypatch.setattr(
            "superlocalmemory.infra.process_reaper.find_slm_processes",
            lambda: [proc],
        )

        def mock_kill(pid, **kwargs):
            return {"pid": pid, "killed": False, "method": "error", "error": "Permission denied"}

        monkeypatch.setattr(
            "superlocalmemory.infra.process_reaper.kill_orphan", mock_kill
        )

        result = cleanup_all_orphans(default_reaper_config, force=True)
        assert result["killed"] == 0
        assert len(result["errors"]) == 1

    def test_cleanup_orphan_kill_error(
        self, default_reaper_config, monkeypatch
    ) -> None:
        """Normal mode handles kill errors in orphan processing."""
        from superlocalmemory.infra.process_reaper import (
            ReaperConfig,
            SlmProcessInfo,
            cleanup_all_orphans,
        )

        config = ReaperConfig(orphan_age_threshold_hours=0.0)

        orphan = SlmProcessInfo(
            pid=11111, ppid=99999,
            start_time=time.time() - 7200,
            command="python -m superlocalmemory.mcp.server",
            is_orphan=True, parent_name="", age_hours=2.0,
        )

        monkeypatch.setattr(
            "superlocalmemory.infra.process_reaper.find_slm_processes",
            lambda: [orphan],
        )

        def mock_kill(pid, **kwargs):
            return {"pid": pid, "killed": False, "method": "error", "error": "Permission denied"}

        monkeypatch.setattr(
            "superlocalmemory.infra.process_reaper.kill_orphan", mock_kill
        )

        result = cleanup_all_orphans(config)
        assert result["killed"] == 0
        assert len(result["errors"]) == 1

    def test_empty_system(self, default_reaper_config, monkeypatch) -> None:
        """No SLM processes found returns clean result."""
        from superlocalmemory.infra.process_reaper import cleanup_all_orphans

        monkeypatch.setattr(
            "superlocalmemory.infra.process_reaper.find_slm_processes",
            lambda: [],
        )

        result = cleanup_all_orphans(default_reaper_config)
        assert result["total_found"] == 0
        assert result["orphans_found"] == 0
        assert result["killed"] == 0


# ---------------------------------------------------------------------------
# Coverage gap tests: _check_parent subprocess error (lines 156-157)
# ---------------------------------------------------------------------------
class TestCheckParentSubprocessError:
    """Cover the except (SubprocessError, OSError) in _check_parent."""

    def test_check_parent_ps_subprocess_error(self, monkeypatch) -> None:
        """When ps -p fails with SubprocessError, parent_name stays empty."""
        from superlocalmemory.infra.process_reaper import _check_parent

        # os.kill must succeed (parent alive), then subprocess.run must fail
        original_kill = os.kill

        def mock_kill(pid, sig):
            if pid == 54321 and sig == 0:
                return None  # alive
            return original_kill(pid, sig)

        monkeypatch.setattr("os.kill", mock_kill)
        monkeypatch.setattr(
            "subprocess.run",
            MagicMock(side_effect=subprocess.SubprocessError("mocked ps fail")),
        )

        is_orphan, name = _check_parent(54321)
        assert is_orphan is False
        assert name == ""  # Failed to read name

    def test_check_parent_ps_os_error(self, monkeypatch) -> None:
        """When ps -p fails with OSError, parent_name stays empty."""
        from superlocalmemory.infra.process_reaper import _check_parent

        original_kill = os.kill

        def mock_kill(pid, sig):
            if pid == 54322 and sig == 0:
                return None
            return original_kill(pid, sig)

        monkeypatch.setattr("os.kill", mock_kill)
        monkeypatch.setattr(
            "subprocess.run",
            MagicMock(side_effect=OSError("mocked os error")),
        )

        is_orphan, name = _check_parent(54322)
        assert is_orphan is False
        assert name == ""


# ---------------------------------------------------------------------------
# Coverage gap tests: find_slm_processes empty line (line 186)
# ---------------------------------------------------------------------------
class TestFindSlmProcessesEmptyLine:
    """Cover the empty-line continue in find_slm_processes."""

    def test_ps_output_with_empty_lines(self, monkeypatch) -> None:
        """Empty lines in ps output are skipped gracefully."""
        from superlocalmemory.infra.process_reaper import find_slm_processes

        fake_output = (
            "  PID  PPID                     STARTED COMMAND\n"
            "\n"  # Empty line that should be skipped
            "   \n"  # Whitespace-only line
            "99998     1 Mon Mar 30 14:25:01 2026 python -m superlocalmemory.mcp.server\n"
        )

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = fake_output
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock_result)

        result = find_slm_processes()
        pids = [p.pid for p in result]
        assert 99998 in pids


# ---------------------------------------------------------------------------
# Coverage gap tests: find_slm_processes ValueError/IndexError (lines 237-239)
# ---------------------------------------------------------------------------
class TestFindSlmProcessesParseError:
    """Cover the except (ValueError, IndexError) in ps line parsing."""

    def test_ps_output_malformed_pid(self, monkeypatch) -> None:
        """Lines with non-integer PID trigger ValueError, are skipped."""
        from superlocalmemory.infra.process_reaper import find_slm_processes

        fake_output = (
            "  PID  PPID                     STARTED COMMAND\n"
            "  abc     1 Mon Mar 30 14:25:01 2026 python -m superlocalmemory.mcp.server\n"
        )

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = fake_output
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock_result)

        result = find_slm_processes()
        assert result == []

    def test_ps_output_too_short_line(self, monkeypatch) -> None:
        """Lines with too few fields trigger IndexError, are skipped."""
        from superlocalmemory.infra.process_reaper import find_slm_processes

        fake_output = (
            "  PID  PPID                     STARTED COMMAND\n"
            "  superlocalmemory\n"  # Too short, triggers IndexError
        )

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = fake_output
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock_result)

        result = find_slm_processes()
        assert result == []


# ---------------------------------------------------------------------------
# Coverage gap tests: find_orphans skips parent PID (line 272)
# ---------------------------------------------------------------------------
class TestFindOrphansSkipsParent:
    """Cover the p.pid == os.getppid() continue in find_orphans."""

    def test_never_returns_parent(self, monkeypatch) -> None:
        """find_orphans never includes the parent process."""
        from superlocalmemory.infra.process_reaper import (
            ReaperConfig,
            SlmProcessInfo,
            find_orphans,
        )

        config = ReaperConfig(orphan_age_threshold_hours=0.0)

        parent_proc = SlmProcessInfo(
            pid=os.getppid(),
            ppid=1,
            start_time=time.time() - 7200,
            command="python -m superlocalmemory.mcp.server",
            is_orphan=True,
            parent_name="",
            age_hours=2.0,
        )

        monkeypatch.setattr(
            "superlocalmemory.infra.process_reaper.find_slm_processes",
            lambda: [parent_proc],
        )

        orphans = find_orphans(config)
        assert len(orphans) == 0


# ---------------------------------------------------------------------------
# Coverage gap tests: kill_orphan PermissionError on probe (lines 320-321)
# ---------------------------------------------------------------------------
class TestKillOrphanPermissionOnProbe:
    """Cover PermissionError when probing PID existence."""

    def test_kill_orphan_permission_error_on_probe(self, monkeypatch) -> None:
        """PermissionError on os.kill(pid, 0) returns refused."""
        from superlocalmemory.infra.process_reaper import kill_orphan

        # Capture real values BEFORE monkeypatching
        real_pid = os.getpid()
        real_ppid = os.getppid()
        fake_pid = real_pid + 10000

        def mock_kill(pid, sig):
            if sig == 0:
                raise PermissionError("Operation not permitted")
            raise AssertionError("Should not reach SIGTERM")

        monkeypatch.setattr("os.kill", mock_kill)
        monkeypatch.setattr("os.getpid", lambda: real_pid)
        monkeypatch.setattr("os.getppid", lambda: real_ppid)

        result = kill_orphan(fake_pid)
        assert result["killed"] is False
        assert result["method"] == "refused"
        assert "Permission denied" in result["error"]


# ---------------------------------------------------------------------------
# Coverage gap tests: kill_orphan SIGTERM errors (lines 329-335)
# ---------------------------------------------------------------------------
class TestKillOrphanSigtermErrors:
    """Cover ProcessLookupError and PermissionError on SIGTERM."""

    def test_kill_orphan_process_dies_during_sigterm(self, monkeypatch) -> None:
        """Process dies between probe and SIGTERM (ProcessLookupError)."""
        from superlocalmemory.infra.process_reaper import kill_orphan

        call_count = 0

        def mock_kill(pid, sig):
            nonlocal call_count
            call_count += 1
            if sig == 0:
                return None  # alive on probe
            if sig == signal.SIGTERM:
                raise ProcessLookupError("No such process")
            raise AssertionError("Unexpected signal")

        monkeypatch.setattr("os.kill", mock_kill)
        fake_pid = os.getpid() + 10000

        result = kill_orphan(fake_pid)
        assert result["killed"] is False
        assert result["method"] == "already_dead"
        assert result["error"] is None

    def test_kill_orphan_permission_error_on_sigterm(self, monkeypatch) -> None:
        """PermissionError on SIGTERM returns refused."""
        from superlocalmemory.infra.process_reaper import kill_orphan

        def mock_kill(pid, sig):
            if sig == 0:
                return None  # alive on probe
            if sig == signal.SIGTERM:
                raise PermissionError("Operation not permitted")
            raise AssertionError("Unexpected signal")

        monkeypatch.setattr("os.kill", mock_kill)
        fake_pid = os.getpid() + 10000

        result = kill_orphan(fake_pid)
        assert result["killed"] is False
        assert result["method"] == "refused"
        assert "Permission denied" in result["error"]


# ---------------------------------------------------------------------------
# Coverage gap tests: kill_orphan SIGKILL errors (lines 367-373)
# ---------------------------------------------------------------------------
class TestKillOrphanSigkillErrors:
    """Cover ProcessLookupError and PermissionError on SIGKILL."""

    def test_kill_orphan_dies_before_sigkill(self, monkeypatch) -> None:
        """Process dies after SIGTERM, before SIGKILL (ProcessLookupError)."""
        from superlocalmemory.infra.process_reaper import kill_orphan

        sigterm_sent = False

        def mock_kill(pid, sig):
            nonlocal sigterm_sent
            if sig == 0:
                if sigterm_sent:
                    # During the wait loop, process is still alive
                    return None
                return None  # alive on initial probe
            if sig == signal.SIGTERM:
                sigterm_sent = True
                return None  # SIGTERM accepted
            if sig == signal.SIGKILL:
                raise ProcessLookupError("No such process")
            raise AssertionError(f"Unexpected signal {sig}")

        # Mock sleep and monotonic to fast-forward through the wait loop
        monkeypatch.setattr("os.kill", mock_kill)
        # Make the wait loop expire immediately
        call_idx = [0]

        def mock_monotonic():
            call_idx[0] += 1
            if call_idx[0] <= 1:
                return 0.0  # start
            return 100.0  # expired (deadline passed)

        monkeypatch.setattr("time.monotonic", mock_monotonic)
        monkeypatch.setattr("time.sleep", lambda _: None)
        fake_pid = os.getpid() + 10000

        result = kill_orphan(fake_pid, graceful_timeout_seconds=1.0)
        assert result["killed"] is True
        assert result["method"] == "sigterm"  # died before SIGKILL

    def test_kill_orphan_permission_error_on_sigkill(self, monkeypatch) -> None:
        """PermissionError on SIGKILL returns refused."""
        from superlocalmemory.infra.process_reaper import kill_orphan

        def mock_kill(pid, sig):
            if sig == 0:
                return None  # alive
            if sig == signal.SIGTERM:
                return None  # accepted but doesn't die
            if sig == signal.SIGKILL:
                raise PermissionError("Operation not permitted")
            raise AssertionError(f"Unexpected signal {sig}")

        monkeypatch.setattr("os.kill", mock_kill)
        call_idx = [0]

        def mock_monotonic():
            call_idx[0] += 1
            if call_idx[0] <= 1:
                return 0.0
            return 100.0  # expire wait loop

        monkeypatch.setattr("time.monotonic", mock_monotonic)
        monkeypatch.setattr("time.sleep", lambda _: None)
        fake_pid = os.getpid() + 10000

        result = kill_orphan(fake_pid, graceful_timeout_seconds=1.0)
        assert result["killed"] is False
        assert result["method"] == "refused"
        assert "Permission denied" in result["error"]


# ---------------------------------------------------------------------------
# Coverage gap tests: reap_stale_on_startup orphan kill path (lines 406-436)
# ---------------------------------------------------------------------------
class TestReapStaleOrphanKillPath:
    """Cover the orphan detection and kill path in reap_stale_on_startup."""

    def test_startup_kills_tracked_orphan(
        self, tmp_pid_file: Path, monkeypatch
    ) -> None:
        """Kills tracked orphan whose parent is dead and age > threshold."""
        from superlocalmemory.infra.pid_manager import PidManager
        from superlocalmemory.infra.process_reaper import (
            ReaperConfig,
            reap_stale_on_startup,
        )

        config = ReaperConfig(orphan_age_threshold_hours=0.0)
        mgr = PidManager(tmp_pid_file)
        mgr.register(55555, 1)  # ppid=1 => orphan

        # Patch the started_at to be old
        records = mgr.read_all()
        # Manually rewrite with old timestamp
        import json
        from datetime import UTC, datetime, timedelta

        old_time = (datetime.now(UTC) - timedelta(hours=5)).isoformat()
        data = {"pids": [{"pid": 55555, "ppid": 1, "started_at": old_time}]}
        tmp_pid_file.write_text(json.dumps(data))

        # os.kill(55555, 0) must succeed (process alive)
        original_kill = os.kill

        def mock_kill(pid, sig):
            if pid == 55555:
                if sig == 0:
                    return None  # alive
                return None  # SIGTERM accepted
            return original_kill(pid, sig)

        monkeypatch.setattr("os.kill", mock_kill)

        # _check_parent(1) returns (True, "init") => orphan
        # kill_orphan must be mocked to avoid real signals
        monkeypatch.setattr(
            "superlocalmemory.infra.process_reaper.kill_orphan",
            lambda pid, **kw: {"pid": pid, "killed": True, "method": "sigterm", "error": None},
        )

        # Mock find_orphans to return empty (no untracked)
        monkeypatch.setattr(
            "superlocalmemory.infra.process_reaper.find_orphans",
            lambda config: [],
        )

        result = reap_stale_on_startup(config, mgr)
        assert int(result["orphans_found"]) >= 1
        assert int(result["orphans_killed"]) >= 1

    def test_startup_tracked_orphan_kill_error(
        self, tmp_pid_file: Path, monkeypatch
    ) -> None:
        """Error killing tracked orphan appends to errors list."""
        from superlocalmemory.infra.pid_manager import PidManager
        from superlocalmemory.infra.process_reaper import (
            ReaperConfig,
            reap_stale_on_startup,
        )

        config = ReaperConfig(orphan_age_threshold_hours=0.0)
        mgr = PidManager(tmp_pid_file)

        import json
        from datetime import UTC, datetime, timedelta

        old_time = (datetime.now(UTC) - timedelta(hours=5)).isoformat()
        data = {"pids": [{"pid": 55556, "ppid": 1, "started_at": old_time}]}
        tmp_pid_file.write_text(json.dumps(data))

        original_kill = os.kill

        def mock_kill(pid, sig):
            if pid == 55556 and sig == 0:
                return None  # alive
            return original_kill(pid, sig)

        monkeypatch.setattr("os.kill", mock_kill)

        monkeypatch.setattr(
            "superlocalmemory.infra.process_reaper.kill_orphan",
            lambda pid, **kw: {"pid": pid, "killed": False, "method": "refused", "error": "Permission denied"},
        )
        monkeypatch.setattr(
            "superlocalmemory.infra.process_reaper.find_orphans",
            lambda config: [],
        )

        result = reap_stale_on_startup(config, mgr)
        assert int(result["orphans_found"]) >= 1
        assert int(result["orphans_killed"]) == 0
        assert len(result["errors"]) >= 1

    def test_startup_tracked_orphan_bad_timestamp(
        self, tmp_pid_file: Path, monkeypatch
    ) -> None:
        """Bad started_at timestamp defaults age to 0.0, skipping kill."""
        from superlocalmemory.infra.pid_manager import PidManager
        from superlocalmemory.infra.process_reaper import (
            ReaperConfig,
            reap_stale_on_startup,
        )

        config = ReaperConfig(orphan_age_threshold_hours=0.0)
        mgr = PidManager(tmp_pid_file)

        import json

        # Write a record with an unparseable timestamp
        data = {"pids": [{"pid": 55559, "ppid": 1, "started_at": "INVALID-TIMESTAMP"}]}
        tmp_pid_file.write_text(json.dumps(data))

        original_kill = os.kill

        def mock_kill(pid, sig):
            if pid == 55559 and sig == 0:
                return None  # alive
            return original_kill(pid, sig)

        monkeypatch.setattr("os.kill", mock_kill)

        # _check_parent(1) => (True, "init") for orphan detection
        # kill_orphan mock
        monkeypatch.setattr(
            "superlocalmemory.infra.process_reaper.kill_orphan",
            lambda pid, **kw: {"pid": pid, "killed": True, "method": "sigterm", "error": None},
        )
        monkeypatch.setattr(
            "superlocalmemory.infra.process_reaper.find_orphans",
            lambda config: [],
        )

        result = reap_stale_on_startup(config, mgr)
        # age_hours=0.0 from bad timestamp, but threshold is also 0.0
        # so 0.0 > 0.0 is False => should NOT be killed
        assert result["registered_pid"] == os.getpid()

    def test_startup_permission_error_on_pid_check(
        self, tmp_pid_file: Path, monkeypatch
    ) -> None:
        """PermissionError on os.kill(pid, 0) continues without removing."""
        from superlocalmemory.infra.pid_manager import PidManager
        from superlocalmemory.infra.process_reaper import (
            ReaperConfig,
            reap_stale_on_startup,
        )

        config = ReaperConfig(orphan_age_threshold_hours=0.0)
        mgr = PidManager(tmp_pid_file)
        mgr.register(55557, 99999)

        original_kill = os.kill

        def mock_kill(pid, sig):
            if pid == 55557 and sig == 0:
                raise PermissionError("Operation not permitted")
            return original_kill(pid, sig)

        monkeypatch.setattr("os.kill", mock_kill)
        monkeypatch.setattr(
            "superlocalmemory.infra.process_reaper.find_orphans",
            lambda config: [],
        )

        result = reap_stale_on_startup(config, mgr)
        # PID should NOT have been removed (PermissionError = keep it)
        records = mgr.read_all()
        pids = [r.pid for r in records]
        assert os.getpid() in pids  # self registered


# ---------------------------------------------------------------------------
# Coverage gap tests: reap_stale_on_startup untracked orphan scan (lines 446-462)
# ---------------------------------------------------------------------------
class TestReapStaleUntrackedOrphans:
    """Cover the untracked orphan scan in reap_stale_on_startup."""

    def test_startup_kills_untracked_orphan(
        self, tmp_pid_file: Path, monkeypatch
    ) -> None:
        """Finds and kills an orphan not in the PID file."""
        from superlocalmemory.infra.pid_manager import PidManager
        from superlocalmemory.infra.process_reaper import (
            ReaperConfig,
            SlmProcessInfo,
            reap_stale_on_startup,
        )

        config = ReaperConfig(orphan_age_threshold_hours=0.0)
        mgr = PidManager(tmp_pid_file)
        # Empty PID file -- no tracked processes

        untracked_orphan = SlmProcessInfo(
            pid=77777, ppid=1,
            start_time=time.time() - 18000,
            command="python -m superlocalmemory.mcp.server",
            is_orphan=True, parent_name="init", age_hours=5.0,
        )

        monkeypatch.setattr(
            "superlocalmemory.infra.process_reaper.find_orphans",
            lambda cfg: [untracked_orphan],
        )
        monkeypatch.setattr(
            "superlocalmemory.infra.process_reaper.kill_orphan",
            lambda pid, **kw: {"pid": pid, "killed": True, "method": "sigterm", "error": None},
        )

        result = reap_stale_on_startup(config, mgr)
        assert int(result["orphans_found"]) >= 1
        assert int(result["orphans_killed"]) >= 1

    def test_startup_untracked_orphan_kill_error(
        self, tmp_pid_file: Path, monkeypatch
    ) -> None:
        """Error killing untracked orphan appends to errors."""
        from superlocalmemory.infra.pid_manager import PidManager
        from superlocalmemory.infra.process_reaper import (
            ReaperConfig,
            SlmProcessInfo,
            reap_stale_on_startup,
        )

        config = ReaperConfig(orphan_age_threshold_hours=0.0)
        mgr = PidManager(tmp_pid_file)

        untracked_orphan = SlmProcessInfo(
            pid=77778, ppid=1,
            start_time=time.time() - 18000,
            command="python -m superlocalmemory.mcp.server",
            is_orphan=True, parent_name="init", age_hours=5.0,
        )

        monkeypatch.setattr(
            "superlocalmemory.infra.process_reaper.find_orphans",
            lambda cfg: [untracked_orphan],
        )
        monkeypatch.setattr(
            "superlocalmemory.infra.process_reaper.kill_orphan",
            lambda pid, **kw: {"pid": pid, "killed": False, "method": "refused", "error": "Permission denied"},
        )

        result = reap_stale_on_startup(config, mgr)
        assert int(result["orphans_found"]) >= 1
        assert int(result["orphans_killed"]) == 0
        assert len(result["errors"]) >= 1

    def test_startup_untracked_scan_exception(
        self, tmp_pid_file: Path, monkeypatch
    ) -> None:
        """Exception in find_orphans is caught gracefully."""
        from superlocalmemory.infra.pid_manager import PidManager
        from superlocalmemory.infra.process_reaper import (
            ReaperConfig,
            reap_stale_on_startup,
        )

        config = ReaperConfig(orphan_age_threshold_hours=0.0)
        mgr = PidManager(tmp_pid_file)

        monkeypatch.setattr(
            "superlocalmemory.infra.process_reaper.find_orphans",
            MagicMock(side_effect=RuntimeError("ps command exploded")),
        )

        # Should not raise
        result = reap_stale_on_startup(config, mgr)
        assert result["registered_pid"] == os.getpid()


# ---------------------------------------------------------------------------
# Coverage gap tests: cleanup_all_orphans force skips self (line 503)
# ---------------------------------------------------------------------------
class TestCleanupForceSkipsSelf:
    """Cover the p.pid == os.getpid() continue in force mode."""

    def test_force_skips_self_pid(self, default_reaper_config, monkeypatch) -> None:
        """Force mode skips the current process."""
        from superlocalmemory.infra.process_reaper import (
            SlmProcessInfo,
            cleanup_all_orphans,
        )

        self_proc = SlmProcessInfo(
            pid=os.getpid(), ppid=os.getppid(),
            start_time=time.time() - 3600,
            command="python -m superlocalmemory.mcp.server",
            is_orphan=False, parent_name="node", age_hours=1.0,
        )
        other_proc = SlmProcessInfo(
            pid=88888, ppid=22222,
            start_time=time.time() - 3600,
            command="python -m superlocalmemory.mcp.server",
            is_orphan=False, parent_name="node", age_hours=1.0,
        )

        monkeypatch.setattr(
            "superlocalmemory.infra.process_reaper.find_slm_processes",
            lambda: [self_proc, other_proc],
        )

        killed_pids: list[int] = []

        def mock_kill(pid, **kwargs):
            killed_pids.append(pid)
            return {"pid": pid, "killed": True, "method": "sigterm", "error": None}

        monkeypatch.setattr(
            "superlocalmemory.infra.process_reaper.kill_orphan", mock_kill
        )

        result = cleanup_all_orphans(default_reaper_config, force=True)
        # Self should NOT be killed
        assert os.getpid() not in killed_pids
        assert 88888 in killed_pids
