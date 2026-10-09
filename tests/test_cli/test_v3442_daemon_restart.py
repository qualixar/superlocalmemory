# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for v3.4.42 — `slm restart` Step 3 self-deadlock fix (B1).

The bug:
  cmd_restart Step 2 acquired daemon.lock via fcntl.flock(LOCK_EX | LOCK_NB),
  then Step 3 called ensure_daemon() which tried to acquire the SAME lock from
  a separate fd in the same process. BSD-style flock blocks per-fd even within
  one process, so the second flock fails with EWOULDBLOCK, ensure_daemon falls
  into its "wait for someone else" branch, times out at 60s, and reports
  "failed to start" — even though no real failure occurred and a follow-up
  CLI call would successfully start the daemon.

The fix:
  Extract `_start_daemon_subprocess()` from ensure_daemon() — it does the
  raw subprocess spawn + PID/port write + wait-for-ready, WITHOUT taking
  the lock. Callers that already hold daemon.lock (cmd_restart Step 3)
  invoke this helper directly. ensure_daemon() still uses it after its own
  lock acquisition, preserving the public-API contract for unrelated callers.
"""

from __future__ import annotations

from itertools import chain, repeat
from unittest.mock import MagicMock, patch


class TestStartDaemonSubprocessHelper:
    """The new helper extracted in v3.4.42."""

    def test_helper_exists_and_is_importable(self) -> None:
        """B1: _start_daemon_subprocess is defined and callable in cli.daemon."""
        from superlocalmemory.cli import daemon as _daemon
        assert hasattr(_daemon, "_start_daemon_subprocess")
        assert callable(_daemon._start_daemon_subprocess)

    def test_returns_true_immediately_when_daemon_already_running(self) -> None:
        """B1: helper short-circuits when daemon is already up — no subprocess spawn."""
        from superlocalmemory.cli import daemon as _daemon

        with patch.object(_daemon, "is_daemon_running", return_value=True):
            with patch("subprocess.Popen") as popen:
                result = _daemon._start_daemon_subprocess()

        assert result is True
        popen.assert_not_called()  # never spawned a subprocess

    def test_spawns_subprocess_and_leaves_pid_port_to_the_daemon_when_down(self, tmp_path) -> None:
        """B1: helper spawns daemon and delegates wait; the daemon writes the record."""
        from superlocalmemory.cli import daemon as _daemon

        fake_proc = MagicMock()
        fake_proc.pid = 99999

        with patch.object(_daemon, "is_daemon_running", return_value=False), \
             patch.object(_daemon, "_PID_FILE", tmp_path / "daemon.pid"), \
             patch.object(_daemon, "_PORT_FILE", tmp_path / "daemon.port"), \
             patch("subprocess.Popen", return_value=fake_proc) as popen, \
             patch.object(_daemon, "_wait_for_daemon", return_value=True) as wait:
            result = _daemon._start_daemon_subprocess()

        assert result is True
        # One daemon launch. (On Windows, writing daemon.json also runs icacls
        # through subprocess to make it owner-only; that is not a launch.)
        launches = [c for c in popen.call_args_list
                    if "superlocalmemory.server.unified_daemon" in c.args[0]]
        assert len(launches) == 1, popen.call_args_list
        wait.assert_called_once()
        # The parent writes no pid/port mirrors; the daemon that owns the data
        # folder does.
        assert not (tmp_path / "daemon.pid").exists()
        assert not (tmp_path / "daemon.port").exists()

    def test_returns_false_when_wait_for_daemon_times_out(self, tmp_path) -> None:
        """B1: helper propagates _wait_for_daemon's timeout result honestly."""
        from superlocalmemory.cli import daemon as _daemon

        fake_proc = MagicMock()
        fake_proc.pid = 12345

        with patch.object(_daemon, "is_daemon_running", return_value=False), \
             patch.object(_daemon, "_PID_FILE", tmp_path / "daemon.pid"), \
             patch.object(_daemon, "_PORT_FILE", tmp_path / "daemon.port"), \
             patch("subprocess.Popen", return_value=fake_proc), \
             patch.object(_daemon, "_wait_for_daemon", return_value=False):
            result = _daemon._start_daemon_subprocess()

        assert result is False

    def test_refuses_to_spawn_when_a_listener_already_owns_the_port(self) -> None:
        from superlocalmemory.cli import daemon as _daemon

        with patch.object(_daemon, "is_daemon_running", return_value=False), patch.object(
            _daemon, "_has_tcp_listener", return_value=True,
        ), patch("subprocess.Popen") as popen:
            assert not _daemon._start_daemon_subprocess()

        popen.assert_not_called()


class TestEnsureDaemonStillWorks:
    """ensure_daemon() must keep its existing semantics for non-restart callers."""

    def test_ensure_daemon_short_circuit_when_running(self) -> None:
        """ensure_daemon returns True without locking when daemon is up."""
        from superlocalmemory.cli import daemon as _daemon

        with patch.object(_daemon, "is_daemon_running", return_value=True):
            assert _daemon.ensure_daemon() is True

    def test_ensure_daemon_delegates_to_helper_after_lock(
        self, tmp_path, monkeypatch,
    ) -> None:
        """ensure_daemon now delegates the actual spawn to the helper."""
        from superlocalmemory.cli import daemon as _daemon

        monkeypatch.setenv("SLM_TEST_ALLOW_DAEMON_SPAWN", "1")
        with patch.object(_daemon, "is_daemon_running", side_effect=[False, False]), \
             patch.object(_daemon, "_LOCK_FILE", tmp_path / "daemon.lock"), \
             patch.object(_daemon, "_start_daemon_subprocess", return_value=True) as helper:
            assert _daemon.ensure_daemon() is True

        helper.assert_called_once()


class TestOwnedDaemonShutdownWait:
    """Restart must wait for port release, not merely descriptor removal."""

    def test_wait_requires_process_exit_and_port_release(self) -> None:
        from types import SimpleNamespace

        from superlocalmemory.cli import daemon as _daemon

        descriptor = SimpleNamespace(port=18765)
        with patch.object(
            _daemon, "_descriptor_process_is_alive",
            side_effect=chain([True, False], repeat(False)),
        ), patch.object(
            _daemon, "_is_port_available", side_effect=[False, True],
        ), patch.object(_daemon.time, "sleep"):
            assert _daemon.wait_for_owned_daemon_shutdown(descriptor, timeout=1)

    def test_wait_refuses_restart_when_port_remains_owned(self) -> None:
        from types import SimpleNamespace

        from superlocalmemory.cli import daemon as _daemon

        descriptor = SimpleNamespace(port=18765)
        with patch.object(
            _daemon, "_descriptor_process_is_alive", return_value=False,
        ), patch.object(
            _daemon, "_is_port_available", return_value=False,
        ), patch.object(_daemon.time, "monotonic", side_effect=[0, 0, 2]), patch.object(
            _daemon.time, "sleep",
        ):
            assert not _daemon.wait_for_owned_daemon_shutdown(descriptor, timeout=1)

    def test_wait_tracks_verified_legacy_pid_and_custom_port(self) -> None:
        from superlocalmemory.cli import daemon as _daemon

        with patch.object(
            _daemon,
            "_is_verified_legacy_process",
            side_effect=chain([True, False], repeat(False)),
        ) as process_alive, patch.object(
            _daemon, "_is_port_available", side_effect=[False, True],
        ) as port_available, patch.object(_daemon.time, "sleep"):
            assert _daemon.wait_for_owned_daemon_shutdown(
                None,
                timeout=1,
                legacy_pid=7771,
                legacy_port=18766,
            )

        process_alive.assert_called_with(7771)
        port_available.assert_called_with(18766)

    def test_windows_port_release_probe_requests_exclusive_bind(self) -> None:
        from superlocalmemory.cli import daemon as _daemon

        candidate = MagicMock()
        candidate.__enter__.return_value = candidate
        candidate.__exit__.return_value = False
        with patch.object(_daemon.sys, "platform", "win32"), patch.object(
            _daemon.socket,
            "SO_EXCLUSIVEADDRUSE",
            0x4,
            create=True,
        ), patch.object(
            _daemon.socket, "socket", return_value=candidate,
        ):
            assert _daemon._is_port_available(18767)

        candidate.setsockopt.assert_called_once_with(
            _daemon.socket.SOL_SOCKET,
            0x4,
            1,
        )
        candidate.bind.assert_called_once_with(("127.0.0.1", 18767))

    def test_zombie_descriptor_process_is_not_treated_as_alive(self) -> None:
        from types import SimpleNamespace

        import psutil

        from superlocalmemory.cli import daemon as _daemon

        process = MagicMock()
        process.is_running.return_value = True
        process.status.return_value = psutil.STATUS_ZOMBIE
        descriptor = SimpleNamespace(pid=99991, process_create_time=1.0)
        with patch.object(_daemon, "_is_pid_alive", return_value=True), patch(
            "psutil.Process", return_value=process,
        ):
            assert _daemon._descriptor_process_is_alive(descriptor) is False


class TestRestartStep3UsesHelperNotEnsureDaemon:
    """B1 regression guard: Step 3 must NOT call ensure_daemon (would self-deadlock)."""

    def test_cmd_restart_imports_start_daemon_subprocess_not_ensure_daemon_in_step3(self) -> None:
        """The fix is observable in the source: Step 3 imports the helper, not ensure_daemon."""
        from pathlib import Path
        commands_py = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "superlocalmemory"
            / "cli"
            / "commands.py"
        )
        text = commands_py.read_text(encoding="utf-8")

        # Find the cmd_restart function body
        start = text.index("def cmd_restart(")
        end = text.index("\ndef ", start + 1)
        body = text[start:end]

        # Step 3 block must import _start_daemon_subprocess and call it
        assert "_start_daemon_subprocess" in body, (
            "cmd_restart Step 3 must call _start_daemon_subprocess (the v3.4.42 fix), "
            "not ensure_daemon (which would self-deadlock on the held lock)."
        )
        # And must NOT import ensure_daemon for Step 3 (the buggy form)
        assert "from superlocalmemory.cli.daemon import ensure_daemon" not in body, (
            "cmd_restart must not import ensure_daemon — that re-introduces the "
            "v3.4.13→v3.4.41 self-deadlock bug."
        )


def test_restart_never_scans_or_kills_machine_wide_processes() -> None:
    """V3.7 restart is namespace-owned, never process-name-wide."""
    import inspect

    from superlocalmemory.cli.commands import cmd_restart

    source = inspect.getsource(cmd_restart)
    assert "process_iter" not in source
    assert "pkill" not in source
    assert "stop_daemon" in source
