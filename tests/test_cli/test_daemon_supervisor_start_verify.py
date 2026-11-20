# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A service start counts only when the service manager really runs the daemon.

``launchctl bootstrap`` exiting 0 used to be taken as "started": a job that
was loaded but could not run (disabled, or an interpreter that exits at once)
then cost the client a 60 s wait and a failed start with no direct fallback.
The job is now kicked after loading, and a service that reports no process
for a grace period is treated as not running, so the client starts the
daemon directly with the service's settings. A service process that is still
starting is never doubled. launchd is an in-memory fake.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from superlocalmemory.cli import daemon
from superlocalmemory.cli import daemon_supervisor as supervisor
from superlocalmemory.cli import service_installer as service
from superlocalmemory.cli.daemon_supervisor import JobState, SupervisorJob
from tests.test_cli._service_manager_fakes import FakeLaunchd

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX service simulation")

PORT = "19295"


@pytest.fixture
def world(monkeypatch, tmp_path):
    home, root = tmp_path / "home", tmp_path / "root"
    home.mkdir()
    root.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SLM_DATA_DIR", str(root))
    monkeypatch.setenv("SLM_DAEMON_PORT", PORT)
    monkeypatch.delenv("SLM_SERVICE_LABEL", raising=False)
    fake = FakeLaunchd()
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setattr(supervisor, "current_platform", lambda: "macos")
    monkeypatch.setattr(supervisor, "_NO_PID_GRACE_S", 0.0, raising=False)
    path = home / "Library" / "LaunchAgents" / f"{service.service_identity()}.plist"
    path.parent.mkdir(parents=True)
    path.write_text(service._macos_plist_content())
    fake.path, fake.root = path, root  # installed, not loaded (after a bootout)

    monkeypatch.setattr(daemon, "is_daemon_running", lambda: False)
    monkeypatch.setattr(daemon, "_has_tcp_listener", lambda port: False)
    monkeypatch.setattr(daemon, "assert_no_durable_root_conflict", lambda: None)
    popen = MagicMock()
    popen.return_value.poll.return_value = None
    monkeypatch.setattr("subprocess.Popen", popen)
    fake.popen = popen

    def wait(timeout=60, proc=None):
        """Health never comes from the service; a direct child is healthy."""
        if proc is popen.return_value:
            return True
        for _ in range(timeout * 2):
            if proc is not None and proc.poll() is not None:
                return False
        return False

    monkeypatch.setattr(daemon, "_wait_for_daemon", wait)
    return fake


def _start() -> bool:
    return daemon._start_daemon_subprocess(port=int(PORT))


def test_loaded_job_that_cannot_run_falls_back_to_a_direct_start(world):
    world.kickstart_rc = 113  # loaded, but launchd will not run it (disabled)
    assert _start() is True
    assert [c[1] for c in world.calls if c[1] in {"bootstrap", "kickstart"}] == [
        "bootstrap", "kickstart"]
    world.popen.assert_called_once()
    env = world.popen.call_args.kwargs["env"]
    assert env["SLM_DATA_DIR"] == str(world.root) and env["SLM_DAEMON_PORT"] == PORT


def test_service_whose_daemon_exits_at_once_falls_back_without_the_full_wait(world):
    world.start_pid = None  # kickstart succeeds; the program dies immediately
    assert _start() is True
    world.popen.assert_called_once()


def test_service_daemon_that_is_running_but_not_ready_is_never_doubled(world):
    world.start_pid = 4242  # the service's process is alive, just slow
    assert _start() is False
    world.popen.assert_not_called()


def test_a_loaded_and_running_job_is_kickstarted_not_restarted(world):
    world.load(world.path, pid=4242)
    assert supervisor.request_start(supervisor.find_job(int(PORT))).how == "kickstarted"
    kicks = [c for c in world.calls if c[1] == "kickstart"]
    assert kicks and all("-k" not in c for c in kicks)


# --- the service process watcher ---------------------------------------------------

class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


@pytest.mark.parametrize("status", ["running", "unknown", "stale"])
def test_watcher_never_declares_a_death_it_cannot_observe(monkeypatch, status):
    monkeypatch.setattr(supervisor, "_NO_PID_GRACE_S", 5.0)
    monkeypatch.setattr(supervisor, "job_state", lambda job: JobState(status, 1))
    clock = _Clock()
    watcher = supervisor.ServiceProcess(SupervisorJob("macos", "j", Path("/x"), 1), clock)
    for clock.now in (0.0, 10.0, 1000.0):
        assert watcher.poll() is None


def test_watcher_on_windows_never_reports_an_exit(monkeypatch):
    monkeypatch.setattr(supervisor, "_NO_PID_GRACE_S", 0.0)
    monkeypatch.setattr(supervisor, "job_state", lambda job: JobState("idle"))
    watcher = supervisor.ServiceProcess(SupervisorJob("windows", "j", Path("/x"), 1))
    assert watcher.poll() is None


def test_watcher_waits_out_the_grace_and_a_pid_resets_it(monkeypatch):
    monkeypatch.setattr(supervisor, "_NO_PID_GRACE_S", 5.0)
    states = {"now": JobState("idle")}
    monkeypatch.setattr(supervisor, "job_state", lambda job: states["now"])
    clock = _Clock()
    watcher = supervisor.ServiceProcess(SupervisorJob("macos", "j", Path("/x"), 1), clock)
    clock.now = 4.9
    assert watcher.poll() is None, "a brief gap before launchd spawns is not a death"
    states["now"] = JobState("running", 7)
    clock.now = 6.0
    assert watcher.poll() is None
    states["now"] = JobState("idle")
    clock.now = 10.0
    assert watcher.poll() is None, "the grace restarts after the last seen process"
    clock.now = 11.1
    assert watcher.poll() == 1
    states["now"] = JobState("running", 8)
    assert watcher.poll() == 1, "an exit, once reported, is final"
