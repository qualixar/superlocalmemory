# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A service that accepts the start but runs nothing must not block auto-start.

On Windows the logon task has no daemon PID to report, so the readiness wait
cannot see that ``schtasks /Run`` started nothing (for example when the task's
wrapper points at a Python that was removed). Auto-start used to wait 60 s
and give up with no direct start, where the release before always started the
daemon. It now starts the daemon directly with the task's settings, but only
when nothing owns the data folder or the port: a daemon that is starting
(instance lock), one that published its record, or anything listening on the
port is never doubled.

The real ``_start_daemon_subprocess`` and readiness wait run here. Only Task
Scheduler (``subprocess.run``), the direct spawn (``Popen``) and the 0.5 s
sleep between readiness checks are replaced. No real service manager is used.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from superlocalmemory.cli import daemon
from superlocalmemory.cli import daemon_supervisor as supervisor
from superlocalmemory.cli import service_installer as service
from superlocalmemory.cli.daemon_supervisor import JobState, StartOutcome, SupervisorJob
from superlocalmemory.infra.data_root import state_path
from superlocalmemory.infra.instance_lock import get_instance_lock
from tests.test_cli._service_manager_fakes import FakeTaskScheduler

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX simulation of Windows")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def world(monkeypatch, tmp_path):
    home, root = tmp_path / "home", tmp_path / "root"
    home.mkdir()
    root.mkdir()
    port = _free_port()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SLM_DATA_DIR", str(root))
    monkeypatch.setenv("SLM_DAEMON_PORT", str(port))
    monkeypatch.setenv("SLM_AGENT_ID", "client-agent")
    monkeypatch.delenv("SLM_SERVICE_LABEL", raising=False)
    scheduler = FakeTaskScheduler()
    monkeypatch.setattr(subprocess, "run", scheduler)
    monkeypatch.setattr(supervisor, "current_platform", lambda: "windows")
    wrapper = state_path(f"start-daemon-{service.service_identity()}.vbs")
    wrapper.write_text(service._windows_vbs_content())
    scheduler.tasks[service.service_identity()] = wrapper

    popen = MagicMock()
    popen.return_value.poll.return_value = None
    monkeypatch.setattr("subprocess.Popen", popen)
    real_wait = daemon._wait_for_daemon
    world = SimpleNamespace(scheduler=scheduler, popen=popen, root=root, port=port,
                            during_wait=lambda: None)

    def wait(timeout=60, proc=None):
        if proc is popen.return_value:
            return True  # the directly started daemon comes up
        world.during_wait()  # what the service's start did meanwhile
        return real_wait(timeout=timeout, proc=proc)

    monkeypatch.setattr(daemon, "_wait_for_daemon", wait)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    return world


def _start(world) -> bool:
    return daemon._start_daemon_subprocess(port=world.port)


def _ran_the_task(world) -> bool:
    return any(call[1] == "/Run" for call in world.scheduler.calls)


def test_task_that_starts_nothing_falls_back_to_a_direct_start(world, caplog):
    started = _start(world)
    assert world.popen.call_count == 1, "no direct start after the task started nothing"
    assert started is True
    assert _ran_the_task(world), "the installed task is still asked first"
    world.popen.assert_called_once()
    env = world.popen.call_args.kwargs["env"]
    assert env["SLM_DATA_DIR"] == str(world.root) and env["SLM_DAEMON_PORT"] == str(world.port)
    assert "SLM_AGENT_ID" not in env, "a fallback start is still scrubbed"
    # A broken task is never silent: every fallback names it and the fix.
    assert service.service_identity() in caplog.text
    assert "slm serve install" in caplog.text


# --- CRIT: the fallback never doubles a daemon that is on its way ---------------------

def test_daemon_that_holds_the_instance_lock_is_never_doubled(world):
    lock = get_instance_lock()
    world.during_wait = lambda: lock.try_acquire()  # the task's daemon is starting
    try:
        assert _start(world) is False
    finally:
        lock.release()
    world.popen.assert_not_called()


def test_daemon_that_published_its_record_is_never_doubled(world, monkeypatch):
    from superlocalmemory.infra.daemon_identity import build_descriptor

    record = build_descriptor(port=world.port, version="test", pid=4242, state="ready")
    published = {"record": None}
    world.during_wait = lambda: published.update(record=record)
    monkeypatch.setattr(daemon, "read_descriptor", lambda: published["record"])
    monkeypatch.setattr(daemon, "_descriptor_process_is_alive", lambda d: d is record)
    monkeypatch.setattr(daemon, "_fetch_health", lambda port: None)  # not answering yet
    assert _start(world) is False
    world.popen.assert_not_called()


def test_anything_listening_on_the_port_is_never_doubled(world):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    world.during_wait = lambda: (listener.bind(("127.0.0.1", world.port)), listener.listen())
    try:
        assert _start(world) is False
    finally:
        listener.close()
    world.popen.assert_not_called()


def test_task_scheduler_that_refuses_the_run_still_starts_directly(world):
    world.scheduler.run_rc = 1
    assert _start(world) is True
    world.popen.assert_called_once()


# --- the same guard on macOS and Linux ------------------------------------------------

def _service_job(platform: str) -> SupervisorJob:
    from pathlib import Path

    return SupervisorJob(platform, "com.example.slm.test", Path("/nonexistent"), 1)


@pytest.mark.parametrize("platform", ["macos", "linux"])
def test_manager_that_cannot_be_asked_after_the_start_falls_back_when_nothing_appeared(
    monkeypatch, platform,
):
    monkeypatch.setattr(supervisor, "find_job", lambda port: _service_job(platform))
    monkeypatch.setattr(supervisor, "request_start", lambda job: StartOutcome(True, "started"))
    monkeypatch.setattr(supervisor, "job_state", lambda job: JobState("unknown"))
    ready, job = supervisor.start_through_service(
        1, lambda proc: False, nothing_started=lambda: True)
    assert ready is None and job is not None


@pytest.mark.parametrize("platform", ["macos", "linux"])
def test_service_process_that_is_alive_is_never_doubled(monkeypatch, platform):
    monkeypatch.setattr(supervisor, "find_job", lambda port: _service_job(platform))
    monkeypatch.setattr(supervisor, "request_start", lambda job: StartOutcome(True, "started"))
    monkeypatch.setattr(supervisor, "job_state", lambda job: JobState("running", 4242))
    ready, _ = supervisor.start_through_service(
        1, lambda proc: False, nothing_started=lambda: True)
    assert ready is False


def test_ownership_check_that_fails_means_no_fallback(monkeypatch):
    monkeypatch.setattr(supervisor, "find_job", lambda port: _service_job("windows"))
    monkeypatch.setattr(supervisor, "request_start", lambda job: StartOutcome(True, "started"))
    monkeypatch.setattr(supervisor, "job_state", lambda job: JobState("idle"))

    def broken():
        raise OSError("data folder unreadable")

    ready, _ = supervisor.start_through_service(1, lambda proc: False, nothing_started=broken)
    assert ready is False


def test_callers_that_cannot_check_ownership_never_fall_back(monkeypatch):
    monkeypatch.setattr(supervisor, "find_job", lambda port: _service_job("windows"))
    monkeypatch.setattr(supervisor, "request_start", lambda job: StartOutcome(True, "started"))
    monkeypatch.setattr(supervisor, "job_state", lambda job: JobState("idle"))
    ready, _ = supervisor.start_through_service(1, lambda proc: False)
    assert ready is False
