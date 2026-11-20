# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Auto-start goes through the installed service; a direct start is scrubbed.

Reproduces the split-brain seen after an upgrade: with the service unloaded,
an already-running MCP client auto-started a detached daemon that carried the
client's environment (agent name, channel, tool profile) and none of the
service's own settings, then held the port so the service's daemon exited.
Every service-manager call here is mocked; nothing reaches launchd/systemd.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from superlocalmemory.cli import daemon
from superlocalmemory.cli import daemon_supervisor as supervisor
from superlocalmemory.cli.daemon_supervisor import JobState, StartOutcome, SupervisorJob
from tests._portable import GUARD_ENV

CLIENT_ENV = {
    "SLM_AGENT_ID": "agent-under-test",
    "SLM_CLIENT_CHANNEL": "mcp",
    "SLM_MCP_PROFILE": "whole",
    "SLM_SKIP_DEP_CHECK": "1",
    "SLM_DAEMON_INSTANCE_ID": "client-instance",
    "SLM_DAEMON_CAPABILITY": "client-capability",
    "SLM_HOOK_DAEMON_TIMEOUT": "3",
}


def _job(port: int, platform: str = "macos", **environment: str) -> SupervisorJob:
    return SupervisorJob(platform, "com.example.slm.test", Path("/nonexistent/job.plist"),
                         port, dict(environment))


@pytest.fixture
def start_env(monkeypatch, free_port):
    for key, value in CLIENT_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("SLM_RANKING", "client-ranking")
    monkeypatch.setattr(daemon, "is_daemon_running", lambda: False)
    monkeypatch.setattr(daemon, "_has_tcp_listener", lambda port: False)
    monkeypatch.setattr(daemon, "assert_no_durable_root_conflict", lambda: None)
    popen = MagicMock()
    popen.return_value.poll.return_value = None
    monkeypatch.setattr("subprocess.Popen", popen)
    monkeypatch.setattr(daemon, "_wait_for_daemon", lambda timeout=60, proc=None: True)
    return SimpleNamespace(port=free_port, popen=popen)


# --- (a) the service starts its own daemon ---------------------------------------

def test_installed_service_is_asked_to_start_and_nothing_is_spawned(start_env, monkeypatch):
    job = _job(start_env.port, SLM_RANKING="v2-ensemble")
    monkeypatch.setattr(supervisor, "find_job", lambda port: job if port == start_env.port else None)
    asked = MagicMock(return_value=StartOutcome(True, "kickstarted"))
    monkeypatch.setattr(supervisor, "request_start", asked)

    assert daemon._start_daemon_subprocess(port=start_env.port) is True
    asked.assert_called_once_with(job)
    start_env.popen.assert_not_called()


def test_service_start_that_never_becomes_healthy_reports_failure(start_env, monkeypatch):
    """A daemon owns the folder but never answers: never start a second one."""
    job = _job(start_env.port)
    monkeypatch.setattr(supervisor, "find_job", lambda port: job)
    monkeypatch.setattr(supervisor, "request_start",
                        lambda j: StartOutcome(True, "kickstarted"))
    monkeypatch.setattr(daemon, "_wait_for_daemon", lambda timeout=60, proc=None: False)
    monkeypatch.setattr(daemon, "_data_folder_is_owned", lambda: True)
    assert daemon._start_daemon_subprocess(port=start_env.port) is False
    start_env.popen.assert_not_called()


def test_failed_service_start_falls_back_to_scrubbed_spawn_with_service_settings(
    start_env, monkeypatch, caplog,
):
    job = _job(start_env.port, SLM_RANKING="v2-ensemble", SLM_AGENT_ID="never-from-a-job")
    monkeypatch.setattr(supervisor, "find_job", lambda port: job)
    monkeypatch.setattr(supervisor, "request_start",
                        lambda j: StartOutcome(False, "failed", "domain unavailable"))

    assert daemon._start_daemon_subprocess(port=start_env.port) is True
    start_env.popen.assert_called_once()
    env = start_env.popen.call_args.kwargs["env"]
    assert env["SLM_RANKING"] == "v2-ensemble", "service settings must reach the daemon"
    for key in ("SLM_AGENT_ID", "SLM_CLIENT_CHANNEL", "SLM_MCP_PROFILE",
                "SLM_SKIP_DEP_CHECK", "SLM_HOOK_DAEMON_TIMEOUT"):
        assert key not in env, key
    assert env["SLM_DAEMON_INSTANCE_ID"] != "client-instance"
    assert env["SLM_DAEMON_CAPABILITY"] != "client-capability"
    assert job.name in caplog.text and "directly" in caplog.text


# --- (b) a direct start never inherits the client ------------------------------

def test_direct_spawn_without_service_never_inherits_client_identity(start_env, monkeypatch):
    monkeypatch.setattr(supervisor, "find_job", lambda port: None)
    asked = MagicMock()
    monkeypatch.setattr(supervisor, "request_start", asked)

    assert daemon._start_daemon_subprocess(port=start_env.port) is True
    asked.assert_not_called()
    env = start_env.popen.call_args.kwargs["env"]
    for key in ("SLM_AGENT_ID", "SLM_CLIENT_CHANNEL", "SLM_MCP_PROFILE",
                "SLM_SKIP_DEP_CHECK", "SLM_HOOK_DAEMON_TIMEOUT"):
        assert key not in env, key
    assert env["SLM_DAEMON_INSTANCE_ID"] != "client-instance"
    assert env["SLM_RANKING"] == "client-ranking", "operator settings are not client identity"
    assert env["OMP_NUM_THREADS"] == "1"
    assert os.environ["SLM_AGENT_ID"] == "agent-under-test", "the client's own env is untouched"


def test_daemon_environment_is_pure_and_service_values_win():
    base = {"SLM_AGENT_ID": "a", "SLM_RANKING": "client", "PATH": "/x"}
    job = _job(1, SLM_RANKING="service", SLM_MCP_PROFILE="never")
    env = supervisor.daemon_environment(base, job)
    assert env == {"SLM_RANKING": "service", "PATH": "/x"}
    assert base == {"SLM_AGENT_ID": "a", "SLM_RANKING": "client", "PATH": "/x"}


def test_daemon_environment_drops_every_caller_session_and_keeps_the_daemons_settings():
    caller = {"SLM_SESSION_ID": "s", "CLAUDE_SESSION_ID": "c", "CLAUDE_CODE_SESSION_ID": "cc",
              "SLM_AGENT_ID": "a", "SLM_CLIENT_CHANNEL": "mcp", "SLM_USER_SESSION": "u"}
    settings = {"SLM_DATA_DIR": "/data", "SLM_DAEMON_PORT": "9999", "SLM_RANKING": "r",
                "OPENROUTER_API_KEY": "k1", "OPENAI_API_KEY": "k2", "ANTHROPIC_API_KEY": "k3",
                "PATH": "/x", "HOME": "/h", **GUARD_ENV}
    assert supervisor.daemon_environment({**caller, **settings}) == settings
    # The service definition's own settings still win, and still never carry a session.
    job = _job(9999, SLM_DATA_DIR="/service-data", OPENAI_API_KEY="service",
               SLM_REQUIRE_CREDENTIALS="1", CLAUDE_SESSION_ID="x")
    env = supervisor.daemon_environment({**caller, **settings}, job)
    assert env == {**settings, "SLM_DATA_DIR": "/service-data", "OPENAI_API_KEY": "service",
                   "SLM_REQUIRE_CREDENTIALS": "1"}


# --- the service-manager commands ------------------------------------------------

class _Runner:
    def __init__(self, codes: dict[str, int]):
        self.codes, self.calls = codes, []

    def __call__(self, argv):
        self.calls.append(argv)
        return SimpleNamespace(returncode=self.codes.get(argv[1], 0), stdout="", stderr="")


def _loaded(monkeypatch, loaded_states):
    """Stub the verified job state: True is an owned loaded job, False none."""
    states = iter(loaded_states)
    monkeypatch.setattr(supervisor, "job_state", lambda job: JobState(
        "idle" if next(states) else "not_loaded"))


def _macos(monkeypatch, loaded_states, codes=None):
    _loaded(monkeypatch, loaded_states)
    runner = _Runner(codes or {})
    monkeypatch.setattr(supervisor, "_run", runner)
    monkeypatch.setattr(supervisor, "_macos_domain", lambda: "gui/501")
    return runner


def test_macos_loaded_job_is_kickstarted_without_restart(monkeypatch):
    runner = _macos(monkeypatch, [True])
    outcome = supervisor.request_start(_job(1))
    assert outcome == StartOutcome(True, "kickstarted")
    assert runner.calls == [["launchctl", "kickstart", "gui/501/com.example.slm.test"]]
    assert "-k" not in runner.calls[0], "a running job must never be killed"


def test_macos_unloaded_job_is_loaded_not_bypassed(monkeypatch):
    runner = _macos(monkeypatch, [False, True])
    outcome = supervisor.request_start(_job(1))
    assert outcome == StartOutcome(True, "loaded")
    # Loaded is not running: the job is kicked too (a no-op if RunAtLoad ran it).
    assert runner.calls == [["launchctl", "bootstrap", "gui/501", "/nonexistent/job.plist"],
                            ["launchctl", "kickstart", "gui/501/com.example.slm.test"]]


def test_macos_concurrent_loader_wins_then_job_is_kickstarted(monkeypatch):
    runner = _macos(monkeypatch, [False, True], {"bootstrap": 5})
    assert supervisor.request_start(_job(1)) == StartOutcome(True, "kickstarted")
    assert [c[1] for c in runner.calls] == ["bootstrap", "kickstart"]


def test_macos_load_failure_is_reported_not_raised(monkeypatch):
    _macos(monkeypatch, [False, False], {"bootstrap": 119})
    outcome = supervisor.request_start(_job(1))
    assert not outcome.started and "119" in outcome.detail


def test_linux_and_windows_use_their_service_managers(monkeypatch):
    _loaded(monkeypatch, [True, True])
    runner = _Runner({})
    monkeypatch.setattr(supervisor, "_run", runner)
    assert supervisor.request_start(_job(1, "linux")).started
    assert supervisor.request_start(_job(1, "windows")).started
    assert runner.calls == [["systemctl", "--user", "start", "com.example.slm.test"],
                            ["schtasks", "/Run", "/TN", "com.example.slm.test"]]


def test_service_manager_missing_is_a_failed_outcome(monkeypatch):
    def missing(argv):
        raise FileNotFoundError(argv[0])

    _loaded(monkeypatch, [True])
    monkeypatch.setattr(supervisor, "_run", missing)
    outcome = supervisor.request_start(_job(1, "linux"))
    assert not outcome.started and "systemctl" in outcome.detail


def test_job_without_a_verified_definition_is_never_trusted(monkeypatch):
    """The manager's raw answer is not ownership: a PID it reports for a job
    that was never checked against its file is not the service's daemon."""
    from superlocalmemory.cli import service_live

    monkeypatch.setattr(service_live, "launchctl_list",
                        lambda name: {"Label": name, "PID": 4242})
    monkeypatch.setattr(supervisor, "_run", lambda argv: SimpleNamespace(
        returncode=0, stdout="LoadState=loaded\nMainPID=77\n", stderr=""))
    for platform in ("macos", "linux"):
        assert supervisor.job_pid(_job(1, platform)) is None
        assert supervisor.job_state(_job(1, platform)).status == "unknown"
        assert not supervisor.request_start(_job(1, platform)).started
    assert supervisor.job_pid(_job(1, "windows")) is None


def test_jobs_are_matched_by_port_and_carry_their_definition_environment(monkeypatch):
    import plistlib

    from superlocalmemory.cli import service_inventory

    def row(name, port):
        content = plistlib.dumps({"Label": name, "EnvironmentVariables": {
            "SLM_DAEMON_PORT": str(port), "SLM_RANKING": "v2-ensemble"}}).decode()
        return service_inventory.ServiceDefinition(name, Path(f"/x/{name}.plist"),
                                                   port, content, ())

    monkeypatch.setattr(supervisor, "current_platform", lambda: "macos")
    monkeypatch.setattr(service_inventory, "definitions",
                        lambda platform: [row("a", 9001), row("b", 9002)])
    jobs = supervisor.jobs_for(9002)
    assert [j.name for j in jobs] == ["b"]
    assert jobs[0].environment["SLM_RANKING"] == "v2-ensemble"
    assert supervisor.find_job(9999) is None


def test_discovery_failure_means_no_service(monkeypatch):
    from superlocalmemory.cli import service_inventory

    monkeypatch.setattr(supervisor, "current_platform", lambda: "linux")

    def broken(platform):
        raise OSError("systemctl missing")

    monkeypatch.setattr(service_inventory, "definitions", broken)
    assert supervisor.find_job(8765) is None


def test_linux_and_windows_definition_environment_parsing():
    unit = ("[Unit]\nDescription=x\n[Service]\nExecStart=/p -m m\n"
            'Environment="SLM_DATA_DIR=/d"\nEnvironment="SLM_RANKING=v2"\n')
    assert supervisor._linux_environment(unit) == {"SLM_DATA_DIR": "/d", "SLM_RANKING": "v2"}
    vbs = ('Set WshShell = CreateObject("WScript.Shell")\n'
           'WshShell.Environment("Process")("SLM_RANKING") = "v2"\n'
           'WshShell.Run """p"" -m m", 0, False\n')
    assert supervisor._windows_environment(vbs) == {"SLM_RANKING": "v2"}


# --- thread settings: the service's value wins, the client's never does -------------

def test_service_thread_settings_win_over_the_direct_start_defaults(start_env, monkeypatch):
    job = _job(start_env.port, OMP_NUM_THREADS="4", KMP_DUPLICATE_LIB_OK="FALSE")
    monkeypatch.setattr(supervisor, "find_job", lambda port: job)
    monkeypatch.setattr(supervisor, "request_start",
                        lambda j: StartOutcome(False, "failed", "refused"))
    assert daemon._start_daemon_subprocess(port=start_env.port) is True
    env = start_env.popen.call_args.kwargs["env"]
    assert env["OMP_NUM_THREADS"] == "4"
    assert env["KMP_DUPLICATE_LIB_OK"] == "FALSE"


def test_client_thread_settings_never_reach_the_daemon(start_env, monkeypatch):
    """Serial OpenMP prevents a known crash; a client's shell value must not undo it."""
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "FALSE")
    monkeypatch.setattr(supervisor, "find_job", lambda port: None)
    assert daemon._start_daemon_subprocess(port=start_env.port) is True
    env = start_env.popen.call_args.kwargs["env"]
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["KMP_DUPLICATE_LIB_OK"] == "TRUE"


# --- #337: auto-start and an installer never change the service at once -----------

def _hold_service_lock(home: Path):
    """Hold the service-mutation lock the way another process's installer does."""
    import fcntl
    import os

    path = home / ".slm-service-mutation.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


@pytest.mark.skipif(sys.platform == "win32", reason="flock")
def test_auto_start_skips_the_service_while_an_installer_holds_it(monkeypatch, tmp_path):
    import os

    monkeypatch.setenv("HOME", str(tmp_path))
    calls: list = []
    monkeypatch.setattr(supervisor, "_run", lambda argv: calls.append(argv))
    fd = _hold_service_lock(tmp_path)
    try:
        outcome = supervisor.request_start(_job(1))
    finally:
        os.close(fd)
    assert outcome.started is False
    assert outcome.how == "busy"
    assert "install" in outcome.detail
    assert calls == []  # launchctl was never asked to load the file being removed


@pytest.mark.skipif(sys.platform == "win32", reason="flock")
def test_auto_start_holds_the_service_lock_while_it_asks_the_manager(monkeypatch, tmp_path):
    from superlocalmemory.infra import service_mutation

    monkeypatch.setenv("HOME", str(tmp_path))
    seen: list = []

    def starter(job):
        seen.append(getattr(service_mutation._HELD, "lock", None) is not None)
        return StartOutcome(True, "started")

    monkeypatch.setitem(supervisor._STARTERS, "linux", starter)
    assert supervisor.request_start(_job(1, "linux")).started
    assert seen == [True]
    assert getattr(service_mutation._HELD, "lock", None) is None  # released after
