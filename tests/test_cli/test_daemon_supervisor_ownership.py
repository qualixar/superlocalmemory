# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Auto-start and doctor trust a loaded job only when it is the installed one.

A definition edited without a reload, or an install that wrote a new file but
failed to load it, leaves the service manager holding a *stale* job under the
same name. Asking that job to start ran the old interpreter and settings, and
doctor reported its process as owned. The loaded job must match the file
(label, program arguments, data folder and port) before it is started or
trusted. launchd and systemd are in-memory fakes; nothing real is touched.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from superlocalmemory.cli import daemon_supervisor as supervisor
from superlocalmemory.cli import service_installer as service
from tests.test_cli._service_manager_fakes import FakeLaunchd, FakeSystemd

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX service simulation")

PORT = "19294"


@pytest.fixture
def home(monkeypatch, tmp_path):
    home, root = tmp_path / "home", tmp_path / "root with spaces"
    home.mkdir()
    root.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SLM_DATA_DIR", str(root))
    monkeypatch.setenv("SLM_DAEMON_PORT", PORT)
    monkeypatch.delenv("SLM_SERVICE_LABEL", raising=False)
    return home


@pytest.fixture(params=["legacy", "xml"])
def launchd(request, home, monkeypatch):
    """A loaded, owned LaunchAgent. ``xml`` isolates the ownership logic from
    the text parser; ``legacy`` is what launchd really prints."""
    fake = FakeLaunchd(fmt=request.param)
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setattr(supervisor, "current_platform", lambda: "macos")
    path = home / "Library" / "LaunchAgents" / f"{service.service_identity()}.plist"
    path.parent.mkdir(parents=True)
    path.write_text(service._macos_plist_content())
    fake.load(path)
    fake.path = path
    return fake


def _make_stale(fake: FakeLaunchd) -> None:
    """launchd still holds the job as first loaded; the file has since changed."""
    job = fake.jobs[fake.path.stem]
    job.plist = {**job.plist,
                 "ProgramArguments": ["/old/python", *job.plist["ProgramArguments"][1:]]}


def _job():
    (job,) = supervisor.jobs_for(int(PORT))
    return job


def test_owned_loaded_job_is_kickstarted(launchd):
    outcome = supervisor.request_start(_job())
    assert outcome.started and outcome.how == "kickstarted"
    assert any(call[1] == "kickstart" for call in launchd.calls)


def test_stale_loaded_job_is_never_started(launchd):
    _make_stale(launchd)
    outcome = supervisor.request_start(_job())
    assert not outcome.started
    assert "does not match" in outcome.detail
    assert not any(call[1] in {"kickstart", "bootstrap"} for call in launchd.calls)


def test_stale_loaded_job_falls_back_to_a_direct_start_with_the_files_settings(launchd):
    _make_stale(launchd)
    ready, job = supervisor.start_through_service(int(PORT), lambda *a, **k: True)
    assert ready is None, "the caller must start the daemon directly"
    assert job is not None and job.environment["SLM_DAEMON_PORT"] == PORT


def test_pid_of_a_stale_job_is_not_the_services(launchd):
    launchd.jobs[launchd.path.stem].pid = 777
    assert supervisor.job_pid(_job()) == 777
    _make_stale(launchd)
    assert supervisor.job_pid(_job()) is None
    assert supervisor.job_state(_job()).status == "stale"


def test_job_loaded_under_the_name_by_another_data_folder_is_not_ours(launchd, tmp_path):
    job = launchd.jobs[launchd.path.stem]
    job.pid = 778
    job.plist = {**job.plist, "EnvironmentVariables": {
        **job.plist["EnvironmentVariables"], "SLM_DATA_DIR": str(tmp_path)}}
    assert supervisor.job_pid(_job()) is None
    assert not supervisor.job_loaded(_job())


# --- Linux ----------------------------------------------------------------------

@pytest.fixture
def systemd(home, monkeypatch):
    fake = FakeSystemd(home=home)
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setattr(supervisor, "current_platform", lambda: "linux")
    path = fake.path(service.service_identity())
    path.parent.mkdir(parents=True)
    path.write_text(service._linux_service_content())
    fake.units[path.stem] = path.read_text()
    fake.unit = path.stem
    return fake


def test_linux_owned_unit_is_started(systemd):
    assert supervisor.request_start(_job()).started
    assert ["systemctl", "--user", "start", systemd.unit] in systemd.calls


def test_linux_unit_edited_without_reload_is_never_started(systemd):
    # systemd still runs the unit as loaded; the file now names another python.
    path = systemd.path(systemd.unit)
    loaded = systemd.units[systemd.unit]
    edited = loaded.replace(f"ExecStart={service.get_python_path()} ", "ExecStart=/new/python ")
    assert edited != loaded
    path.write_text(edited)
    job = _job()
    outcome = supervisor.request_start(job)
    assert not outcome.started and "does not match" in outcome.detail
    assert not any("start" in call for call in systemd.calls)
    systemd.pids[systemd.unit] = 99
    assert supervisor.job_pid(job) is None


def test_linux_pid_of_the_owned_unit(systemd):
    systemd.pids[systemd.unit] = 4343
    assert supervisor.job_pid(_job()) == 4343


# --- slm doctor -------------------------------------------------------------------

@pytest.fixture
def live_daemon(monkeypatch):
    """A live daemon record for pid 777 on the service's port."""
    from types import SimpleNamespace

    from superlocalmemory.cli import daemon as client
    from superlocalmemory.cli import doctor_supervisor

    record = SimpleNamespace(pid=777, port=int(PORT))
    monkeypatch.setattr(client, "read_descriptor", lambda: record)
    monkeypatch.setattr(client, "_descriptor_process_is_alive", lambda d: True)
    monkeypatch.setattr(doctor_supervisor, "_environment_of", lambda pid: {})
    return doctor_supervisor


def test_doctor_passes_the_owned_jobs_process(launchd, live_daemon):
    launchd.jobs[launchd.path.stem].pid = 777
    assert live_daemon.check_supervisor_ownership().status == "PASS"


def test_doctor_never_calls_a_stale_jobs_process_owned(launchd, live_daemon):
    launchd.jobs[launchd.path.stem].pid = 777
    _make_stale(launchd)
    found = live_daemon.check_supervisor_ownership()
    assert found.status == "WARN"
    assert "does not match its installed definition" in found.detail
    assert "slm serve install" in found.fix


# A WARN must name a fix that can work. When the service manager cannot be
# asked (no user bus, no GUI domain over a remote shell), `slm restart` changes
# nothing, so the ownership check is skipped on both platforms instead.

def test_doctor_skips_when_launchd_cannot_be_asked(launchd, live_daemon):
    launchd.query_rc = 1
    assert live_daemon.check_supervisor_ownership() is None


def test_doctor_skips_on_linux_without_a_user_bus(systemd, live_daemon):
    systemd.bus_rc = 1
    assert live_daemon.check_supervisor_ownership() is None


def test_doctor_skips_when_the_user_bus_goes_away_after_discovery(
    systemd, live_daemon, monkeypatch,
):
    jobs = supervisor.jobs_for(int(PORT))
    monkeypatch.setattr(supervisor, "jobs_for", lambda port: jobs)
    systemd.bus_rc = 1
    assert live_daemon.check_supervisor_ownership() is None


def test_doctor_still_warns_when_the_manager_answers_and_runs_another_process(
    launchd, live_daemon,
):
    launchd.jobs[launchd.path.stem].pid = 888  # the service runs something else
    found = live_daemon.check_supervisor_ownership()
    assert found.status == "WARN" and "slm restart" in found.fix
    launchd.jobs[launchd.path.stem].pid = None  # the service runs nothing
    assert live_daemon.check_supervisor_ownership().status == "WARN"


def test_doctor_passes_a_daemon_run_by_this_folders_service_on_another_port(
    live_daemon, monkeypatch,
):
    """#338: the daemon on this port may be run by the folder's other service."""
    from pathlib import Path

    from superlocalmemory.cli.daemon_supervisor import JobState, SupervisorJob

    here = SupervisorJob("macos", "svc-a", Path("/x/a.plist"), int(PORT), {})
    there = SupervisorJob("macos", "svc-b", Path("/x/b.plist"), int(PORT) + 1, {})
    monkeypatch.setattr(supervisor, "jobs_for",
                        lambda port: [here] if port == int(PORT) else [here, there])
    states = {"svc-a": JobState("idle", None), "svc-b": JobState("running", 777)}
    monkeypatch.setattr(supervisor, "job_state", lambda job: states[job.name])
    found = live_daemon.check_supervisor_ownership()
    assert found.status == "PASS", found
    assert "svc-b" in found.detail
