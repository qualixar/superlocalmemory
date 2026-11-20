# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A service-started daemon takes the folder back from an unowned daemon, once.

Before: the service's daemon found the folder held, exited 0, and the service
manager (which restarts only on failure) never tried again, so an unowned
daemon with a client's environment served indefinitely. The service-manager
queries are mocked; the stop is mocked; no real daemon is touched.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from superlocalmemory.cli import daemon as client
from superlocalmemory.cli import daemon_supervisor as supervisor
from superlocalmemory.cli.daemon_supervisor import SupervisorJob
from superlocalmemory.infra import instance_lock
from superlocalmemory.server import supervisor_handover as handover

PORT = 18765
FOREIGN_PID = 999_001
_REAL_STOP = client.stop_daemon


def _job(name="svc"):
    return SupervisorJob("macos", name, Path(f"/x/{name}.plist"), PORT, {})


@pytest.fixture
def world(monkeypatch):
    state = SimpleNamespace(
        jobs=[_job()], pids={"svc": os.getpid()}, held=True,
        holder=SimpleNamespace(pid=FOREIGN_PID, port=PORT), alive=True,
        stop=MagicMock(return_value=True), freed=True,
    )
    monkeypatch.setattr(supervisor, "jobs_for", lambda port: state.jobs if port == PORT else [])
    monkeypatch.setattr(supervisor, "job_pid", lambda job: state.pids.get(job.name))
    monkeypatch.setattr(instance_lock, "instance_lock_is_held", lambda *a: state.held)
    monkeypatch.setattr(instance_lock, "wait_until_instance_lock_free",
                        lambda timeout_s=0: state.freed)
    monkeypatch.setattr(client, "read_descriptor", lambda: state.holder)
    monkeypatch.setattr(client, "_descriptor_process_is_alive", lambda d: state.alive)
    monkeypatch.setattr(client, "stop_daemon", state.stop)
    return state


def test_service_daemon_takes_over_from_an_unowned_daemon(world, caplog):
    assert handover.take_over_if_supervised(PORT) == "took_over"
    world.stop.assert_called_once()
    assert str(FOREIGN_PID) in caplog.text


def test_a_daemon_the_service_does_not_run_never_takes_over(world):
    world.pids = {"svc": 12345}  # the service runs some other process
    assert handover.take_over_if_supervised(PORT) == "unsupervised"
    world.stop.assert_not_called()


def test_no_installed_service_changes_nothing(world):
    world.jobs = []
    assert handover.take_over_if_supervised(PORT) == "unsupervised"
    world.stop.assert_not_called()


def test_free_folder_needs_no_hand_over(world):
    world.held = False
    assert handover.take_over_if_supervised(PORT) == "free"
    world.stop.assert_not_called()


def test_unprovable_holder_is_never_stopped(world):
    world.holder = None
    assert handover.take_over_if_supervised(PORT) == "holder_unknown"
    world.alive, world.holder = False, SimpleNamespace(pid=FOREIGN_PID, port=PORT)
    assert handover.take_over_if_supervised(PORT) == "holder_unknown"
    world.stop.assert_not_called()


def test_holder_run_by_another_service_is_left_alone(world):
    world.jobs = [_job("svc"), _job("legacy")]
    world.pids = {"svc": os.getpid(), "legacy": FOREIGN_PID}
    assert handover.take_over_if_supervised(PORT) == "holder_supervised"
    world.stop.assert_not_called()


def test_refused_hand_over_is_reported_once_and_never_raises(world, caplog):
    world.stop.return_value = False
    assert handover.take_over_if_supervised(PORT) == "stop_failed"
    assert "slm restart" in caplog.text
    world.stop.side_effect = RuntimeError("boom")
    assert handover.take_over_if_supervised(PORT) == "error"


def test_start_server_offers_hand_over_before_waiting_for_the_folder(monkeypatch):
    """Wiring: the check runs in the daemon start path, before the lock wait."""
    from superlocalmemory.server import unified_daemon

    order = []
    monkeypatch.setattr(handover, "take_over_if_supervised",
                        lambda port: order.append(("handover", port)) or "unsupervised")

    def refuse(lock, wait_s):
        order.append(("acquire", wait_s))
        return False

    monkeypatch.setattr(unified_daemon, "acquire_with_backoff", refuse)
    monkeypatch.setattr(unified_daemon, "install_thread_dump_signal", lambda: None)
    unified_daemon.start_server(port=PORT)
    assert [step for step, _ in order] == ["handover", "acquire"]
    assert order[0] == ("handover", PORT)


# --- the stop is bound to the holder that was checked ---------------------------

def _record(pid, instance):
    return SimpleNamespace(pid=pid, port=PORT, instance_id=instance, state="ready")


@pytest.fixture
def real_stop(world, monkeypatch):
    """The real ``stop_daemon``, with only its HTTP request and wait faked."""
    sent = []
    monkeypatch.setattr(client, "stop_daemon", _REAL_STOP)
    monkeypatch.setattr(client, "daemon_request", lambda method, path, **kw: (
        sent.append(kw.get("expected_descriptor")) or {"status": "stopping"}))
    monkeypatch.setattr(client, "wait_for_owned_daemon_shutdown", lambda *a, **k: True)
    return sent


def test_hand_over_never_stops_a_daemon_that_replaced_the_checked_one(world, real_stop,
                                                                       monkeypatch):
    checked, replacement = _record(FOREIGN_PID, "unowned"), _record(999_002, "newcomer")
    reads = iter([checked])
    monkeypatch.setattr(client, "read_descriptor", lambda: next(reads, replacement))
    assert handover.take_over_if_supervised(PORT) == "stop_failed"
    assert real_stop == [], "the record changed after the check: nothing may be stopped"


def test_hand_over_stops_exactly_the_checked_daemon(world, real_stop, monkeypatch):
    checked = _record(FOREIGN_PID, "unowned")
    monkeypatch.setattr(client, "read_descriptor", lambda: checked)
    assert handover.take_over_if_supervised(PORT) == "took_over"
    assert [d.instance_id for d in real_stop] == ["unowned"]


def test_plain_stop_still_stops_whatever_daemon_owns_the_folder(real_stop, monkeypatch):
    monkeypatch.setattr(client, "read_descriptor", lambda: _record(4, "current"))
    monkeypatch.setattr(client, "_descriptor_process_is_alive", lambda d: True)
    assert client.stop_daemon() is True
    assert [d.instance_id for d in real_stop] == ["current"]


# --- slm doctor --------------------------------------------------------------------

@pytest.fixture
def doctor_world(world, monkeypatch):
    from superlocalmemory.cli import doctor_supervisor

    world.jobs = [SupervisorJob("macos", "svc", Path("/x/svc.plist"), PORT,
                                {"SLM_RANKING": "v2-ensemble", "PATH": "/usr/bin"})]
    world.pids = {"svc": None}

    def state(job):
        pid = world.pids.get(job.name)
        return supervisor.JobState("running", pid) if pid else supervisor.JobState("idle")

    monkeypatch.setattr(supervisor, "job_state", state)
    monkeypatch.setattr(doctor_supervisor, "_environment_of",
                        lambda pid: {"SLM_AGENT_ID": "x", "PATH": "/other"})
    return doctor_supervisor


def test_doctor_flags_a_daemon_its_service_does_not_own(doctor_world):
    found = doctor_world.check_supervisor_ownership()
    assert found.status == "WARN"
    assert "not owned by its service svc" in found.detail
    assert "SLM_RANKING" in found.detail and "PATH" not in found.detail
    assert "v2-ensemble" not in found.detail, "values are never printed"
    assert "slm restart" in found.fix


def test_doctor_passes_a_service_owned_daemon(doctor_world, world):
    world.pids = {"svc": FOREIGN_PID}
    assert doctor_world.check_supervisor_ownership().status == "PASS"


def test_doctor_is_silent_without_a_service_or_a_daemon(doctor_world, world):
    world.jobs = []
    assert doctor_world.check_supervisor_ownership() is None
    world.jobs, world.holder = [_job()], None
    assert doctor_world.check_supervisor_ownership() is None
