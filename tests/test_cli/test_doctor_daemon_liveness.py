"""Doctor diagnoses daemon availability without starting it or waiting for boot."""

import json
import os
from argparse import Namespace
from dataclasses import replace
from unittest.mock import MagicMock

import psutil
import pytest

from superlocalmemory.cli import commands, daemon, doctor_daemon
from superlocalmemory.infra.daemon_identity import build_descriptor, write_descriptor


@pytest.fixture
def probes(monkeypatch):
    health = MagicMock(return_value=None)
    request = MagicMock(return_value={})
    def fetch(record, path, timeout):
        limit = (doctor_daemon.HEALTH_TIMEOUT if path == "/health"
                 else doctor_daemon.FEATURES_TIMEOUT)
        assert 0 < timeout <= limit
        return health(record.port, timeout=timeout) if path == "/health" else request(record)

    monkeypatch.setattr(doctor_daemon, "_fetch_json", fetch)
    # Unit tests inspect the worker's checks; socket/CLI tests exercise its deadline.
    monkeypatch.setattr(doctor_daemon, "check_daemon", doctor_daemon._inspect_daemon)
    # Any old readiness helper would wait or probe more than once.
    def forbidden(*a, **k):
        raise AssertionError("doctor must not start or wait for the daemon")

    for name in ("ensure_daemon", "is_daemon_running", "_verified_legacy_health"):
        monkeypatch.setattr(daemon, name, forbidden)
    monkeypatch.setattr(daemon, "daemon_request", request)
    monkeypatch.setattr("superlocalmemory.cli.features_cmd.daemon_request", request)
    monkeypatch.setattr("superlocalmemory.core.component_healer.heal_missing",
                        lambda *a, **k: {"healed": 0, "attempted": 0, "failed": 0})
    # Full doctor tests should not load a model or run a real worker.
    proc = MagicMock()
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: proc)
    monkeypatch.setattr(commands, "_readline_with_timeout", lambda *a: ('{"ok": true}', None))
    return health, request


def report(capsys, **flags):
    try:
        commands.cmd_doctor(Namespace(json=True, **flags))
    except SystemExit as exc:
        assert exc.code == 1
    data = json.loads(capsys.readouterr().out)["data"]
    return next(c for c in data["checks"] if c["name"] == "Daemon")


@pytest.mark.parametrize("flags", [{}, {"quick": True}, {"deep": True}, {"fix": True}])
def test_missing_daemon_is_visible_in_every_mode(probes, capsys, flags):
    row = report(capsys, **flags)
    assert row["status"] in {"WARN", "FAIL"}
    assert row["fix"] == "slm serve start"
    probes[0].assert_not_called()
    probes[1].assert_not_called()


def test_live_healthy_daemon_passes_one_short_probe(probes, capsys, free_port):
    record = build_descriptor(port=free_port, version="4.1.25", pid=os.getpid())
    write_descriptor(record)
    probes[0].return_value = record.public_health_fields()
    assert report(capsys, quick=True)["status"] == "PASS"
    probes[0].assert_called_once()
    assert probes[0].call_args.args == (free_port,)
    assert 0 < probes[0].call_args.kwargs["timeout"] <= 0.5
    probes[1].assert_called_once_with(record)


def test_dead_record_fails_with_restart(probes, capsys, free_port, monkeypatch):
    record = build_descriptor(port=free_port, version="4.1.25", pid=os.getpid())
    write_descriptor(record)
    monkeypatch.setattr("psutil.Process", MagicMock(side_effect=psutil.NoSuchProcess(999)))
    row = report(capsys, quick=True)
    assert row["status"] == "FAIL"
    assert "dead" in row["detail"] or "no longer running" in row["detail"]
    assert row["fix"] == "slm restart"
    probes[0].assert_not_called()


@pytest.mark.parametrize("reply", [None, {"service": "foreign"}])
def test_live_but_unhealthy_or_foreign_daemon_fails(probes, capsys, free_port, reply):
    write_descriptor(build_descriptor(port=free_port, version="4.1.25", pid=os.getpid()))
    probes[0].return_value = reply
    row = report(capsys, quick=True)
    assert row["status"] == "FAIL" and row["fix"] == "slm restart"
    assert probes[0].call_count == 1
    probes[1].assert_not_called()


def test_recycled_pid_never_counts_as_the_daemon(probes, capsys, free_port, monkeypatch):
    record = build_descriptor(port=free_port, version="4.1.25", pid=os.getpid())
    write_descriptor(replace(record, process_start_token="test:old-process"))
    monkeypatch.setattr("superlocalmemory.cli.doctor_daemon.process_start_token_for",
                        lambda pid: "test:new-process")
    probes[0].return_value = record.public_health_fields()
    row = report(capsys, quick=True)
    assert row["status"] == "FAIL" and row["fix"] == "slm restart"
    assert "reused" in row["detail"]
    probes[0].assert_not_called()


def test_broken_descriptor_is_reported(probes, capsys, tmp_path):
    (tmp_path / "daemon.json").write_text("{bad json")
    row = report(capsys, quick=True)
    assert row["status"] == "FAIL" and row["fix"] == "slm restart"
    probes[0].assert_not_called()


def test_inspection_failure_warns_and_other_checks_continue(probes, capsys, monkeypatch):
    monkeypatch.setattr(daemon, "read_descriptor", MagicMock(side_effect=OSError("unreadable")))
    row = report(capsys, quick=True)
    assert row["status"] == "WARN"
    assert row["fix"] == "slm serve start"


def test_daemon_row_is_also_printed_in_text(probes, capsys):
    try:
        commands.cmd_doctor(Namespace(json=False, quick=True))
    except SystemExit as exc:
        assert exc.code == 1
    out = capsys.readouterr().out
    assert "[WARN] Daemon:" in out and "slm serve start" in out


@pytest.mark.parametrize("healthy", [False, True])
def test_old_record_creation_time_drift_needs_health_proof(
    probes, free_port, monkeypatch, healthy,
):
    from superlocalmemory.cli.doctor_daemon import check_daemon

    record = build_descriptor(port=free_port, version="4.1.25", pid=os.getpid())
    record = replace(record, process_start_token=None, process_create_time=0)
    write_descriptor(record)
    monkeypatch.setattr("superlocalmemory.cli.doctor_daemon.process_start_token_for",
                        lambda pid: None)
    probes[0].return_value = record.public_health_fields() if healthy else None
    check = check_daemon()
    assert check.status == ("PASS" if healthy else "FAIL")
    if not healthy:
        assert "identity" in check.detail
        assert check.fix == "slm restart"
    assert probes[0].call_count == 1


def test_zombie_record_is_dead_without_health(probes, free_port, monkeypatch):
    from superlocalmemory.cli.doctor_daemon import check_daemon

    write_descriptor(build_descriptor(port=free_port, version="4.1.25", pid=os.getpid()))
    process = MagicMock()
    process.status.return_value = psutil.STATUS_ZOMBIE
    monkeypatch.setattr("psutil.Process", lambda pid: process)
    check = check_daemon()
    assert check.status == "FAIL" and "no longer running" in check.detail
    probes[0].assert_not_called()


def test_features_timeout_on_a_fresh_daemon_retries_then_says_still_starting(
        probes, monkeypatch, free_port):
    import time

    from superlocalmemory.cli.doctor_daemon import ProbeTimeout

    record = build_descriptor(port=free_port, version="4.1.25", pid=os.getpid())
    write_descriptor(record)
    budgets = []

    def fetch(rec, path, timeout):
        if path == "/health":
            return rec.public_health_fields()
        budgets.append(timeout)
        raise ProbeTimeout

    monkeypatch.setattr(doctor_daemon, "_fetch_json", fetch)
    result = doctor_daemon._inspect_daemon(time.monotonic() + doctor_daemon.CHECK_TIMEOUT)
    assert result.status == "PASS" and result.feature_warning == ""
    assert budgets == [doctor_daemon.FEATURES_TIMEOUT, doctor_daemon.WARMUP_TIMEOUT]
    assert "still starting" in result.feature_note and "slm doctor again" in result.feature_note


def test_a_fresh_daemon_whose_features_answer_on_the_retry_is_clean(monkeypatch, free_port):
    import time

    from superlocalmemory.cli.doctor_daemon import ProbeTimeout

    write_descriptor(build_descriptor(port=free_port, version="4.1.25", pid=os.getpid()))
    calls = []

    def fetch(rec, path, timeout):
        if path == "/health":
            return rec.public_health_fields()
        calls.append(timeout)
        if len(calls) == 1:
            raise ProbeTimeout
        return {"media": {}}

    monkeypatch.setattr(doctor_daemon, "_fetch_json", fetch)
    result = doctor_daemon._inspect_daemon(time.monotonic() + doctor_daemon.CHECK_TIMEOUT)
    assert (result.feature_warning, result.feature_note) == ("", "")


def test_an_old_daemon_that_does_not_answer_still_warns_without_retry(monkeypatch, free_port):
    import time

    from superlocalmemory.cli.doctor_daemon import ProbeTimeout

    old = time.time() - 3600
    write_descriptor(build_descriptor(
        port=free_port, version="4.1.25", pid=os.getpid(), started_at=old))
    calls = []

    def fetch(rec, path, timeout):
        if path == "/health":
            return rec.public_health_fields()
        calls.append(timeout)
        raise ProbeTimeout

    monkeypatch.setattr(doctor_daemon, "_fetch_json", fetch)
    result = doctor_daemon._inspect_daemon(time.monotonic() + doctor_daemon.CHECK_TIMEOUT)
    assert len(calls) == 1
    assert "did not answer within" in result.feature_warning and result.feature_note == ""


def test_doctor_prints_still_starting_as_a_pass_with_no_restart_advice(
        probes, capsys, monkeypatch):
    note = "Optional feature status is still starting - run slm doctor again in a minute"
    monkeypatch.setattr(doctor_daemon, "check_daemon", lambda: doctor_daemon.DaemonCheck(
        "PASS", "Running and healthy (pid 1)", feature_note=note))
    from tests.helpers.doctor import run_doctor_report

    run_doctor_report(Namespace(json=True, quick=True))
    rows = {c["name"]: c for c in json.loads(capsys.readouterr().out)["data"]["checks"]}
    assert rows["Daemon features"]["status"] == "PASS"
    assert rows["Daemon features"]["detail"] == note
    assert rows["Daemon features"]["fix"] == ""


def test_features_get_a_sane_budget_and_slowness_never_says_restart(
        monkeypatch, free_port, capsys, probes):
    import time


    assert doctor_daemon.FEATURES_TIMEOUT >= 2.0 > doctor_daemon.HEALTH_TIMEOUT
    assert doctor_daemon.CHECK_TIMEOUT >= doctor_daemon.FEATURES_TIMEOUT + doctor_daemon.HEALTH_TIMEOUT
    write_descriptor(build_descriptor(port=free_port, version="4.1.25", pid=os.getpid(),
                                      started_at=time.time() - 3600))
    slow = doctor_daemon.DaemonCheck(
        "PASS", "Running and healthy (pid 1)",
        feature_warning="Optional feature status did not answer within 2s")
    monkeypatch.setattr(doctor_daemon, "check_daemon", lambda: slow)
    from tests.helpers.doctor import run_doctor_report

    run_doctor_report(Namespace(json=True, quick=True))
    rows = {c["name"]: c for c in json.loads(capsys.readouterr().out)["data"]["checks"]}
    assert rows["Daemon features"]["status"] == "WARN"
    assert "restart" not in rows["Daemon features"]["fix"]
