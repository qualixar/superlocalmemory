# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The macOS service checks read the job record launchd really prints.

``launchctl list <label>`` prints legacy text that ``plutil`` rejects (array
items and the closing brace end in ';') and that never carries the job's
environment. Every check that read it failed closed on a real Mac: a loaded,
running service looked foreign, so ``slm serve uninstall`` and a reinstall
refused to unload it and auto-start never recognised the service. The
environment is read from ``launchctl print`` instead. No real launchd is used.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from superlocalmemory.cli import service_installer as service
from superlocalmemory.cli import service_inventory, service_live
from superlocalmemory.cli.launchctl_record import parse_list_output, print_environment
from tests.test_cli._service_manager_fakes import FakeLaunchd

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX launchd simulation")

# Shape captured from `launchctl list <label>` on macOS 26 (paths replaced).
REAL_LIST_TEXT = """{
\t"StandardOutPath" = "/Users/u/.superlocalmemory/logs/service.log";
\t"LimitLoadToSessionType" = "Aqua";
\t"StandardErrorPath" = "/Users/u/.superlocalmemory/logs/service-error.log";
\t"Label" = "com.qualixar.superlocalmemory";
\t"OnDemand" = true;
\t"LastExitStatus" = 0;
\t"PID" = 37912;
\t"Program" = "/Users/u/venv/bin/python";
\t"ProgramArguments" = (
\t\t"/Users/u/venv/bin/python";
\t\t"-m";
\t\t"superlocalmemory.server.unified_daemon";
\t\t"--start";
\t\t"--port=8765";
\t);
};
"""


def test_launchctl_list_text_is_parsed_not_rejected():
    record = parse_list_output(REAL_LIST_TEXT)
    assert record["Label"] == "com.qualixar.superlocalmemory"
    assert record["PID"] == 37912 and record["OnDemand"] is True
    assert record["ProgramArguments"][1:] == [
        "-m", "superlocalmemory.server.unified_daemon", "--start", "--port=8765"]
    assert "EnvironmentVariables" not in record, "launchd never prints it here"


@pytest.mark.parametrize("broken", ["", "{", '{ "a" = ; };', '{ "a" = "b" };', "( 1; 2; );"])
def test_unreadable_list_text_fails_closed(broken):
    with pytest.raises(ValueError):
        parse_list_output(broken)


def test_print_environment_is_the_jobs_own_block_only():
    text = ("gui/501/x = {\n\tinherited environment = {\n\t\tSECRET => s\n\t}\n\n"
            "\tdefault environment = {\n\t\tPATH => /usr/bin\n\t}\n\n"
            "\tenvironment = {\n\t\tSLM_DATA_DIR => /a b/c\n\t\tSLM_DAEMON_PORT => 8765\n\t}\n}\n")
    assert print_environment(text) == {"SLM_DATA_DIR": "/a b/c", "SLM_DAEMON_PORT": "8765"}
    with pytest.raises(ValueError):
        print_environment("gui/501/x = {\n\tpid = 1\n}\n")


@pytest.fixture
def launchd(monkeypatch, tmp_path):
    home, root = tmp_path / "home", tmp_path / "root with spaces"
    home.mkdir()
    root.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SLM_DATA_DIR", str(root))
    monkeypatch.setenv("SLM_DAEMON_PORT", "19293")
    monkeypatch.delenv("SLM_SERVICE_LABEL", raising=False)
    fake = FakeLaunchd()
    monkeypatch.setattr(subprocess, "run", fake)
    path = home / "Library" / "LaunchAgents" / f"{service.service_identity()}.plist"
    path.parent.mkdir(parents=True)
    path.write_text(service._macos_plist_content())
    fake.load(path, pid=37912)
    fake.home, fake.root, fake.path = home, root, path
    return fake


def _row():
    (row,) = service_inventory.definitions("macos")
    return row


def test_loaded_service_is_recognised_from_launchds_real_answers(launchd):
    row = _row()
    assert service_live.verified_live("macos", row) is True
    record = service_live.macos_owned_record(row)
    assert record["PID"] == 37912
    assert any(call[1] == "print" for call in launchd.calls), "environment came from print"


def test_uninstall_unloads_a_loaded_service_on_a_real_mac(launchd):
    assert service.uninstall_macos() is True
    assert ["launchctl", "unload", str(launchd.path)] in launchd.calls
    assert not launchd.path.exists()


def test_job_of_another_data_folder_under_the_same_label_is_still_refused(launchd):
    foreign = launchd.root.parent / "foreign"
    foreign.mkdir()
    job = launchd.jobs[launchd.path.stem]
    job.plist = {**job.plist, "EnvironmentVariables": {
        **job.plist["EnvironmentVariables"], "SLM_DATA_DIR": str(foreign)}}
    with pytest.raises(service_live.JobMismatch):
        service_live.verified_live("macos", _row())
    assert service.uninstall_macos() is False
    assert not any(call[1] == "unload" for call in launchd.calls)
    assert launchd.path.exists()


def test_launchd_that_cannot_be_read_is_a_query_failure_not_a_mismatch(launchd):
    row = _row()
    launchd.query_rc = 1
    with pytest.raises(service_live.ManagerQueryFailed):
        service_live.macos_owned_record(row)
