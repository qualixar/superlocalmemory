# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""D-49 on the command line: `slm flags kill on|off` goes through the daemon, which records it as a
decision (channel cli). With no daemon the switch still moves, and the CLI says it was not recorded:
the CLI never opens the store for writing itself (the #386 write boundary)."""

from __future__ import annotations

from argparse import Namespace

from superlocalmemory.cli import daemon as daemon_mod
from superlocalmemory.cli import flags_cmd
from superlocalmemory.core.feature_flags import KILL_ENV, kill_switch_on


def test_with_a_daemon_the_change_goes_through_its_recorded_route(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.delenv(KILL_ENV, raising=False)
    calls: list[tuple] = []
    answer = {"success": True, "kill_switch": {"on": True}}
    monkeypatch.setattr(daemon_mod, "daemon_request",
                        lambda method, path, body=None, **_: calls.append((method, path, body)) or answer)
    flags_cmd.cmd_flags(Namespace(action="kill", value="on", json=False))
    assert calls == [("POST", "/api/flags/kill", {"on": True})]


def test_without_a_daemon_the_switch_moves_and_the_cli_says_it_was_not_recorded(
        tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.delenv(KILL_ENV, raising=False)

    def absent(*_a, **_k):
        raise daemon_mod.DaemonNotFound("not running")

    monkeypatch.setattr(daemon_mod, "daemon_request", absent)
    flags_cmd.cmd_flags(Namespace(action="kill", value="on", json=False))
    assert kill_switch_on(tmp_path, {})
    assert "not in the list of decisions" in capsys.readouterr().err


def test_no_daemon_answer_means_not_recorded_and_the_file_is_set(tmp_path, monkeypatch, capsys) -> None:
    """daemon_request answers None when nothing runs: that is not a recorded change."""
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.delenv(KILL_ENV, raising=False)
    monkeypatch.setattr(daemon_mod, "daemon_request", lambda *a, **k: None)
    flags_cmd.cmd_flags(Namespace(action="kill", value="on", json=False))
    assert kill_switch_on(tmp_path, {})
    assert "not in the list of decisions" in capsys.readouterr().err
