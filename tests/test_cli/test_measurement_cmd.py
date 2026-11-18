# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm measurement``: the notice, each flag in plain words, one off switch (D-28)."""

from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace

import pytest

from superlocalmemory.cli import measurement_cmd
from superlocalmemory.core.feature_flags import FeatureFlags
from superlocalmemory.outcomes import disclosure


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    for name in ("SLM_FEATURE_OUTCOMES_HOLDOUT", "SLM_FEATURE_OUTCOMES_SWAPS",
                 "SLM_FEATURE_OUTCOMES_RECEIPTS", "SLM_4230_OFF"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(measurement_cmd, "_home", lambda: tmp_path)
    return tmp_path


def _args(action="status", **kw):
    return Namespace(action=action, json=kw.pop("json", False), receipts=kw.pop("receipts", False))


def test_status_prints_the_notice_the_first_time_only(_home, capsys) -> None:
    assert measurement_cmd.cmd_measurement(_args()) == 0
    first = capsys.readouterr().out
    assert "SuperLocalMemory now measures" in first and "slm measurement off" in first
    assert disclosure.pending(_home) is False
    measurement_cmd.cmd_measurement(_args())
    second = capsys.readouterr().out
    assert "SuperLocalMemory now measures" not in second
    assert "Withholding: on" in second and "Usage check: off" in second


def test_status_json_carries_the_notice_but_does_not_count_as_seen(_home, capsys) -> None:
    assert measurement_cmd.cmd_measurement(_args(json=True)) == 0
    out = json.loads(capsys.readouterr().out)
    data = out["data"]
    assert out["success"] is True and data["pending"] is True and data["measuring"] is True
    assert "slm measurement off" in data["notice"]
    assert {f["id"] for f in data["flags"]} >= {"outcomes.holdout", "outcomes.swaps"}
    assert disclosure.pending(_home) is True  # an agent reading it is not a person seeing it


def test_off_then_on_changes_only_the_two_interventions(_home, capsys) -> None:
    assert measurement_cmd.cmd_measurement(_args("off")) == 0
    flags = FeatureFlags(_home, environ={})
    assert not flags.enabled("outcomes.holdout") and not flags.enabled("outcomes.swaps")
    assert flags.enabled("outcomes.receipts")
    assert "off" in capsys.readouterr().out.lower()
    measurement_cmd.cmd_measurement(_args("on"))
    assert FeatureFlags(_home, environ={}).enabled("outcomes.holdout")


def test_off_with_receipts_turns_the_receipts_off_too(_home) -> None:
    measurement_cmd.cmd_measurement(_args("off", receipts=True))
    assert not FeatureFlags(_home, environ={}).enabled("outcomes.receipts")


def test_off_json_names_what_changed(_home, capsys) -> None:
    measurement_cmd.cmd_measurement(_args("off", json=True))
    data = json.loads(capsys.readouterr().out)["data"]
    assert data["changed"] == {"outcomes.holdout": "off", "outcomes.swaps": "off"}
    assert data["measuring"] is False


def test_off_twice_says_there_was_nothing_to_change(_home, capsys) -> None:
    measurement_cmd.cmd_measurement(_args("off"))
    capsys.readouterr()
    measurement_cmd.cmd_measurement(_args("off"))
    assert "already off" in capsys.readouterr().out.lower()


def test_the_kill_switch_is_reported(_home, capsys) -> None:
    from superlocalmemory.core.feature_flags import set_kill_switch

    set_kill_switch(_home, True)
    measurement_cmd.cmd_measurement(_args())
    out = capsys.readouterr().out
    assert "kill switch" in out.lower() and "Measuring: no" in out


def test_the_parser_accepts_the_three_actions_and_defaults_to_status() -> None:
    import argparse

    root = argparse.ArgumentParser()
    measurement_cmd.register_measurement_parsers(root.add_subparsers(dest="command"))
    assert root.parse_args(["measurement"]).action == "status"
    ns = root.parse_args(["measurement", "off", "--receipts", "--json"])
    assert (ns.action, ns.receipts, ns.json) == ("off", True, True)
    with pytest.raises(SystemExit):
        root.parse_args(["measurement", "sideways"])


def test_the_real_cli_is_wired_and_listed_in_help(tmp_path) -> None:
    env = {"SLM_DATA_DIR": str(tmp_path), "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    done = subprocess.run([sys.executable, "-m", "superlocalmemory.cli.main", "measurement",
                           "status", "--json"], capture_output=True, text=True, env=env, timeout=120)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["data"]["measuring"] is True
    assert not (tmp_path / "disclosures.json").exists()  # JSON does not acknowledge


def test_setup_and_upgrade_hosts_show_the_notice_once_without_a_prompt(_home, capsys, monkeypatch) -> None:
    import builtins

    monkeypatch.setattr(builtins, "input", lambda *_: (_ for _ in ()).throw(AssertionError("prompted")))
    assert measurement_cmd.announce_if_pending() is True
    assert "slm measurement off" in capsys.readouterr().err
    # pytest's stderr is not a terminal, so it is shown again until a person sees it
    assert disclosure.pending(_home) is True
    measurement_cmd.cmd_measurement(_args())
    capsys.readouterr()
    assert measurement_cmd.announce_if_pending() is False


def test_a_failure_to_show_the_notice_never_breaks_setup(monkeypatch) -> None:
    def broken():
        raise RuntimeError("no home")

    monkeypatch.setattr(measurement_cmd, "_home", broken)
    assert measurement_cmd.announce_if_pending() is False


def test_cmd_setup_and_upgrade_hosts_call_it() -> None:
    import inspect

    from superlocalmemory.cli import commands, host_upgrades

    assert "announce_if_pending" in inspect.getsource(commands.cmd_setup)
    assert "announce_if_pending" in inspect.getsource(host_upgrades.cmd_upgrade_hosts)
