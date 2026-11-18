# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm proposals``: anyone reads; only a person at a terminal decides (WS-6, I-10)."""

from __future__ import annotations

import json
from argparse import Namespace

import pytest

from superlocalmemory.cli import proposals_cmd

SET = {"set_id": "s1", "state": "proposed", "title": "Tidy", "basis": None, "proposals": [
    {"proposal_id": "p1", "kind": "retire", "state": "proposed",
     "summary": "Retire “weekly” because “monthly” replaces it"}]}


def _wire(monkeypatch, *, tty=True, answers=()):
    from superlocalmemory.cli import daemon

    sent = []

    def fake_request(method, path, body=None, **flags):
        sent.append((method, path, body))
        if method == "GET" and path.startswith("/api/proposals/"):
            return {"success": True, "set": SET}
        if method == "GET":
            return {"success": True, "count": 1, "sets": [SET]}
        return {"success": True, "set": {**SET, "state": "applied"}}

    monkeypatch.setattr(daemon, "daemon_request", fake_request)
    monkeypatch.setattr(proposals_cmd, "_daemon", lambda: True)
    monkeypatch.setattr("sys.stdin.isatty", lambda: tty)
    monkeypatch.setattr("sys.stdout.isatty", lambda: tty)
    replies = iter(answers)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(replies))
    return sent


def _args(sub="list", **kw):
    base = dict(proposals_command=sub, profile="", json=False, state="proposed", limit=50,
                set_id="s1")
    return Namespace(**{**base, **kw})


def test_listing_works_anywhere_and_is_read_only(monkeypatch, capsys) -> None:
    sent = _wire(monkeypatch, tty=False)
    assert proposals_cmd.cmd_proposals(_args()) == 0
    out = capsys.readouterr().out
    assert "Retire “weekly” because “monthly” replaces it" in out and "1 waiting" in out
    assert [m for m, _p, _b in sent] == ["GET"] and "state=proposed" in sent[0][1]


def test_json_listing_is_for_agents(monkeypatch, capsys) -> None:
    _wire(monkeypatch, tty=False)
    assert proposals_cmd.cmd_proposals(_args(json=True)) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["data"]["sets"][0]["set_id"] == "s1"


@pytest.mark.parametrize("sub", ["apply", "reject", "rollback"])
def test_without_a_terminal_a_decision_is_refused_and_asks_nothing(monkeypatch, capsys, sub) -> None:
    sent = _wire(monkeypatch, tty=False)
    assert proposals_cmd.cmd_proposals(_args(sub)) == 1
    assert sent == [] and "asks a person at a terminal" in capsys.readouterr().err


def test_a_person_sees_the_set_confirms_and_it_is_sent_with_the_cli_channel(
        monkeypatch, capsys) -> None:
    sent = _wire(monkeypatch, answers=["y"])
    assert proposals_cmd.cmd_proposals(_args("apply")) == 0
    out = capsys.readouterr().out
    assert "Retire “weekly”" in out and "Applied. Undo it any time with: slm proposals rollback s1" in out
    assert sent[-1] == ("POST", "/api/proposals/s1/apply", {"channel": "cli"})


@pytest.mark.parametrize("reply", ["n", "", "maybe"])
def test_anything_but_yes_changes_nothing(monkeypatch, capsys, reply) -> None:
    sent = _wire(monkeypatch, answers=[reply])
    assert proposals_cmd.cmd_proposals(_args("apply")) == 0
    assert [m for m, _p, _b in sent] == ["GET"] and "Nothing was changed" in capsys.readouterr().out


def test_a_closed_input_stream_is_a_no(monkeypatch) -> None:
    sent = _wire(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda _p="": (_ for _ in ()).throw(EOFError()))
    assert proposals_cmd.cmd_proposals(_args("rollback")) == 0
    assert [m for m, _p, _b in sent] == ["GET"]


def test_a_conflict_is_reported_as_a_refusal(monkeypatch, capsys) -> None:
    from superlocalmemory.cli import daemon

    _wire(monkeypatch, answers=["y"])
    real = daemon.daemon_request

    def refusing(method, path, body=None, **flags):
        if method == "POST":
            raise daemon.DaemonConflict("This set is applied, not waiting for a decision.")
        return real(method, path, body, **flags)

    monkeypatch.setattr(daemon, "daemon_request", refusing)
    assert proposals_cmd.cmd_proposals(_args("apply")) == 1
    assert "This set is applied" in capsys.readouterr().err


def test_the_command_is_registered_and_dispatched() -> None:
    import argparse

    from superlocalmemory.cli.commands import dispatch

    parser = argparse.ArgumentParser()
    proposals_cmd.register_proposals_parsers(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["proposals", "show", "s1", "--json"])
    assert args.proposals_command == "show" and args.set_id == "s1" and args.json
    assert callable(dispatch)
