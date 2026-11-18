# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm proposals``: anyone reads and decides; recorded and reversible (WS-6, D-49)."""

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
@pytest.mark.parametrize("tty", [True, False])
def test_a_decision_needs_no_terminal_and_asks_nothing(monkeypatch, sub, tty) -> None:
    """D-49: an agent's shell tool decides the same way a person does; nothing is prompted."""
    sent = _wire(monkeypatch, tty=tty)
    monkeypatch.setattr("builtins.input", lambda _p="": pytest.fail("asked a question"))
    assert proposals_cmd.cmd_proposals(_args(sub)) == 0
    method, path, body = sent[-1]
    assert (method, path) == ("POST", f"/api/proposals/s1/{sub}") and body["by"]
    assert "channel" not in body  # the channel is the connection's, never a parameter


def test_the_set_is_shown_and_the_way_back_is_named(monkeypatch, capsys) -> None:
    _wire(monkeypatch, tty=False)
    assert proposals_cmd.cmd_proposals(_args("apply")) == 0
    out = capsys.readouterr().out
    assert "Retire “weekly”" in out and "Undo it any time with: slm proposals rollback s1" in out


def test_an_agents_own_id_is_the_name_recorded(monkeypatch) -> None:
    monkeypatch.setenv("SLM_AGENT_ID", "codex")
    sent = _wire(monkeypatch, tty=False)
    proposals_cmd.cmd_proposals(_args("apply"))
    assert sent[-1][2]["by"] == "agent:codex"


def test_a_role_that_may_not_write_is_told_so(monkeypatch, capsys) -> None:
    from superlocalmemory.cli import daemon

    _wire(monkeypatch)
    real = daemon.daemon_request

    def refusing(method, path, body=None, **flags):
        if method == "POST":
            raise daemon.DaemonRefused(403, path)
        return real(method, path, body, **flags)

    monkeypatch.setattr(daemon, "daemon_request", refusing)
    assert proposals_cmd.cmd_proposals(_args("apply")) == 1
    assert "your role" in capsys.readouterr().err


def test_a_conflict_is_reported_as_a_refusal(monkeypatch, capsys) -> None:
    from superlocalmemory.cli import daemon

    _wire(monkeypatch)
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
