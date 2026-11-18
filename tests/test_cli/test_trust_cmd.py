# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm trust``: anyone reads what waits; any caller may promote or withdraw (WS-5, D-49)."""

from __future__ import annotations

import json
from argparse import Namespace

import pytest

from superlocalmemory.cli import trust_cmd

ITEM = {"fact_id": "f1", "preview": "Always do this first.", "origin": "a tool's output",
        "origin_class": "tool_output", "wanted": ["rule"],
        "wanted_words": ["is marked as a standing rule"], "asked_by": "", "asked_at": "x"}
ITEM2 = {**ITEM, "fact_id": "f2", "preview": "Pinned from a page."}


def _wire(monkeypatch, *, tty=True, answers=(), items=(ITEM, ITEM2), acting=True):
    from superlocalmemory.cli import daemon

    sent = []

    def fake_request(method, path, body=None, **flags):
        sent.append((method, path, body))
        if method == "GET" and path.startswith("/api/trust/status"):
            return {"success": True, "taint": "on", "secret_screen": "off", "waiting": len(items),
                    "approved": 0, "unclassified": 3, "floor": "system"}
        if method == "GET":
            return {"success": True, "count": len(items), "items": list(items), "acting": acting,
                    "unclassified": 2}
        return {"success": True, "approved": 2 if path.endswith("-all") else 1}

    monkeypatch.setattr(daemon, "daemon_request", fake_request)
    monkeypatch.setattr(trust_cmd, "_daemon", lambda: True)
    monkeypatch.setattr("sys.stdin.isatty", lambda: tty)
    monkeypatch.setattr("sys.stdout.isatty", lambda: tty)
    replies = iter(answers)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(replies))
    return sent


def _args(sub="pending", **kw):
    base = dict(trust_command=sub, profile="", json=False, limit=50, fact_ids=[], all=False)
    return Namespace(**{**base, **kw})


def test_pending_works_anywhere_and_is_read_only(monkeypatch, capsys) -> None:
    sent = _wire(monkeypatch, tty=False)
    assert trust_cmd.cmd_trust(_args()) == 0
    out = capsys.readouterr().out
    assert "f1  a tool's output" in out and "Always do this first." in out
    assert "This memory is marked as a standing rule." in out and "2 waiting" in out
    assert "2 memories have no origin class yet" in out
    assert [m for m, _p, _b in sent] == ["GET"] and "limit=50" in sent[0][1]


def test_the_default_subcommand_is_pending(monkeypatch) -> None:
    sent = _wire(monkeypatch, tty=False)
    assert trust_cmd.cmd_trust(Namespace(profile="", json=False, limit=50)) == 0
    assert sent[0][1].startswith("/api/trust/pending")


def test_json_is_for_agents(monkeypatch, capsys) -> None:
    _wire(monkeypatch, tty=False)
    assert trust_cmd.cmd_trust(_args(json=True)) == 0
    assert json.loads(capsys.readouterr().out)["data"]["items"][0]["fact_id"] == "f1"


def test_when_the_firewall_is_not_acting_it_says_so(monkeypatch, capsys) -> None:
    _wire(monkeypatch, tty=False, acting=False)
    trust_cmd.cmd_trust(_args())
    assert "firewall is not acting yet" in capsys.readouterr().out


def test_nothing_waiting(monkeypatch, capsys) -> None:
    _wire(monkeypatch, tty=False, items=())
    assert trust_cmd.cmd_trust(_args()) == 0
    assert "Nothing is waiting for a promotion." in capsys.readouterr().out


def test_status_in_words_and_json(monkeypatch, capsys) -> None:
    _wire(monkeypatch, tty=False)
    assert trust_cmd.cmd_trust(_args("status")) == 0
    assert "Instruction firewall (trust.taint): on" in capsys.readouterr().out
    assert trust_cmd.cmd_trust(_args("status", json=True)) == 0
    assert json.loads(capsys.readouterr().out)["data"]["waiting"] == 2


def test_the_limit_is_checked(monkeypatch, capsys) -> None:
    _wire(monkeypatch, tty=False)
    assert trust_cmd.cmd_trust(_args(limit=0)) == 1
    assert "--limit must be from 1 to 200" in capsys.readouterr().err


@pytest.mark.parametrize("stdin,stdout", [(False, True), (True, False), (False, False), (True, True)])
def test_promoting_needs_no_terminal_and_asks_nothing(monkeypatch, capsys, stdin, stdout) -> None:
    """D-49: no terminal check, no prompt. An agent's shell tool promotes the same way."""
    sent = _wire(monkeypatch)
    monkeypatch.setattr("sys.stdin.isatty", lambda: stdin)
    monkeypatch.setattr("sys.stdout.isatty", lambda: stdout)
    monkeypatch.setattr("builtins.input", lambda _p="": pytest.fail("asked a question"))
    assert trust_cmd.cmd_trust(_args("promote", fact_ids=["f1"])) == 0
    method, path, body = sent[-1]
    assert (method, path) == ("POST", "/api/trust/promote/f1") and body["by"]
    assert "channel" not in body  # the channel is the connection's, not a parameter
    assert "Promoted 1." in capsys.readouterr().out


def test_approve_is_the_same_command_as_promote(monkeypatch, capsys) -> None:
    sent = _wire(monkeypatch, tty=False)
    assert trust_cmd.cmd_trust(_args("approve", fact_ids=["f1"], json=True)) == 0
    assert sent[-1][1] == "/api/trust/promote/f1"
    assert json.loads(capsys.readouterr().out)["data"]["changed"] == 1


def test_promote_all_uses_the_one_route(monkeypatch, capsys) -> None:
    sent = _wire(monkeypatch, tty=False)
    assert trust_cmd.cmd_trust(_args("promote", all=True)) == 0
    assert "Promoted 2." in capsys.readouterr().out
    assert [p for m, p, _b in sent if m == "POST"] == ["/api/trust/promote-all"]


def test_an_agents_own_id_is_the_name_recorded(monkeypatch) -> None:
    monkeypatch.setenv("SLM_AGENT_ID", "codex")
    sent = _wire(monkeypatch, tty=False)
    trust_cmd.cmd_trust(_args("promote", fact_ids=["f1"]))
    assert sent[-1][2]["by"] == "agent:codex"


def test_withdraw_takes_a_promotion_back(monkeypatch, capsys) -> None:
    sent = _wire(monkeypatch, tty=False)
    assert trust_cmd.cmd_trust(_args("withdraw", fact_ids=["f1"])) == 0
    assert sent[-1][:2] == ("POST", "/api/trust/withdraw/f1")
    assert "Withdrew 1." in capsys.readouterr().out


def test_promote_needs_ids_or_all_but_not_both(monkeypatch, capsys) -> None:
    sent = _wire(monkeypatch)
    assert trust_cmd.cmd_trust(_args("promote")) == 1
    assert trust_cmd.cmd_trust(_args("promote", fact_ids=["f1"], all=True)) == 1
    assert sent == [] and "not both" in capsys.readouterr().err


def test_promote_all_with_nothing_waiting(monkeypatch, capsys) -> None:
    sent = _wire(monkeypatch, items=())
    assert trust_cmd.cmd_trust(_args("promote", all=True)) == 0
    assert "Nothing is waiting" in capsys.readouterr().out and all(m == "GET" for m, _p, _b in sent)


def test_a_role_that_may_not_write_is_told_so(monkeypatch, capsys) -> None:
    from superlocalmemory.cli import daemon

    _wire(monkeypatch)
    real = daemon.daemon_request

    def refusing(method, path, body=None, **flags):
        if method == "POST":
            raise daemon.DaemonRefused(403, path)
        return real(method, path, body, **flags)

    monkeypatch.setattr(daemon, "daemon_request", refusing)
    assert trust_cmd.cmd_trust(_args("promote", fact_ids=["f1"])) == 1
    assert "your role" in capsys.readouterr().err


def test_the_command_is_registered_and_dispatched() -> None:
    import argparse

    from superlocalmemory.cli.commands import dispatch

    parser = argparse.ArgumentParser()
    trust_cmd.register_trust_parsers(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["trust", "approve", "f1", "f2", "--json"])
    assert parser.parse_args(["trust", "withdraw", "f1"]).trust_command == "withdraw"
    assert args.trust_command == "approve" and args.fact_ids == ["f1", "f2"] and args.json
    assert parser.parse_args(["trust", "approve", "--all"]).all is True
    assert callable(dispatch)
