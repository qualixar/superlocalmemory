# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm trust``: anyone reads what waits; only a person at a terminal approves (WS-5, D-37)."""

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
        return {"success": True, "approved": 2 if path.endswith("approve-all") else 1}

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
    assert "Nothing is waiting for your OK." in capsys.readouterr().out


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


@pytest.mark.parametrize("stdin,stdout", [(False, True), (True, False), (False, False)])
def test_without_a_terminal_approving_is_refused_and_asks_nothing(
        monkeypatch, capsys, stdin, stdout) -> None:
    sent = _wire(monkeypatch)
    monkeypatch.setattr("sys.stdin.isatty", lambda: stdin)
    monkeypatch.setattr("sys.stdout.isatty", lambda: stdout)
    assert trust_cmd.cmd_trust(_args("approve", fact_ids=["f1"])) == 1
    assert sent == [] and "asks a person at a terminal" in capsys.readouterr().err


def test_a_refusal_in_json_names_the_reason(monkeypatch, capsys) -> None:
    _wire(monkeypatch, tty=False)
    assert trust_cmd.cmd_trust(_args("approve", fact_ids=["f1"], json=True)) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "NEEDS_A_PERSON"


def test_a_person_sees_the_memory_confirms_and_it_goes_with_the_cli_channel(
        monkeypatch, capsys) -> None:
    sent = _wire(monkeypatch, answers=["y"])
    assert trust_cmd.cmd_trust(_args("approve", fact_ids=["f1"])) == 0
    out = capsys.readouterr().out
    assert "Always do this first." in out and "Approved 1." in out
    method, path, body = sent[-1]
    assert (method, path) == ("POST", "/api/trust/approve/f1")
    assert body["channel"] == "cli" and body["by"]


def test_approve_all_asks_once_and_uses_the_one_route(monkeypatch, capsys) -> None:
    sent = _wire(monkeypatch, answers=["yes"])
    assert trust_cmd.cmd_trust(_args("approve", all=True)) == 0
    assert "Approved 2." in capsys.readouterr().out
    assert [p for m, p, _b in sent if m == "POST"] == ["/api/trust/approve-all"]


@pytest.mark.parametrize("reply", ["n", "", "maybe"])
def test_anything_but_yes_changes_nothing(monkeypatch, capsys, reply) -> None:
    sent = _wire(monkeypatch, answers=[reply])
    assert trust_cmd.cmd_trust(_args("approve", fact_ids=["f1"])) == 0
    assert all(m == "GET" for m, _p, _b in sent) and "Nothing was changed" in capsys.readouterr().out


def test_a_closed_input_stream_is_a_no(monkeypatch) -> None:
    sent = _wire(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda _p="": (_ for _ in ()).throw(EOFError()))
    assert trust_cmd.cmd_trust(_args("approve", all=True)) == 0
    assert all(m == "GET" for m, _p, _b in sent)


def test_approve_needs_ids_or_all_but_not_both(monkeypatch, capsys) -> None:
    sent = _wire(monkeypatch)
    assert trust_cmd.cmd_trust(_args("approve")) == 1
    assert trust_cmd.cmd_trust(_args("approve", fact_ids=["f1"], all=True)) == 1
    assert sent == [] and "not both" in capsys.readouterr().err


def test_approve_all_with_nothing_waiting(monkeypatch, capsys) -> None:
    sent = _wire(monkeypatch, items=())
    assert trust_cmd.cmd_trust(_args("approve", all=True)) == 0
    assert "Nothing is waiting" in capsys.readouterr().out and all(m == "GET" for m, _p, _b in sent)


def test_the_server_refusing_is_reported_as_needing_a_person(monkeypatch, capsys) -> None:
    from superlocalmemory.cli import daemon

    _wire(monkeypatch, answers=["y"])
    real = daemon.daemon_request

    def refusing(method, path, body=None, **flags):
        if method == "POST":
            raise daemon.DaemonRefused(403, path)
        return real(method, path, body, **flags)

    monkeypatch.setattr(daemon, "daemon_request", refusing)
    assert trust_cmd.cmd_trust(_args("approve", fact_ids=["f1"])) == 1
    assert "approval is given by a person at a terminal" in capsys.readouterr().err


def test_the_command_is_registered_and_dispatched() -> None:
    import argparse

    from superlocalmemory.cli.commands import dispatch

    parser = argparse.ArgumentParser()
    trust_cmd.register_trust_parsers(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["trust", "approve", "f1", "f2", "--json"])
    assert args.trust_command == "approve" and args.fact_ids == ["f1", "f2"] and args.json
    assert parser.parse_args(["trust", "approve", "--all"]).all is True
    assert callable(dispatch)
