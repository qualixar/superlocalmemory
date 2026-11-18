# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm audit``: any caller labels; a person at a terminal, an agent without one (W2, D-49)."""

from __future__ import annotations

from argparse import Namespace

import pytest

from superlocalmemory.cli import audit_cmd

ITEM = {"item_id": "h1", "question": "Memory A was saved before memory B. What does B do to A?",
        "memories": [{"name": "A", "fact_id": "a", "content": "weekly", "said_at": "2026-03-01"},
                     {"name": "B", "fact_id": "b", "content": "monthly", "said_at": "2026-11-14"}],
        "options": [{"label": "replaces", "text": "B replaces A"}, {"label": "cannot_tell", "text": "I can't tell"}]}


def _wire(monkeypatch, *, tty=True, answers=(), pending=None):
    from superlocalmemory.cli import daemon

    sent = []

    def fake_request(method, path, body=None, **flags):
        sent.append((method, path, body))
        if method == "GET":
            return pending if pending is not None else {"success": True, "count": 1, "items": [ITEM]}
        return {"success": True, "labelled": 1}

    monkeypatch.setattr(daemon, "daemon_request", fake_request)
    monkeypatch.setattr(audit_cmd, "_daemon", lambda: True)
    monkeypatch.setattr("sys.stdin.isatty", lambda: tty)
    monkeypatch.setattr("sys.stdout.isatty", lambda: tty)
    replies = iter(answers)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(replies))
    return sent


def _args(**kw):
    base = dict(count=False, profile="", json=False, list_queue=False, answer=None,
                retract=None, model="")
    return Namespace(**{**base, **kw})


def test_without_a_terminal_it_lists_the_queue_and_asks_nothing(monkeypatch, capsys) -> None:
    """D-49: no terminal gate. An agent reads the queue and labels with --answer."""
    sent = _wire(monkeypatch, tty=False)
    monkeypatch.setattr("builtins.input", lambda _p="": pytest.fail("asked a question"))
    assert audit_cmd.cmd_audit(_args()) == 0
    out = capsys.readouterr().out
    assert "weekly" in out and "slm audit --answer ITEM_ID LABEL" in out
    assert [m for m, _p, _b in sent] == ["GET"]


def test_the_queue_as_json_for_agents_carries_the_agents_model(monkeypatch, capsys) -> None:
    sent = _wire(monkeypatch, tty=False)
    assert audit_cmd.cmd_audit(_args(list_queue=True, json=True, model="claude sonnet")) == 0
    assert "model=claude%20sonnet" in sent[0][1]
    import json

    assert json.loads(capsys.readouterr().out)["data"]["items"][0]["item_id"] == "h1"


def test_an_agent_answers_without_a_terminal_and_names_its_model(monkeypatch, capsys) -> None:
    monkeypatch.setenv("SLM_AGENT_ID", "codex")
    sent = _wire(monkeypatch, tty=False)
    monkeypatch.setattr("superlocalmemory.cli.daemon.daemon_request",
                        lambda m, p, b=None, **f: sent.append((m, p, b)) or
                        {"success": True, "labelled": 2, "counted": 2})
    assert audit_cmd.cmd_audit(_args(answer=["h1", "replaces"], model="gpt-x")) == 0
    assert sent[-1] == ("POST", "/api/audit/h1/answer",
                        {"label": "replaces", "model": "gpt-x", "by": "agent:codex"})
    assert "counts toward 2" in capsys.readouterr().out


def test_an_answer_with_no_model_is_saved_but_says_it_does_not_count(monkeypatch, capsys) -> None:
    sent = _wire(monkeypatch, tty=False)
    monkeypatch.setattr("superlocalmemory.cli.daemon.daemon_request",
                        lambda m, p, b=None, **f: sent.append((m, p, b)) or
                        {"success": True, "labelled": 2, "counted": 0})
    assert audit_cmd.cmd_audit(_args(answer=["h1", "same"])) == 0
    assert sent[-1][2]["model"] == "" and "does not count" in capsys.readouterr().out


def test_a_label_can_be_retracted(monkeypatch, capsys) -> None:
    sent = _wire(monkeypatch, tty=False)
    monkeypatch.setattr("superlocalmemory.cli.daemon.daemon_request",
                        lambda m, p, b=None, **f: sent.append((m, p, b)) or
                        {"success": True, "retracted": 2})
    assert audit_cmd.cmd_audit(_args(retract="h1")) == 0
    assert sent[-1][:2] == ("POST", "/api/audit/h1/retract")
    assert "Label taken back" in capsys.readouterr().out


def test_count_works_anywhere_and_writes_nothing(monkeypatch, capsys) -> None:
    sent = _wire(monkeypatch, tty=False)
    assert audit_cmd.cmd_audit(_args(count=True)) == 0
    assert [m for m, _p, _b in sent] == ["GET"] and "1 question(s) waiting" in capsys.readouterr().out


def test_a_numbered_answer_is_saved_as_the_option_label(monkeypatch, capsys) -> None:
    sent = _wire(monkeypatch, answers=["7", "1"])
    assert audit_cmd.cmd_audit(_args()) == 0
    out = capsys.readouterr().out
    assert "A (2026-03-01): weekly" in out and "Please type one of the numbers" in out
    method, path, body = sent[-1]
    assert (method, path) == ("POST", "/api/audit/h1/answer")
    assert body["label"] == "replaces" and body["model"] == "human"  # a person at the terminal
    assert "Saved 1 answer(s)" in out


@pytest.mark.parametrize("reply", ["s", "q"])
def test_skip_or_stop_saves_nothing(monkeypatch, reply) -> None:
    sent = _wire(monkeypatch, answers=[reply])
    assert audit_cmd.cmd_audit(_args()) == 0
    assert [m for m, _p, _b in sent] == ["GET"]


def test_an_empty_queue_says_so(monkeypatch, capsys) -> None:
    _wire(monkeypatch, pending={"success": True, "count": 0, "items": []})
    assert audit_cmd.cmd_audit(_args()) == 0
    assert "Nothing to answer right now" in capsys.readouterr().out
