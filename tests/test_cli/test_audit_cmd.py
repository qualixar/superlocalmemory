# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm audit``: a person answers at a terminal; a script or agent cannot (W2, R-16)."""

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
    return Namespace(count=False, profile="", json=False, **kw)


def test_without_a_terminal_it_refuses_and_asks_nothing(monkeypatch, capsys) -> None:
    sent = _wire(monkeypatch, tty=False)
    assert audit_cmd.cmd_audit(_args()) == 1
    assert sent == [] and "asks a person at a terminal" in capsys.readouterr().err


def test_count_works_anywhere_and_writes_nothing(monkeypatch, capsys) -> None:
    sent = _wire(monkeypatch, tty=False)
    assert audit_cmd.cmd_audit(Namespace(count=True, profile="", json=False)) == 0
    assert [m for m, _p, _b in sent] == ["GET"] and "1 question(s) waiting" in capsys.readouterr().out


def test_a_numbered_answer_is_saved_as_the_option_label(monkeypatch, capsys) -> None:
    sent = _wire(monkeypatch, answers=["7", "1"])
    assert audit_cmd.cmd_audit(_args()) == 0
    out = capsys.readouterr().out
    assert "A (2026-03-01): weekly" in out and "Please type one of the numbers" in out
    assert sent[-1] == ("POST", "/api/audit/h1/answer", {"label": "replaces", "channel": "cli"})
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
