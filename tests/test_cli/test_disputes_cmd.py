# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm disputes settle|reopen``: any caller; recorded; the way back is named (D-49)."""

from __future__ import annotations

from argparse import Namespace

import pytest

from superlocalmemory.cli import disputes_cmd


@pytest.fixture
def sent(monkeypatch):
    from superlocalmemory.cli import daemon

    calls = []

    def fake(method, path, body=None, **flags):
        calls.append((method, path, body))
        if path.endswith("/settle"):
            return {"success": True, "dispute_id": "d1", "resolution": "b", "retired": ["f1"]}
        return {"success": True, "dispute_id": "d1", "restored": ["f1"]}

    monkeypatch.setattr(daemon, "daemon_request", fake)
    monkeypatch.setattr(disputes_cmd, "_daemon", lambda: True)
    monkeypatch.setattr("superlocalmemory.core.admission.gate_cli_mutation", lambda kind: None)
    monkeypatch.delenv("SLM_AGENT_ID", raising=False)
    return calls


def _args(sub, **kw):
    return Namespace(disputes_command=sub, profile="", json=False, dispute_id="d1", keep="f2", **kw)


def test_settle_sends_no_channel_and_names_the_way_back(sent, capsys) -> None:
    assert disputes_cmd.cmd_disputes(_args("settle")) == 0
    method, path, body = sent[-1]
    assert (method, path) == ("POST", "/api/disputes/d1/settle")
    assert body["keep"] == "f2" and "channel" not in body and body["by"]
    assert "slm disputes reopen d1" in capsys.readouterr().out


def test_reopen_takes_the_answer_back_and_names_the_agent(sent, capsys, monkeypatch) -> None:
    monkeypatch.setenv("SLM_AGENT_ID", "codex")
    assert disputes_cmd.cmd_disputes(_args("reopen")) == 0
    method, path, body = sent[-1]
    assert (method, path, body["by"]) == ("POST", "/api/disputes/d1/reopen", "agent:codex")
    assert "The retired memory is back" in capsys.readouterr().out


def test_reopen_is_registered() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    disputes_cmd.register_disputes_parsers(parser.add_subparsers(dest="command"))
    assert parser.parse_args(["disputes", "reopen", "d1"]).disputes_command == "reopen"
