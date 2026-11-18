# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W3 open item: before-web recall goes through the hook queue when the daemon is
up, so it is a ``hook_recall`` delivery of this conversation (receipt, usage join,
never shown twice); otherwise it keeps the old command path."""

from __future__ import annotations

import io
import json
from pathlib import Path

from superlocalmemory.hooks import before_web_hook as bw


class _Queue:
    asked: list = []

    def __init__(self, path) -> None:
        pass

    def enqueue(self, **kw):
        _Queue.asked.append(kw)
        return "r1"

    def poll_result(self, request_id, timeout_s):
        assert timeout_s <= bw._RECALL_TIMEOUT_SEC
        return {"ok": True, "results": [{"content": "JWT tokens expire in 1h", "score": 0.9,
                                         "fact_id": "f1"}]}

    def close(self) -> None:
        pass


def _socket(monkeypatch, up: bool) -> None:
    monkeypatch.setattr("superlocalmemory.hooks.hook_daemon.owned_socket_path",
                        lambda: Path("/tmp/slm.sock") if up else None)


def test_with_the_daemon_up_the_queue_answers_for_this_conversation(monkeypatch) -> None:
    _Queue.asked = []
    _socket(monkeypatch, True)
    monkeypatch.setattr("superlocalmemory.core.recall_queue.RecallQueue", _Queue)
    monkeypatch.setattr("superlocalmemory.hooks.auto_recall_hook._detect_mode", lambda: "A")
    monkeypatch.setattr("superlocalmemory.hooks.auto_recall_hook._format_envelope",
                        lambda results, response=None: {"hookSpecificOutput": {
                            "additionalContext": "EVIDENCE " + results[0]["content"]}})
    assert bw._queue_context("jwt expiry", "conv-1") == "EVIDENCE JWT tokens expire in 1h"
    (asked,) = _Queue.asked
    assert asked["session_id"] == "conv-1" and asked["agent_id"] == "before_web_hook"
    assert asked["limit_n"] == bw._RECALL_LIMIT


def test_without_the_daemon_it_keeps_the_command_path(monkeypatch, capsys) -> None:
    _socket(monkeypatch, False)
    monkeypatch.setattr(bw, "_run_recall", lambda q: "x" * 80)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"session_id": "conv-1", "tool_input": {"query": "jwt expiry policy"}})))
    assert bw.main() == 0
    assert "SLM PRE-WEB RECALL" in capsys.readouterr().out


def test_the_queue_answer_is_used_in_the_reminder(monkeypatch, capsys) -> None:
    monkeypatch.setattr(bw, "_queue_context", lambda q, sid: "QUEUED EVIDENCE")
    monkeypatch.setattr(bw, "_run_recall", lambda q: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"session_id": "conv-1", "tool_input": {"query": "jwt expiry policy"}})))
    assert bw.main() == 0
    assert "QUEUED EVIDENCE" in capsys.readouterr().out
