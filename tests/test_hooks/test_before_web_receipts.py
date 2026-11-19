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
    monkeypatch.setattr(bw, "_run_recall", lambda q, **k: "x" * 80)
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


# -- 20 Nov 2026: the hook timed out on most busy web calls, and filled a correct
# "nothing relevant" with weak matches. Measured on the author's Mac: 42 of 50 hook
# timeouts in one day were before_web (Claude Code kills it at 5 s); the queue's 3 s
# wait plus a fresh `slm recall` with its own 3 s could not fit.

def _stdin(monkeypatch) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"session_id": "conv-1", "tool_input": {"query": "sqlite busy timeout"}})))


def test_a_queue_that_found_nothing_relevant_is_the_answer(monkeypatch, capsys) -> None:
    """The queue injects only what the answer check supports, as the per-prompt hook
    does. Its empty answer must not be replaced by the command path's weak matches."""
    called: list[str] = []  # main() swallows exceptions (fail-open), so record calls
    monkeypatch.setattr(bw, "_queue_context", lambda q, sid: "")
    monkeypatch.setattr(bw, "_run_recall", lambda *a, **k: called.append("fell back") or "z" * 80)
    _stdin(monkeypatch)
    assert bw.main() == 0
    assert called == []
    assert capsys.readouterr().out == ""


def test_only_an_unavailable_queue_falls_back(monkeypatch, capsys) -> None:
    seen: list[float] = []
    monkeypatch.setattr(bw, "_queue_context", lambda q, sid: None)
    monkeypatch.setattr(bw, "_run_recall",
                        lambda q, timeout: seen.append(timeout) or "y" * 80)
    _stdin(monkeypatch)
    assert bw.main() == 0
    assert "SLM PRE-WEB RECALL" in capsys.readouterr().out
    assert seen and 0 < seen[0] <= bw._HOOK_BUDGET_SEC


def test_the_whole_hook_fits_inside_the_host_timeout(monkeypatch, capsys) -> None:
    """A queue that used up the budget leaves no time for a second recall."""
    clock = iter([100.0, 100.0 + bw._HOOK_BUDGET_SEC - 0.5])
    monkeypatch.setattr(bw.time, "monotonic", lambda: next(clock))
    called: list[str] = []
    monkeypatch.setattr(bw, "_queue_context", lambda q, sid: None)
    monkeypatch.setattr(bw, "_run_recall", lambda *a, **k: called.append("over") or "z" * 80)
    _stdin(monkeypatch)
    assert bw.main() == 0
    assert called == []
    assert capsys.readouterr().out == ""
    assert bw._HOOK_BUDGET_SEC < 5.0  # the plugin's PreToolUse timeout for this hook


def test_queue_states(monkeypatch) -> None:
    """None = unavailable (fall back allowed); "" = answered with nothing relevant."""
    _socket(monkeypatch, False)
    assert bw._queue_context("q", "c") is None

    class _Silent(_Queue):
        def poll_result(self, request_id, timeout_s):
            return None  # nobody answered in time

    class _Empty(_Queue):
        def poll_result(self, request_id, timeout_s):
            return {"ok": True, "results": []}

    _socket(monkeypatch, True)
    monkeypatch.setattr("superlocalmemory.hooks.auto_recall_hook._detect_mode", lambda: "A")
    monkeypatch.setattr("superlocalmemory.core.recall_queue.RecallQueue", _Silent)
    assert bw._queue_context("q", "c") is None
    monkeypatch.setattr("superlocalmemory.core.recall_queue.RecallQueue", _Empty)
    monkeypatch.setattr("superlocalmemory.hooks.auto_recall_hook._format_envelope",
                        lambda results, response=None: {})
    assert bw._queue_context("q", "c") == ""
