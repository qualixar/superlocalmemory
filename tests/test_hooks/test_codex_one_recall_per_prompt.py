# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""4.2.30: one Codex prompt yields one recall, whichever hooks are registered.

The Codex plugin's ``hooks.json`` runs ``slm hook auto_recall`` on UserPromptSubmit, and the
installed ``codex-prompt`` hook (``slm codex install``) runs the same action as a child. Both are
kept (a plugin-only user needs the first, an installer-only user the second); the per-turn marker in
``auto_recall_hook`` (A-17) lets only the first one recall. This test runs the real ``codex-prompt``
handler, takes the payload it hands its child, and shows the plugin's hook and that child are one turn.
"""

from __future__ import annotations

import io
import json
import sys
import threading

from superlocalmemory.hooks import auto_recall_hook as recall_hook
from superlocalmemory.hooks import codex_handlers
from superlocalmemory.hooks.hook_deadline import BoundedResult

PAYLOAD = {"prompt": "why did we pick sqlite for the queue?", "session_id": "codex-77",
           "cwd": "/work/project"}


def _prompt_children(monkeypatch) -> dict[str, str]:
    """Run ``codex-prompt``; return the stdin each ``slm hook <action>`` child was given."""
    seen: dict[str, str] = {}
    lock = threading.Lock()

    def fake_run_bounded(command, *, deadline, input_text="", registry=None, **_kw):
        with lock:
            seen[command[-1]] = input_text
        return BoundedResult(0, "{}", "", False)

    monkeypatch.setattr(codex_handlers, "run_bounded", fake_run_bounded)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(PAYLOAD)))
    monkeypatch.setattr(codex_handlers.Watchdog, "emit", lambda self, text: None)
    codex_handlers.hook_codex_prompt()
    return seen


def _run_recall_hook(monkeypatch, stdin_text: str) -> dict:
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin_text))
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    assert recall_hook.main() == 0
    return json.loads(out.getvalue() or "{}")


def test_codex_prompt_hands_its_child_the_same_turn_the_plugin_hook_gets(monkeypatch) -> None:
    calls: list[str] = []

    def socket_first(prompt, session_id, budget=None):
        calls.append(session_id)
        return {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                       "additionalContext": "memories"}}

    monkeypatch.setattr(recall_hook, "_try_socket_first", socket_first)
    monkeypatch.setattr(recall_hook, "_switched_off", lambda: False)
    monkeypatch.setattr(recall_hook, "_per_prompt_off", lambda: False)

    children = _prompt_children(monkeypatch)
    assert set(children) == {"auto_recall", "user_prompt_rehash", "topic_shift"}
    child_payload = json.loads(children["auto_recall"])
    assert child_payload["prompt"] == PAYLOAD["prompt"]
    assert child_payload["session_id"] == PAYLOAD["session_id"]

    plugin_answer = _run_recall_hook(monkeypatch, json.dumps(PAYLOAD))  # the plugin's hook
    child_answer = _run_recall_hook(monkeypatch, children["auto_recall"])  # codex-prompt's child

    assert "hookSpecificOutput" in plugin_answer and child_answer == {}
    assert calls == ["codex-77"], "one prompt, one recall"


def test_the_order_does_not_matter_either(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(recall_hook, "_try_socket_first",
                        lambda p, s, budget=None: calls.append(s) or {"hookSpecificOutput": {
                            "hookEventName": "UserPromptSubmit", "additionalContext": "m"}})
    monkeypatch.setattr(recall_hook, "_switched_off", lambda: False)
    monkeypatch.setattr(recall_hook, "_per_prompt_off", lambda: False)
    children = _prompt_children(monkeypatch)

    first = _run_recall_hook(monkeypatch, children["auto_recall"])
    second = _run_recall_hook(monkeypatch, json.dumps(PAYLOAD))

    assert bool(first) and second == {} and calls == ["codex-77"]
