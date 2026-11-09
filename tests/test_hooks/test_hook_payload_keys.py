# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
"""Every ``slm hook`` reads Claude's snake_case keys, whichever host calls it.

Grok CLI runs the Claude Code plugin's hooks but sends camelCase keys
(``sessionId``, ``toolName``, ``toolInput``); only ``hook_event_name`` and
``tool_response`` have snake_case aliases (Grok CLI 1.0.50, user guide
"Hooks"). The SLM hooks read ``session_id``/``tool_name``/``tool_input``, so in
Grok they ran and recorded nothing (14 Oct 2026). The dispatcher now adds the
snake_case keys before any handler reads stdin; keys already present win.
"""

from __future__ import annotations

import io
import json

from superlocalmemory.hooks.payload_keys import normalize_payload, normalize_stdin

GROK = {
    "hookEventName": "post_tool_use",
    "hook_event_name": "PostToolUse",
    "sessionId": "abc-123",
    "cwd": "/w",
    "workspaceRoot": "/w",
    "toolName": "run_terminal_command",
    "toolInput": {"command": "npm test"},
    "tool_response": {"ok": True},
}


def test_camel_case_keys_gain_their_snake_case_names():
    out = normalize_payload(GROK)
    assert out["session_id"] == "abc-123"
    assert out["tool_name"] == "run_terminal_command"
    assert out["tool_input"] == {"command": "npm test"}


def test_existing_snake_case_keys_win_and_claude_payloads_are_unchanged():
    out = normalize_payload(GROK)
    assert out["hook_event_name"] == "PostToolUse"  # Claude's value, not Grok's
    claude = {"session_id": "s", "tool_name": "Bash", "tool_input": {"command": "ls"}}
    assert normalize_payload(claude) == claude
    both = {"session_id": "keep", "sessionId": "other"}
    assert normalize_payload(both)["session_id"] == "keep"


def test_stdin_is_rewritten_once_and_non_json_passes_through(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(GROK)))
    normalize_stdin()
    import sys
    assert json.loads(sys.stdin.read())["tool_name"] == "run_terminal_command"

    monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
    normalize_stdin()
    assert sys.stdin.read() == "not json"

    monkeypatch.setattr("sys.stdin", io.StringIO("[1, 2]"))
    normalize_stdin()
    assert sys.stdin.read() == "[1, 2]"


def test_the_dispatcher_normalises_before_the_handler_reads(monkeypatch):
    import sys

    from superlocalmemory.hooks import hook_handlers

    seen = {}

    def fake_checkpoint():
        seen.update(json.loads(sys.stdin.read()))

    monkeypatch.setattr(hook_handlers, "_hook_checkpoint", fake_checkpoint)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(GROK)))
    hook_handlers.handle_hook("checkpoint")
    assert seen["tool_input"] == {"command": "npm test"}
