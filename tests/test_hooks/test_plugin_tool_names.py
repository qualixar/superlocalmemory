# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""SLM's hook texts and gate work under both names its MCP tools can have (20 Nov 2026).

Installed through the Claude Code plugin, the tools are named
``mcp__plugin_superlocalmemory_superlocalmemory__<tool>``; registered by hand they are
``mcp__superlocalmemory__<tool>``. The session-start hint said
``select:mcp__superlocalmemory__session_init``, which finds nothing for a plugin-only
user (found on the author's Mac, where a second hand-made entry hid it), and the optional
gate let only the short names through, so a plugin user could never get past it.
"""

from __future__ import annotations

import io
import json
import time
from unittest.mock import patch

import pytest

from superlocalmemory.hooks import hook_handlers as hh
from superlocalmemory.hooks import topic_shift_hook as ts

PLUGIN = "mcp__plugin_superlocalmemory_superlocalmemory__"
SHORT = "mcp__superlocalmemory__"


@pytest.mark.parametrize("name", [SHORT + "session_init", PLUGIN + "session_init",
                                  PLUGIN + "recall", SHORT + "remember"])
def test_both_names_are_slm_tools(name) -> None:
    assert hh.is_slm_tool(name)


@pytest.mark.parametrize("name", ["Bash", "mcp__other__session_init",
                                  "mcp__plugin_other_superlocalmemory__recall",
                                  "mcp__superlocalmemory_evil__recall", "", SHORT, PLUGIN])
def test_other_tools_are_not(name) -> None:
    assert not hh.is_slm_tool(name)


def test_the_hint_finds_the_tool_under_either_name() -> None:
    hint = hh._SESSION_INIT_HINT
    assert "select:" not in hint, "an exact select names only one of the two installs"
    assert 'ToolSearch query: "superlocalmemory session_init"' in hint
    assert PLUGIN + "session_init" in hint and SHORT + "session_init" in hint


def test_no_reminder_names_only_the_short_tool() -> None:
    for text in (ts._SHIFT_REMINDER,):
        assert SHORT not in text or PLUGIN in text


def test_the_gate_lets_a_plugin_user_call_session_init(tmp_path, monkeypatch) -> None:
    start = tmp_path / "start"
    start.write_text(str(int(time.time())), encoding="utf-8")
    monkeypatch.setattr(hh, "_MARKER", str(tmp_path / "missing-marker"))
    monkeypatch.setattr(hh, "_START_TIME", str(start))
    stdin = io.StringIO(json.dumps({"tool_name": PLUGIN + "session_init"}))
    with patch("sys.stdin", stdin), patch("sys.stdin.isatty", return_value=False):
        with pytest.raises(SystemExit) as exc:
            hh.handle_hook("gate")
    assert exc.value.code == 0
