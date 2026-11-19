# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""F-04 (19 Nov 2026): a Claude Code hook is SLM's only when it equals a command SLM has written.

Before, ``"slm hook" in command`` decided. The upgrade then replaced a person's edited line with the
stock one (losing an ``SLM_PROFILE=work`` prefix, a script after it, a raised timeout) and deleted
commands that only mention the words (``echo rslm hook finished``). Now ownership is by identity:
the command must equal one SLM wrote (current or in any earlier release, whitespace and temp-path
normalised) and the hook object must be as SLM wrote it. Everything else is the person's and stays.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from superlocalmemory.hooks import _hook_identity as ident
from superlocalmemory.hooks import claude_code_hooks as claude
from superlocalmemory.hooks import host_hook_upgrade as upgrade

VERSION = "4.2.30"
TMP = "/var/folders/k3/4mty_8291733jsf1k89hmkr80000gn/T"
HASH = "f0a66cd7e6f4edf2"

#: What earlier releases wrote on a Mac or Linux and on Windows, with that computer's own temp
#: folder, taken from the git history of hooks/claude_code_hooks.py (v3.0 to 4.2.19).
HISTORIC = [
    ("slm hook start 2>/dev/null || true", 15000),
    ("slm hook stop 2>/dev/null || true", 10000),
    ("slm session close 2>/dev/null || true", 15),  # 0cf344b0, 3.7.x: the one Fable named
    (f"slm hook start 2>/dev/null || true; touch {TMP}/slm-session-start-time-{HASH}", 15),
    (f"slm hook stop 2>/dev/null || true; rm -f {TMP}/slm-session-initialized-{HASH} "
     f"{TMP}/slm-session-start-time-{HASH}", 10),
    (f"touch {TMP}/slm-session-initialized", 500),  # before the data-root suffix
    (f"test -f {TMP}/slm-session-initialized-{HASH} || test ! -f {TMP}/slm-session-start-time-{HASH}"
     " || { echo '[SLM] Call mcp__superlocalmemory__session_init first'; exit 2; }", 2),
    ('cmd /c "slm hook stop 2>NUL || exit /b 0"', 10000),
    (f'cmd /c "echo.>"\\Users\\ab\\AppData\\Local\\Temp\\slm-session-initialized-{HASH}""', 2),
    ('cmd /c "echo.>\\Users\\ab\\AppData\\Local\\Temp\\slm-session-initialized"', 500),
    ("/Users/someone/.superlocalmemory/hooks/slm-session-start.sh", None),
    ("/home/someone/.superlocalmemory/hooks/slm-auto-capture.sh", None),
    ("slm hook mandate 2>/dev/null || true", 5),
    ("slm  hook   topic_shift 2>/dev/null || true", 3),  # whitespace is not an edit
]


def _hook(command: str, timeout: int | None = 5, **extra) -> dict:
    hook: dict = {"type": "command", "command": command, **extra}
    if timeout is not None:
        hook["timeout"] = timeout
    return hook


@pytest.mark.parametrize("command,timeout", HISTORIC)
def test_every_command_an_earlier_release_wrote_is_still_slms(command, timeout) -> None:
    assert ident.is_known_command(command)
    assert ident.is_slm_hook(_hook(command, timeout))


@pytest.mark.parametrize("platform", ["darwin", "win32"])
@pytest.mark.parametrize("gate", [False, True])
def test_every_command_the_current_code_writes_is_owned(monkeypatch, platform, gate) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    rows = [h for entries in claude._hook_definitions(include_gate=gate).values()
            for e in entries for h in e["hooks"]]
    assert len(rows) >= (15 if gate else 13)
    for hook in rows:
        assert ident.is_slm_hook(hook), hook["command"]


@pytest.mark.parametrize("command", [
    "echo rslm hook finished",
    "echo use slm hooks",
    "tmux new -s my-slm-session",
    "~/bin/notify-llm hook-done; echo rslm hook finished >> ~/log.txt",
    'cd "$CLAUDE_PROJECT_DIR" && SLM_PROFILE=work slm hook start; ./scripts/my-banner.sh',
    "SLM_PROFILE=work slm hook start 2>/dev/null || true",
    "slm hook start 2>/dev/null || true; ./my-extra.sh",
    "slm hook nonsense 2>/dev/null || true",
    "bash ~/.superlocalmemory/hooks/start.sh",
    "test -f /tmp/slm-session-initialized",
    "",
])
def test_anything_else_is_the_persons(command) -> None:
    assert not ident.is_known_command(command)
    assert not ident.is_slm_hook(_hook(command, 5))


def test_an_edited_hook_object_is_not_owned_even_with_slms_command() -> None:
    command = "slm hook checkpoint 2>/dev/null || true"
    assert ident.is_slm_hook(_hook(command, 5))
    assert not ident.is_slm_hook(_hook(command, 60))  # a raised timeout
    assert not ident.is_slm_hook(_hook(command, 1))  # a lowered one
    assert not ident.is_slm_hook(_hook(command, 5, async_=True))  # an extra key
    assert not ident.is_slm_hook(_hook(command, 5, shell="zsh"))
    assert not ident.is_slm_hook({"type": "prompt", "command": command, "timeout": 5})
    assert not ident.is_slm_hook(_hook(command, True))  # True is not 5 seconds
    assert ident.is_slm_hook(_hook(command, None))  # no timeout written: the host default
    assert not ident.is_slm_hook("slm hook checkpoint")


# -- the three cases Fable ran, through the real upgrade -------------------------------------------

def _stock() -> dict:
    return {"theme": "dark", "hooks": claude._hook_definitions()}


def _run(tmp_path: Path, settings: dict) -> tuple[dict, dict]:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    result = upgrade.upgrade_claude_settings(path, version=VERSION)
    return result, json.loads(path.read_text(encoding="utf-8"))


def _commands(data: dict, event: str) -> list[str]:
    return [h["command"] for e in data["hooks"][event] for h in e["hooks"]]


def test_a_line_with_a_profile_prefix_and_a_script_after_it_keeps_every_word(tmp_path) -> None:
    edited = 'cd "$CLAUDE_PROJECT_DIR" && SLM_PROFILE=work slm hook start; ./scripts/my-banner.sh'
    settings = _stock()
    settings["hooks"]["SessionStart"][1]["hooks"][0]["command"] = edited
    settings["hooks"]["PostToolUse"] = settings["hooks"]["PostToolUse"][:2]  # no cue yet: a real upgrade
    _, data = _run(tmp_path, settings)
    start_lines = [c for c in _commands(data, "SessionStart") if "slm hook start" in c]
    assert start_lines == [edited]  # theirs, once: SLM does not add a second copy beside it
    assert any("cue" in c for c in _commands(data, "PostToolUse")), "the rest was still upgraded"


def test_commands_that_only_mention_the_words_are_never_deleted(tmp_path) -> None:
    theirs = [
        "~/bin/notify-llm hook-done; echo rslm hook finished >> ~/log.txt",
        "echo use slm hooks",
        "tmux new -s my-slm-session",
    ]
    settings = _stock()
    settings["hooks"]["Stop"] = [{"hooks": [_hook(c, None)]} for c in theirs] + settings["hooks"]["Stop"]
    settings["hooks"]["PostToolUse"] = settings["hooks"]["PostToolUse"][:2]
    _, data = _run(tmp_path, settings)
    stop = _commands(data, "Stop")
    for command in theirs:
        assert command in stop
    assert sum("slm hook stop " in c for c in stop) == 1  # SLM's own, replaced in place, not doubled


def test_a_raised_timeout_stays_and_the_hook_is_not_added_again(tmp_path) -> None:
    settings = _stock()
    settings["hooks"]["SessionStart"][1]["hooks"][0]["timeout"] = 60
    settings["hooks"]["PostToolUse"] = settings["hooks"]["PostToolUse"][:2]
    _, data = _run(tmp_path, settings)
    starts = [h for e in data["hooks"]["SessionStart"] for h in e["hooks"]
              if "slm hook start" in h["command"]]
    assert len(starts) == 1 and starts[0]["timeout"] == 60


def test_the_second_run_over_the_result_changes_nothing(tmp_path) -> None:
    settings = _stock()
    settings["hooks"]["SessionStart"][1]["hooks"][0]["command"] = "SLM_PROFILE=work slm hook start"
    settings["hooks"]["PostToolUse"] = settings["hooks"]["PostToolUse"][:2]
    first, data = _run(tmp_path, settings)
    assert first["status"] == "updated"
    again = upgrade.upgrade_claude_settings(tmp_path / "settings.json", version=VERSION)
    assert again["status"] == "current"
    assert json.loads((tmp_path / "settings.json").read_text(encoding="utf-8")) == data


def test_an_old_release_marker_with_another_computers_temp_folder_is_replaced(tmp_path) -> None:
    settings = {"hooks": {"SessionStart": [{"hooks": [
        _hook(f"slm hook start 2>/dev/null || true; touch /tmp/elsewhere/slm-session-start-time-{HASH}",
              15)]}]}}
    result, data = _run(tmp_path, settings)
    assert result["status"] == "updated"
    assert sum("slm hook start" in c for c in _commands(data, "SessionStart")) == 1
    assert "/tmp/elsewhere" not in json.dumps(data)


def test_a_file_whose_only_slm_line_is_edited_counts_as_having_no_slm_hooks() -> None:
    settings = {"hooks": {"SessionStart": [{"hooks": [_hook("SLM_PROFILE=work slm hook start")]}]}}
    assert not claude.settings_have_slm_hooks(settings)
    assert claude.settings_have_slm_hooks({"hooks": claude._hook_definitions()})


def test_remove_deletes_only_what_slm_wrote() -> None:
    mine = "echo use slm hooks"
    settings = _stock()
    settings["hooks"]["Stop"].append({"hooks": [_hook(mine, None)]})
    cleaned = claude._remove_slm_hooks(settings)
    assert _commands(cleaned, "Stop") == [mine]
