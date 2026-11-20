# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
"""The Muse CLI plugin, as shipped in muse-plugin/ (14 Oct 2026).

Muse CLI 1.4.7 imports Claude Code plugins, but refused ours: it rejects short
MCP names in a skill's ``allowed-tools`` (8 of 17 skills), the ``${ComSpec:-}``
launcher in ``.mcp.json``, and it drops SessionStart output that starts with
``[`` as malformed JSON, which is how ``slm hook start`` begins. The Muse build
removes the first two and runs the JSON-emitting SessionStart action instead.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MUSE = REPO / "muse-plugin"


def _commands() -> list[str]:
    data = json.loads((MUSE / "hooks" / "hooks.json").read_text(encoding="utf-8"))
    return [h["command"] for groups in data["hooks"].values()
            for g in groups for h in g.get("hooks", [])]


def test_muse_plugin_is_in_sync_with_its_builder():
    result = subprocess.run(["node", str(REPO / "scripts" / "build-muse-plugin.mjs"), "--check"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_no_skill_declares_allowed_tools():
    skills = sorted(MUSE.glob("skills/*/SKILL.md"))
    assert len(skills) == 17
    offenders = [s.parent.name for s in skills
                 if "\nallowed-tools:" in s.read_text(encoding="utf-8").split("\n---\n", 1)[0]]
    assert offenders == []


def test_no_mcp_launcher_and_no_claude_permissions_file():
    assert not (MUSE / ".mcp.json").exists()
    assert not (MUSE / "settings.json").exists()
    manifest = json.loads((MUSE / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    assert "mcpServers" not in manifest
    assert "slm connect muse" in manifest["description"]


def test_session_start_answers_in_json():
    commands = _commands()
    assert any(" hook codex-start " in c for c in commands)
    assert not any(" hook start " in c for c in commands)


def test_codex_start_prints_hook_specific_output_json():
    """The action the Muse hook runs answers with an object Muse can parse."""
    from superlocalmemory.hooks import codex_handlers

    text = codex_handlers._start_output("SLM session_init OK: s", "[BEGIN UNTRUSTED SLM EVIDENCE v1]")
    data = json.loads(text)
    assert data["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "[BEGIN UNTRUSTED" in data["hookSpecificOutput"]["additionalContext"]


@pytest.mark.skipif(shutil.which("muse") is None, reason="Muse CLI not installed")
def test_muse_validates_the_plugin_without_errors():
    result = subprocess.run(["muse", "plugins", "validate", str(MUSE)],
                            capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
    assert "severity=error" not in result.stdout + result.stderr


def test_agents_name_slm_tools_the_way_muse_does():
    for agent in sorted(MUSE.glob("agents/*.md")):
        head = agent.read_text(encoding="utf-8").split("\n---\n", 1)[0]
        tools = [t.strip() for t in head.split("\ntools:", 1)[1].split("\n", 1)[0].split(",")]
        bare = [t for t in tools if t and t[0].islower() and not t.startswith("mcp__")]
        assert bare == [], (agent.name, bare)
