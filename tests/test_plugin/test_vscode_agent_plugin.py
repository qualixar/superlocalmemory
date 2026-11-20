# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
"""The VS Code agent plugin and the marketplace VS Code finds first (14 Oct 2026).

VS Code's agent plugins accept Claude Code plugins but only replace
${CLAUDE_PLUGIN_ROOT} and ${PLUGIN_ROOT}; our Claude plugin's
"${ComSpec:-...}" launcher cannot start there. VS Code reads
.github/plugin/marketplace.json before .claude-plugin/marketplace.json, so that
file lists vscode-agent-plugin/: the same skills and agents, a plain `slm mcp`.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "vscode-agent-plugin"


def test_in_sync_with_its_builder():
    r = subprocess.run(["node", str(REPO / "scripts" / "build-vscode-agent-plugin.mjs"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_the_server_is_a_plain_slm_mcp_with_no_placeholders():
    text = (OUT / ".mcp.json").read_text(encoding="utf-8")
    assert "${" not in text
    server = json.loads(text)["mcpServers"]["superlocalmemory"]
    assert server["command"] == "slm" and server["args"] == ["mcp"]
    assert "SLM_MCP_PROFILE" not in server.get("env", {})


def test_skills_and_agents_ship_but_claude_only_hooks_do_not():
    assert len(list(OUT.glob("skills/*/SKILL.md"))) == 17
    assert len(list(OUT.glob("agents/*.md"))) == 4
    assert not (OUT / "hooks").exists() and not (OUT / "scripts").exists()


def test_the_github_marketplace_lists_it_and_claude_keeps_its_own():
    gh = json.loads((REPO / ".github" / "plugin" / "marketplace.json").read_text(encoding="utf-8"))
    assert [p["source"] for p in gh["plugins"]] == ["./vscode-agent-plugin"]
    claude = json.loads((REPO / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
    assert "./plugin" in [p["source"] for p in claude["plugins"]]
