# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
"""Claude Code sub-agents must name SLM tools by their full MCP name.

Claude Code matches an agent's ``tools:`` entries by full name only
(``mcp__<server>__<tool>``, and ``mcp__plugin_<plugin>_<server>__<tool>`` for a
server bundled in a plugin). A bare ``recall`` matches nothing, so the four SLM
agents (memory, optimize and governance advisors, loop runner) received only
``Read`` and ``Bash`` and never the memory tools.

``plugin-src/agents`` stays host-neutral (bare names, read directly by Codex,
Antigravity, Hermes and Copilot); the Claude Code build rewrites the line.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PLUGIN = REPO / "plugin"
BUILDER = REPO / "scripts" / "build-plugin.mjs"
HELPER = REPO / "scripts" / "agent-tool-names.mjs"

# Trees whose agents Claude Code reads, so every SLM tool must be spelled in full.
CLAUDE_FORMAT_TREES = ("plugin", "muse-plugin", "claude-code-assets")


def _tools(agent: Path) -> list[str]:
    head = agent.read_text(encoding="utf-8").split("\n---\n", 1)[0]
    line = next(ln for ln in head.splitlines() if ln.startswith("tools:"))
    return [t.strip() for t in line[len("tools:"):].split(",") if t.strip()]


def _agents(tree: str) -> list[Path]:
    found = sorted((REPO / tree / "agents").glob("*.md"))
    assert len(found) == 4, (tree, [p.name for p in found])
    return found


def _prefixes() -> tuple[str, str]:
    """Derived from the shipped manifest and .mcp.json, not hard-coded."""
    plugin = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))["name"]
    servers = json.loads((PLUGIN / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]
    assert len(servers) == 1, list(servers)
    (server,) = servers
    clean = lambda s: re.sub(r"[^A-Za-z0-9_-]", "_", s)  # noqa: E731 - the documented rule
    return f"mcp__{clean(server)}__", f"mcp__plugin_{clean(plugin)}_{clean(server)}__"


def _node(expr: str) -> str:
    result = subprocess.run(
        ["node", "--input-type=module", "-e",
         f"import * as h from {json.dumps(HELPER.as_uri())}; process.stdout.write({expr});"],
        capture_output=True, text=True, check=True)
    return result.stdout


@pytest.mark.parametrize("tree", CLAUDE_FORMAT_TREES)
def test_no_agent_names_an_slm_tool_bare(tree):
    for agent in _agents(tree):
        bare = [t for t in _tools(agent) if not t[0].isupper() and not t.startswith("mcp__")]
        assert bare == [], f"{tree}/agents/{agent.name} has bare tool names Claude Code cannot resolve: {bare}"


def test_every_slm_tool_resolves_in_a_plugin_install_and_a_direct_mcp_install():
    direct, plugin = _prefixes()
    for agent in _agents("plugin"):
        tools = _tools(agent)
        slm = [t for t in tools if t.startswith("mcp__")]
        assert slm, f"{agent.name} declares no SLM tools at all"
        direct_names = {t[len(direct):] for t in slm if t.startswith(direct)}
        plugin_names = {t[len(plugin):] for t in slm if t.startswith(plugin)}
        assert direct_names, f"{agent.name} has no direct-MCP spelling ({direct}<tool>)"
        assert direct_names == plugin_names, (agent.name, direct_names ^ plugin_names)
        assert len(slm) == len(direct_names) + len(plugin_names), f"{agent.name} has a stray MCP entry"
        assert len(tools) == len(set(tools)), f"{agent.name} repeats a tool"


def test_built_in_tools_are_left_alone():
    for agent in _agents("plugin"):
        source = REPO / "plugin-src" / "agents" / agent.name
        keep = lambda path: [t for t in _tools(path) if t[0].isupper()]  # noqa: E731
        assert keep(agent) == keep(source), agent.name


def test_the_source_stays_bare_so_other_hosts_are_not_affected():
    for agent in (REPO / "plugin-src" / "agents").glob("*.md"):
        assert not [t for t in _tools(agent) if t.startswith("mcp__")], agent.name


def test_plugin_tree_is_in_sync_with_its_builder():
    result = subprocess.run(["node", str(BUILDER), "--check"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_transform_is_idempotent_and_hosts_derive_the_spelling_they_need():
    md = "---\nname: a\ntools: recall, Read, slm_loop_run\n---\nbody\n"
    claude = _node(f"h.claudeCodeAgentTools({json.dumps(md)}, 'superlocalmemory')")
    assert "mcp__superlocalmemory__recall" in claude
    assert "mcp__plugin_superlocalmemory_superlocalmemory__slm_loop_run" in claude
    assert _node(f"h.claudeCodeAgentTools({json.dumps(claude)}, 'superlocalmemory')") == claude
    assert _node(f"h.directAgentTools({json.dumps(claude)})").count("mcp__plugin_") == 0
    assert _node(f"h.bareAgentTools({json.dumps(claude)})") == md


def _registered_tools() -> set[str]:
    """Every tool name any MCP profile can expose (the server's own registry)."""
    from superlocalmemory.mcp.profiles import _PROFILE_DEFINITIONS
    return set().union(*_PROFILE_DEFINITIONS.values())


def test_muse_agents_name_each_slm_tool_exactly_once_as_a_real_direct_tool():
    """Muse installs SLM as a plain MCP server, so the only valid spelling is
    ``mcp__superlocalmemory__<tool>`` where ``<tool>`` is a registered tool.
    A double prefix (the plugin form nested inside the direct form) used to pass."""
    direct, _plugin = _prefixes()
    real = _registered_tools()
    assert "recall" in real  # guard: the registry loaded and is not empty
    for agent in _agents("muse-plugin"):
        slm = [t for t in _tools(agent) if t.startswith("mcp__")]
        assert slm, f"{agent.name} declares no SLM tools at all"
        for entry in slm:
            assert entry.startswith(direct), (agent.name, entry)
            name = entry[len(direct):]
            assert "mcp__" not in name and "plugin_" not in name, (agent.name, entry)
            assert name in real, f"{agent.name}: {entry} is not a tool the server registers"
        assert len(slm) == len(set(slm)), f"{agent.name} repeats an SLM tool"


def test_claude_code_assets_agents_are_the_direct_form_of_plugin_src_and_in_sync():
    """``slm upgrade-hosts`` installs ``claude-code-assets/agents`` into a plugin-less Claude Code.
    It is generated from plugin-src by the same transform as Muse's, so the two cannot differ."""
    builder = REPO / "scripts" / "build-claude-code-assets.mjs"
    checked = subprocess.run(["node", str(builder), "--check"], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stdout + checked.stderr
    direct, _plugin = _prefixes()
    real = _registered_tools()
    for agent in _agents("claude-code-assets"):
        source = (REPO / "plugin-src" / "agents" / agent.name).read_text(encoding="utf-8")
        assert agent.read_text(encoding="utf-8") == _node(f"h.directAgentTools({json.dumps(source)})")
        assert agent.read_text(encoding="utf-8") == (REPO / "muse-plugin" / "agents" / agent.name).read_text(
            encoding="utf-8"), "same spelling as the Muse build, the other plain-MCP consumer"
        for entry in (t for t in _tools(agent) if not t[0].isupper()):
            assert entry.startswith(direct) and entry[len(direct):] in real, (agent.name, entry)


def test_claude_plugin_tools_are_real_registered_tools_in_both_spellings():
    direct, plugin = _prefixes()
    real = _registered_tools()
    for agent in _agents("plugin"):
        for entry in (t for t in _tools(agent) if t.startswith("mcp__")):
            prefix = direct if entry.startswith(direct) else plugin
            assert entry.startswith(prefix), (agent.name, entry)
            assert entry[len(prefix):] in real, (agent.name, entry)
