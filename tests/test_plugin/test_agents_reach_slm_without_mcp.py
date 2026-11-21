# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
"""SLM's agents and ``/slm-loop`` must still reach memory where MCP is blocked.

Some hosts refuse local MCP servers by policy. There, an agent's ``mcp__…`` tools
simply do not exist, and an agent without Bash (or one that never heard of
``slm tool``) has no way to reach SuperLocalMemory at all. ``slm tool <name>``
calls every MCP tool from a shell with the same arguments and checks, so each
shipped agent needs Bash and has to be told about that route.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# The source of truth plus every generated tree an agent host reads.
AGENT_TREES = ("plugin-src", "plugin", "muse-plugin", "claude-code-assets")


def _agents(tree: str) -> list[Path]:
    found = sorted((REPO / tree / "agents").glob("*.md"))
    assert len(found) == 4, (tree, [p.name for p in found])
    return found


def _tools(agent: Path) -> list[str]:
    head = agent.read_text(encoding="utf-8").split("\n---\n", 1)[0]
    line = next(ln for ln in head.splitlines() if ln.startswith("tools:"))
    return [t.strip() for t in line[len("tools:"):].split(",") if t.strip()]


@pytest.mark.parametrize("tree", AGENT_TREES)
def test_every_agent_has_bash(tree):
    for agent in _agents(tree):
        assert "Bash" in _tools(agent), f"{tree}/agents/{agent.name} cannot run `slm` without Bash"


@pytest.mark.parametrize("tree", AGENT_TREES)
def test_every_agent_names_the_slm_tool_route(tree):
    for agent in _agents(tree):
        text = agent.read_text(encoding="utf-8")
        assert "When SLM's MCP tools are not in your tool list" in text, agent.name
        for needed in ("slm tool list", "slm tool describe", "slm tool <name> --json",
                       "slm status", "slm-cli"):
            assert needed in text, f"{tree}/agents/{agent.name} does not mention {needed!r}"
        assert "There is no CLI form that returns a session_id" not in text, agent.name


@pytest.mark.parametrize("tree", ("plugin-src", "plugin", "muse-plugin"))
def test_slm_loop_command_gives_the_cli_route(tree):
    command = REPO / tree / "commands" / "slm-loop.md"
    text = command.read_text(encoding="utf-8")
    assert "slm tool slm_loop_run" in text, command
    assert "slm loop history" in text, command
