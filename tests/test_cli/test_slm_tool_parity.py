# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm tool list`` and ``describe`` are the MCP server's ``tools/list``, for every tool set.

Each case runs in a fresh process, because ``mcp/server.py`` fixes its tool
surface when it is imported: the process imports the server with the case's
environment, builds the ``slm tool`` host for the matching ``--tool-set``, and
reports both lists. Names, titles, descriptions and input schemas must be equal.
A second group runs the real ``slm tool`` entry point and checks what it prints.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

_PROBE = r"""
import asyncio, json, sys
from superlocalmemory.mcp import server as s
from superlocalmemory.cli import tool_host

def shape(tools):
    return {t.name: [t.title, t.description, t.input_schema] for t in tools}

host = tool_host.build_host(sys.argv[1] or None)
print(json.dumps({
    "mcp": shape(asyncio.run(s.server.list_tools())),
    "cli": shape(tool_host.listed_tools(host)),
    "registered_mcp": sorted(t.name for t in s.server._tool_manager.list_tools()),
    "registered_cli": sorted(t.name for t in tool_host.registered_tools(host)),
}, default=str))
"""


def _env(tmp_path: Path, extra: dict[str, str]) -> dict[str, str]:
    from tests._portable import child_env_base

    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("SLM_MCP_", "SLM_CLIENT_CHANNEL"))}
    env.update(child_env_base(tmp_path / "home"))
    env.update({
        "SLM_DATA_DIR": str(tmp_path / "data"),
        "PYTHONPATH": str(REPO / "src"), "SLM_DISABLE_WARMUP_SIDE_EFFECTS": "1",
        # The stdio watchdog threads stay off; the tool surface does not depend on them.
        "SLM_MCP_EMBEDDED": "1", "SLM_NON_INTERACTIVE": "1", "CI": "1",
    })
    env.update(extra)
    return env


def _probe(tmp_path: Path, extra: dict[str, str], tool_set: str) -> dict:
    proc = subprocess.run([sys.executable, "-c", _PROBE, tool_set], capture_output=True,
                          text=True, env=_env(tmp_path, extra), cwd=str(REPO), timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("server_env, tool_set", [
    ({"SLM_MCP_PROFILE": "core"}, "core"),
    ({"SLM_MCP_PROFILE": "code"}, "code"),
    ({"SLM_MCP_PROFILE": "full"}, "full"),
    ({"SLM_MCP_PROFILE": "power"}, "power"),
    ({"SLM_MCP_PROFILE": "mesh"}, "mesh"),
    ({"SLM_MCP_ALL_TOOLS": "1"}, "whole"),
    ({"SLM_MCP_ALL_TOOLS": "1"}, ""),          # slm tool's default: every tool
    ({}, "mcp"),                               # an MCP connection with no settings
    ({"SLM_MCP_PROFILE": "code"}, "mcp"),      # ... with a tool set in its environment
    ({"SLM_MCP_TOOLS": "recall,remember,mesh_send"}, "mcp"),
])
def test_slm_tool_lists_exactly_what_the_mcp_server_lists(tmp_path, server_env, tool_set):
    seen = _probe(tmp_path, server_env, tool_set)
    assert sorted(seen["cli"]) == sorted(seen["mcp"]), (
        set(seen["cli"]) ^ set(seen["mcp"]))
    for name, (title, description, schema) in seen["mcp"].items():
        assert seen["cli"][name] == [title, description, schema], name


def test_both_register_the_same_112_tools(tmp_path):
    seen = _probe(tmp_path, {"SLM_MCP_ALL_TOOLS": "1"}, "")
    assert seen["registered_cli"] == seen["registered_mcp"]
    assert len(seen["registered_cli"]) == 112


# -- the entry point --------------------------------------------------------------------------

def _slm(tmp_path: Path, *args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "superlocalmemory.cli.main", "tool", *args],
                          capture_output=True, text=True, input=stdin, timeout=300,
                          env=_env(tmp_path, {}), cwd=str(tmp_path))


def test_list_and_describe_print_the_registry_as_json(tmp_path):
    listed = _slm(tmp_path, "list", "--json")
    assert listed.returncode == 0, listed.stderr[-2000:]
    data = json.loads(listed.stdout)
    names = [t["name"] for t in data["tools"]]
    assert data["count"] == len(names) and names == sorted(names) and "recall" in names
    described = _slm(tmp_path, "describe", "recall", "--json")
    assert described.returncode == 0, described.stderr[-2000:]
    record = json.loads(described.stdout)
    assert record["input_schema"]["required"] == ["query"]
    assert record["cli"] == "slm tool recall --query <query>"
    core = json.loads(_slm(tmp_path, "list", "--json", "--tool-set", "core").stdout)
    assert core["count"] == 20


@pytest.mark.parametrize("args, code", [
    (["recall", "--query", "q", "--limit", "lots"], "invalid_arguments"),
    (["recall", "--json", '{"query": "q", "channel": "dashboard"}'], "invalid_arguments"),
    (["remember", "--content", "x", "--agent-id-source", "connection"], "invalid_arguments"),
    (["recal", "--query", "q"], "unknown_tool"),
    (["build_code_graph", "--tool-set", "core", "--repo-path", "/x"], "not_in_tool_set"),
    (["recall", "--tool-set", "banana", "--query", "q"], "unknown_tool_set"),
])
def test_a_call_that_cannot_be_made_exits_2_with_a_structured_error(tmp_path, args, code):
    proc = _slm(tmp_path, *args)
    assert proc.returncode == 2, (proc.stdout, proc.stderr[-2000:])
    body = json.loads(proc.stdout)
    assert body["success"] is False and body["code"] == code, body


def test_a_tool_help_is_its_description(tmp_path):
    proc = _slm(tmp_path, "build_code_graph", "--help")
    assert proc.returncode == 0
    assert "Build a complete code knowledge graph" in proc.stdout
    assert "--repo-path" in proc.stdout


def test_a_tool_that_needs_no_daemon_answers_on_stdout_only(tmp_path):
    proc = _slm(tmp_path, "get_version")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert json.loads(proc.stdout)["success"] is True


def test_arguments_on_stdin(tmp_path):
    proc = _slm(tmp_path, "slm_compress", "-", stdin=json.dumps({"content": "x " * 400}))
    assert proc.returncode in (0, 1), proc.stderr[-2000:]
    assert isinstance(json.loads(proc.stdout), dict)


def test_an_mcp_server_says_mcp_even_when_started_from_slm_tool(tmp_path):
    """A daemon started by ``slm tool`` inherits ``SLM_CLIENT_CHANNEL=cli``; the MCP server
    it embeds is still the MCP door, so its own daemon requests must say ``mcp``."""
    probe = ("import os\nfrom superlocalmemory.mcp import server\n"
             "from superlocalmemory.core.agent_attribution import client_channel\n"
             "print(client_channel())\n")
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                          env=_env(tmp_path, {"SLM_CLIENT_CHANNEL": "cli"}), cwd=str(REPO),
                          timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().splitlines()[-1] == "mcp"
