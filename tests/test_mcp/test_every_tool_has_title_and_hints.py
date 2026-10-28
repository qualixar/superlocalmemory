# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Every MCP tool declares a title and all four behaviour hints (issue #300).

Anthropic's Connectors Directory wants a ``title`` plus ``readOnlyHint`` /
``destructiveHint`` on every tool; OpenAI's app review wants all four hints as
explicit booleans (null reads as "unknown" and fails review) and checks them
against what the tool does. Without hints the MCP default for a tool is
``destructiveHint=true``, so an unannotated tool looks like a deletion.

The tests build the real server, register every tool through the same
``register_*`` functions the product uses, and read what a client would be told.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from superlocalmemory.mcp.http_transport import SLMFastMCP

MCP_DIR = Path(__file__).resolve().parents[2] / "src" / "superlocalmemory" / "mcp"

HINT_FIELDS = ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint")

# Tools that delete, erase, overwrite or rewrite what a person has stored, or
# stop a process. A client must be able to ask before running one.
DESTRUCTIVE_TOOLS = frozenset({
    "delete_memory", "update_memory", "review_correction", "forget",
    "compact_memories", "set_retention_policy", "quantize",
    "consolidate_cognitive", "run_maintenance", "resolve_operation",
    "contradict_assertion", "evolve_skill", "manage_view", "reap_processes",
    "apply_refactor", "mesh_state",
})

# Tools that only read: no hint may say otherwise.
READ_ONLY_TOOLS = frozenset({
    "recall", "search", "fetch", "list_recent", "get_status", "expand",
    "get_media", "media_status", "mesh_peers", "mesh_status", "slm_cache_get",
    "run_view", "get_memory_summary", "list_corrections", "health",
})

# Tools that reach outside this machine (a link, a hosted model).
OPEN_WORLD_TOOLS = frozenset({
    "remember_media", "remember_document", "evolve_skill", "slm_loop_run",
})


def _register_every_tool() -> SLMFastMCP:
    from superlocalmemory.mcp.tools_active import register_active_tools
    from superlocalmemory.mcp.tools_brain import register_brain_tools
    from superlocalmemory.mcp.tools_code_graph import register_code_graph_tools
    from superlocalmemory.mcp.tools_context import register_prestage_tool
    from superlocalmemory.mcp.tools_core import register_core_tools
    from superlocalmemory.mcp.tools_evolution import register_evolution_tools
    from superlocalmemory.mcp.tools_kinds import register_kind_tools
    from superlocalmemory.mcp.tools_learning import register_learning_tools
    from superlocalmemory.mcp.tools_loops import register_loop_tools
    from superlocalmemory.mcp.tools_media import (
        register_document_tools,
        register_media_tools,
    )
    from superlocalmemory.mcp.tools_media_upload import register_upload_link_tool
    from superlocalmemory.mcp.tools_mesh import register_mesh_tools
    from superlocalmemory.mcp.tools_ops import register_ops_tools
    from superlocalmemory.mcp.tools_optimize import register_optimize_tools
    from superlocalmemory.mcp.tools_summaries import register_summary_tools
    from superlocalmemory.mcp.tools_v3 import register_v3_tools
    from superlocalmemory.mcp.tools_v28 import register_v28_tools
    from superlocalmemory.mcp.tools_v33 import register_v33_tools
    from superlocalmemory.mcp.tools_views import register_view_tools

    server = SLMFastMCP("SLM hints test")
    get_engine = lambda: None  # noqa: E731 - registration never calls it
    register_core_tools(server, get_engine)
    register_v28_tools(server, get_engine)
    register_v3_tools(server, get_engine)
    register_active_tools(server, get_engine)
    register_v33_tools(server, get_engine)
    register_code_graph_tools(server, get_engine)
    register_mesh_tools(server, get_engine)
    register_learning_tools(server, get_engine)
    register_evolution_tools(server, get_engine)
    register_optimize_tools(server)
    register_loop_tools(server, get_engine)
    register_ops_tools(server, get_engine)
    register_brain_tools(server, get_engine)
    register_summary_tools(server, get_engine)
    register_view_tools(server, get_engine)
    register_kind_tools(server, get_engine)
    register_media_tools(server)
    register_document_tools(server)
    register_upload_link_tool(server)
    register_prestage_tool(server, lambda *a, **k: [])
    return server


def _hint(tool, camel: str):
    ann = tool.annotations
    snake = "".join(f"_{c.lower()}" if c.isupper() else c for c in camel)
    return getattr(ann, snake, None) if ann is not None else None


def _source_tool_names() -> set[str]:
    """Every function decorated with ``@server.tool(...)`` under mcp/."""
    names: set[str] = set()
    for path in MCP_DIR.glob("tools_*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for deco in node.decorator_list:
                if isinstance(deco, ast.Call) and ast.unparse(deco.func) == "server.tool":
                    names.add(node.name)
    return names


@pytest.fixture(scope="module")
def tools() -> dict[str, object]:
    return {t.name: t for t in _register_every_tool()._tool_manager.list_tools()}


def test_the_registration_helper_reaches_every_tool_in_the_source(tools) -> None:
    """If a tool is missing here, this test file cannot vouch for it."""
    missing = _source_tool_names() - set(tools)
    assert not missing, f"tools in source but not registered by this test: {sorted(missing)}"


def test_every_tool_has_a_human_title(tools) -> None:
    bad = {
        name: tool.title
        for name, tool in tools.items()
        if not (tool.title or "").strip() or tool.title == name or "_" in tool.title
    }
    assert not bad, f"tools without a plain-words title: {bad}"


def test_every_tool_title_is_unique(tools) -> None:
    titles = [tool.title for tool in tools.values()]
    dupes = sorted({t for t in titles if titles.count(t) > 1})
    assert not dupes, f"two tools share a title: {dupes}"


@pytest.mark.parametrize("field", HINT_FIELDS)
def test_every_tool_sets_each_hint_as_an_explicit_bool(tools, field) -> None:
    bad = {
        name: _hint(tool, field)
        for name, tool in tools.items()
        if not isinstance(_hint(tool, field), bool)
    }
    assert not bad, f"{field} must be True or False on every tool, not unset: {bad}"


def test_a_read_only_tool_is_never_also_destructive(tools) -> None:
    bad = [
        name for name, tool in tools.items()
        if _hint(tool, "readOnlyHint") is True and _hint(tool, "destructiveHint") is True
    ]
    assert not bad, f"read-only and destructive at once: {bad}"


def test_known_destructive_tools_say_so(tools) -> None:
    wrong = {
        name: _hint(tools[name], "destructiveHint")
        for name in sorted(DESTRUCTIVE_TOOLS)
        if _hint(tools[name], "destructiveHint") is not True
    }
    assert not wrong, f"these must have destructiveHint=True: {wrong}"
    assert all(_hint(tools[n], "readOnlyHint") is False for n in DESTRUCTIVE_TOOLS)


def test_known_read_only_tools_say_so(tools) -> None:
    wrong = [
        name for name in sorted(READ_ONLY_TOOLS)
        if _hint(tools[name], "readOnlyHint") is not True
        or _hint(tools[name], "destructiveHint") is not False
    ]
    assert not wrong, f"these must be read-only and non-destructive: {wrong}"


def test_open_world_is_claimed_only_where_a_tool_leaves_the_machine(tools) -> None:
    claimed = {name for name, tool in tools.items() if _hint(tool, "openWorldHint") is True}
    assert claimed == OPEN_WORLD_TOOLS


def test_the_cache_write_is_not_called_read_only(tools) -> None:
    """slm_cache_set writes a value; it used to claim readOnlyHint=True."""
    assert _hint(tools["slm_cache_set"], "readOnlyHint") is False


def test_the_hint_builder_refuses_a_read_only_destructive_tool() -> None:
    from superlocalmemory.mcp.tool_hints import hints

    with pytest.raises(ValueError):
        hints(read_only=True, destructive=True)
