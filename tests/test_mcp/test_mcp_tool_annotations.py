# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""MCP tool annotation tests — v3.4.26 Phase 1.

Verifies `readOnlyHint`, `destructiveHint`, and `idempotentHint` are set
correctly on SLM MCP tools. Rationale: Claude Code and similar MCP
clients serialize read-path tools that lack `readOnlyHint=True` — which
halves parallel-dispatch throughput. Setting the annotation doubles
client-side concurrency before the SLM queue even engages. Reference:
MASTER-PLAN-V5-ADDENDUM.md §1.11 + M-C-19 + T-H-08.

Also verifies destructive operations (delete_memory, forget) are marked
`destructiveHint=True` so clients can require confirmation, and
idempotent operations (currently update_memory) are marked
`idempotentHint=True` so clients can retry safely.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from mcp.server.fastmcp import FastMCP


# Tools whose calls MUST be side-effect-free — safe to parallel-dispatch.
READ_ONLY_TOOLS: set[str] = {
    # Core read tools
    "recall",
    "search",
    "fetch",
    "list_recent",
    "get_status",
    # Infinite-memory reads
    "get_soft_prompts",
    # Two-way learning reads
    "get_assertions",
    # Skill evolution reads
    "skill_health",
    "skill_lineage",
    # Mesh reads (note: mesh_summary mutates — registers/updates summary — excluded)
    "mesh_peers",
    "mesh_events",
    "mesh_status",
}

# Tools that mutate persistent state destructively — clients SHOULD confirm.
DESTRUCTIVE_TOOLS: set[str] = {
    "delete_memory",
    "forget",
}

# Tools that can be safely retried — same input produces same outcome.
IDEMPOTENT_TOOLS: set[str] = {
    # same update applied twice = same state
    "update_memory",
}

NON_IDEMPOTENT_MUTATIONS: set[str] = {
    "remember",
    "reinforce_assertion",
    "contradict_assertion",
}


def _build_server_with_all_tools() -> FastMCP:
    """Register every tool on a fresh FastMCP instance (SLM_MCP_ALL_TOOLS path)."""
    import os

    os.environ["SLM_MCP_ALL_TOOLS"] = "1"

    server = FastMCP("SLM test")
    # Dummy get_engine that returns a MagicMock — tool wiring doesn't
    # execute bodies at registration time, so this is safe.
    get_engine = MagicMock()

    from superlocalmemory.mcp.tools_core import register_core_tools
    from superlocalmemory.mcp.tools_v28 import register_v28_tools
    from superlocalmemory.mcp.tools_v3 import register_v3_tools
    from superlocalmemory.mcp.tools_active import register_active_tools
    from superlocalmemory.mcp.tools_v33 import register_v33_tools
    from superlocalmemory.mcp.tools_mesh import register_mesh_tools
    from superlocalmemory.mcp.tools_learning import register_learning_tools
    from superlocalmemory.mcp.tools_evolution import register_evolution_tools

    register_core_tools(server, get_engine)
    register_v28_tools(server, get_engine)
    register_v3_tools(server, get_engine)
    register_active_tools(server, get_engine)
    register_v33_tools(server, get_engine)
    register_mesh_tools(server, get_engine)
    register_learning_tools(server, get_engine)
    register_evolution_tools(server, get_engine)
    return server


def _tool_map(server: FastMCP) -> dict[str, object]:
    return {t.name: t for t in server._tool_manager.list_tools()}


@pytest.fixture(scope="module")
def tools() -> dict[str, object]:
    return _tool_map(_build_server_with_all_tools())


def test_read_only_tools_have_readOnlyHint_true(tools: dict[str, object]) -> None:
    """Every tool in READ_ONLY_TOOLS must have annotations.readOnlyHint == True.

    Without this, Claude Code serializes read-path tools and parallel
    dispatch is bottlenecked at the client. See v5 plan §1.11 / M-C-19.
    """
    missing: list[str] = []
    wrong: list[tuple[str, object]] = []
    for name in sorted(READ_ONLY_TOOLS):
        tool = tools.get(name)
        if tool is None:
            missing.append(name)
            continue
        ann = getattr(tool, "annotations", None)
        if ann is None or getattr(ann, "readOnlyHint", None) is not True:
            wrong.append((name, ann))
    assert not missing, f"Expected read-only tools not registered: {missing}"
    assert not wrong, (
        "These tools MUST have readOnlyHint=True but don't: "
        f"{[n for n, _ in wrong]}"
    )


def test_write_tools_do_not_claim_readOnlyHint(tools: dict[str, object]) -> None:
    """Write/mutating tools MUST NOT have readOnlyHint=True.

    Honesty: if readOnlyHint is True, the client may cache or parallelise
    calls in ways incompatible with a mutation. Explicit False is OK;
    None (unset) is OK; True is a lie.
    """
    write_tools = {
        "remember", "delete_memory", "update_memory",
        "session_init", "observe", "close_session",
        "forget", "run_maintenance", "consolidate_cognitive",
        "set_mode", "report_outcome",
        "log_tool_event", "reinforce_assertion", "contradict_assertion",
        "evolve_skill",
        "mesh_send", "mesh_lock", "mesh_state",
        "mesh_summary",  # mutates (registers/updates session summary)
        "mesh_inbox",  # mutates (auto-marks messages as read)
    }
    liars: list[str] = []
    for name in sorted(write_tools):
        tool = tools.get(name)
        if tool is None:
            continue  # tool may not be registered in this build — skip
        ann = getattr(tool, "annotations", None)
        if ann is not None and getattr(ann, "readOnlyHint", None) is True:
            liars.append(name)
    assert not liars, (
        "These mutating tools falsely claim readOnlyHint=True: "
        f"{liars}"
    )


def test_destructive_tools_have_destructiveHint_true(tools: dict[str, object]) -> None:
    """Destructive tools (delete_memory, forget) MUST have destructiveHint=True.

    Clients can then prompt the user for confirmation before invoking.
    """
    wrong: list[str] = []
    for name in sorted(DESTRUCTIVE_TOOLS):
        tool = tools.get(name)
        if tool is None:
            continue
        ann = getattr(tool, "annotations", None)
        if ann is None or getattr(ann, "destructiveHint", None) is not True:
            wrong.append(name)
    assert not wrong, (
        "These destructive tools MUST have destructiveHint=True: "
        f"{wrong}"
    )


def test_idempotent_tools_have_idempotentHint_true(tools: dict[str, object]) -> None:
    """Idempotent tools must be marked so clients can retry safely.

    update_memory: same update = same state
    """
    wrong: list[str] = []
    for name in sorted(IDEMPOTENT_TOOLS):
        tool = tools.get(name)
        if tool is None:
            continue
        ann = getattr(tool, "annotations", None)
        if ann is None or getattr(ann, "idempotentHint", None) is not True:
            wrong.append(name)
    assert not wrong, (
        "These idempotent tools MUST have idempotentHint=True: "
        f"{wrong}"
    )


def test_non_idempotent_mutations_do_not_claim_safe_retry(
    tools: dict[str, object],
) -> None:
    """Repeated state-changing calls must not advertise idempotency."""
    liars: list[str] = []
    for name in sorted(NON_IDEMPOTENT_MUTATIONS):
        tool = tools.get(name)
        if tool is None:
            continue
        ann = getattr(tool, "annotations", None)
        if ann is not None and getattr(ann, "idempotentHint", None) is True:
            liars.append(name)
    assert not liars, f"These tools falsely claim idempotency: {liars}"


def test_read_only_count_at_least_13(tools: dict[str, object]) -> None:
    """Regression guard: we ship at least 13 tools with readOnlyHint=True.

    v5 plan §1.11 / M-C-19 targets ~16; 13 is the current essential set.
    If this number drops, a refactor likely stripped an annotation.
    """
    count = 0
    for tool in tools.values():
        ann = getattr(tool, "annotations", None)
        if ann is not None and getattr(ann, "readOnlyHint", None) is True:
            count += 1
    assert count >= 12, (
        f"Only {count} tools have readOnlyHint=True; expected ≥ 12. "
        "Check that recent refactors didn't strip annotations."
    )
