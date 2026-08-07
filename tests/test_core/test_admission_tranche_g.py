# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Tranche G admission tests: read-only annotations + live dynamic coverage.

Tests:
  G1: 32 read-only tools have readOnlyHint=True in the server registry.
  G2: mesh_inbox is in _GATED_MCP_TOOLS (gated mutator).
  G3: mesh_inbox is in _REQUIRED_MCP_GATES.
  G4: Real-registry completeness — coverage_self_check(ENTERPRISE, server=server)
      does NOT raise with SLM_MCP_ALL_TOOLS=1 after all annotations land.
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_server_module(monkeypatch, tmp_path):
    """Reload mcp.server with SLM_MCP_ALL_TOOLS=1 and a tmp data dir.

    Returns the freshly-reloaded module (server attribute = real SLMFastMCP).
    Admission module is NOT reloaded (preserves AdmissionDenied identity).
    """
    monkeypatch.setenv("SLM_MCP_ALL_TOOLS", "1")
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    # Write a minimal personal config so SLMConfig.load() doesn't fail
    (tmp_path / "config.toml").write_text("")

    import superlocalmemory.mcp.server as server_mod
    importlib.reload(server_mod)
    return server_mod


# ---------------------------------------------------------------------------
# Tool names expected in each category
# ---------------------------------------------------------------------------

_READ_ONLY_TOOLS = frozenset({
    # tools_active.py
    "session_init",
    # tools_core.py
    "backup_status", "memory_used", "get_learned_patterns", "get_attribution",
    # tools_v28.py
    "get_lifecycle_status", "get_behavioral_patterns", "audit_trail",
    # tools_v3.py
    "get_version", "get_mode", "health", "consistency_check", "recall_trace",
    # tools_v33.py
    "get_retention_stats",
    # tools_code_graph.py (18 tools)
    "get_blast_radius", "get_review_context", "query_graph",
    "semantic_search_code", "list_graph_stats", "find_large_functions",
    "list_flows", "get_flow", "get_affected_flows",
    "list_communities", "get_community", "get_architecture_overview",
    "detect_changes", "refactor_preview", "code_memory_search",
    "code_entity_history", "enrich_blast_radius", "code_stale_check",
})


# ---------------------------------------------------------------------------
# G1: Every read-only tool has readOnlyHint=True in the server tool registry
# ---------------------------------------------------------------------------

def _read_only_hint(tool: object) -> object:
    """Wire (readOnlyHint) or mcp 2.0 model (read_only_hint) attribute."""
    ann = getattr(tool, "annotations", None)
    if ann is None:
        return None
    val = getattr(ann, "readOnlyHint", None)
    if val is not None:
        return val
    return getattr(ann, "read_only_hint", None)


def test_read_only_tools_annotated(monkeypatch, tmp_path):
    """G1 — 32 tools carry readOnlyHint=True in the MCP server registry."""
    server_mod = _load_server_module(monkeypatch, tmp_path)
    real_server = server_mod.server

    tool_dict = real_server._tool_manager._tools
    missing_annotation = []
    for name in sorted(_READ_ONLY_TOOLS):
        tool = tool_dict.get(name)
        if tool is None:
            missing_annotation.append(f"{name}: NOT REGISTERED")
            continue
        hint = _read_only_hint(tool)
        if hint is not True:
            missing_annotation.append(f"{name}: readOnlyHint={hint!r}")

    assert not missing_annotation, (
        "Tools missing readOnlyHint=True annotation:\n"
        + "\n".join(f"  {m}" for m in missing_annotation)
    )


# ---------------------------------------------------------------------------
# G2: mesh_inbox is in _GATED_MCP_TOOLS (decorated with @admits)
# ---------------------------------------------------------------------------

def test_mesh_inbox_in_gated_tools(monkeypatch, tmp_path):
    """G2 — mesh_inbox is in _GATED_MCP_TOOLS after server registration."""
    _load_server_module(monkeypatch, tmp_path)

    import superlocalmemory.core.admission as adm
    assert "mesh_inbox" in adm._GATED_MCP_TOOLS, (
        "mesh_inbox not found in _GATED_MCP_TOOLS — @admits decorator missing"
    )


# ---------------------------------------------------------------------------
# G3: mesh_inbox is in _REQUIRED_MCP_GATES (static declaration)
# ---------------------------------------------------------------------------

def test_mesh_inbox_in_required_gates():
    """G3 — mesh_inbox is declared in _REQUIRED_MCP_GATES."""
    import superlocalmemory.core.admission as adm
    assert "mesh_inbox" in adm._REQUIRED_MCP_GATES, (
        "mesh_inbox missing from _REQUIRED_MCP_GATES in admission.py"
    )


# ---------------------------------------------------------------------------
# G4: Real-registry completeness proof
# ---------------------------------------------------------------------------

def test_real_registry_completeness(monkeypatch, tmp_path):
    """G4 — coverage_self_check(ENTERPRISE, server=real_server) does not raise.

    This is the machine-checked proof that every mutating MCP tool is gated.
    With SLM_MCP_ALL_TOOLS=1 all 84 tools are registered; after G-tranche
    annotations every read-only tool carries readOnlyHint=True so the dynamic
    check in coverage_self_check finds zero ungated mutators.
    """
    (tmp_path / "config.toml").write_text('[deployment]\nmode = "enterprise"\n')
    monkeypatch.setenv("SLM_MCP_CONFIG", str(tmp_path / "config.toml"))
    server_mod = _load_server_module(monkeypatch, tmp_path)
    real_server = server_mod.server

    from superlocalmemory.core.admission import coverage_self_check
    from superlocalmemory.core.config import DEPLOYMENT_ENTERPRISE

    # Should NOT raise — if it does, the error message names the offending tools.
    coverage_self_check(DEPLOYMENT_ENTERPRISE, server=real_server)
