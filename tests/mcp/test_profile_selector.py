# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Tests for WP-01: MCP Profile Selector

"""Tests for _PROFILE_DEFINITIONS and _resolve_profile_allowed (WP-01).

RED-first per TDD contract. All tests import symbols directly from the
module; no server side-effects are exercised.
"""

from __future__ import annotations

import importlib
import logging
import sys
import types
from typing import Callable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_module():
    """Import server module with embedded+warmup suppressed."""
    import os
    os.environ["SLM_MCP_EMBEDDED"] = "1"
    os.environ.setdefault("SLM_DISABLE_WARMUP_SIDE_EFFECTS", "1")
    mod_name = "superlocalmemory.mcp.server"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    return importlib.import_module(mod_name)


# ---------------------------------------------------------------------------
# RED-1: _PROFILE_DEFINITIONS keys are exactly {core, code, full, power, mesh}
#        ("whole" must be ABSENT)
# ---------------------------------------------------------------------------

def test_profile_definitions_keys_exact():
    mod = _get_module()
    assert hasattr(mod, "_PROFILE_DEFINITIONS"), "_PROFILE_DEFINITIONS not found in server module"
    keys = set(mod._PROFILE_DEFINITIONS.keys())
    assert keys == {"core", "code", "full", "power", "mesh"}, (
        f"Keys mismatch: got {keys}"
    )
    assert "whole" not in keys, "'whole' must be absent from _PROFILE_DEFINITIONS"


# ---------------------------------------------------------------------------
# RED-2: core == exactly the 14 names in LLD §5
# ---------------------------------------------------------------------------

_EXPECTED_CORE = frozenset({
    "remember", "recall", "search", "fetch", "list_recent", "update_memory", "forget",
    "session_init", "close_session",
    "slm_compress", "slm_retrieve", "slm_cache_set", "slm_cache_get", "slm_optimize_stats",
})


def test_profile_core_exact():
    mod = _get_module()
    core = mod._PROFILE_DEFINITIONS["core"]
    assert core == _EXPECTED_CORE, (
        f"core diff — extra: {core - _EXPECTED_CORE}, missing: {_EXPECTED_CORE - core}"
    )
    assert len(core) == 14, f"core must be 14 names, got {len(core)}"


# ---------------------------------------------------------------------------
# RED-3: code == core | 6 code-graph names (==20)
# ---------------------------------------------------------------------------

_CODE_EXTRA = frozenset({
    "build_code_graph", "get_blast_radius", "query_graph",
    "semantic_search_code", "get_review_context", "detect_changes",
})


def test_profile_code_exact():
    mod = _get_module()
    code = mod._PROFILE_DEFINITIONS["code"]
    expected = _EXPECTED_CORE | _CODE_EXTRA
    assert code == expected, (
        f"code diff — extra: {code - expected}, missing: {expected - code}"
    )
    assert len(code) == 20, f"code must be 20 names, got {len(code)}"


# ---------------------------------------------------------------------------
# RED-4: full == 38 and ⊇ core memory names; built from explicit 30+8 literal
# ---------------------------------------------------------------------------

_EXPECTED_FULL_MESH = frozenset({
    "mesh_summary", "mesh_peers", "mesh_send", "mesh_inbox",
    "mesh_state", "mesh_lock", "mesh_events", "mesh_status",
})

_EXPECTED_FULL_BASE = frozenset({
    "remember", "recall", "search", "fetch", "list_recent", "delete_memory",
    "update_memory", "get_status", "session_init", "observe", "close_session",
    "report_feedback", "forget", "run_maintenance", "consolidate_cognitive",
    "get_soft_prompts", "set_mode", "report_outcome", "log_tool_event",
    "get_assertions", "reinforce_assertion", "contradict_assertion",
    "evolve_skill", "skill_health", "skill_lineage",
    "slm_compress", "slm_retrieve", "slm_cache_set", "slm_cache_get", "slm_optimize_stats",
})

_EXPECTED_FULL = _EXPECTED_FULL_BASE | _EXPECTED_FULL_MESH


def test_profile_full_exact():
    mod = _get_module()
    full = mod._PROFILE_DEFINITIONS["full"]
    assert full == _EXPECTED_FULL, (
        f"full diff — extra: {full - _EXPECTED_FULL}, missing: {_EXPECTED_FULL - full}"
    )
    assert len(full) == 38, f"full must be 38 names, got {len(full)}"
    # Must ⊇ core memory names
    core = mod._PROFILE_DEFINITIONS["core"]
    assert core <= full, f"full must be a superset of core; missing from full: {core - full}"


# ---------------------------------------------------------------------------
# RED-5: power == 50 and ⊇ full
# ---------------------------------------------------------------------------

_POWER_EXTRA = frozenset({
    "get_version", "get_mode", "health", "consistency_check", "recall_trace",
    "get_lifecycle_status", "set_retention_policy", "compact_memories",
    "get_behavioral_patterns", "audit_trail", "quantize", "get_retention_stats",
})

_EXPECTED_POWER = _EXPECTED_FULL | _POWER_EXTRA


def test_profile_power_exact():
    mod = _get_module()
    power = mod._PROFILE_DEFINITIONS["power"]
    assert power == _EXPECTED_POWER, (
        f"power diff — extra: {power - _EXPECTED_POWER}, missing: {_EXPECTED_POWER - power}"
    )
    assert len(power) == 50, f"power must be 50 names, got {len(power)}"
    full = mod._PROFILE_DEFINITIONS["full"]
    assert full <= power, f"power must be a superset of full; missing: {full - power}"


# ---------------------------------------------------------------------------
# RED-6: mesh == 8, all names start with "mesh_"
# ---------------------------------------------------------------------------

def test_profile_mesh_exact():
    mod = _get_module()
    mesh = mod._PROFILE_DEFINITIONS["mesh"]
    assert mesh == _EXPECTED_FULL_MESH, (
        f"mesh diff — extra: {mesh - _EXPECTED_FULL_MESH}, missing: {_EXPECTED_FULL_MESH - mesh}"
    )
    assert len(mesh) == 8, f"mesh must be 8 names, got {len(mesh)}"
    non_mesh = {n for n in mesh if not n.startswith("mesh_")}
    assert not non_mesh, f"All mesh names must start with 'mesh_'; offenders: {non_mesh}"


# ---------------------------------------------------------------------------
# RED-7 (MANDATORY): Every profile name maps to a real registered tool
# ---------------------------------------------------------------------------

class _NameCollector:
    """Duck-typed FastMCP server that records func.__name__ for every tool decorated."""

    def __init__(self):
        self.collected: set[str] = set()

    def tool(self, *args, **kwargs):
        def decorator(func):
            self.collected.add(func.__name__)
            return func
        return decorator

    def resource(self, *args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def __getattr__(self, name):
        # Return a no-op callable for anything else (e.g. server.title, etc.)
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **kw: None


def test_every_profile_name_is_a_real_registered_tool():
    """RED-7: Guard against silent drops in _FilteredServer (v3.6.11 lesson).

    Builds a _NameCollector, drives all register_*_tools through it, then
    asserts every name in every profile ∈ collected set.
    """
    from superlocalmemory.mcp.tools_core import register_core_tools
    from superlocalmemory.mcp.tools_v28 import register_v28_tools
    from superlocalmemory.mcp.tools_v3 import register_v3_tools
    from superlocalmemory.mcp.tools_active import register_active_tools
    from superlocalmemory.mcp.tools_v33 import register_v33_tools
    from superlocalmemory.mcp.tools_code_graph import register_code_graph_tools
    from superlocalmemory.mcp.tools_mesh import register_mesh_tools
    from superlocalmemory.mcp.tools_learning import register_learning_tools
    from superlocalmemory.mcp.tools_evolution import register_evolution_tools
    from superlocalmemory.mcp.tools_optimize import register_optimize_tools

    collector = _NameCollector()
    get_engine_stub = lambda: None  # noqa: E731

    register_core_tools(collector, get_engine_stub)
    register_v28_tools(collector, get_engine_stub)
    register_v3_tools(collector, get_engine_stub)
    register_active_tools(collector, get_engine_stub)
    register_v33_tools(collector, get_engine_stub)
    register_code_graph_tools(collector, get_engine_stub)
    register_mesh_tools(collector, get_engine_stub)
    register_learning_tools(collector, get_engine_stub)
    register_evolution_tools(collector, get_engine_stub)
    register_optimize_tools(collector)

    mod = _get_module()
    all_profile_names: set[str] = set()
    for profile_name, profile_set in mod._PROFILE_DEFINITIONS.items():
        for tool_name in profile_set:
            all_profile_names.add(tool_name)

    missing = all_profile_names - collector.collected
    assert not missing, (
        f"Profile names NOT found in any register_*_tools body "
        f"(would be silently dropped by _FilteredServer): {sorted(missing)}"
    )


# ---------------------------------------------------------------------------
# Resolver tests
# ---------------------------------------------------------------------------

def test_resolve_profile_empty_string_returns_none():
    mod = _get_module()
    assert hasattr(mod, "_resolve_profile_allowed"), "_resolve_profile_allowed not found"
    result = mod._resolve_profile_allowed("", mod._PROFILE_DEFINITIONS, mod._ESSENTIAL_TOOLS)
    assert result is None, f"'' should return None, got {result!r}"


def test_resolve_profile_whole_returns_none():
    mod = _get_module()
    result = mod._resolve_profile_allowed("whole", mod._PROFILE_DEFINITIONS, mod._ESSENTIAL_TOOLS)
    assert result is None, f"'whole' should return None (raw server), got {result!r}"


def test_resolve_profile_known_returns_frozenset():
    mod = _get_module()
    result = mod._resolve_profile_allowed("core", mod._PROFILE_DEFINITIONS, mod._ESSENTIAL_TOOLS)
    assert result == mod._PROFILE_DEFINITIONS["core"], (
        f"'core' should return definitions['core'], got {result!r}"
    )
    assert isinstance(result, frozenset)


def test_resolve_profile_deprecated_alias_returns_canonical_and_warns(caplog):
    mod = _get_module()
    with caplog.at_level(logging.WARNING, logger="superlocalmemory.mcp.server"):
        result = mod._resolve_profile_allowed(
            "full38", mod._PROFILE_DEFINITIONS, mod._ESSENTIAL_TOOLS
        )
    assert result == mod._PROFILE_DEFINITIONS["full"]
    warning_texts = caplog.messages  # documented pytest API (getMessage-interpolated)
    assert any("full38" in t and "full" in t for t in warning_texts), (
        f"Expected warning mapping 'full38' to 'full', got: {warning_texts}"
    )


def test_resolve_profile_all_published_legacy_aliases():
    mod = _get_module()
    expected = {
        "core14": mod._PROFILE_DEFINITIONS["core"],
        "code20": mod._PROFILE_DEFINITIONS["code"],
        "mesh8": mod._PROFILE_DEFINITIONS["mesh"],
        "full38": mod._PROFILE_DEFINITIONS["full"],
        "power50": mod._PROFILE_DEFINITIONS["power"],
        "whole81": None,
    }
    for alias, allowed in expected.items():
        assert mod._resolve_profile_allowed(
            alias, mod._PROFILE_DEFINITIONS, mod._ESSENTIAL_TOOLS
        ) == allowed


def test_resolve_profile_unknown_fails_closed():
    mod = _get_module()
    import pytest

    with pytest.raises(ValueError, match=r"banana.*core.*whole"):
        mod._resolve_profile_allowed(
            "banana", mod._PROFILE_DEFINITIONS, mod._ESSENTIAL_TOOLS
        )


# ---------------------------------------------------------------------------
# RED-8: Default target uses _FilteredServer over _ESSENTIAL_TOOLS
# ---------------------------------------------------------------------------

def test_default_target_uses_essential_filtered_server():
    """With no profile env, _target must be a _FilteredServer over _ESSENTIAL_TOOLS."""
    import os
    # Ensure no profile or all-tools env is set
    for key in ("SLM_MCP_PROFILE", "SLM_MCP_ALL_TOOLS", "SLM_MCP_TOOLS"):
        os.environ.pop(key, None)
    os.environ["SLM_MCP_EMBEDDED"] = "1"

    mod_name = "superlocalmemory.mcp.server"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    mod = importlib.import_module(mod_name)

    target = mod._target
    # _target must NOT be the raw server
    assert target is not mod.server, (
        "_target should be a _FilteredServer, not the raw server, when no env vars set"
    )
    # Must be a _FilteredServer
    assert isinstance(target, mod._FilteredServer), (
        f"_target should be _FilteredServer, got {type(target).__name__}"
    )
    # Must be filtered on _ESSENTIAL_TOOLS
    assert target._allowed == mod._ESSENTIAL_TOOLS, (
        f"Default _FilteredServer._allowed must equal _ESSENTIAL_TOOLS; "
        f"diff: {target._allowed.symmetric_difference(mod._ESSENTIAL_TOOLS)}"
    )
