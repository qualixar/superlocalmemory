# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""MCP profile definitions — pure data, no side effects.

Extracted from mcp/server.py (v3.8.0) so the daemon can import profile
metadata without triggering FastMCP tool registration or engine warmup.

server.py re-exports all names from this module for backward compatibility.
Do NOT import FastMCP, MemoryEngine, or any heavy dependency here.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# v3.6.14 WP-01: Named profile definitions
# ---------------------------------------------------------------------------

_PROFILE_CORE: frozenset[str] = frozenset({  # 14
    "remember", "recall", "search", "fetch", "list_recent", "update_memory", "forget",
    "session_init", "close_session",
    "slm_compress", "slm_retrieve", "slm_cache_set", "slm_cache_get", "slm_optimize_stats",
})

_PROFILE_CODE: frozenset[str] = _PROFILE_CORE | frozenset({  # 21
    "build_code_graph", "get_blast_radius", "query_graph",
    "semantic_search_code", "get_review_context", "detect_changes",
    # switch_profile lets a plugin/IDE session change the active workspace over
    # MCP (the plugin ships SLM_MCP_PROFILE=code, so it must be here). The
    # underlying route is RBAC member-gated, so company-mode isolation holds.
    "switch_profile",
})

_PROFILE_FULL_MESH: frozenset[str] = frozenset({  # 8
    "mesh_summary", "mesh_peers", "mesh_send", "mesh_inbox",
    "mesh_state", "mesh_lock", "mesh_events", "mesh_status",
})

_PROFILE_FULL: frozenset[str] = frozenset({  # 31 base — EXPLICIT literal, NOT runtime _ESSENTIAL_TOOLS (OQ-2)
    "remember", "recall", "search", "fetch", "list_recent", "delete_memory", "update_memory",
    "get_status", "session_init", "observe", "close_session", "report_feedback", "forget",
    "run_maintenance", "consolidate_cognitive", "get_soft_prompts", "set_mode", "report_outcome",
    "log_tool_event", "get_assertions", "reinforce_assertion", "contradict_assertion",
    "evolve_skill", "skill_health", "skill_lineage", "switch_profile",
    "slm_compress", "slm_retrieve", "slm_cache_set", "slm_cache_get", "slm_optimize_stats",
}) | _PROFILE_FULL_MESH  # 39

_PROFILE_POWER: frozenset[str] = _PROFILE_FULL | frozenset({  # 51
    "get_version", "get_mode", "health", "consistency_check", "recall_trace",
    "get_lifecycle_status", "set_retention_policy", "compact_memories",
    "get_behavioral_patterns", "audit_trail", "quantize", "get_retention_stats",
})

_PROFILE_MESH: frozenset[str] = _PROFILE_FULL_MESH  # 8

# Canonical name → frozenset mapping.  "whole" is intentionally absent —
# it maps to the raw server (all tools, D-2 LOCKED).
_PROFILE_DEFINITIONS: dict[str, frozenset[str]] = {
    "core": _PROFILE_CORE,
    "code": _PROFILE_CODE,
    "full": _PROFILE_FULL,
    "power": _PROFILE_POWER,
    "mesh": _PROFILE_MESH,
}

# Compatibility aliases published by the v3.6 README.  Stale client
# configurations have one deterministic meaning; migration warnings fire
# at server startup.  Any other value is a configuration error (fail closed).
_PROFILE_ALIASES: dict[str, str] = {
    "core14": "core",
    # 3.8.0: switch_profile added to code/full/power (counts +1). Old count-
    # suffixed names kept so a v3.6/3.7 config still resolves (back-compat).
    "code20": "code",
    "code21": "code",
    "full38": "full",
    "full39": "full",
    "power50": "power",
    "power51": "power",
    "mesh8": "mesh",
    "whole81": "whole",
}

# Plain-English descriptions for UI display.
# Rules: no internal jargon (no POMDP, Fisher-Rao, TurboQuant, etc.),
# one sentence, user-facing language only.
PROFILE_DESCRIPTIONS: dict[str, str] = {
    "core": "Essential memory: store, recall, search, sessions",
    "code": "Core + code-graph tools + profile switching (default for IDE coding agents)",
    "full": "All everyday memory, optimization, and mesh tools",
    "power": "Everything in full plus advanced governance/behavioral tools",
    "mesh": "Cross-device mesh coordination only",
}
