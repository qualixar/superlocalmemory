# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Which tools a connection sees: the essential set, named tool sets and their precedence.

Moved out of ``mcp/server.py`` so ``slm tool`` applies exactly the filter the
MCP server applies, without importing the stdio server. ``server.py`` keeps its
old names (``_ESSENTIAL_TOOLS``, ``_resolve_profile_allowed``) as aliases.

Precedence (unchanged): ``SLM_MCP_ALL_TOOLS=1`` > ``SLM_MCP_TOOLS`` allowlist >
tool set (``SLM_MCP_PROFILE``, else the one saved by ``slm tools set``) >
the essential default. ``None`` means the raw server: every registered tool.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Mapping

# The tool-set deprecation warning is a server start-up message: keep the server's logger.
logger = logging.getLogger("superlocalmemory.mcp.server")

#: The default surface before mesh and image tools are added (mirrors ``full``).
ESSENTIAL_BASE: frozenset[str] = frozenset({
    # Core memory operations (8)
    "remember",
    "recall",
    "search",
    "fetch",
    "expand",
    "list_recent",
    "delete_memory",
    "update_memory",
    "get_status",
    "switch_profile",
    # Session lifecycle (3)
    "session_init",
    "observe",
    "close_session",
    # Feedback / learning signals — reachable Dash-Core path for
    # thumbs-up / pin / drift signals.
    "report_feedback",
    # v4.0.2 portable Brain evidence: profile-scoped receipt reads/writes.
    "get_brain_evidence_status",
    "record_agent_experience",
    "record_cognitive_turn",
    "finalize_cognitive_turn",
    # v4.0.4: explicit, optional observation from the separately installed
    # Bounded Loops MCP producer. It never participates in recall/ranking.
    "observe_bounded_loop_evidence",
    # Update, review, and list form one core correction lifecycle.
    "review_correction",
    "list_corrections",
    # 4.2.30 (D-37): an agent that meets a dispute in recall must be able to
    # list and settle it with the user's answer.
    "list_disputes",
    "settle_dispute",
    # v4.0.8 (#113): readable summaries. Present here as well as in the named
    # profiles because this set is the FALLBACK surface — it must mirror
    # ``full``, and a tool that ships in the smallest profile ("core") cannot be
    # missing from the fallback without a client silently losing it.
    "get_memory_summary",
    # 4.1.21 (#113): saved views — run (read) and manage (write). Mirrors
    # ``full`` in mcp/profiles.py, as this set must.
    "run_view",
    "manage_view",
    # Memory management (2)
    "forget",
    "run_maintenance",
    # NOTE: prestage_context IS registered (see register_prestage_tool below)
    # but is deliberately absent from the default surface. Brain evidence is
    # intentionally present: it is the portable contract for installed agents.
    # Infinite memory + learning (4)
    "consolidate_cognitive",
    "get_soft_prompts",
    "set_mode",
    "report_outcome",
    # v3.4.7: Two-way learning (4)
    "log_tool_event",
    "get_assertions",
    "reinforce_assertion",
    "contradict_assertion",
    # v3.4.11: Skill evolution (3)
    "evolve_skill",
    "skill_health",
    "skill_lineage",
    # v3.6.11: Surface B Optimize tools (5)
    "slm_compress",
    "slm_retrieve",
    "slm_cache_set",
    "slm_cache_get",
    "slm_optimize_stats",
    # v3.8.0: bounded-loop tools (3) — CLI + /slm-loop command + MCP. Kept in
    # the default exposure so any MCP client discovers gated loops, not just
    # profile=code/full/power sessions.
    "slm_loop_run",
    "slm_loop_history",
    "slm_loop_show",
    # 4.1.19 WP8: memory-kind management (4). Varun's call (2026-10-03):
    # default visibility, not power-only — every normal session gets these,
    # matching mcp/profiles.py's _PROFILE_KINDS in code/full (this set must
    # mirror full exactly; see test_prestage_registered.py).
    "set_memory_kind",
    "memory_kinds_status",
    "review_memory_kinds",
    "confirm_memory_kinds",
})

_MESH_TOOLS: frozenset[str] = frozenset({
    "mesh_summary", "mesh_peers", "mesh_send", "mesh_inbox", "mesh_wait",
    "mesh_state", "mesh_lock", "mesh_events", "mesh_status",
})


def mesh_tools_enabled(env: Mapping[str, str] | None = None) -> bool:
    """v3.4.4: mesh tools are on with ``mesh_enabled`` in config or ``SLM_MCP_MESH_TOOLS=1``."""
    env = os.environ if env is None else env
    if env.get("SLM_MCP_MESH_TOOLS", "").lower() in ("1", "true"):
        return True
    try:
        from superlocalmemory.core.config import SLMConfig

        return getattr(SLMConfig.load(), "mesh_enabled", True)  # default True in v3.4.3+
    except Exception:
        return True  # Safe default — mesh broker is always in daemon


def media_tools_enabled() -> bool:
    """Image and document tools are listed by default only while images are on."""
    try:
        from superlocalmemory.runtimes import features

        return bool(features.media_enabled())
    except Exception:
        return False


def essential_tools(env: Mapping[str, str] | None = None) -> frozenset[str]:
    """The default surface: the essential base, plus mesh and image tools when on."""
    tools = set(ESSENTIAL_BASE)
    if mesh_tools_enabled(env):
        tools |= _MESH_TOOLS
    if media_tools_enabled():
        from superlocalmemory.mcp import profiles

        tools |= profiles._MEDIA_TOOLS
    return frozenset(tools)


def resolve_profile_allowed(
    profile: str,
    definitions: dict[str, frozenset[str]],
    essential: frozenset[str],
) -> frozenset[str] | None:
    """Resolve a profile name to its allowed tool frozenset, or None for raw-server routes.

    Returns:
        None  — for "" (no selection) or "whole" (raw server, all tools).
        frozenset — the profile's tool set for known profiles.

    Raises:
        ValueError: if ``profile`` is neither canonical nor an explicitly
            supported compatibility alias.
    """
    from superlocalmemory.mcp.profiles import _PROFILE_ALIASES

    canonical = _PROFILE_ALIASES.get(profile, profile)
    if canonical != profile:
        logger.warning(
            "SLM_MCP_PROFILE=%r is deprecated; use %r instead.",
            profile,
            canonical,
        )
    if not canonical or canonical == "whole":
        return None
    if canonical in definitions:
        return definitions[canonical]
    from superlocalmemory.mcp.profile_names import (
        UnknownProfileError,
        unknown_profile_message,
    )

    raise UnknownProfileError(unknown_profile_message(profile))


def surface_for(
    *,
    all_tools: bool,
    allowlist: str,
    profile: str,
    essential: frozenset[str],
    resolve: Callable[..., frozenset[str] | None] = resolve_profile_allowed,
) -> frozenset[str] | None:
    """The allowed tool names for one connection's settings, or ``None`` for every tool."""
    from superlocalmemory.mcp.profiles import (
        _MEDIA_TOOLS,
        _PROFILE_ALIASES,
        _PROFILE_DEFINITIONS,
    )

    if all_tools:
        return None
    if allowlist:
        return frozenset(t.strip() for t in allowlist.split(",") if t.strip())
    if profile == "whole":
        return None
    if profile:
        allowed = resolve(profile, _PROFILE_DEFINITIONS, essential)
        if allowed is None:
            return None
        canonical = _PROFILE_ALIASES.get(profile, profile)
        return allowed | (_MEDIA_TOOLS if canonical in {"full", "power"} else frozenset())
    return essential | _MEDIA_TOOLS


def connection_surface(env: Mapping[str, str] | None = None) -> frozenset[str] | None:
    """What an MCP connection started now, with this environment, would be allowed."""
    from superlocalmemory.mcp.profile_names import effective_profile

    env = os.environ if env is None else env
    return surface_for(
        all_tools=env.get("SLM_MCP_ALL_TOOLS") == "1",
        allowlist=env.get("SLM_MCP_TOOLS", "").strip(),
        profile=effective_profile(),
        essential=essential_tools(env),
    )


__all__ = [
    "ESSENTIAL_BASE", "connection_surface", "essential_tools", "media_tools_enabled",
    "mesh_tools_enabled", "resolve_profile_allowed", "surface_for",
]
