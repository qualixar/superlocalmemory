# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Shared MCP utilities — single source of truth for helpers used
across tools_core, tools_active, tools_v28, tools_v3, tools_v33.

V3.3.12: Extracted _emit_event to eliminate code duplication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from superlocalmemory.infra.data_root import state_path


def emit_event(event_type: str, payload: dict | None = None,
               source_agent: str = "mcp_client") -> None:
    """Emit an event to the EventBus (best-effort, never raises)."""
    try:
        from superlocalmemory.infra.event_bus import EventBus
        bus = EventBus.get_instance(state_path("memory.db"))
        bus.emit(event_type, payload=payload, source_agent=source_agent,
                 source_protocol="mcp")
    except Exception:
        pass


@dataclass
class MCPMutationAuthorization:
    """One capability-owned MCP mutation authorized by engine policy."""

    engine: Any
    operation: str
    context: dict[str, str]
    _completed: bool = False

    @property
    def actor_id(self) -> str:
        return self.context["agent_id"]

    def complete(self) -> None:
        """Emit the matching post hook once the mutation succeeds."""
        if self._completed:
            return
        self.engine._hooks.run_post(self.operation, self.context)
        self._completed = True


def authorize_mcp_mutation(
    engine: Any,
    operation: str,
    *,
    mutation_source: str,
    profile_id: str | None = None,
    fact_id: str = "",
    content_preview: str = "",
) -> MCPMutationAuthorization:
    """Derive local authority and run policy before a direct MCP write.

    MCP URL/env agent identifiers remain audit attribution only.  The trusted
    principal is derived from the private local install capability, matching
    the canonical ingestion and recall-worker mutation boundaries.
    """
    if operation not in {"update", "delete"}:
        raise ValueError("MCP mutations must use update or delete policy")

    from superlocalmemory.core.engine_ingestion import local_trusted_actor_id
    from superlocalmemory.mcp.agent_context import get_current_agent_id

    source_agent_id = get_current_agent_id(env_fallback=True)
    context = {
        "operation": operation,
        "agent_id": local_trusted_actor_id("mcp"),
        "source_agent_id": source_agent_id,
        "mutation_source": mutation_source,
        "profile_id": profile_id or engine.profile_id,
    }
    if fact_id:
        context["fact_id"] = fact_id
    if content_preview:
        context["content_preview"] = content_preview[:100]
    engine._hooks.run_pre(operation, context)
    return MCPMutationAuthorization(engine, operation, context)


__all__ = [
    "MCPMutationAuthorization",
    "authorize_mcp_mutation",
    "emit_event",
]
