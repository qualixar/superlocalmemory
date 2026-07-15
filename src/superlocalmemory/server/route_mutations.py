# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Authenticated policy boundary for non-ingestion HTTP mutations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, Request


@dataclass
class RouteMutationAuthorization:
    """One pre-authorized mutation whose post hook is emitted on success."""

    engine: Any
    operation: str
    context: dict[str, str]
    _completed: bool = False

    @property
    def actor_id(self) -> str:
        return self.context["agent_id"]

    def complete(self) -> None:
        """Record a successful mutation exactly once."""
        if self._completed:
            return
        self.engine._hooks.run_post(self.operation, self.context)
        self._completed = True


def authorize_route_mutation(
    request: Request,
    *,
    operation: str,
    source_agent_id: str,
    profile_id: str | None = None,
    fact_id: str = "",
    content_preview: str = "",
) -> RouteMutationAuthorization:
    """Consume the authenticated principal and run the engine pre-hook.

    ``source_agent_id`` is a code-owned route label, never caller identity.
    The security principal comes from middleware or a verified credential.
    """
    if operation not in {"update", "delete"}:
        raise ValueError("route mutations must use update or delete policy")

    from superlocalmemory.server.routes.helpers import get_engine_lazy
    from superlocalmemory.server.write_identity import authenticated_request_actor

    app_state = request.app.state
    actor_id = authenticated_request_actor(
        request,
        getattr(app_state, "daemon_descriptor", None),
        actor_kind=source_agent_id,
    )
    engine = get_engine_lazy(app_state)
    if engine is None or not hasattr(engine, "_hooks"):
        raise HTTPException(503, detail="Engine not initialized")

    effective_profile = profile_id or engine.profile_id
    context = {
        "operation": operation,
        "agent_id": actor_id,
        "source_agent_id": source_agent_id,
        "profile_id": effective_profile,
    }
    if fact_id:
        context["fact_id"] = fact_id
    if content_preview:
        context["content_preview"] = content_preview[:100]

    try:
        engine._hooks.run_pre(operation, context)
    except Exception as exc:
        raise HTTPException(403, detail="Write authorization rejected") from exc
    return RouteMutationAuthorization(engine, operation, context)


__all__ = ["RouteMutationAuthorization", "authorize_route_mutation"]
