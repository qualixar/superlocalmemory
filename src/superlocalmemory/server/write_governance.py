# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The trust and policy checks a text save passes, for routes that save other kinds of content."""

from __future__ import annotations

import hashlib
import logging

from fastapi import HTTPException, Request

from superlocalmemory.core.actor_context import ActorContext, Transport
from superlocalmemory.core.operation_policy_registry import _DEFAULT_REGISTRY as _registry
from superlocalmemory.core.operation_request import OperationKind
from superlocalmemory.server.rbac_enforce import resolve_actor_roles

logger = logging.getLogger(__name__)


def _policy_mode(request: Request) -> str:
    rbac = getattr(request.app.state, "rbac", None)
    try:
        return "company" if rbac is not None and rbac.user_count() > 0 else "local"
    except Exception:  # noqa: BLE001 - an unreadable mode counts as local, as for text saves
        return "local"


def _actor_context(request: Request, engine, actor_id: str, profile: str) -> ActorContext:
    token = (request.headers.get("x-slm-user-session", "")
             or (request.cookies.get("slm_session", "") if request.cookies else "")) or ""
    return ActorContext(
        principal_id=actor_id, roles=resolve_actor_roles(request, profile=profile),
        active_profile_id=engine._profile_id, transport=Transport.HTTP,
        client_host=(request.client.host if request.client is not None else "") or "",
        session_token_hash=hashlib.sha256(token.encode()).hexdigest()[:16] if token else "")


def _enforce(request: Request, engine, kind: OperationKind, hook_payload: dict, *, actor_id: str,
             profile: str, what: str) -> None:
    try:
        engine._hooks.run_pre("store" if kind is OperationKind.REMEMBER else "delete", hook_payload)
    except Exception as exc:  # noqa: BLE001 - any refusal from the hook stops the action
        logger.info("%s refused by the trust hook (%s)", what, type(exc).__name__)
        raise HTTPException(403, detail=f"This {what} was refused by the workspace trust policy.") from None
    decision = _registry.evaluate(kind, _actor_context(request, engine, actor_id, profile), _policy_mode(request))
    if not decision.allowed:
        raise HTTPException(403, detail=f"This {what} is not allowed by the workspace policy.")


def enforce_remember_governance(request: Request, engine, *, actor_id: str, profile: str, preview: str) -> None:
    """Run the trust pre-hook and the REMEMBER policy; 403 when either refuses. Stores nothing."""
    _enforce(request, engine, OperationKind.REMEMBER,
             {"operation": "store", "agent_id": actor_id, "profile_id": profile, "content_preview": preview[:100]},
             actor_id=actor_id, profile=profile, what="save")


def enforce_forget_governance(request: Request, engine, *, actor_id: str, profile: str, target: str) -> None:
    """Run the trust pre-hook and the FORGET policy for taking ``target`` out of view; 403 when refused."""
    _enforce(request, engine, OperationKind.FORGET,
             {"operation": "delete", "agent_id": actor_id, "profile_id": profile, "fact_id": target},
             actor_id=actor_id, profile=profile, what="removal")
