# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Who may change state over HTTP once team accounts are on.

``read_gates`` answers the read side. This is the write side: one place that
every POST, PUT, PATCH and DELETE passes through before the route runs, so a
route added later without its own role check is still refused to a caller who
may not change anything.

Before this, about fifteen state-changing routes (``/observe``, the mesh, the
Optimize settings, compaction, tiering, heal, profile switch, restart ...) only
proved a machine credential. In company mode ("Require login") the install
token is not authority: ``GET /internal/token`` hands it to any local process,
so such a process could change the shared workspace unnamed, and a signed-in
read-only ``viewer`` could lock files and change settings.

The gate runs ahead of body validation, so a refusal is a 401 or 403 and never
a 422 or 500 that shows the handler was reached first. Routes that already ask
for a role of their own still do; this is the floor under them, not a
replacement. With no team accounts, or when login is optional and the caller
has no session, nothing changes: the machine operator keeps full access.

The tables below are the whole policy. A route with no entry needs ``write`` on
a workspace the caller belongs to (``scope="any"``: the route names its own
workspace and decides the exact one). Routes that act on the active workspace
only are listed with ``scope="active"`` and are checked against it. Routes
that change one setting for the whole installation (the login policy, the
provider, Optimize, Jev consent, restarts ...) are ``scope="global"``: MANAGE
on one workspace is not enough, the caller must administer every workspace or
be the machine owner (PR 248 review, finding 1).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from superlocalmemory.access.rbac import Permission, Role, permissions_for_role

logger = logging.getLogger(__name__)

_MUTATING = frozenset({"POST", "PUT", "PATCH", "DELETE"})


@dataclass(frozen=True)
class Rule:
    """What a caller must hold to change state on a route.

    ``scope``: ``active`` = the active workspace (a ``?profile=`` the route does
    not read must never move the check); ``named`` = the workspace in
    ``?profile_id=`` when the route honours it, else the active one; ``any`` =
    some workspace the caller belongs to (the route picks the workspace itself
    and enforces it); ``path`` = the workspace in the URL; ``global`` = every workspace
    (installation-wide state). ``owner_permission``
    is what the machine owner needs when it has no session (defaults to
    ``permission``). ``machine_ok`` lets a machine
    principal (daemon capability, fleet secret) through without a session, as
    the mesh read gate does.
    """

    permission: Permission
    scope: str = "any"
    owner_permission: Permission | None = None
    machine_ok: bool = False


#: Not role-gated here; each has its own, stronger or different, check.
OPEN_EXACT = {
    "/api/rbac/login": "signing in is how a person gets a session; it cannot require one",
    "/api/rbac/logout": "ends the caller's own session (its route has the machine guard)",
    "/internal/prewarm": "hook hot path with its own four gates; fills a derived cache only",
    "/api/v3/embed": "computes vectors for text the caller sends and stores nothing",
    "/stop": "needs the private daemon capability (break-glass), checked by the route",
}
OPEN_PREFIXES = (
    "/mcp/",      # mounted MCP app: each tool is admitted by its own decorator
    "/v1/",       # LLM provider proxy: carries provider keys, not SLM roles
    "/v1beta/",
)

_READ = Rule(Permission.READ)
_GLOBAL = Rule(Permission.MANAGE, "global")
_WRITE_ACTIVE = Rule(Permission.WRITE, "active")
_MESH = Rule(Permission.WRITE, "active", machine_ok=True)
_WRITE_NAMED = Rule(Permission.WRITE, "named")

#: Exact paths, first.
_EXACT: dict[str, Rule] = {
    # POSTs that only read memory: a viewer may call them.
    "/api/search": _READ,
    "/api/v3/chat/stream": _READ,
    "/api/v3/recall/trace": _READ,
    "/session/open": _READ,
    # Installation-wide state: one setting for every workspace.
    **{path: _GLOBAL for path in (
        "/api/rbac/policy",
        "/api/memory-kinds/settings",
        "/api/learning/backup",
        "/api/learning/migrate-legacy",
        "/maintenance/run",
        "/maintenance/compact",
        "/maintenance/regraph",
        "/api/daemon/restart",
        "/api/daemon/doctor",
        "/api/daemon/repair",
        "/api/v3/components/heal",
        "/api/v3/ide/connect",
        "/api/v3/ollama/validate",
        "/api/v3/mode",
        "/api/v3/mode/set",
        "/api/v3/scope/config",
        "/api/v3/runtime/config",
        "/api/v3/auto-capture/config",
        "/api/v3/auto-recall/config",
        "/api/v3/auto-invoke/config",
        "/api/v3/daemon/config",
        "/api/v3/forgetting/config",
        "/api/v3/graph/config",
        "/api/v3/mesh/config",
        "/api/v3/storage/config",
        "/api/v3/trust/config",
        "/api/v3/ratelimit",
        "/api/v3/features/media/enable",
        "/api/v3/features/media/disable",
        "/api/v3/answer-check/mode",
        "/api/v3/answer-check/save",
        "/api/evolution/enable",
        "/api/evolution/disable",
        "/api/evolution/config",
    )},
    **{path: Rule(Permission.MANAGE, "active") for path in (
        "/api/evolution/run", "/api/evolution/approve", "/api/evolution/rollback",
    )},
    "/api/profiles/create": Rule(Permission.MANAGE, "any"),
    # Writes into the active workspace that name no workspace of their own.
    "/observe": _WRITE_ACTIVE,
    "/session/close": _WRITE_ACTIVE,
    "/consolidate/cognitive": _WRITE_ACTIVE,
    "/api/v3/learning/consolidate": _WRITE_ACTIVE,
    "/api/v3/forgetting/run": _WRITE_ACTIVE,
    "/api/v3/graph/run-communities": _WRITE_ACTIVE,
}

#: Path prefixes, in order.
_PREFIXES: tuple[tuple[str, Rule], ...] = (
    ("/api/v3/local-apps/", _GLOBAL),
    ("/api/integrity/repair", _GLOBAL),
    ("/api/optimize", _GLOBAL),
    ("/api/v3/provider", _GLOBAL),
    ("/api/v3/embedding", _GLOBAL),
    ("/api/v3/answer-check/jev/", _GLOBAL),
    ("/api/v3/answer-check/laya/", _GLOBAL),
    ("/api/backup", _GLOBAL),
    ("/api/upgrade/", _GLOBAL),
    ("/api/adapters/", _GLOBAL),
    ("/api/rbac", Rule(Permission.MANAGE, "any")),
    ("/mesh/", _MESH),
    ("/maintenance/", _WRITE_ACTIVE),
    ("/api/lifecycle/", _WRITE_ACTIVE),
    ("/api/tiers/", _WRITE_NAMED),  # tiers read ?profile_id= themselves
    ("/api/v3/consolidation/", _WRITE_ACTIVE),
)

_PROFILE_SWITCH = re.compile(r"^/api/profiles/([^/]+)/switch$")
_PROFILE_DELETE = re.compile(r"^/api/profiles/([^/]+)$")
_DEFAULT = Rule(Permission.WRITE, "any")


def rule_for(method: str, path: str) -> Rule | None:
    """The rule for one request, or ``None`` when this layer leaves it alone."""
    if method not in _MUTATING:
        return None
    if path in OPEN_EXACT or path.startswith(OPEN_PREFIXES) or path == "/mcp":
        return None
    if path in _EXACT:
        return _EXACT[path]
    if method == "POST" and _PROFILE_SWITCH.match(path):
        # A user enters a workspace they belong to; the owner must administer.
        return Rule(Permission.READ, "path", owner_permission=Permission.MANAGE)
    if method == "DELETE" and _PROFILE_DELETE.match(path):
        return Rule(Permission.MANAGE, "path")
    for prefix, rule in _PREFIXES:
        if path.startswith(prefix):
            return rule
    return _DEFAULT


def _refusal(status: int, message: str):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=status, content={"error": message, "detail": message})


def _session_token(request) -> str:
    return (request.headers.get("x-slm-user-session", "")
            or (request.cookies.get("slm_session", "") if request.cookies else ""))


def _target_profile(request, rule: Rule) -> str | None:
    """The workspace a ``active`` or ``path`` rule is checked against."""
    if rule.scope == "path":
        found = (_PROFILE_SWITCH.match(request.url.path)
                 or _PROFILE_DELETE.match(request.url.path))
        return found.group(1) if found else None
    if rule.scope == "named":
        return request.query_params.get("profile_id", "").strip() or None
    return None


def _user_holds(rbac: Any, user: dict, rule: Rule, profile: str | None) -> bool:
    uid = user["user_id"]
    if rule.scope == "global":
        from superlocalmemory.server.rbac_enforce import holds_global_manage

        return holds_global_manage(rbac, uid)
    if rule.scope == "any":
        for row in rbac.list_user_profiles(uid):
            try:
                if rule.permission in permissions_for_role(Role(row["role"])):
                    return True
            except ValueError:
                continue
        return False
    if profile is None:
        from superlocalmemory.server.rbac_enforce import _active_profile

        profile = _active_profile()
    return bool(rbac.has_permission(uid, profile, rule.permission))


def _machine_principal(request, app_state) -> bool:
    """A computer's own agent (daemon capability) or a mesh node (fleet secret)."""
    from superlocalmemory.server.access_gate import mesh_secret_ok
    from superlocalmemory.server.read_gates import _capability_ok

    headers = {k.lower(): v for k, v in request.headers.items()}
    return _capability_ok(request, app_state) or mesh_secret_ok(headers, app_state)


def _check_owner(request, app_state, rule: Rule, profile: str | None):
    """No session: the machine owner. Company mode needs a person or break-glass."""
    from fastapi import HTTPException

    from superlocalmemory.server.rbac_enforce import require_permission

    if rule.machine_ok and _machine_principal(request, app_state):
        return None
    try:
        require_permission(request, rule.owner_permission or rule.permission, profile=profile)
    except HTTPException as exc:
        return _refusal(exc.status_code, str(exc.detail))
    return None


def mutation_gate(request, app_state):
    """``None`` to allow; otherwise the refusal response (401, 403 or 503)."""
    rule = rule_for(request.method, request.url.path)
    rbac = getattr(app_state, "rbac", None)
    if rule is None or rbac is None:
        return None
    from fastapi import HTTPException

    from superlocalmemory.server.rbac_enforce import resolve_principal

    try:
        profile = _target_profile(request, rule)
        if not _session_token(request) and not rbac.require_login():
            return None  # personal mode: the operator may change anything
        principal = resolve_principal(request)
        if principal["kind"] != "user":
            return _check_owner(request, app_state, rule, profile)
        if _user_holds(rbac, principal, rule, profile):
            return None
    except HTTPException as exc:
        return _refusal(exc.status_code, str(exc.detail))
    except Exception:  # noqa: BLE001 - an unreadable policy is a no, not a yes
        logger.warning("mutation gate: authorization could not be read", exc_info=True)
        return _refusal(503, "authorization temporarily unavailable")
    if rule.scope == "global":
        from superlocalmemory.server.rbac_enforce import GLOBAL_MANAGE_MESSAGE

        return _refusal(403, GLOBAL_MANAGE_MESSAGE)
    return _refusal(
        403, f"Your role does not allow '{rule.permission.value}' on this workspace.",
    )


__all__ = ["OPEN_EXACT", "OPEN_PREFIXES", "Rule", "mutation_gate", "rule_for"]
