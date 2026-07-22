# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 — RBAC / teams (C3)

"""RBAC enforcement boundary for HTTP routes.

This is the single place that turns "who is calling" + "what are they trying to
do" into an allow/deny decision, on top of the existing machine-auth layer
(write_identity). It is deliberately small so every mutation route calls the
same code path — the research warning was explicit: an RBAC layer that is
defined but not consistently called is worse than none.

Principal model
---------------
* **user**  — a logged-in dashboard user (valid session token). Always enforced
  against their role on the active profile.
* **owner** — the machine operator (already proved machine auth via
  write_identity; no user session). In personal mode the owner bypasses RBAC
  (all permissions). When the org turns on *require_login* (company mode) the
  owner bypass is disabled and a user session is mandatory.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request

from superlocalmemory.access.rbac import Permission, Role, permissions_for_role

_SESSION_HEADER = "X-SLM-User-Session"
_SESSION_COOKIE = "slm_session"

OWNER_PRINCIPAL = {
    "kind": "owner",
    "user_id": "owner",
    "username": "owner",
    "display_name": "Machine Owner",
}


def get_rbac_engine(app_state: Any) -> Any | None:
    return getattr(app_state, "rbac", None)


def _session_token(request: Request) -> str:
    tok = request.headers.get(_SESSION_HEADER, "")
    if tok:
        return tok
    try:
        return request.cookies.get(_SESSION_COOKIE, "") or ""
    except Exception:
        return ""


def resolve_principal(request: Request) -> dict:
    """Resolve the caller to a user (valid session) or the machine owner."""
    rbac = get_rbac_engine(request.app.state)
    token = _session_token(request)
    if rbac is not None and token:
        user = rbac.resolve_session(token)
        if user:
            return {"kind": "user", **user}
    return dict(OWNER_PRINCIPAL)


def _active_profile() -> str:
    from superlocalmemory.server.routes.helpers import get_active_profile

    return get_active_profile()


def require_permission(
    request: Request,
    permission: Permission,
    *,
    profile: str | None = None,
) -> dict:
    """Authorize ``permission`` on ``profile`` (default: active profile).

    Returns the principal on success. Raises 401 when a login is required but
    absent, or 403 when the user's role does not grant the permission.
    """
    rbac = get_rbac_engine(request.app.state)
    principal = resolve_principal(request)
    require_login = bool(rbac is not None and rbac.require_login())
    prof = profile or _active_profile()

    if principal["kind"] == "owner":
        # The machine operator is root — they always retain MANAGE (they have
        # shell access to the box regardless), so company mode can never lock
        # administration out of the dashboard. require_login only gates the
        # owner's DATA operations, forcing per-user login for read/write/etc.
        if require_login and permission != Permission.MANAGE:
            raise HTTPException(
                401,
                detail="Login required: this workspace enforces per-user access.",
            )
        return principal  # personal mode — operator is owner

    # Logged-in user: always enforced against their role.
    if rbac is not None and rbac.has_permission(principal["user_id"], prof, permission):
        return principal
    raise HTTPException(
        403,
        detail=(
            f"Your role does not allow '{permission.value}' on this workspace."
        ),
    )


def require_manage(request: Request, *, profile: str | None = None) -> dict:
    """Guard for user/role administration (MANAGE permission)."""
    return require_permission(request, Permission.MANAGE, profile=profile)


def principal_info(request: Request) -> dict:
    """Rich identity for /whoami: principal + role + effective permissions on
    the active profile. Never raises — used by the dashboard to render UI."""
    rbac = get_rbac_engine(request.app.state)
    principal = resolve_principal(request)
    prof = _active_profile()
    info = {
        "kind": principal["kind"],
        "user_id": principal["user_id"],
        "username": principal["username"],
        "display_name": principal.get("display_name", principal["username"]),
        "profile": prof,
        "rbac_active": bool(rbac is not None and rbac.user_count() > 0),
        "require_login": bool(rbac is not None and rbac.require_login()),
    }
    if principal["kind"] == "owner":
        # Owner has every permission (personal mode) unless login is required.
        info["role"] = "owner"
        info["permissions"] = [p.value for p in Permission]
        return info
    role = rbac.get_role(principal["user_id"], prof) if rbac is not None else None
    info["role"] = role.value if role else None
    info["permissions"] = (
        [p.value for p in permissions_for_role(role)] if role else []
    )
    return info
