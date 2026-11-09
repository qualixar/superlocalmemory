# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 — RBAC / teams (C3)

"""Dashboard-facing RBAC API: login/logout/whoami + user & role administration.

Mounted at /api/rbac/*. Every route first proves machine auth (the dashboard
holds the install token / the operator is on loopback) via
require_http_mutation_actor, then applies RBAC where relevant:

  * login/logout/whoami/status — machine auth only (identity bootstrap).
  * user + membership + policy admin — MANAGE permission (owner or an admin).

Creating the first user works out of the box: with zero users the caller is the
implicit owner, who has MANAGE.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from superlocalmemory.access.rbac import LastAdminError, RbacError
from superlocalmemory.access.rbac_admin import (
    AuditUnavailable,
    AuthorityError,
    audited,
    check_identity_authority,
    record,
)
from superlocalmemory.server.rbac_enforce import (
    get_rbac_engine,
    principal_info,
    require_global_manage,
    require_manage,
    resolve_principal,
)

logger = logging.getLogger("superlocalmemory.routes.rbac")
router = APIRouter(prefix="/api/rbac", tags=["rbac"])

_SESSION_COOKIE = "slm_session"

# Login throttle: lock an account after repeated failures within a window, to
# blunt password spraying / brute force (scrypt alone allows ~20 tries/s).
_LOGIN_MAX_FAILS = 5
_LOGIN_WINDOW_SEC = 300
# SEC-M-01: cap tracked usernames so a spray of unique usernames (one failed
# login each) cannot grow this dict without bound (memory-exhaustion DoS).
_LOGIN_MAX_TRACKED = 10_000
_login_fails: dict[str, list] = {}
_login_lock = threading.Lock()


def _login_blocked(username: str) -> bool:
    now = time.time()
    with _login_lock:
        fails = [t for t in _login_fails.get(username, []) if now - t < _LOGIN_WINDOW_SEC]
        # SEC-M-01: evict entries that decayed to empty instead of parking a
        # zero-length list forever (only successful logins used to pop them).
        if fails:
            _login_fails[username] = fails
        else:
            _login_fails.pop(username, None)
        # Overflow guard: drop the oldest-inserted key if we exceed the cap.
        if len(_login_fails) > _LOGIN_MAX_TRACKED:
            _login_fails.pop(next(iter(_login_fails)), None)
        return len(fails) >= _LOGIN_MAX_FAILS


def _login_note(username: str, ok: bool) -> None:
    with _login_lock:
        if ok:
            _login_fails.pop(username, None)
        else:
            _login_fails.setdefault(username, []).append(time.time())


def _is_browser(request: Request) -> bool:
    h = request.headers
    return bool(h.get("referer") or h.get("origin") or h.get("cookie")
                or h.get("sec-fetch-mode"))


def _cookie_secure(request: Request) -> bool:
    return (request.headers.get("x-forwarded-proto", "").lower() == "https"
            or os.environ.get("SLM_DASHBOARD_HTTPS") == "1")


# -- models --

class LoginRequest(BaseModel):
    username: str
    password: str


class CreateUserRequest(BaseModel):
    username: str
    password: str
    display_name: str = ""
    role: Optional[str] = None      # optional membership on the active profile
    profile_id: Optional[str] = None


class UpdateUserRequest(BaseModel):
    display_name: Optional[str] = None
    password: Optional[str] = None
    status: Optional[str] = None
    # Self-service password change proves the current password (OPS-27).
    current_password: Optional[str] = None


class MemberRequest(BaseModel):
    user_id: str
    role: str
    profile_id: Optional[str] = None


class RemoveMemberRequest(BaseModel):
    user_id: str
    profile_id: Optional[str] = None


class PolicyRequest(BaseModel):
    require_login: bool


# -- helpers --

def _machine_guard(request: Request) -> None:
    """Prove machine auth (operator / dashboard) before any RBAC route.

    In company mode (require_login on) a credential is mandatory: the loopback
    shortcut for credential-less callers is off.
    """
    from superlocalmemory.server.rbac_enforce import require_machine_credential
    from superlocalmemory.server.write_identity import require_http_mutation_actor

    rbac = get_rbac_engine(request.app.state)
    if rbac is not None and rbac.require_login():
        require_machine_credential(request)
        return
    broker = getattr(request.app.state, "mesh_broker", None)
    require_http_mutation_actor(
        request,
        getattr(request.app.state, "daemon_descriptor", None),
        actor_kind="rbac-route",
        mesh_secret=getattr(broker, "_shared_secret", None) if broker else None,
    )


def _engine(request: Request):
    rbac = get_rbac_engine(request.app.state)
    if rbac is None:
        raise HTTPException(503, detail="RBAC subsystem not initialized")
    return rbac


def _active_profile() -> str:
    from superlocalmemory.server.routes.helpers import get_active_profile

    return get_active_profile()


def _session_token(request: Request) -> str:
    return (request.headers.get("X-SLM-User-Session", "")
            or (request.cookies.get(_SESSION_COOKIE, "") or ""))


def _http_error(exc: Exception) -> HTTPException:
    """One mapping for refusals from account administration."""
    if isinstance(exc, AuditUnavailable):
        return HTTPException(503, detail=str(exc))
    if isinstance(exc, AuthorityError):
        return HTTPException(403, detail=str(exc))
    if isinstance(exc, LastAdminError):
        return HTTPException(409, detail=str(exc))
    # SEC-H-02 carve-out: RbacError messages are intentional domain validation
    # messages (e.g. "Username already exists"), safe for the admin console.
    return HTTPException(400, detail=str(exc))


def _account_id(rbac, username: str) -> str:
    """The account a sign-in named, for the audit trail (never the password)."""
    wanted = (username or "").strip()
    for user in rbac.list_users():
        if user["username"] == wanted:
            return str(user["user_id"])
    return "unknown-username"


# -- identity --

@router.post("/login")
async def login(req: LoginRequest, request: Request, response: Response):
    _machine_guard(request)
    uname = (req.username or "").strip()
    rbac = _engine(request)
    if _login_blocked(uname):
        record(rbac, "login", actor="anonymous", outcome="throttled",
               account=_account_id(rbac, uname))
        raise HTTPException(429, detail="Too many attempts. Try again later.")
    user = rbac.verify_credentials(req.username, req.password)
    if not user:
        _login_note(uname, ok=False)
        record(rbac, "login", actor="anonymous", outcome="failed",
               account=_account_id(rbac, uname))
        raise HTTPException(401, detail="Invalid username or password")
    _login_note(uname, ok=True)
    token = rbac.create_session(user["user_id"])
    record(rbac, "login", actor=f"user:{user['user_id']}", outcome="ok",
           session="created")
    # HttpOnly cookie so dashboard JS never has to hold the session token;
    # Secure when served over HTTPS (behind a TLS proxy).
    response.set_cookie(
        _SESSION_COOKIE, token, httponly=True, samesite="strict", path="/",
        secure=_cookie_secure(request),
    )
    # Only echo the raw token to non-browser (CLI) clients that cannot use the
    # cookie — browsers rely on the HttpOnly cookie and must not see it in the
    # response body (proxy logs / devtools capture).
    body = {"ok": True, "user": user}
    if not _is_browser(request):
        body["token"] = token
    return body


@router.post("/logout")
async def logout(request: Request, response: Response):
    _machine_guard(request)
    rbac = _engine(request)
    token = _session_token(request)
    if token:
        user = rbac.resolve_session(token)
        rbac.revoke_session(token)
        from superlocalmemory.server.caller_workspace import session_workspaces

        session_workspaces(request.app.state).forget(token)
        if user:
            record(rbac, "logout", actor=f"user:{user['user_id']}", outcome="ok",
                   session="revoked")
    response.delete_cookie(_SESSION_COOKIE, path="/")
    return {"ok": True}


@router.get("/whoami")
def whoami(request: Request):
    _machine_guard(request)
    return principal_info(request)


@router.get("/status")
def status(request: Request):
    _machine_guard(request)
    rbac = _engine(request)
    return {
        "rbac_active": rbac.user_count() > 0,
        "require_login": rbac.require_login(),
        "user_count": rbac.user_count(),
    }


# -- user administration (MANAGE) --

@router.get("/users")
def list_users(request: Request):
    _machine_guard(request)
    principal = require_manage(request)
    rbac = _engine(request)
    all_users = rbac.list_users()
    if principal.get("kind") == "owner":
        return {"users": all_users}
    # A workspace admin only sees users who share a workspace they MANAGE
    # (usernames/display names are PII — don't leak other tenants' rosters).
    from superlocalmemory.access.rbac import Permission
    mgr = {p["profile_id"] for p in rbac.list_user_profiles(principal["user_id"])
           if rbac.has_permission(principal["user_id"], p["profile_id"], Permission.MANAGE)}
    visible = [
        u for u in all_users
        if u["user_id"] == principal["user_id"]
        or any(m["profile_id"] in mgr for m in rbac.list_user_profiles(u["user_id"]))
    ]
    return {"users": visible}


@router.post("/users")
async def create_user(req: CreateUserRequest, request: Request):
    _machine_guard(request)
    principal = require_manage(request)
    rbac = _engine(request)
    prof = req.profile_id or _active_profile()
    # Granting a role on a workspace requires MANAGE on THAT workspace.
    if req.role:
        require_manage(request, profile=prof)

    def _create():
        user = rbac.create_user(
            req.username, req.password, display_name=req.display_name,
            created_by=principal["username"],
        )
        if req.role:
            rbac.set_membership(prof, user["user_id"], req.role,
                                added_by=principal["username"])
        return user

    try:
        user = audited(rbac, "user_create", principal, _create,
                       profile=prof if req.role else "", new_role=req.role or None)
    except (RbacError, AuditUnavailable) as e:
        logger.warning("rbac create_user rejected: %s", e)
        raise _http_error(e)
    return {"ok": True, "user": user}


def _changed_fields(req: UpdateUserRequest) -> list[str]:
    fields = []
    if req.status is not None:
        fields.append("status")
    if req.display_name is not None:
        fields.append("display_name")
    if req.password is not None:
        fields.append("password")
    return fields


def _apply_update(rbac, user_id: str, req: UpdateUserRequest) -> None:
    if req.status is not None:
        rbac.set_status(user_id, req.status)
    if req.display_name is not None:
        rbac.set_display_name(user_id, req.display_name)
    if req.password is not None:
        rbac.set_password(user_id, req.password)


@router.patch("/users/{user_id}")
async def update_user(user_id: str, req: UpdateUserRequest, request: Request):
    """Change an account's password, status or name (global to the account).

    The computer's owner, the person themselves (a password change proves the
    current one), or an administrator of every workspace the account belongs
    to. A scoped administrator changes roles instead (OPS-27).
    """
    _machine_guard(request)
    principal = resolve_principal(request)
    if principal.get("kind") != "user":
        principal = require_manage(request)
    rbac = _engine(request)
    fields = _changed_fields(req)
    try:
        for field in fields:
            check_identity_authority(rbac, principal, user_id, field=field,
                                     current_password=req.current_password)
        audited(rbac, "user_update", principal, lambda: _apply_update(rbac, user_id, req),
                target_user=user_id, fields=fields, new_status=req.status,
                sessions_revoked=bool(req.password is not None or req.status == "disabled"))
    except (RbacError, AuditUnavailable) as e:
        logger.warning("rbac update_user rejected: %s", e)
        raise _http_error(e)
    return {"ok": True, "user": rbac.get_user(user_id)}


@router.delete("/users/{user_id}")
async def delete_user(user_id: str, request: Request):
    _machine_guard(request)
    principal = require_manage(request)
    rbac = _engine(request)
    try:
        check_identity_authority(rbac, principal, user_id, field="delete")
        if rbac.get_user(user_id) is None:
            raise HTTPException(404, detail="User not found.")
        audited(rbac, "user_delete", principal, lambda: rbac.delete_user(user_id),
                target_user=user_id)
    except (RbacError, AuditUnavailable) as e:
        logger.warning("rbac delete_user rejected: %s", e)
        raise _http_error(e)
    return {"ok": True}


# -- membership administration (MANAGE) --

@router.get("/members")
def list_members(request: Request, profile_id: Optional[str] = None):
    _machine_guard(request)
    prof = profile_id or _active_profile()
    # MANAGE on the TARGET workspace, not merely the active one — otherwise an
    # admin of one workspace could enumerate another's members.
    require_manage(request, profile=prof)
    return {"profile_id": prof, "members": _engine(request).list_members(prof)}


@router.post("/members")
async def set_member(req: MemberRequest, request: Request):
    _machine_guard(request)
    prof = req.profile_id or _active_profile()
    principal = require_manage(request, profile=prof)
    rbac = _engine(request)
    old = rbac.get_role(req.user_id, prof)
    try:
        m = audited(rbac, "role_change", principal,
                    lambda: rbac.set_membership(prof, req.user_id, req.role,
                                                added_by=principal["username"]),
                    profile=prof, target_user=req.user_id,
                    old_role=old.value if old else None, new_role=req.role)
    except (RbacError, AuditUnavailable) as e:
        logger.warning("rbac set_membership rejected: %s", e)
        raise _http_error(e)
    return {"ok": True, "membership": m}


@router.delete("/members")
async def remove_member(req: RemoveMemberRequest, request: Request):
    _machine_guard(request)
    prof = req.profile_id or _active_profile()
    principal = require_manage(request, profile=prof)
    rbac = _engine(request)
    old = rbac.get_role(req.user_id, prof)
    try:
        audited(rbac, "membership_remove", principal,
                lambda: rbac.remove_membership(prof, req.user_id),
                profile=prof, target_user=req.user_id,
                old_role=old.value if old else None)
    except (RbacError, AuditUnavailable) as e:
        logger.warning("rbac remove_membership rejected: %s", e)
        raise _http_error(e)
    return {"ok": True}


# -- policy (MANAGE) --

@router.post("/policy")
async def set_policy(req: PolicyRequest, request: Request):
    _machine_guard(request)
    # One setting for every workspace: MANAGE on the active one is not enough.
    principal = require_global_manage(request)
    rbac = _engine(request)
    old = rbac.require_login()
    try:
        audited(rbac, "policy_change", principal,
                lambda: rbac.set_require_login(req.require_login),
                setting="require_login", old_value=old, new_value=req.require_login)
    except (RbacError, AuditUnavailable) as e:
        raise _http_error(e)
    return {"ok": True, "require_login": req.require_login}


__all__ = ["router"]
