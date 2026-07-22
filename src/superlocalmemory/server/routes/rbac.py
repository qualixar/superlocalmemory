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

from superlocalmemory.access.rbac import RbacError
from superlocalmemory.server.rbac_enforce import (
    get_rbac_engine, principal_info, require_manage, resolve_principal,
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
    """Prove machine auth (operator / dashboard) before any RBAC route."""
    from superlocalmemory.server.write_identity import require_http_mutation_actor

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


def _require_authority_over_user(request: Request, target_user_id: str) -> None:
    """Reject modifying a user the caller has no authority over (IDOR guard).

    The machine owner is root. A logged-in admin may only act on a user who
    shares at least one workspace on which the admin holds MANAGE.
    """
    from superlocalmemory.access.rbac import Permission
    from superlocalmemory.server.rbac_enforce import resolve_principal

    principal = resolve_principal(request)
    if principal["kind"] == "owner":
        return
    rbac = _engine(request)
    target_profiles = rbac.list_user_profiles(target_user_id)
    if any(rbac.has_permission(principal["user_id"], p["profile_id"], Permission.MANAGE)
           for p in target_profiles):
        return
    raise HTTPException(403, detail="You have no authority over this user.")


# -- identity --

@router.post("/login")
async def login(req: LoginRequest, request: Request, response: Response):
    _machine_guard(request)
    uname = (req.username or "").strip()
    if _login_blocked(uname):
        raise HTTPException(429, detail="Too many attempts. Try again later.")
    rbac = _engine(request)
    user = rbac.verify_credentials(req.username, req.password)
    if not user:
        _login_note(uname, ok=False)
        raise HTTPException(401, detail="Invalid username or password")
    _login_note(uname, ok=True)
    token = rbac.create_session(user["user_id"])
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
        rbac.revoke_session(token)
    response.delete_cookie(_SESSION_COOKIE, path="/")
    return {"ok": True}


@router.get("/whoami")
async def whoami(request: Request):
    _machine_guard(request)
    return principal_info(request)


@router.get("/status")
async def status(request: Request):
    _machine_guard(request)
    rbac = _engine(request)
    return {
        "rbac_active": rbac.user_count() > 0,
        "require_login": rbac.require_login(),
        "user_count": rbac.user_count(),
    }


# -- user administration (MANAGE) --

@router.get("/users")
async def list_users(request: Request):
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
    # Granting a role on a workspace requires MANAGE on THAT workspace.
    if req.role:
        require_manage(request, profile=req.profile_id or _active_profile())
    try:
        user = rbac.create_user(
            req.username, req.password, display_name=req.display_name,
            created_by=principal["username"],
        )
        if req.role:
            rbac.set_membership(
                req.profile_id or _active_profile(), user["user_id"],
                req.role, added_by=principal["username"],
            )
    except RbacError as e:
        # SEC-H-02 carve-out: RbacError messages are intentional domain
        # validation messages (e.g. "Username already exists") and are safe
        # to surface to the admin console. Log for audit trail.
        logger.warning("rbac create_user rejected: %s", e)
        raise HTTPException(400, detail=str(e))
    return {"ok": True, "user": user}


@router.patch("/users/{user_id}")
async def update_user(user_id: str, req: UpdateUserRequest, request: Request):
    _machine_guard(request)
    require_manage(request)
    _require_authority_over_user(request, user_id)
    rbac = _engine(request)
    try:
        if req.display_name is not None or req.status is not None:
            if req.status is not None:
                rbac.set_status(user_id, req.status)
            if req.display_name is not None:
                # display_name update via direct engine call
                conn = rbac._conn()
                try:
                    conn.execute(
                        "UPDATE rbac_users SET display_name=? WHERE user_id=?",
                        (req.display_name, user_id),
                    )
                    conn.commit()
                finally:
                    conn.close()
        if req.password is not None:
            rbac.set_password(user_id, req.password)
    except RbacError as e:
        logger.warning("rbac update_user rejected: %s", e)
        raise HTTPException(400, detail=str(e))
    return {"ok": True, "user": rbac.get_user(user_id)}


@router.delete("/users/{user_id}")
async def delete_user(user_id: str, request: Request):
    _machine_guard(request)
    require_manage(request)
    _require_authority_over_user(request, user_id)
    try:
        _engine(request).delete_user(user_id)
    except RbacError as e:
        logger.warning("rbac delete_user not found: %s", e)
        raise HTTPException(404, detail=str(e))
    return {"ok": True}


# -- membership administration (MANAGE) --

@router.get("/members")
async def list_members(request: Request, profile_id: Optional[str] = None):
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
    try:
        m = rbac.set_membership(prof, req.user_id, req.role,
                                added_by=principal["username"])
    except RbacError as e:
        logger.warning("rbac set_membership rejected: %s", e)
        raise HTTPException(400, detail=str(e))
    return {"ok": True, "membership": m}


@router.delete("/members")
async def remove_member(req: RemoveMemberRequest, request: Request):
    _machine_guard(request)
    prof = req.profile_id or _active_profile()
    require_manage(request, profile=prof)
    _engine(request).remove_membership(prof, req.user_id)
    return {"ok": True}


# -- policy (MANAGE) --

@router.post("/policy")
async def set_policy(req: PolicyRequest, request: Request):
    _machine_guard(request)
    require_manage(request)
    _engine(request).set_require_login(req.require_login)
    return {"ok": True, "require_login": req.require_login}
