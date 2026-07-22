# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SLM Mesh — FastAPI routes for P2P agent communication.

Mounted at /mesh/* in the unified daemon. Uses MeshBroker for all operations.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import asyncio
import re
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/mesh", tags=["mesh"])


# -- Request models --

class RegisterRequest(BaseModel):
    session_id: str
    summary: str = ""
    host: str = "127.0.0.1"
    port: int = 0
    project_path: str = ""
    agent_type: str = "unknown"


class DeregisterRequest(BaseModel):
    peer_id: str


class HeartbeatRequest(BaseModel):
    peer_id: str


class SummaryRequest(BaseModel):
    peer_id: str
    summary: str


class SendRequest(BaseModel):
    from_peer: str = ""
    to: str = ""
    to_peer: str = ""  # v3.4.6: accept both 'to' and 'to_peer' for compatibility
    content: str
    type: str = "text"


class ReadRequest(BaseModel):
    message_ids: list[int]


class StateSetRequest(BaseModel):
    key: str
    value: str
    set_by: str


class LockRequest(BaseModel):
    file_path: str
    locked_by: str
    action: str  # acquire, release, query


# -- Helpers --

def _get_broker(request: Request):
    broker = getattr(request.app.state, 'mesh_broker', None)
    if broker is None:
        raise HTTPException(503, detail="Mesh broker not initialized")
    # Check if mesh is enabled
    config = getattr(request.app.state, 'config', None)
    if config and not getattr(config, 'mesh_enabled', True):
        raise HTTPException(503, detail="Mesh disabled in config")
    # v3.6.12 (mesh-1 security): SLM_MESH_SHARED_SECRET was read by the broker but
    # never verified on inbound mesh HTTP calls. When a secret is configured,
    # require it (constant-time) from NON-loopback callers via X-Mesh-Secret.
    # The local MCP client always calls over loopback and is exempt, so this is
    # zero-change for single-machine use and closes the LAN mesh auth bypass.
    secret = getattr(broker, "_shared_secret", None)
    if secret:
        client_host = request.client.host if request.client else ""
        if client_host not in ("127.0.0.1", "::1", "localhost"):
            import hmac
            from superlocalmemory.core.security_primitives import verify_install_token

            # Path 1: install token — dashboard/browser callers hold this and
            # should not need the mesh secret exposed in JS.
            install_token = request.headers.get("x-install-token", "")
            if install_token and verify_install_token(install_token):
                return broker

            # Path 2: mesh secret — remote agents / remote_sync.py / LAN peers.
            # Accept X-Mesh-Secret (legacy v3.6.12 header) OR
            # Authorization: Bearer <secret> (RFC 7617 canonical form).
            presented = (
                request.headers.get("x-mesh-secret")
                or request.headers.get("authorization", "").removeprefix("Bearer ").strip()
            )
            if not presented or not hmac.compare_digest(presented, secret):
                raise HTTPException(401, detail="invalid or missing credential")
            return broker

    # Loopback is a transport property, not an identity.  Every mesh read and
    # write must prove the install/API/process capability because peer inboxes,
    # coordination state, and project paths can be sensitive.
    from superlocalmemory.server.write_identity import require_write_actor

    require_write_actor(
        request,
        getattr(request.app.state, "daemon_descriptor", None),
        actor_kind="mesh-route",
    )
    return broker


def _active_profile() -> str:
    """Resolve the tenant (profile) for this mesh request.

    The mesh is a per-tenant coordination bus: peers, messages, shared state,
    and locks are all scoped to the active profile so one tenant never sees
    another's coordination traffic. Resolved from the request ContextVar set by
    ProfileRuntimeMiddleware (falls back to the configured active profile).
    """
    from superlocalmemory.server.routes.helpers import get_active_profile

    return get_active_profile()


_SECRET_KEY = re.compile(
    r"(?:^|[_\-.])(api[_\-.]?key|secret|token|password|credential)(?:$|[_\-.])",
    re.IGNORECASE,
)


def _reject_secret_state(key: str, value: str) -> None:
    """Refuse secret material in the plaintext coordination store."""
    from superlocalmemory.core.security_primitives import redact_secrets

    if _SECRET_KEY.search(key) or redact_secrets(value) != value:
        raise HTTPException(
            422,
            detail="Mesh state is coordination metadata; secret values are prohibited",
        )


# -- Routes --

@router.post("/register")
async def register(req: RegisterRequest, request: Request):
    broker = _get_broker(request)
    if not req.session_id:
        raise HTTPException(400, detail="session_id required")
    return broker.register_peer(
        req.session_id, req.summary, req.host, req.port,
        req.project_path, req.agent_type, profile_id=_active_profile(),
    )


@router.post("/deregister")
async def deregister(req: DeregisterRequest, request: Request):
    broker = _get_broker(request)
    result = broker.deregister_peer(req.peer_id, profile_id=_active_profile())
    if not result.get("ok"):
        raise HTTPException(404, detail=result.get("error", "peer not found"))
    return result


def _peer_activity_counts(request: Request, session_ids: list[str]) -> dict:
    """Per-session {tool_count, memory_count} for the active profile.

    Each mesh peer IS an agent session, so its activity is the tool events it
    logged and the memories it contributed under the current profile. Batched
    to a single grouped query per table. Fail-soft: returns {} on any error so
    the peer list still renders.
    """
    if not session_ids:
        return {}
    try:
        from superlocalmemory.server.routes.helpers import (
            get_engine_lazy, get_active_profile,
        )
        engine = get_engine_lazy(request.app.state)
        if engine is None:
            return {}
        profile = get_active_profile()
        placeholders = ",".join("?" * len(session_ids))
        params = [profile, *session_ids]
        counts: dict[str, dict] = {sid: {"tool_count": 0, "memory_count": 0}
                                   for sid in session_ids}
        for table, key in (("tool_events", "tool_count"),
                            ("memories", "memory_count")):
            try:
                rows = engine._db.execute(
                    f"SELECT session_id, COUNT(*) AS n FROM {table} "
                    f"WHERE profile_id = ? AND session_id IN ({placeholders}) "
                    "GROUP BY session_id",
                    tuple(params),
                )
                for r in rows:
                    d = dict(r)
                    sid = d.get("session_id")
                    if sid in counts:
                        counts[sid][key] = d.get("n", 0)
            except Exception:
                continue
        return counts
    except Exception:
        return {}


@router.get("/peers")
async def peers(request: Request):
    broker = _get_broker(request)
    peer_list = broker.list_all_peers(_active_profile())
    session_ids = [p.get("session_id") for p in peer_list if p.get("session_id")]
    counts = _peer_activity_counts(request, session_ids)
    for p in peer_list:
        c = counts.get(p.get("session_id"), {})
        p["tool_count"] = c.get("tool_count", 0)
        p["memory_count"] = c.get("memory_count", 0)
    return {"peers": peer_list}


@router.post("/heartbeat")
async def heartbeat(req: HeartbeatRequest, request: Request):
    broker = _get_broker(request)
    result = broker.heartbeat(req.peer_id, profile_id=_active_profile())
    if not result.get("ok"):
        raise HTTPException(404, detail=result.get("error", "peer not found"))
    return result


@router.post("/summary")
async def summary(req: SummaryRequest, request: Request):
    broker = _get_broker(request)
    result = broker.update_summary(req.peer_id, req.summary,
                                   profile_id=_active_profile())
    if not result.get("ok"):
        raise HTTPException(404, detail=result.get("error", "peer not found"))
    return result


@router.post("/send")
async def send(req: SendRequest, request: Request):
    broker = _get_broker(request)
    to_target = req.to_peer or req.to  # v3.4.6: accept both field names
    if not to_target:
        raise HTTPException(400, detail="'to' or 'to_peer' required")
    # Resolve the tenant on the event loop (the request ContextVar is set here);
    # a worker thread would not inherit it.
    profile = _active_profile()
    # send_message may make a blocking httpx call (up to 10s) when delivering
    # to a remote peer. Offload to a worker thread so a slow/dead peer network
    # never stalls the daemon event loop for all other users. The broker opens
    # a fresh SQLite connection per call, so this is thread-safe.
    result = await asyncio.to_thread(
        broker.send_message, req.from_peer, to_target, req.content,
        req.type, "", profile,
    )
    if not result.get("ok"):
        status = 413 if "too large" in result.get("error", "") else 404
        raise HTTPException(status, detail=result.get("error", ""))
    return result


@router.get("/inbox/{peer_id}")
async def inbox(peer_id: str, request: Request, project_path: str = ""):
    broker = _get_broker(request)
    return {"messages": broker.get_inbox(peer_id, project_path,
                                         profile_id=_active_profile())}


@router.post("/inbox/{peer_id}/read")
async def mark_read(peer_id: str, req: ReadRequest, request: Request):
    broker = _get_broker(request)
    return broker.mark_read(peer_id, req.message_ids,
                            profile_id=_active_profile())


@router.get("/pending/{peer_id}")
async def pending(peer_id: str, request: Request, project_path: str = ""):
    """Get pending broadcast/project messages for this peer."""
    broker = _get_broker(request)
    messages = broker.get_pending(peer_id, project_path,
                                  profile_id=_active_profile())
    return {"messages": messages, "count": len(messages)}


@router.get("/state")
async def state_all(request: Request):
    broker = _get_broker(request)
    return {"state": broker.get_state(profile_id=_active_profile())}


@router.post("/state")
async def state_set(req: StateSetRequest, request: Request):
    broker = _get_broker(request)
    if not req.key:
        raise HTTPException(400, detail="key required")
    _reject_secret_state(req.key, req.value)
    return broker.set_state(req.key, req.value, req.set_by,
                            profile_id=_active_profile())


@router.get("/state/{key}")
async def state_get(key: str, request: Request):
    broker = _get_broker(request)
    result = broker.get_state_key(key, profile_id=_active_profile())
    if result is None:
        raise HTTPException(404, detail="key not found")
    return result


@router.post("/lock")
async def lock(req: LockRequest, request: Request):
    broker = _get_broker(request)
    if not req.file_path or not req.locked_by:
        raise HTTPException(400, detail="file_path and locked_by required")
    if req.action not in ("acquire", "release", "query"):
        raise HTTPException(400, detail="action must be acquire, release, or query")
    return broker.lock_action(req.file_path, req.locked_by, req.action,
                              profile_id=_active_profile())


@router.get("/events")
async def events(request: Request):
    broker = _get_broker(request)
    return {"events": broker.get_events(profile_id=_active_profile())}


@router.get("/status")
async def status(request: Request):
    broker = _get_broker(request)
    return broker.get_status(profile_id=_active_profile())
