# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SLM Mesh — FastAPI routes for P2P agent communication.

Mounted at /mesh/* in the unified daemon. Uses MeshBroker for all operations.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/mesh", tags=["mesh"])

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
_STALE_AFTER = timedelta(minutes=5)
_EXPIRE_AFTER = timedelta(minutes=30)


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
def register(req: RegisterRequest, request: Request):
    broker = _get_broker(request)
    if not req.session_id:
        raise HTTPException(400, detail="session_id required")
    return broker.register_peer(
        req.session_id, req.summary, req.host, req.port,
        req.project_path, req.agent_type, profile_id=_active_profile(),
    )


@router.post("/deregister")
def deregister(req: DeregisterRequest, request: Request):
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
            get_active_profile,
            get_engine_lazy,
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


def _parse_heartbeat(value: object) -> datetime | None:
    """Parse a stored heartbeat as UTC without trusting malformed values."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _mesh_read_model(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return bounded remote peers and local sessions with read-time liveness.

    The broker persists local agent sessions in ``mesh_peers`` for messaging.
    That storage detail must not make them look like remote mesh neighbours in
    the dashboard. Classification is read-only so an ordinary dashboard GET
    neither mutates a live database nor waits for the five-minute cleanup loop.
    """
    now = datetime.now(timezone.utc)
    remote: list[dict] = []
    local: list[dict] = []
    for record in records:
        heartbeat = _parse_heartbeat(record.get("last_heartbeat"))
        if heartbeat is None:
            continue
        age = now - heartbeat
        if age >= _EXPIRE_AFTER:
            continue
        stale_at = heartbeat + _STALE_AFTER
        expires_at = heartbeat + _EXPIRE_AFTER
        status = "active" if age < _STALE_AFTER else "stale"
        normalized = {
            **record,
            "status": status,
            "stale_at": stale_at.isoformat(),
            "expires_at": expires_at.isoformat(),
        }
        if str(record.get("host") or "").lower() in _LOOPBACK_HOSTS:
            local.append(normalized)
        else:
            remote.append(normalized)
    return remote, local


def _mesh_counts(remote: list[dict], local: list[dict]) -> dict:
    """Expose active counts separately while retaining the legacy total key."""
    active_remote = sum(peer["status"] == "active" for peer in remote)
    active_local = sum(session["status"] == "active" for session in local)
    return {
        "peer_count": active_remote + active_local,
        "active_peer_count": active_remote + active_local,
        "remote_peer_count": active_remote,
        "local_session_count": active_local,
        "stale_peer_count": sum(peer["status"] == "stale" for peer in remote),
        "stale_local_session_count": sum(session["status"] == "stale" for session in local),
    }


@router.get("/peers")
def peers(request: Request, view: str = "all"):
    broker = _get_broker(request)
    if view not in {"all", "remote", "local"}:
        raise HTTPException(422, detail="view must be all, remote, or local")
    remote_peers, local_sessions = _mesh_read_model(
        broker.list_all_peers(_active_profile()),
    )
    peer_list = (
        remote_peers if view == "remote" else
        local_sessions if view == "local" else
        [*remote_peers, *local_sessions]
    )
    session_ids = [p.get("session_id") for p in peer_list if p.get("session_id")]
    counts = _peer_activity_counts(request, session_ids)
    enriched = []
    for p in peer_list:
        c = counts.get(p.get("session_id"), {})
        enriched.append({
            **p,
            "tool_count": c.get("tool_count", 0),
            "memory_count": c.get("memory_count", 0),
        })
    return {
        "peers": enriched,
        "remote_peers": remote_peers,
        "local_sessions": local_sessions,
        "view": view,
        **_mesh_counts(remote_peers, local_sessions),
    }


@router.post("/heartbeat")
def heartbeat(req: HeartbeatRequest, request: Request):
    """Update peer liveness without blocking the daemon's async event loop."""
    broker = _get_broker(request)
    result = broker.heartbeat(req.peer_id, profile_id=_active_profile())
    if not result.get("ok"):
        raise HTTPException(404, detail=result.get("error", "peer not found"))
    return result


@router.post("/summary")
def summary(req: SummaryRequest, request: Request):
    broker = _get_broker(request)
    result = broker.update_summary(req.peer_id, req.summary,
                                   profile_id=_active_profile())
    if not result.get("ok"):
        raise HTTPException(404, detail=result.get("error", "peer not found"))
    return result


@router.post("/send")
def send(req: SendRequest, request: Request):
    broker = _get_broker(request)
    to_target = req.to_peer or req.to  # v3.4.6: accept both field names
    if not to_target:
        raise HTTPException(400, detail="'to' or 'to_peer' required")
    profile = _active_profile()
    # This sync FastAPI route already runs in the worker thread pool, so the
    # broker's SQLite retries and optional remote HTTP delivery cannot block
    # the daemon event loop.
    result = broker.send_message(
        req.from_peer, to_target, req.content, req.type, "", profile,
    )
    if not result.get("ok"):
        status = 413 if "too large" in result.get("error", "") else 404
        raise HTTPException(status, detail=result.get("error", ""))
    return result


@router.get("/inbox/{peer_id}")
def inbox(peer_id: str, request: Request, project_path: str = ""):
    broker = _get_broker(request)
    return {"messages": broker.get_inbox(peer_id, project_path,
                                         profile_id=_active_profile())}


@router.post("/inbox/{peer_id}/read")
def mark_read(peer_id: str, req: ReadRequest, request: Request):
    broker = _get_broker(request)
    return broker.mark_read(peer_id, req.message_ids,
                            profile_id=_active_profile())


@router.get("/pending/{peer_id}")
def pending(peer_id: str, request: Request, project_path: str = ""):
    """Get pending broadcast/project messages for this peer."""
    broker = _get_broker(request)
    messages = broker.get_pending(peer_id, project_path,
                                  profile_id=_active_profile())
    return {"messages": messages, "count": len(messages)}


@router.get("/state")
def state_all(request: Request):
    broker = _get_broker(request)
    return {"state": broker.get_state(profile_id=_active_profile())}


@router.post("/state")
def state_set(req: StateSetRequest, request: Request):
    broker = _get_broker(request)
    if not req.key:
        raise HTTPException(400, detail="key required")
    _reject_secret_state(req.key, req.value)
    return broker.set_state(req.key, req.value, req.set_by,
                            profile_id=_active_profile())


@router.get("/state/{key}")
def state_get(key: str, request: Request):
    broker = _get_broker(request)
    result = broker.get_state_key(key, profile_id=_active_profile())
    if result is None:
        raise HTTPException(404, detail="key not found")
    return result


@router.post("/lock")
def lock(req: LockRequest, request: Request):
    broker = _get_broker(request)
    if not req.file_path or not req.locked_by:
        raise HTTPException(400, detail="file_path and locked_by required")
    if req.action not in ("acquire", "release", "query"):
        raise HTTPException(400, detail="action must be acquire, release, or query")
    return broker.lock_action(req.file_path, req.locked_by, req.action,
                              profile_id=_active_profile())


@router.get("/events")
def events(request: Request):
    broker = _get_broker(request)
    return {"events": broker.get_events(profile_id=_active_profile())}


@router.get("/status")
def status(request: Request):
    broker = _get_broker(request)
    broker_status = broker.get_status(profile_id=_active_profile())
    remote_peers, local_sessions = _mesh_read_model(
        broker.list_all_peers(_active_profile()),
    )
    return {
        **broker_status,
        **_mesh_counts(remote_peers, local_sessions),
    }
