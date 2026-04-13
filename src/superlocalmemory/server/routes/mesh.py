# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the Elastic License 2.0 - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SLM Mesh — FastAPI routes for P2P agent communication.

Mounted at /mesh/* in the unified daemon. Uses MeshBroker for all operations.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

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
    return broker


# -- Routes --

@router.post("/register")
async def register(req: RegisterRequest, request: Request):
    broker = _get_broker(request)
    if not req.session_id:
        raise HTTPException(400, detail="session_id required")
    return broker.register_peer(
        req.session_id, req.summary, req.host, req.port,
        req.project_path, req.agent_type,
    )


@router.post("/deregister")
async def deregister(req: DeregisterRequest, request: Request):
    broker = _get_broker(request)
    result = broker.deregister_peer(req.peer_id)
    if not result.get("ok"):
        raise HTTPException(404, detail=result.get("error", "peer not found"))
    return result


@router.get("/peers")
async def peers(request: Request):
    broker = _get_broker(request)
    return {"peers": broker.list_peers()}


@router.post("/heartbeat")
async def heartbeat(req: HeartbeatRequest, request: Request):
    broker = _get_broker(request)
    result = broker.heartbeat(req.peer_id)
    if not result.get("ok"):
        raise HTTPException(404, detail=result.get("error", "peer not found"))
    return result


@router.post("/summary")
async def summary(req: SummaryRequest, request: Request):
    broker = _get_broker(request)
    result = broker.update_summary(req.peer_id, req.summary)
    if not result.get("ok"):
        raise HTTPException(404, detail=result.get("error", "peer not found"))
    return result


@router.post("/send")
async def send(req: SendRequest, request: Request):
    broker = _get_broker(request)
    to_target = req.to_peer or req.to  # v3.4.6: accept both field names
    if not to_target:
        raise HTTPException(400, detail="'to' or 'to_peer' required")
    result = broker.send_message(req.from_peer, to_target, req.content, req.type)
    if not result.get("ok"):
        status = 413 if "too large" in result.get("error", "") else 404
        raise HTTPException(status, detail=result.get("error", ""))
    return result


@router.get("/inbox/{peer_id}")
async def inbox(peer_id: str, request: Request, project_path: str = ""):
    broker = _get_broker(request)
    return {"messages": broker.get_inbox(peer_id, project_path)}


@router.post("/inbox/{peer_id}/read")
async def mark_read(peer_id: str, req: ReadRequest, request: Request):
    broker = _get_broker(request)
    return broker.mark_read(peer_id, req.message_ids)


@router.get("/pending/{peer_id}")
async def pending(peer_id: str, request: Request, project_path: str = ""):
    """Get pending broadcast/project messages for this peer."""
    broker = _get_broker(request)
    messages = broker.get_pending(peer_id, project_path)
    return {"messages": messages, "count": len(messages)}


@router.get("/state")
async def state_all(request: Request):
    broker = _get_broker(request)
    return {"state": broker.get_state()}


@router.post("/state")
async def state_set(req: StateSetRequest, request: Request):
    broker = _get_broker(request)
    if not req.key:
        raise HTTPException(400, detail="key required")
    return broker.set_state(req.key, req.value, req.set_by)


@router.get("/state/{key}")
async def state_get(key: str, request: Request):
    broker = _get_broker(request)
    result = broker.get_state_key(key)
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
    return broker.lock_action(req.file_path, req.locked_by, req.action)


@router.get("/events")
async def events(request: Request):
    broker = _get_broker(request)
    return {"events": broker.get_events()}


@router.get("/status")
async def status(request: Request):
    broker = _get_broker(request)
    return broker.get_status()
