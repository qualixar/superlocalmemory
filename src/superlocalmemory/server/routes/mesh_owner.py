# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Owner controls for bot messages: list messages and peers, mute, rename, retire a peer.

Local dashboard only: a loopback caller on a loopback host with the same
checks the connection routes use, the daemon capability on every change, and
the manage permission. Events log the peer id and counts, never message text.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict

from superlocalmemory.server.loopback import is_loopback
from superlocalmemory.server.rbac_enforce import require_manage
from superlocalmemory.server.routes.mesh import _active_profile
from superlocalmemory.server.write_identity import require_write_actor

router = APIRouter(prefix="/api/v3/mesh", tags=["mesh-owner"])


class MuteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    muted: bool


class RenameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    display_name: str


def _owner_context(request: Request, *, mutation: bool):
    """Authorize a dashboard call and return the mesh broker."""
    peer = request.client.host if request.client else ""
    if not is_loopback(peer) or not is_loopback(request.url.hostname or ""):
        raise HTTPException(403, "local_dashboard_required")
    origin = request.headers.get("Origin")
    if origin and origin != str(request.url).split(request.url.path)[0]:
        raise HTTPException(403, "same_origin_required")
    if request.headers.get("Sec-Fetch-Site") in {"cross-site", "same-site"}:
        raise HTTPException(403, "same_origin_required")
    if mutation:
        require_write_actor(
            request, getattr(request.app.state, "daemon_descriptor", None),
            actor_kind="mesh-owner",
        )
    profile = _active_profile()
    require_manage(request, profile=profile)
    broker = getattr(request.app.state, "mesh_broker", None)
    if broker is None:
        raise HTTPException(503, detail="Mesh broker not initialized")
    return broker, profile


def _checked(result: dict) -> dict:
    if result.get("ok"):
        return result
    error = result.get("error", "")
    raise HTTPException(404 if "not found" in error else 422, detail=error)


@router.get("/messages")
def list_messages(request: Request, limit: int = Query(50, ge=1, le=500), peer: str = ""):
    broker, profile = _owner_context(request, mutation=False)
    return {"messages": broker.list_messages(limit, peer, profile_id=profile)}


@router.get("/peers")
def list_peers(request: Request):
    broker, profile = _owner_context(request, mutation=False)
    return {"peers": broker.list_peer_overview(profile_id=profile)}


@router.post("/peers/{peer_id}/mute")
def mute_peer(peer_id: str, body: MuteRequest, request: Request):
    broker, profile = _owner_context(request, mutation=True)
    return _checked(broker.set_muted(peer_id, body.muted, profile_id=profile))


@router.patch("/peers/{peer_id}")
def rename_peer(peer_id: str, body: RenameRequest, request: Request):
    broker, profile = _owner_context(request, mutation=True)
    return _checked(broker.rename_peer(peer_id, body.display_name, profile_id=profile))


@router.delete("/peers/{peer_id}")
def retire_peer(peer_id: str, request: Request):
    broker, profile = _owner_context(request, mutation=True)
    return _checked(broker.retire_peer(peer_id, profile_id=profile))
