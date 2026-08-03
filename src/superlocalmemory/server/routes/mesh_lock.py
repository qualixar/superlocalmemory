# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""SLM Mesh — lock-delta HTTP route (3c-2).

Exposes ``GET /mesh/lock/delta`` so remote peers can fetch this node's
live advisory locks for convergence via ``LockCoordinator.resolve()``.

Auth and broker resolution mirror ``routes/mesh.py``.
Do NOT mount this router directly — the delivery lead wires it via
``app.include_router(mesh_lock_routes.router)``.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from superlocalmemory.mesh.lock_protocol import LockCoordinator
from superlocalmemory.mesh.node_identity import get_node_id
from superlocalmemory.server.routes.mesh import _get_broker
from superlocalmemory.server.routes.helpers import get_active_profile

router = APIRouter(prefix="/mesh/lock", tags=["mesh-lock"])


@router.get("/delta")
def lock_delta(
    profile: str = "",
    request: Request = None,  # type: ignore[assignment]
) -> dict:
    """Return this node's live advisory locks for a given tenant profile.

    The response is designed for consumption by a remote peer's
    ``LockCoordinator.resolve()`` call: it includes the raw lock records
    plus the local ``node_id`` so the remote can apply the total-order
    ``(fencing_token, node_id)`` comparison without a separate handshake.

    Query params:
        profile: Tenant profile id.  Falls back to the active profile
                 from the request context var when omitted or blank.

    Returns:
        ``{"locks": [...], "node_id": "<hex>"}``
        ``locks`` is the output of ``LockCoordinator.local_lock_delta()``.
    """
    broker = _get_broker(request)
    profile_id: str = profile.strip() or get_active_profile()
    coordinator = LockCoordinator(broker)
    locks = coordinator.local_lock_delta(profile_id)
    node_id = get_node_id(broker._db_path)
    return {"locks": locks, "node_id": node_id}
