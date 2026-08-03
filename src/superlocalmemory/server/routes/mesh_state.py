# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""SLM Mesh — state-sync delta endpoint (3c-1).

Exposes a single GET route that returns the local ``mesh_state`` rows whose
revision exceeds ``since``, suitable for pull-based LWW convergence.

Mounted by the delivery lead (unified_daemon.py) via ``include_router``.
Auth, broker access, and profile resolution all mirror ``routes/mesh.py``
exactly — no new patterns are introduced.

Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from superlocalmemory.mesh.state_sync import StateSyncer

# Re-use auth + broker helpers verbatim from the main mesh module so that
# security properties (shared-secret, loopback trust, write-identity) apply
# identically to this new endpoint.
from superlocalmemory.server.routes.mesh import _active_profile, _get_broker

router = APIRouter(prefix="/mesh", tags=["mesh-state-sync"])


@router.get("/state/delta")
def state_delta(
    request: Request,
    profile: str = "",
    since: int = 0,
) -> dict:
    """Return local mesh_state rows with revision > ``since`` for LWW sync.

    Query parameters
    ----------------
    profile : str, optional
        Tenant profile id.  Defaults to the currently active profile when
        omitted or empty.
    since : int, optional
        Only rows whose ``revision`` strictly exceeds this value are returned.
        Pass ``0`` (default) to retrieve all rows.

    Response
    --------
    ``{"entries": [...], "node_id": "<local_node_id>"}``

    Each entry carries ``{key, value, set_by, updated_at, revision, node_id}``
    where ``node_id`` is the effective origin node (resolved from
    ``origin_node`` column; BC rows with ``origin_node=''`` resolve to the
    local node's id).
    """
    broker = _get_broker(request)
    resolved_profile = profile if profile else _active_profile()
    syncer = StateSyncer(broker)
    return {
        "entries": syncer.local_delta(resolved_profile, since),
        "node_id": syncer._node_id,
    }
