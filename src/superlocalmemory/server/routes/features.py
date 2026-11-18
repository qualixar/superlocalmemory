# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""``/api/v3/features``: see what is on, turn images and documents on or off.

The install always runs here, in the daemon: the terminal and the dashboard
both call these routes. Reading creates nothing; changing needs the local
credential the dashboard already holds and a caller on this machine.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from superlocalmemory.runtimes import features as feat

logger = logging.getLogger("superlocalmemory.server.routes.features")

router = APIRouter(prefix="/api/v3/features", tags=["features"])


def _data_root() -> Path:
    from superlocalmemory.infra.data_root import canonical_data_root

    return canonical_data_root()


def _media_env():
    from superlocalmemory.runtimes.media_env import media_env

    return media_env(root=_data_root() / "runtimes" / "media")


def _ram_view() -> dict[str, Any]:
    from superlocalmemory.runtimes.media_ram import ram_view

    return ram_view()


def _ram_refusal(precheck: dict[str, Any]) -> str:
    """Why this computer cannot turn images and documents on ("" when it can); decided in one place."""
    from superlocalmemory.runtimes.media_env import media_ram_refusal

    return media_ram_refusal(int(precheck.get("ram_bytes") or 0))


def _media_view(status: dict[str, Any], root: Path, *, fresh: bool = True) -> dict[str, Any]:
    """``fresh=False`` (the read route) serves the size from memory; changes measure it now."""
    env = status.get("env") or {}
    view = {
        "enabled": bool(status.get("enabled")),
        "requested": feat.media_requested(root),
        "env_state": env.get("state", "not_installed"),
        "progress": env.get("progress", 0.0),
        "step": env.get("step", ""),
        "restart_required": feat.restart_required(status),
        "precheck": status.get("precheck", {}),
    }
    view["ram"] = _ram_view()
    from superlocalmemory.server import media_size

    size_of = media_size.measure_now if fresh else media_size.cached_bytes
    view["installed_bytes"] = size_of(root / "runtimes" / "media")
    view["install_estimate_bytes"] = int(2.8 * 1024 ** 3)
    refusal = _ram_refusal(status.get("precheck") or {})
    view["ram_ok"], view["ram_message"] = not refusal, refusal
    if env.get("error_kind"):
        view["error"] = env["error_kind"]
    if status.get("error"):
        view["error"] = status["error"]
    if status.get("refused"):
        view["refused"] = status["refused"]
    return view


def _apps_with_mesh(request: Request) -> int:
    """Live peers (local sessions and remote) the mesh broker knows for this profile."""
    try:
        from superlocalmemory.server.routes.helpers import get_active_profile
        from superlocalmemory.server.routes.mesh import _mesh_counts, _mesh_read_model

        broker = getattr(request.app.state, "mesh_broker", None)
        if broker is None:
            return 0
        remote, local = _mesh_read_model(broker.list_all_peers(get_active_profile()))
        return int(_mesh_counts(remote, local)["active_peer_count"])
    except Exception:  # noqa: BLE001 - a count that cannot be read is 0, never an error page
        return 0


def _overview(request: Request) -> dict[str, Any]:
    root = _data_root()
    status = feat.media_feature_status(root, env=_media_env())
    return {"media": _media_view(status, root, fresh=False), "mesh": {"apps_with_mesh": _apps_with_mesh(request)}}


def _gate(request: Request) -> None:
    """A caller on this machine that holds an issued credential."""
    from superlocalmemory.server import write_identity
    from superlocalmemory.server.loopback import is_loopback

    host = request.client.host if request.client else ""
    if not (is_loopback(host) or (host == "testclient" and write_identity._TEST_ISOLATION_ALLOWED)):
        raise HTTPException(403, detail="Changes to features are only accepted from this machine.")
    write_identity.require_write_actor(
        request, getattr(request.app.state, "daemon_descriptor", None), actor_kind="features")
    from superlocalmemory.server.rbac_enforce import require_global_manage

    require_global_manage(request)


async def _body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 - an empty or unreadable body is an empty request
        return {}
    return body if isinstance(body, dict) else {}


@router.get("")
def get_features(request: Request) -> dict[str, Any]:
    return _overview(request)


@router.post("/media/enable")
async def enable_media_route(request: Request):
    _gate(request)
    body = await _body(request)
    if body.get("yes") is not True:
        raise HTTPException(400, detail="Send {\"yes\": true} to turn images and documents on.")
    source = body.get("source", "dashboard")
    if source not in tuple(s for s in feat.SOURCES if s != "npm"):
        raise HTTPException(400, detail="source must be one of: cli, dashboard, api.")
    root = _data_root()
    status = feat.enable_media(source=source, env=_media_env(), data_root=root)
    view = _media_view(status, root)
    if status.get("refused"):  # this computer cannot run it: a state conflict, not a server fault
        return JSONResponse({"media": view, "detail": status.get("error", "")}, status_code=409)
    code = 500 if status.get("error") else 202
    return JSONResponse({"media": view}, status_code=code)


@router.post("/media/disable")
async def disable_media_route(request: Request):
    _gate(request)
    remove = (await _body(request)).get("remove_files") is True
    root = _data_root()
    status = feat.disable_media(remove_files=remove, env=_media_env(), data_root=root)
    return {"media": _media_view(status, root)}


# -- the optional picture and document models (WS-8) -------------------------------------------------

def _pack_name(body: dict[str, Any]) -> str:
    name = body.get("pack")
    if name not in ("captions", "documents", "regions"):
        raise HTTPException(400, detail="pack must be one of: captions, documents, regions.")
    return name


@router.get("/media/packs")
def get_media_packs() -> dict[str, Any]:
    """Where each optional part stands, with the download size to show before anything is downloaded."""
    from superlocalmemory.runtimes import media_packs

    return {"packs": media_packs.status_all(_data_root(), env=_media_env())}


@router.post("/media/packs/enable")
async def enable_media_pack_route(request: Request):
    """Turn one part on; its model downloads in the background. Needs ``{"pack": ..., "yes": true}``."""
    _gate(request)
    body = await _body(request)
    if body.get("yes") is not True:
        raise HTTPException(400, detail="Send {\"yes\": true} to download the model and turn this on.")
    source = body.get("source", "dashboard")
    if source not in tuple(s for s in feat.SOURCES if s != "npm"):
        raise HTTPException(400, detail="source must be one of: cli, dashboard, api.")
    from superlocalmemory.runtimes import media_packs

    result = media_packs.enable(_pack_name(body), source=source, data_root=_data_root(), env=_media_env())
    refused = result.get("refused")
    if refused:  # this computer or this moment cannot take it: a state conflict, not a server fault
        return JSONResponse({"pack": result, "detail": refused}, status_code=409)
    return JSONResponse({"pack": result}, status_code=202 if result["state"] == "downloading" else 200)


@router.post("/media/packs/disable")
async def disable_media_pack_route(request: Request):
    _gate(request)
    body = await _body(request)
    from superlocalmemory.runtimes import media_packs

    result = media_packs.disable(_pack_name(body), remove_files=body.get("remove_files") is True,
                                 data_root=_data_root(), env=_media_env())
    return {"pack": result}


def register(app: Any) -> None:
    app.include_router(router)
