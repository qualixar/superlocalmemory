# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Folder source routes (local only).

Connecting a folder is two steps: ``POST /sources`` checks it and returns a preview that saves
nothing, ``POST /sources/{id}/confirm`` connects it. Every route answers only for the active
profile's sources. Nothing here reads or writes a file in the folder.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from superlocalmemory import sources
from superlocalmemory.sources import api as sources_api
from superlocalmemory.sources import picker
from superlocalmemory.server.routes.media import _profile, _require_local
from superlocalmemory.sources.roots import RootRefused

router = APIRouter(prefix="/api/v3/sources", tags=["sources"])
_NO_STORE = {"Cache-Control": "no-store"}
_STATUS = {"unknown_source": 404, "remote_access_on": 409, "writer_not_ready": 503,
           "erasure_incomplete": 503, "cannot_save": 500}


class AddRequest(BaseModel):
    path: str = Field(min_length=1, max_length=4096)
    profile_id: str = ""
    kind: str | None = Field(default=None, pattern="^(folder|obsidian)$")


class ReleaseRequest(BaseModel):
    relpath: str = Field(min_length=1, max_length=4096)


async def _call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return await asyncio.to_thread(fn, *args, **kwargs)
    except sources.SourceRefused as exc:
        raise HTTPException(_STATUS.get(exc.code, 409), detail={"code": exc.code, "message": str(exc)}) from None
    except RootRefused as exc:
        raise HTTPException(422, detail={"code": exc.code, "message": str(exc)}) from None
    except sources.HintsNotAvailable:
        raise HTTPException(404, detail="Not found.") from None


async def _context(request: Request, *, write: bool = False, delete: bool = False,
                   manage: bool = False, profile_id: str = "") -> tuple[str, str]:
    """The profile and actor of a call. ``manage`` is for connecting a folder: reading
    the host's files is an operator act, so it needs MANAGE as well as WRITE."""
    from superlocalmemory.access.rbac import Permission
    from superlocalmemory.server.rbac_enforce import require_manage
    from superlocalmemory.server.routes.helpers import require_engine
    from superlocalmemory.server.write_identity import authenticated_request_actor

    _require_local(request)
    actor = authenticated_request_actor(request, actor_kind="http-sources") if (write or delete) else ""
    engine = require_engine(request)
    perm = Permission.DELETE if delete else Permission.WRITE if write else Permission.READ
    profile = _profile(engine, profile_id or request.query_params.get("profile_id", ""), request, perm)
    if manage:
        require_manage(request, profile=profile)
    return profile, actor


async def _owned(profile: str, source_id: str) -> None:
    mine = await _call(sources.list_sources, profile)
    if source_id not in {s.source_id for s in mine}:
        raise HTTPException(404, detail="Not found.")


@router.get("")
async def list_all(request: Request, profile_id: str = ""):
    profile, _ = await _context(request, profile_id=profile_id)
    found = await _call(sources.list_sources, profile)
    return JSONResponse({"sources": [dataclasses.asdict(s) for s in found]}, headers=_NO_STORE)


@router.post("")
async def add(req: AddRequest, request: Request):
    profile, _ = await _context(request, write=True, manage=True, profile_id=req.profile_id)
    preview = await _call(sources.add_source, req.path, profile_id=profile, kind=req.kind)
    return JSONResponse(dataclasses.asdict(preview), headers=_NO_STORE)


@router.post("/pick-folder")
async def pick_folder(request: Request):
    """Open the computer's own folder dialog; the path still goes through the preview and confirm steps."""
    await _context(request, write=True, manage=True)
    await _call(sources_api.refuse_while_remote)
    try:
        path = await asyncio.to_thread(picker.pick_folder)
    except picker.PickerUnavailable:
        raise HTTPException(501, detail={"code": "picker_unavailable",
                                         "message": "This computer cannot open a folder dialog. Type the path instead."}) from None
    except picker.PickerBusy:
        raise HTTPException(409, detail={"code": "picker_busy",
                                         "message": "A folder dialog is already open."}) from None
    body = {"cancelled": True} if path is None else {"path": path}
    return JSONResponse(body, headers=_NO_STORE)


@router.get("/suggestions")
async def suggestions(request: Request):
    """Folders worth one click: Obsidian vaults, Documents, Desktop. Folder paths only, never files."""
    await _context(request, manage=True)
    await _call(sources_api.refuse_while_remote)
    found = await asyncio.to_thread(picker.suggestions)
    return JSONResponse({"suggestions": found}, headers=_NO_STORE)


@router.post("/{source_id}/confirm")
async def confirm(source_id: str, request: Request):
    profile, _ = await _context(request, write=True, manage=True)
    await _call(sources.confirm_source, source_id, via="dashboard", profile_id=profile)
    return JSONResponse({"confirmed": True, "source_id": source_id}, status_code=202)


@router.delete("/{source_id}")
async def remove(source_id: str, request: Request, purge: bool = False):
    profile, _ = await _context(request, delete=True)
    await _owned(profile, source_id)
    await _call(sources.remove_source, source_id, purge=purge)
    return {"removed": True, "source_id": source_id, "purged": purge}


@router.post("/{source_id}/forget-empty")
async def forget_empty(source_id: str, request: Request):
    profile, _ = await _context(request, delete=True)  # it hides memories, as remove does
    await _owned(profile, source_id)
    return await _call(sources.forget_empty, source_id)


@router.post("/{source_id}/rescan")
async def rescan(source_id: str, request: Request):
    profile, _ = await _context(request, write=True)
    await _owned(profile, source_id)
    return JSONResponse(await _call(sources.rescan, source_id), status_code=202)


@router.get("/{source_id}/report")
async def report(source_id: str, request: Request):
    profile, _ = await _context(request)
    await _owned(profile, source_id)
    found = await _call(sources.source_report, source_id)
    return JSONResponse(dataclasses.asdict(found), headers=_NO_STORE)


@router.post("/{source_id}/quarantine/release")
async def release(source_id: str, req: ReleaseRequest, request: Request):
    profile, _ = await _context(request, write=True)
    await _owned(profile, source_id)
    if not await _call(sources.release_file, source_id, req.relpath):
        raise HTTPException(404, detail="Not found.")
    return {"released": True}


@router.post("/{source_id}/hint")
async def hint(source_id: str, request: Request):
    """Watcher hints need the watcher token, which this build does not create."""
    raise HTTPException(404, detail="Not found.")
