# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Image and document routes (local only).

``POST /api/v3/media/remember`` saves an image from a file path on this machine,
base64 data or an https download link, and ``POST /api/v3/documents`` queues a
PDF; neither is part of any remote tool list. The thumbnail, job-status and
document-removal routes answer only for the profile the item belongs to.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import dataclasses
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, model_validator

from superlocalmemory.core.media_fetch import too_large_for_remote
from superlocalmemory.documents import (
    document_index, document_lint, job_status, remove_document, submit_document,
)
from superlocalmemory.media.gc import gc as run_gc
from superlocalmemory.media.ingest import MediaInput, remember_media
from superlocalmemory.server.loopback import is_loopback

router = APIRouter(prefix="/api/v3", tags=["media"])
_ID = re.compile(r"[0-9a-f]{32}")
MAX_JSON_THUMB_BYTES = 32 * 1024
_CODES = {"stored": 200, "duplicate": 200, "warming": 202, "refused": 422, "processing": 202}


class MediaRememberRequest(BaseModel):
    path: str | None = None
    base64: str | None = Field(default=None, max_length=12_000_000)
    download_url: str | None = Field(default=None, max_length=2_048)
    content: str = Field(default="", max_length=24_000)
    tags: str = ""
    profile_id: str = ""
    idempotency_key: str = ""
    session_date: str = ""
    origin: str = Field(default="", max_length=16)

    @model_validator(mode="after")
    def _one_source(self) -> "MediaRememberRequest":
        if sum(bool(v) for v in (self.path, self.base64, self.download_url)) != 1:
            raise ValueError("give exactly one of path, base64 or download_url")
        return self


def _require_local(request: Request) -> None:
    host = request.client.host if request.client else ""
    if not is_loopback(host):
        raise HTTPException(403, detail="Images are available to local callers only.")


def _stricter_for_remote(req: "MediaRememberRequest") -> bool:
    """True when the body says it came from a remote app. Only the in-process tool
    sends that, and a caller that sets it on its own just gets the remote rules."""
    if req.origin != "remote":
        return False
    if req.path:
        raise HTTPException(422, detail="Remote apps cannot name a file on this computer.")
    if too_large_for_remote(req.base64):
        raise HTTPException(422, detail="Pasted data from a remote app is limited to 512 KB.")
    return True


def _profile(engine, requested: str, request: Request, permission) -> str:
    """The profile a call is about, once the caller holds ``permission`` on it.

    The permission is checked first: a caller without it gets the same answer for a
    profile that exists and one that does not, so names cannot be probed.
    """

    from superlocalmemory.server.rbac_enforce import require_permission

    wanted = (requested or "").strip() or engine._profile_id
    require_permission(request, permission, profile=wanted)
    if wanted != engine._profile_id and not engine._db.execute(
            "SELECT 1 AS one FROM profiles WHERE profile_id = ?", (wanted,)):
        raise HTTPException(404, detail="Unknown profile.")
    return wanted


def _require_manage_if(request: Request, profile: str, needed: object) -> None:
    """MANAGE when ``needed``: reading a file off this computer, or deleting stray files, is an operator act."""
    if needed:
        from superlocalmemory.server.rbac_enforce import require_manage

        require_manage(request, profile=profile)


@router.post("/media/remember")
async def remember(req: MediaRememberRequest, request: Request):
    from superlocalmemory.access.rbac import Permission
    from superlocalmemory.server.routes.helpers import require_engine
    from superlocalmemory.server.write_identity import authenticated_request_actor

    _require_local(request)
    remote = _stricter_for_remote(req)
    actor_id = authenticated_request_actor(request, actor_kind="http-media")
    engine = require_engine(request)
    profile = _profile(engine, req.profile_id, request, Permission.WRITE)
    _require_manage_if(request, profile, req.path)
    from superlocalmemory.memory_core import prepare_user_text
    from superlocalmemory.server.write_governance import enforce_remember_governance

    words = prepare_user_text(engine._config, req.content).text if req.content.strip() else ""
    enforce_remember_governance(request, engine, actor_id=actor_id, profile=profile, preview=words)
    runtime = getattr(request.app.state, "canonical_remember_runtime", None)
    if runtime is None:
        raise HTTPException(503, detail="The memory writer is not ready; retry shortly.")
    inp = MediaInput(path=Path(req.path) if req.path else None, base64=req.base64,
                     download_url=req.download_url or None, remote=remote)
    receipt = await asyncio.to_thread(
        remember_media, inp, content=req.content, profile_id=profile, actor_id=actor_id, runtime=runtime,
        config=engine._config, tags=req.tags, session_date=req.session_date, idempotency_key=req.idempotency_key)
    body = dataclasses.asdict(receipt)
    if receipt.status == "refused":
        body["detail"] = receipt.reason  # the 422 reader (daemon_request) shows this, not a generic line
    return JSONResponse(body, status_code=_CODES.get(receipt.status, 200))


class MediaGcRequest(BaseModel):
    profile_id: str = ""
    dry_run: bool = True


@router.post("/media/gc")
async def collect_garbage(req: MediaGcRequest, request: Request):
    """Report (default) or remove image leftovers: rows without a memory, files without a row."""
    from superlocalmemory.access.rbac import Permission
    from superlocalmemory.server.routes.helpers import require_engine

    from superlocalmemory.server.write_identity import authenticated_request_actor

    _require_local(request)
    if not req.dry_run:
        authenticated_request_actor(request, actor_kind="http-media")  # removing needs credentials
    engine = require_engine(request)
    profile = _profile(engine, req.profile_id, request, Permission.DELETE)
    if not req.dry_run:  # removing stray files takes more than removing one's own leftovers
        _require_manage_if(request, profile, True)
    report = await asyncio.to_thread(run_gc, profile, req.dry_run)
    return JSONResponse(dataclasses.asdict(report))


def _decode_cursor(cursor: str) -> tuple[str, str] | None:
    """The (created_at, media_id) a cursor stands for; 400 when it is not one of ours."""
    if not cursor:
        return None
    try:
        stamp, _, media_id = base64.urlsafe_b64decode(cursor.encode("ascii")).decode("utf-8").partition("|")
    except (binascii.Error, UnicodeError, ValueError):
        raise HTTPException(400, detail="Bad cursor.") from None
    if not stamp or not _ID.fullmatch(media_id):
        raise HTTPException(400, detail="Bad cursor.")
    return stamp, media_id


def _encode_cursor(row: dict) -> str:
    return base64.urlsafe_b64encode(f"{row['created_at']}|{row['media_id']}".encode()).decode("ascii")


def _list_page(profile: str, limit: int, after: tuple[str, str] | None) -> dict:
    from superlocalmemory.media import open_media_store

    store = open_media_store()
    if store is None:
        return {"items": [], "next_cursor": None}
    try:
        rows = store.list_images(profile, limit=limit + 1, after=after)
    finally:
        store.close()
    page = rows[:limit]
    return {"items": page, "next_cursor": _encode_cursor(page[-1]) if len(rows) > limit else None}


@router.get("/media")
async def list_images(request: Request, profile_id: str = "", cursor: str = Query("", max_length=200),
                      limit: int = Query(60, ge=1, le=200)):
    """Saved images of this profile, newest first, a page at a time."""
    from superlocalmemory.access.rbac import Permission
    from superlocalmemory.server.routes.helpers import require_engine

    _require_local(request)
    engine = require_engine(request)
    profile = _profile(engine, profile_id, request, Permission.READ)
    after = _decode_cursor(cursor)
    found = await asyncio.to_thread(_list_page, profile, limit, after)
    return JSONResponse(found, headers={"Cache-Control": "no-store"})


@router.get("/media/{media_id}/thumb")
async def thumbnail(media_id: str, request: Request, profile_id: str = "", format: str = ""):
    from superlocalmemory.access.rbac import Permission
    from superlocalmemory.media import open_media_store
    from superlocalmemory.server.routes.helpers import require_engine

    _require_local(request)
    if format not in ("", "json"):
        raise HTTPException(422, detail="format must be empty or json.")
    engine = require_engine(request)
    profile = _profile(engine, profile_id, request, Permission.READ)
    store = open_media_store() if _ID.fullmatch(media_id) else None
    if store is None:
        raise HTTPException(404, detail="Not found.")
    try:
        row = store.get_item(media_id)
    finally:
        store.close()
    if not row or row["profile_id"] != profile or row["state"] != "active" or not row["thumb_webp"]:
        raise HTTPException(404, detail="Not found.")
    if format == "json":
        thumb = bytes(row["thumb_webp"])
        if len(thumb) > MAX_JSON_THUMB_BYTES:
            raise HTTPException(413, detail="Thumbnail is too large to send inline.")
        return JSONResponse({"mime": "image/webp", "base64": base64.b64encode(thumb).decode("ascii")},
                            headers={"Cache-Control": "private, max-age=3600"})
    return Response(bytes(row["thumb_webp"]), media_type="image/webp",
                    headers={"Cache-Control": "private, max-age=3600", "X-Content-Type-Options": "nosniff"})


class DocumentSubmitRequest(MediaRememberRequest):
    base64: str | None = Field(default=None, max_length=34_000_000)
    file_name: str = Field(default="", max_length=255)


@router.post("/documents")
async def submit(req: DocumentSubmitRequest, request: Request):
    from superlocalmemory.access.rbac import Permission
    from superlocalmemory.memory_core import prepare_user_text
    from superlocalmemory.server.routes.helpers import require_engine
    from superlocalmemory.server.write_governance import enforce_remember_governance
    from superlocalmemory.server.write_identity import authenticated_request_actor

    _require_local(request)
    if req.download_url:
        raise HTTPException(422, detail="Links are accepted for images only.")
    _stricter_for_remote(req)
    actor_id = authenticated_request_actor(request, actor_kind="http-media")
    engine = require_engine(request)
    profile = _profile(engine, req.profile_id, request, Permission.WRITE)
    _require_manage_if(request, profile, req.path)
    words = prepare_user_text(engine._config, req.content).text if req.content.strip() else ""
    enforce_remember_governance(request, engine, actor_id=actor_id, profile=profile, preview=words)
    inp = MediaInput(path=Path(req.path) if req.path else None, base64=req.base64, file_name=req.file_name)
    receipt = await asyncio.to_thread(
        submit_document, inp, content=req.content, profile_id=profile, actor_id=actor_id, config=engine._config,
        tags=req.tags, session_date=req.session_date, idempotency_key=req.idempotency_key)
    body = dataclasses.asdict(receipt)
    if receipt.status == "refused":
        body["detail"] = receipt.reason
    return JSONResponse(body, status_code=_CODES.get(receipt.status, 200))


@router.get("/jobs/{job_id}")
async def job(job_id: str, request: Request, profile_id: str = ""):
    from superlocalmemory.access.rbac import Permission
    from superlocalmemory.server.routes.helpers import require_engine

    _require_local(request)
    engine = require_engine(request)
    profile = _profile(engine, profile_id, request, Permission.READ)
    found = await asyncio.to_thread(job_status, job_id, profile) if _ID.fullmatch(job_id) else None
    if found is None:
        raise HTTPException(404, detail="Not found.")
    return JSONResponse(found, headers={"Cache-Control": "no-store"})


@router.get("/documents")
async def documents(request: Request, profile_id: str = "", cursor: str = Query("", max_length=200),
                    limit: int = Query(50, ge=1, le=200)):
    from superlocalmemory.access.rbac import Permission
    from superlocalmemory.server.routes.helpers import require_engine

    _require_local(request)
    engine = require_engine(request)
    profile = _profile(engine, profile_id, request, Permission.READ)
    found = await asyncio.to_thread(document_index, profile, limit, cursor, db=engine._db)
    return JSONResponse(found, headers={"Cache-Control": "no-store"})


@router.get("/documents/lint")
async def lint(request: Request, profile_id: str = ""):
    from superlocalmemory.access.rbac import Permission
    from superlocalmemory.server.routes.helpers import require_engine

    _require_local(request)
    engine = require_engine(request)
    profile = _profile(engine, profile_id, request, Permission.READ)
    found = await asyncio.to_thread(document_lint, profile, db=engine._db)
    return JSONResponse(found, headers={"Cache-Control": "no-store"})


def _eraser(engine):
    """Erase facts through the compliance path (erasure service, receipt, side tables, text scrub)."""
    def erase(profile_id: str, fact_ids: list[str], document_id: str) -> dict:
        from superlocalmemory.compliance.gdpr import GDPRCompliance

        return GDPRCompliance(engine._db, engine=engine).forget_facts(fact_ids, profile_id, subject_id=document_id)
    return erase


@router.delete("/documents/{document_id}")
async def remove(document_id: str, request: Request, profile_id: str = "", hard: bool = False):
    from superlocalmemory.access.rbac import Permission
    from superlocalmemory.server.routes.helpers import require_engine
    from superlocalmemory.server.write_governance import enforce_forget_governance
    from superlocalmemory.server.write_identity import authenticated_request_actor

    _require_local(request)
    actor_id = authenticated_request_actor(request, actor_kind="http-media")
    engine = require_engine(request)
    profile = _profile(engine, profile_id, request, Permission.DELETE)
    if not _ID.fullmatch(document_id):
        raise HTTPException(404, detail="Not found.")
    enforce_forget_governance(request, engine, actor_id=actor_id, profile=profile, target=document_id)
    if hard:
        if not await asyncio.to_thread(remove_document, document_id, profile, hard=True, eraser=_eraser(engine)):
            raise HTTPException(404, detail="Not found, or the erasure was not complete; retry.")
        return {"removed": True, "document_id": document_id, "erased": True}
    runtime = getattr(request.app.state, "canonical_remember_runtime", None)
    if runtime is None or not getattr(runtime, "ready", False):
        raise HTTPException(503, detail="The memory writer is not ready; retry shortly.")
    if not await asyncio.to_thread(remove_document, document_id, profile, runtime=runtime):
        raise HTTPException(404, detail="Not found.")
    return {"removed": True, "document_id": document_id}
