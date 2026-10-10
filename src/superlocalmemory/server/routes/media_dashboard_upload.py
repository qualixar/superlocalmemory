# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The dashboard's file upload (local only): the sizes the pane advertises, streamed.

``POST /api/v3/media/upload?kind=image|pdf`` takes the raw file as the request
body. The JSON routes carry files as base64 inside a JSON document, which caps
them near 9 MB (images) and 25 MB (PDFs); the dashboard promises 25 MB and 100 MB.
Here the body is streamed to a scratch file outside SuperLocalMemory's data
folder, refused the moment it passes the limit, and then handed to the same
image and document ingest the JSON routes use. The caller's rights are those of
pasted data: the local write credential and WRITE on the profile. The caller
never names a path, so there is no MANAGE step; the ingest still checks what the
bytes really are, whatever ``kind`` says.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from superlocalmemory.documents import submit_document
from superlocalmemory.media import files
from superlocalmemory.media.ingest import MediaInput, remember_media
from superlocalmemory.server.routes.media import _CODES, _profile, _require_local, _save_scope
from superlocalmemory.server.write_governance import enforce_remember_governance

router = APIRouter(prefix="/api/v3", tags=["media"])
MB = 1024 * 1024
IMAGE_LIMIT = 25 * MB
PDF_LIMIT = 100 * MB


def _limit(kind: str) -> int:
    return IMAGE_LIMIT if kind == "image" else PDF_LIMIT


def _too_large(kind: str) -> HTTPException:
    noun = "image" if kind == "image" else "PDF"
    return HTTPException(413, detail=f"That {noun} is too large ({_limit(kind) // MB} MB limit).")


def _disk_full() -> HTTPException:
    return HTTPException(507, detail=files.DISK_FULL)


async def _spool(request: Request, kind: str) -> Path:
    """Stream the body into a new scratch folder; returns the folder.

    413 past the limit; 507 when the disk (or quota) fills while it is written.
    """
    limit = _limit(kind)
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise _too_large(kind)
    work: Path | None = None
    try:
        work = Path(tempfile.mkdtemp(prefix="slm-upload-"))
        fd = os.open(work / "upload.bin", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        size = 0
        with os.fdopen(fd, "wb") as fh:
            async for chunk in request.stream():
                size += len(chunk)
                if size > limit:
                    raise _too_large(kind)
                fh.write(chunk)
    except BaseException as exc:
        if work is not None:
            shutil.rmtree(work, ignore_errors=True)
        if files.is_disk_full(exc):
            raise _disk_full() from None
        raise
    return work


def _ingest(kind: str, work: Path, file_name: str, common: dict, runtime: object):
    """Run the shared ingest on the scratch file, off the event loop."""
    path = work / "upload.bin"
    if kind == "image":
        return remember_media(MediaInput(data=path.read_bytes()), runtime=runtime, **common)
    return submit_document(MediaInput(path=path, file_name=file_name), **common)


@router.post("/media/upload")
async def upload(request: Request, kind: str = Query("", max_length=8),
                 file_name: str = Query("", max_length=255), profile_id: str = ""):
    """Save an image or a PDF sent as the raw request body (the dashboard's upload)."""
    from superlocalmemory.access.rbac import Permission
    from superlocalmemory.server.routes.helpers import require_engine
    from superlocalmemory.server.write_identity import authenticated_request_actor

    _require_local(request)
    if kind not in ("image", "pdf"):
        raise HTTPException(422, detail="kind must be image or pdf.")
    actor_id = authenticated_request_actor(request, actor_kind="http-media")
    engine = require_engine(request)
    profile = _profile(engine, profile_id, request, Permission.WRITE)
    scope, shared_with = _save_scope(engine, SimpleNamespace(origin="", scope=None, shared_with=None),
                                     request, profile)
    enforce_remember_governance(request, engine, actor_id=actor_id, profile=profile, preview="")
    runtime = getattr(request.app.state, "canonical_remember_runtime", None)
    if kind == "image" and runtime is None:
        raise HTTPException(503, detail="The memory writer is not ready; retry shortly.")
    work = await _spool(request, kind)
    try:
        common = dict(content="", profile_id=profile, actor_id=actor_id, config=engine._config,
                      scope=scope, shared_with=shared_with)
        receipt = await asyncio.to_thread(_ingest, kind, work, file_name, common, runtime)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    body = dataclasses.asdict(receipt)
    if receipt.status == "refused":
        body["detail"] = receipt.reason
    return JSONResponse(body, status_code=_CODES.get(receipt.status, 200))
