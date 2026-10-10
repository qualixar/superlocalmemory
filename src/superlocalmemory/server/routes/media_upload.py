# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Save a finished one-time upload (local only).

``POST /api/v3/media/uploads/{upload_id}/finish`` is called by the daemon's own
relay code (``remote_connections/upload_relay``) once every byte of a file sent
through an upload link has arrived. It takes no body: the profile, the kind and
the note come from the link, which was bound to one connection, key and profile
when it was made. The save is a remote one in every respect: personal scope, the
remote fetch rules, the same governance and profile checks as any other save.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from superlocalmemory.documents import submit_document
from superlocalmemory.media.ingest import MediaInput, remember_media
from superlocalmemory.media.upload_links import UploadRow, default_links, looks_like
from superlocalmemory.server.routes.media import _CODES, _profile, _require_local

router = APIRouter(prefix="/api/v3", tags=["media"])
_ID = re.compile(r"[0-9a-f]{32}")
_WRONG = "That file is not the kind of file this link was made for."


def _read_checked(path: Path, row: UploadRow) -> bytes:
    """The scratch file's bytes, once it is exactly the file that was sent and still looks like its kind."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as fh:
            if os.fstat(fh.fileno()).st_size != row.total:
                raise HTTPException(422, detail="The file did not arrive whole. Start the upload again.")
            data = fh.read(row.total + 1)
    except OSError:
        raise HTTPException(422, detail="The file did not arrive whole. Start the upload again.") from None
    if len(data) != row.total or not looks_like(row.kind, data[:16]):
        raise HTTPException(422, detail=_WRONG)
    return data


@router.post("/media/uploads/{upload_id}/finish")
async def finish_upload(upload_id: str, request: Request):
    from superlocalmemory.access.rbac import Permission
    from superlocalmemory.memory_core import prepare_user_text
    from superlocalmemory.server.routes.helpers import require_engine
    from superlocalmemory.server.write_governance import enforce_remember_governance
    from superlocalmemory.server.write_identity import authenticated_request_actor

    _require_local(request)
    actor_id = authenticated_request_actor(request, actor_kind="http-media")
    links = default_links()
    row = await asyncio.to_thread(links.get, upload_id) if _ID.fullmatch(upload_id) else None
    if row is None or row.state != "finishing":
        raise HTTPException(404, detail="Not found.")
    engine = require_engine(request)
    profile = _profile(engine, row.profile_id, request, Permission.WRITE)
    words = prepare_user_text(engine._config, row.note).text if row.note.strip() else ""
    enforce_remember_governance(request, engine, actor_id=actor_id, profile=profile, preview=words)
    runtime = getattr(request.app.state, "canonical_remember_runtime", None)
    if row.kind == "image" and runtime is None:
        raise HTTPException(503, detail="The memory writer is not ready; retry shortly.")
    data = await asyncio.to_thread(_read_checked, links.temp_path(upload_id), row)
    inp = MediaInput(data=data, remote=True)
    common = dict(content=row.note, profile_id=profile, actor_id=actor_id, config=engine._config,
                  idempotency_key=f"upload:{upload_id}", scope="personal", shared_with=())
    if row.kind == "image":
        receipt = await asyncio.to_thread(remember_media, inp, runtime=runtime, **common)
    else:
        receipt = await asyncio.to_thread(submit_document, inp, **common)
    body = dataclasses.asdict(receipt)
    if receipt.status == "refused":
        body["detail"] = receipt.reason
    return JSONResponse(body, status_code=_CODES.get(receipt.status, 200))
