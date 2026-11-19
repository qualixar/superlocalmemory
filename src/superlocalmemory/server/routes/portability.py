# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Bring memories in from other apps: preview, import, progress, cancel, undo (WS-9, D-19, D-49).

    GET  /api/import/runs                   recent runs, the switch and the limits   (review-read)
    GET  /api/import/runs/{run_id}          one run and its progress                 (review-read)
    POST /api/import/preview                what a file holds; writes nothing         (review-read)
    POST /api/import/run                    start an import in the background        (memory-update)
    POST /api/import/runs/{run_id}/cancel   stop it; what was saved is kept          (memory-update)
    POST /api/import/runs/{run_id}/undo     take the whole run back, exactly         (memory-update)
    POST /api/import/switch                 turn the pack on or off (this computer)  (global manage)

The pack is on by default behind ``portability.import`` (a person's own off or the kill switch
stops it); every call but the two reads and the switch says which switch is off. A file
arrives as ``{"filename", "data_base64"}`` (at most
``constants.MAX_UPLOAD_BYTES`` after decoding: the size cap is a routing constant, and a request
that says it is bigger is refused before its body is read), or as ``{"path"}`` from a caller on
this computer. The bytes are staged privately between preview and run (``portability/staging``).
Starting and undoing are decisions by any caller (D-49): who and the channel come from the
connection (``decision_door``), they are recorded as kind ``import`` in the decision log, and a
caller from another machine is governed by its role. Counts and plain words only: no route
returns memory text.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from superlocalmemory.portability import constants as C

logger = logging.getLogger(__name__)

router = APIRouter(tags=["import"])

#: A base64 body is a third bigger than the bytes in it; a little slack for the JSON around them.
_BODY_CAP = C.MAX_UPLOAD_BYTES * 4 // 3 + 64 * 1024


class SourceBody(BaseModel):
    filename: str = Field(default="", max_length=240)
    data_base64: str = ""
    path: str = Field(default="", max_length=2000)
    upload_id: str = Field(default="", max_length=64)
    format: str = Field(default="", max_length=40)
    profile_id: str | None = None
    #: Legacy-style label of the caller; the channel and actor come from the connection.
    by: str = Field(default="", max_length=80)
    wait_seconds: float = Field(default=0.0, ge=0.0, le=60.0)


async def _body(request: Request) -> SourceBody:
    """The JSON body, read in pieces and refused past the cap before it is all in memory."""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > _BODY_CAP:
        raise HTTPException(413, detail=_TOO_BIG)
    received = bytearray()
    async for chunk in request.stream():
        received += chunk
        if len(received) > _BODY_CAP:
            raise HTTPException(413, detail=_TOO_BIG)
    try:
        raw = json.loads(bytes(received) or b"{}")
        return SourceBody.model_validate(raw if isinstance(raw, dict) else {})
    except ValueError:
        raise HTTPException(422, detail="That request could not be read.") from None


_TOO_BIG = (f"That file is larger than {C.MAX_UPLOAD_BYTES // (1024 * 1024)} MB. Split it, or "
            "import it from a path on this computer with 'slm import'.")


def _docs(request: Request, body: SourceBody):
    """The docs a request names: a staged upload, uploaded bytes, or a local path."""
    from superlocalmemory.portability import readers, staging
    from superlocalmemory.server import decision_door

    try:
        if body.upload_id:
            return staging.get(body.upload_id), body.filename or "upload"
        if body.path:
            if not decision_door.peer_is_local(request):
                raise HTTPException(403, detail="A path on this computer can only be named by a "
                                                "caller on this computer. Upload the file instead.")
            return readers.from_path(body.path), body.filename or body.path
        if body.data_base64:
            try:
                data = base64.b64decode(body.data_base64, validate=True)
            except (binascii.Error, ValueError):
                raise HTTPException(422, detail="That file data could not be read.") from None
            return readers.from_bytes(body.filename, data), body.filename or "upload"
    except readers.IntakeRefused as exc:
        raise HTTPException(422, detail=str(exc)) from None
    raise HTTPException(422, detail="Choose a file to import.")


def _refuse(exc: Exception) -> HTTPException:
    """Plain words for a refusal, with the switch named when the pack is off."""
    from superlocalmemory.portability.model import ImportFormatError
    from superlocalmemory.portability.service import ImportOff, ImportRefused

    if isinstance(exc, ImportOff):
        # 409, not 403: the pack being off is a state, not a role. The words name the switch, and
        # the header names it for programs.
        return HTTPException(409, detail=str(exc), headers={"X-SLM-Switch": C.FLAG})
    if isinstance(exc, ImportRefused):
        return HTTPException(409, detail=str(exc))
    if isinstance(exc, ImportFormatError):
        return HTTPException(422, detail=str(exc))
    raise exc


def _profile(value: str | None):
    from superlocalmemory.server.routes import memories as m

    return m._routed_profile(value or None)


@router.get("/api/import/runs")
def list_runs(request: Request, profile_id: str = "", limit: int = 20):
    """Recent runs of the profile, the switch and the limits. Works while the pack is off."""
    from superlocalmemory.portability import service
    from superlocalmemory.server.routes import memories as m

    try:
        engine, target = m._authorize_review_read(request, _profile(profile_id))
        return {"success": True, "profile_id": target, **service.overview(engine, target, limit)}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Import list error")


@router.get("/api/import/runs/{run_id}")
def show_run(request: Request, run_id: str, profile_id: str = ""):
    from superlocalmemory.portability import service
    from superlocalmemory.server.routes import memories as m

    try:
        engine, target = m._authorize_review_read(request, _profile(profile_id))
        found = service.get_run(engine, target, run_id)
        if found is None:
            raise HTTPException(404, detail="No such import.")
        return {"success": True, "profile_id": target, "run": found}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Import status error")


@router.post("/api/import/preview")
async def preview(request: Request):
    """Read a file and say what is in it. Nothing is saved; the file is kept privately for Import."""
    from superlocalmemory.portability import service, staging
    from superlocalmemory.server.routes import memories as m

    body = await _body(request)
    try:
        engine, target = m._authorize_review_read(request, _profile(body.profile_id))

        def work() -> dict:
            docs, _label = _docs(request, body)
            _parsed, view = service.preview(engine, target, docs, body.format or None)
            view["upload_id"] = body.upload_id or staging.put(docs)
            return view

        view = await asyncio.to_thread(work)
        return {"success": True, "profile_id": target, **view}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise _known(exc, m, "Import preview error")


def _known(exc: Exception, m: Any, detail: str) -> HTTPException:
    from superlocalmemory.portability.model import ImportFormatError
    from superlocalmemory.portability.service import ImportOff, ImportRefused

    if isinstance(exc, (ImportOff, ImportRefused, ImportFormatError)):
        return _refuse(exc)
    return m._canonical_mutation_error(exc, detail)


@router.post("/api/import/run")
async def start_run(request: Request):
    """Start the import in the background; returns at once with the run to watch."""
    from superlocalmemory.portability import service, staging
    from superlocalmemory.server import decision_door
    from superlocalmemory.server.routes import memories as m

    body = await _body(request)
    try:
        engine, target, ctx = m._authorize_memory_mutation(
            request, "update", "import", run_pre_hook=False, profile=_profile(body.profile_id))
        who = decision_door.decider_for(request, str(ctx.get("agent_id") or ""), body.by)
        runtime = getattr(request.app.state, "canonical_remember_runtime", None)
        if runtime is None:
            raise HTTPException(503, detail="The memory writer is not ready; try again shortly.")

        def work() -> dict:
            docs, label = _docs(request, body)
            return service.start_run(
                engine, runtime, profile_id=target, docs=docs, fmt=body.format or None,
                label=label, decider=who,
                on_finish=(lambda: staging.drop(body.upload_id)) if body.upload_id else None)

        run = await asyncio.to_thread(work)
        logger.info("import %s started in %s via %s", run.get("run_id", "")[-8:], target,
                    who.channel)
        return {"success": True, "profile_id": target, "run": run}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise _known(exc, m, "Import start error")


@router.post("/api/import/runs/{run_id}/cancel")
def cancel_run(request: Request, run_id: str, body: SourceBody | None = None):
    from superlocalmemory.portability import service
    from superlocalmemory.server.routes import memories as m

    body = body or SourceBody()
    try:
        engine, target, _ctx = m._authorize_memory_mutation(
            request, "update", run_id, run_pre_hook=False, profile=_profile(body.profile_id))
        try:
            return {"success": True, "profile_id": target,
                    "run": service.cancel(engine, target, run_id)}
        except LookupError:
            raise HTTPException(404, detail="No such import.") from None
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise _known(exc, m, "Import cancel error")


@router.post("/api/import/runs/{run_id}/undo")
def undo_run(request: Request, run_id: str, body: SourceBody | None = None):
    """Take the whole run back in the background (recorded). ``wait_seconds`` waits for the end."""
    from superlocalmemory.core.mutations import purge_profile_context_cache
    from superlocalmemory.portability import service
    from superlocalmemory.server import decision_door
    from superlocalmemory.server.routes import memories as m

    body = body or SourceBody()
    try:
        engine, target, ctx = m._authorize_memory_mutation(
            request, "update", run_id, run_pre_hook=False, profile=_profile(body.profile_id))
        who = decision_door.decider_for(request, str(ctx.get("agent_id") or ""), body.by)
        runtime = getattr(request.app.state, "canonical_remember_runtime", None)
        try:
            run = service.start_undo(engine, runtime, profile_id=target, run_id=run_id,
                                     decider=who, wait_s=body.wait_seconds)
        except LookupError:
            raise HTTPException(404, detail="No such import.") from None
        purge_profile_context_cache(engine, target)
        return {"success": True, "profile_id": target, "run": run}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise _known(exc, m, "Import undo error")


class SwitchBody(BaseModel):
    state: str = Field(default="", max_length=8)


@router.post("/api/import/switch")
def switch(request: Request, body: SwitchBody):
    """Turn the pack on or off (``features.json``). This computer only, like any setting."""
    from superlocalmemory.core.feature_flags import FeatureFlags, FlagState, write_flag
    from superlocalmemory.infra.data_root import canonical_data_root
    from superlocalmemory.portability import service
    from superlocalmemory.server import decision_door, write_identity
    from superlocalmemory.server.rbac_enforce import require_global_manage

    if not decision_door.peer_is_local(request):
        raise HTTPException(403, detail="This setting can only be changed from this machine.")
    write_identity.require_write_actor(
        request, getattr(request.app.state, "daemon_descriptor", None), actor_kind="features")
    require_global_manage(request)
    if body.state not in ("on", "off"):
        raise HTTPException(422, detail="Say on or off.")
    home = canonical_data_root()
    if body.state == "on" and FeatureFlags(home).killed:
        raise HTTPException(409, detail=service.KILLED_WORDS)
    write_flag(home, C.FLAG, FlagState(body.state))
    return {"success": True, **service.flag_view(home)}
