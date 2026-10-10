# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""``/api/v3/embedding/reindex``: switch the embedding model in the background.

    POST /api/v3/embedding/reindex                  start a switch -> 202 + job
    GET  /api/v3/embedding/reindex                  progress, live and previous model
    POST /api/v3/embedding/reindex/rollback         back to the previous model -> 202
    POST /api/v3/embedding/reindex/cancel           stop a running switch
    POST /api/v3/embedding/reindex/forget-previous  free the previous vectors

The dashboard's two save routes (PUT /embedding/config, POST /mode/set) call
:func:`queue_if_new_space` too: a save that changes the embedding space
starts a job instead of hot-swapping into a blocking re-embed. One job at a
time; a second request gets 409 naming the running one.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v3/embedding/reindex", tags=["embedding"])

_NOT_RUNNING = ("the background re-index runs inside the SLM daemon, which is not "
                "serving this request; start it with: slm serve")


def _runner(request: Request) -> Any:
    return getattr(request.app.state, "embedding_reindex", None)


def _no_runner() -> JSONResponse:
    return JSONResponse({"error": "daemon_required", "detail": _NOT_RUNNING}, status_code=409)


def _invalid(code: str, message: str) -> JSONResponse:
    # 422 with {code, message}: the CLI shows the reason instead of "not running".
    return JSONResponse({"error": code, "detail": {"code": code, "message": message}},
                        status_code=422)


def _conflict(exc: Any) -> JSONResponse:
    job = exc.job
    return JSONResponse({
        "error": "reindex_running", "job_id": job["job_id"], "state": job["state"],
        "detail": (f"a re-index is already {job['state']} (job {job['job_id']}: "
                   f"{job['from_signature']} -> {job['to_signature']}). Wait for it, or "
                   "stop it with: slm embedder cancel"),
    }, status_code=409)


def _accepted(job: dict) -> JSONResponse:
    return JSONResponse({"success": True, "accepted": True, "job": job,
                         "detail": (f"re-indexing {job['total']} memories with "
                                    f"{job['to']} in the background; recall keeps using "
                                    f"{job['from']} until it is done. Progress: "
                                    "slm embedder status")},
                        status_code=202)


def target_config(live: Any, body: dict) -> Any:
    """The embedding config a request asks for, starting from the live one."""
    from dataclasses import replace

    from superlocalmemory.core.embedding_providers import resolve_embedding_provider

    model = str(body.get("model_name") or body.get("model") or live.model_name).strip()
    provider = resolve_embedding_provider(str(body.get("provider") or ""), model, live.provider)
    dim = int(body.get("dimension") or 0) or (live.dimension if model == live.model_name else 0)
    if not model:
        raise ValueError("a model name is required")
    if not 64 <= dim <= 8192:
        raise ValueError("the new model's vector size is needed: pass --dimension N "
                         "(64-8192), for example 768 or 384")
    endpoint = str(body.get("api_endpoint", live.api_endpoint) or "")
    same_place = provider == live.provider and endpoint == live.api_endpoint
    return replace(
        live, provider=provider, model_name=model, dimension=dim, api_endpoint=endpoint,
        api_key=str(body.get("api_key") or (live.api_key if same_place else "")),
        ollama_model=model if provider == "ollama" else live.ollama_model,
    )


def queue_if_new_space(request: Request, live: Any, target: Any, *,
                       force: bool = False) -> JSONResponse | None:
    """None: apply the save as before. Otherwise the response to send instead.

    Same space (or a renamed alias) -> None. No daemon runner (direct use) ->
    None, the save is persisted and the daemon queues the job when it starts.
    ``force`` with the same width declares the vectors already match: the new
    name becomes the live space without a re-index (the old escape hatch).
    """
    from superlocalmemory.core.embedding_reindex import NoChange, Refused
    from superlocalmemory.storage import embedding_spaces as sp
    from superlocalmemory.storage.embedding_reindex_jobs import JobConflict

    runner = _runner(request)
    if runner is None or sp.same_space(sp.signature_of(live), sp.signature_of(target)):
        return None
    if force and target.dimension == live.dimension:
        _declare_equivalent(runner, target)
        return None
    try:
        return _accepted(runner.request_switch(target))
    except NoChange:
        return None
    except JobConflict as exc:
        return _conflict(exc)
    except Refused as exc:
        return JSONResponse({"error": "reindex_refused", "detail": str(exc)}, status_code=409)


def saved_switch_note(request: Request, live: Any, target: Any) -> dict:
    """``needs_reindex`` + ``message`` for a save answered 200, not 202.

    With the daemon's runner a real change of space answers 202 (see
    :func:`queue_if_new_space`), so a 200 there re-indexes nothing: the same
    space, an alias, a declared-equivalent ``force``, or no change. Without
    the runner the save is persisted, every engine stays on the live model,
    and the daemon queues the job when it loads the configuration.
    """
    from superlocalmemory.core.embedding_reindex import pending_switch_message, space_changed

    if _runner(request) is not None or not space_changed(live, target):
        return {"needs_reindex": False, "message": ""}
    return {"needs_reindex": True, "message": pending_switch_message(live, target)}


def _declare_equivalent(runner: Any, target: Any) -> None:
    from superlocalmemory.core.embedding_reindex_steps import write_txn
    from superlocalmemory.storage import embedding_spaces as sp
    from superlocalmemory.storage.embedding_migrator import _write_stored_signature

    conn = sp.connect(runner.db_path)
    try:
        with write_txn(conn, runner.db_path):
            row = sp.read_space(conn)
            sp.write_space(conn, sp.signature_of(target), sp.public_config(target),
                           row.get("prev_signature") if row else None, None)
    finally:
        conn.close()
    _write_stored_signature(runner.data_root, sp.signature_of(target))


def _live_config(request: Request) -> Any:
    from superlocalmemory.core.config import SLMConfig

    engine = getattr(request.app.state, "engine", None)
    config = getattr(engine, "_config", None) or getattr(request.app.state, "config", None)
    return (config or SLMConfig.load()).embedding


@router.post("")
async def start_switch(request: Request):
    from superlocalmemory.server.rbac_enforce import require_manage

    require_manage(request)
    runner = _runner(request)
    if runner is None:
        return _no_runner()
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("request body must be a JSON object")
        live = _live_config(request)
        target = target_config(live, body)
    except (ValueError, TypeError) as exc:
        return _invalid("invalid_request", str(exc))
    from superlocalmemory.storage import embedding_spaces as sp

    if sp.same_space(sp.signature_of(live), sp.signature_of(target)):
        return _invalid("no_change", f"{target.model_name} is already the embedding model")
    response = await asyncio.to_thread(queue_if_new_space, request, live, target)
    return response or _invalid("no_change", f"{target.model_name} is already in use")


@router.get("")
async def reindex_status(request: Request):
    runner = _runner(request)
    if runner is None:
        return _no_runner()
    return await asyncio.to_thread(runner.status)


async def _call(request: Request, method: str, accepted: bool) -> JSONResponse:
    from superlocalmemory.core.embedding_reindex import Refused
    from superlocalmemory.server.rbac_enforce import require_manage
    from superlocalmemory.storage.embedding_reindex_jobs import JobConflict

    require_manage(request)
    runner = _runner(request)
    if runner is None:
        return _no_runner()
    try:
        result = await asyncio.to_thread(getattr(runner, method))
    except JobConflict as exc:
        return _conflict(exc)
    except Refused as exc:
        return JSONResponse({"error": "refused", "detail": str(exc)}, status_code=409)
    if accepted:
        return _accepted(result)
    return JSONResponse({"success": True, **({"job": result} if "job_id" in result else result)})


@router.post("/rollback")
async def rollback(request: Request):
    return await _call(request, "request_rollback", accepted=True)


@router.post("/cancel")
async def cancel(request: Request):
    return await _call(request, "cancel", accepted=False)


@router.post("/forget-previous")
async def forget_previous(request: Request):
    return await _call(request, "forget_previous", accepted=False)


__all__ = ["queue_if_new_space", "router", "saved_switch_note", "target_config"]
