# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-01 §4.4 / §4.5

"""POST /internal/prewarm — populates the context cache for a session.

S8-SK-02 fix: Wave 2A shipped ``hooks/prewarm_auth.authorize`` (gates
loopback → origin → install-token → body-size) and unit-tested it, but
no FastAPI route mounted it. The hot-path ``post_tool_async_hook`` POSTs
to ``/internal/prewarm`` after every tool call to refresh the
``active_brain_cache`` row for the current session/topic. Without a
route registered here, those POSTs 404'd silently, the cache never
populated, and every ``UserPromptSubmit`` ended up a structural miss.

Design notes
------------
* All 4 gates from LLD-01 §4.4 run before any engine work: loopback
  peer, absence of browser ``Origin`` header, valid install-token, body
  <= ``MAX_BODY_BYTES``. On any gate failure we return the decision's
  status code with ``application/json`` error envelope and do not touch
  the engine. This is LLD-07 SEC-HR-03 applied at the edge.
* The route is async; the actual cache write (``ContextCache.upsert``)
  is synchronous SQLite and runs on the default executor via
  ``asyncio.to_thread`` so we never block the event loop.
* Body schema is intentionally narrow: ``{"session_id": str,
  "prompt": str, "content": str, "fact_ids": list[str]}``. Missing or
  wrong-type fields produce 400.
* Never raises past this function. Any unexpected exception is caught,
  logged at ``debug`` to avoid log flooding under a hostile peer, and
  returned as 500 JSON. The hot path ``post_tool_async_hook`` treats
  any non-2xx as "fire-and-forget, try again later", so degradation is
  graceful.
"""

from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["internal"])


_ALLOWED_BODY_KEYS = frozenset({"session_id", "prompt", "content", "fact_ids"})


@router.post("/internal/prewarm")
async def prewarm(request: Request) -> JSONResponse:
    """Write (or refresh) a context-cache entry for the caller's session.

    Gates (LLD-01 §4.4):
      1. Loopback-only client (127.0.0.1 / ::1 / localhost).
      2. Reject browser-originated calls (any ``Origin`` header).
      3. Install-token present and constant-time-verified.
      4. Body <= ``MAX_BODY_BYTES``.

    On success, returns ``{"ok": true}``. On any failure, returns the
    AuthDecision's status with a terse JSON body. Never exposes engine
    error detail to the caller.
    """
    # Gates 1-3 first (cheap; reject hostile peers before reading body).
    try:
        from superlocalmemory.hooks.prewarm_auth import (
            MAX_BODY_BYTES,
            authorize,
            check_body_size,
        )
    except Exception as exc:  # pragma: no cover — primitives always present
        logger.debug("prewarm: auth primitives unimportable: %s", exc)
        return JSONResponse({"error": "server_error"}, status_code=500)

    client_host = request.client.host if request.client else ""
    headers = {k.lower(): v for k, v in request.headers.items()}

    decision = authorize(client_host=client_host, headers=headers)
    if not decision.allowed:
        return JSONResponse(
            {"error": decision.reason}, status_code=decision.status,
        )

    # Gate 4 — read body with a hard size cap. FastAPI/Starlette has no
    # cheap way to check the Content-Length up front in all servers, so
    # we read at most MAX_BODY_BYTES+1 and reject if we got more.
    try:
        body_bytes = await request.body()
    except Exception as exc:  # pragma: no cover
        logger.debug("prewarm: body read failed: %s", exc)
        return JSONResponse({"error": "bad_body"}, status_code=400)
    ok, reason = check_body_size(body_bytes)
    if not ok:
        return JSONResponse({"error": reason}, status_code=413)

    try:
        import json as _json
        payload = _json.loads(body_bytes or b"{}")
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)

    if not isinstance(payload, dict):
        return JSONResponse({"error": "invalid_json"}, status_code=400)

    # Narrow contract: reject unknown keys to keep the surface small.
    # S10-SEC-N-02: fixed error tag, never echo attacker-supplied keys.
    unknown = set(payload.keys()) - _ALLOWED_BODY_KEYS
    if unknown:
        return JSONResponse(
            {"error": "unknown_keys"}, status_code=400,
        )

    session_id = payload.get("session_id")
    prompt = payload.get("prompt")
    content = payload.get("content")
    fact_ids = payload.get("fact_ids") or []
    if not isinstance(session_id, str) or not session_id:
        return JSONResponse({"error": "session_id_required"}, status_code=400)
    if not isinstance(prompt, str) or not prompt:
        return JSONResponse({"error": "prompt_required"}, status_code=400)
    if not isinstance(content, str) or not content:
        return JSONResponse({"error": "content_required"}, status_code=400)
    if not isinstance(fact_ids, list) or not all(
        isinstance(f, str) for f in fact_ids
    ):
        return JSONResponse({"error": "fact_ids_list"}, status_code=400)

    try:
        topic_sig = await asyncio.to_thread(_compute_topic_sig, prompt)
        await asyncio.to_thread(
            _upsert_cache,
            session_id=session_id,
            topic_sig=topic_sig,
            content=content,
            fact_ids=fact_ids,
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("prewarm: upsert failed: %s", exc)
        return JSONResponse({"error": "upsert_failed"}, status_code=500)

    return JSONResponse({"ok": True})


def _compute_topic_sig(prompt: str) -> str:
    """Lazy import so module import is free of hot-path SLM modules."""
    from superlocalmemory.core.topic_signature import compute_topic_signature
    return compute_topic_signature(prompt)


def _upsert_cache(
    *, session_id: str, topic_sig: str,
    content: str, fact_ids: list[str],
) -> None:
    from superlocalmemory.core.context_cache import CacheEntry, ContextCache
    cache = ContextCache()
    try:
        cache.upsert(CacheEntry(
            session_id=session_id,
            topic_sig=topic_sig,
            content=content,
            fact_ids=tuple(fact_ids),
            provenance="prewarm_post_tool",
            computed_at=int(time.time()),
        ))
    finally:
        cache.close()


__all__ = ("router",)
