# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""POST /internal/hook-events — hooks hand their events to the daemon.

Called by ``hooks/event_delivery.py`` from the PostToolUse, UserPromptSubmit
and Stop hooks of every host. The caller must hold this daemon instance's
private capability (``X-SLM-Daemon-Capability`` + ``X-SLM-Target-Instance``,
what ``daemon_request`` sends). The events are validated, appended to the
spool and the writer thread is woken. The ack is sent only once the events
are on disk, so a daemon that dies after acking loses nothing; the writer
drains the spool when it starts again.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["internal"])

#: 32 events x a few KB each, with room to spare.
MAX_BODY_BYTES = 256 * 1024


@router.post("/internal/hook-events")
async def accept_hook_events(request: Request) -> JSONResponse:
    from superlocalmemory.server.write_identity import require_daemon_actor

    require_daemon_actor(request, getattr(request.app.state, "daemon_descriptor", None))
    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        raise HTTPException(413, detail="hook event batch too large")
    from superlocalmemory.server.hook_event_writer import HookEventBatch

    try:
        batch = HookEventBatch.model_validate_json(body)
    except (ValidationError, ValueError):
        raise HTTPException(422, detail="invalid hook event batch") from None

    import asyncio

    from superlocalmemory.server import hook_event_writer

    written, dropped = await asyncio.to_thread(hook_event_writer.accept_batch, batch)
    return JSONResponse({"ok": True, "accepted": written, "dropped": dropped})


__all__ = ("MAX_BODY_BYTES", "router")
