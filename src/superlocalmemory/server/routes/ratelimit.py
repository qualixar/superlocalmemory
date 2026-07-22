# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com

"""Dashboard-editable rate limits (task #47).

The soak found the loopback write limiter (correctly) returning 429 under
heavy multi-system load. This exposes the write/read/window ceilings so an
operator can raise them from the Governance panel:

  GET /api/v3/ratelimit  — current effective limits (+ derived loopback)
  PUT /api/v3/ratelimit  — set write/read/window; applied at runtime (no
                           restart) and persisted to config.json

Runtime apply goes through rate_limiter.set_limits(), which reconfigures every
registered enforcement limiter live. Persistence reuses config_api's config.json
helpers; load_persisted_limits() re-applies the saved override at daemon start.
Admin-only (RBAC MANAGE), matching config_api.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from superlocalmemory.infra.rate_limiter import (
    _loopback_read,
    _loopback_write,
    get_limits,
    set_limits,
)
from superlocalmemory.server.routes.config_api import (
    _atomic_write,
    _read_config,
    _require_admin,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v3", tags=["ratelimit"])

_CONFIG_KEY = "rate_limit"
_MIN, _MAX_REQ, _MAX_WINDOW = 1, 100_000, 3600


class RateLimitUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    write: int | None = Field(default=None, ge=_MIN, le=_MAX_REQ)
    read: int | None = Field(default=None, ge=_MIN, le=_MAX_REQ)
    window: int | None = Field(default=None, ge=_MIN, le=_MAX_WINDOW)


def _effective() -> dict:
    cur = get_limits()
    return {
        "write": cur["write"],
        "read": cur["read"],
        "window": cur["window"],
        "loopback_write": _loopback_write(cur["write"]),
        "loopback_read": _loopback_read(cur["read"]),
    }


def load_persisted_limits() -> None:
    """Re-apply the persisted rate-limit override at daemon start (fail-open)."""
    try:
        saved = (_read_config().get(_CONFIG_KEY) or {})
        if not isinstance(saved, dict):
            return
        w = saved.get("write")
        r = saved.get("read")
        win = saved.get("window")
        if w is None and r is None and win is None:
            return
        set_limits(write=w, read=r, window=win)
        logger.info("Rate limits restored from config.json: %s", get_limits())
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("load_persisted_limits skipped: %s", exc)


@router.get("/ratelimit")
def get_ratelimit() -> JSONResponse:
    return JSONResponse(_effective())


@router.put("/ratelimit")
async def put_ratelimit(request: Request) -> JSONResponse:
    _require_admin(request)
    try:
        raw = await request.json()
    except Exception:
        raw = {}
    try:
        payload = RateLimitUpdate(**(raw or {}))
    except Exception as exc:
        return JSONResponse(status_code=422, content={"error": str(exc)})

    if payload.write is None and payload.read is None and payload.window is None:
        return JSONResponse(
            status_code=422,
            content={"error": "Provide at least one of write, read, window."},
        )

    # Runtime apply (reconfigures every live limiter — no restart).
    effective = set_limits(
        write=payload.write, read=payload.read, window=payload.window,
    )

    # Persist so the override survives restart.
    try:
        cfg = _read_config()
        cfg[_CONFIG_KEY] = {
            "write": effective["write"],
            "read": effective["read"],
            "window": effective["window"],
        }
        _atomic_write(cfg)
    except Exception as exc:
        logger.warning("rate-limit persist failed (applied at runtime): %s", exc)

    return JSONResponse({"success": True, **_effective()})
