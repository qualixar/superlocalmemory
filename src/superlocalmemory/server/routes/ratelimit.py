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
    _read_config,
    _require_admin,
    _update_config,
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


def _require_read(request: Request) -> None:
    from superlocalmemory.access.rbac import Permission
    from superlocalmemory.server.rbac_enforce import require_permission
    from superlocalmemory.server.routes.helpers import get_active_profile

    require_permission(request, Permission.READ, profile=get_active_profile())


@router.get("/ratelimit")
def get_ratelimit(request: Request) -> JSONResponse:
    _require_read(request)
    return JSONResponse(_effective())


@router.put("/ratelimit")
def put_ratelimit(
    request: Request,
    payload: RateLimitUpdate,
) -> JSONResponse:
    _require_admin(request)

    if payload.write is None and payload.read is None and payload.window is None:
        return JSONResponse(
            status_code=422,
            content={"error": "Provide at least one of write, read, window."},
        )

    current = get_limits()
    requested = {
        "write": payload.write if payload.write is not None else current["write"],
        "read": payload.read if payload.read is not None else current["read"],
        "window": payload.window if payload.window is not None else current["window"],
    }
    # Persist first. A response must never say success for a runtime-only
    # setting that silently disappears after restart.
    try:
        _update_config(
            lambda cfg: cfg.update({_CONFIG_KEY: dict(requested)}),
        )
    except Exception as exc:
        logger.warning("rate-limit persist failed: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": "Rate-limit configuration was not persisted."},
        )

    # Runtime apply reconfigures every live limiter; no restart is required.
    set_limits(**requested)
    return JSONResponse({"success": True, **_effective()})
