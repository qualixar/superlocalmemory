# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Correction cases a user's action overtook: see them, put one back.

When a user deletes, replaces or edits a memory, a correction that SLM itself
proposed and nobody reviewed no longer stands in the way: it is closed as
"overtaken by a user action" (core/overtaken_cases.py). These routes are the
undo surface for that: ``slm corrections overtaken`` and
``slm corrections restore-overtaken CASE_ID`` call them, and the MCP
``list_corrections`` tool gets the same list through ``GET /api/corrections``.

Authorized exactly like listing and reviewing corrections: the caller's role
on the profile named (or the active one), before that profile's existence is
revealed. Another profile's case is "not found", like a missing one. Ids and
lifecycle metadata only; no memory text is ever stored or returned.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["corrections"])

_MAX_LIMIT = 500


class DismissRequest(BaseModel):
    """``slm corrections dismiss``: which pending machine cases, and dry run or not."""

    profile_id: str | None = None
    reasons: list[str] = Field(default_factory=list, max_length=8)
    older_than_days: int | None = Field(default=None, ge=0)
    before: str | None = Field(default=None, max_length=64)
    apply: bool = False


def overtaken_for(engine: Any, profile_id: str, limit: int = 100) -> list[dict]:
    """The profile's overtaken cases, newest first (shared with /api/corrections)."""
    from superlocalmemory.core.overtaken_cases import listing

    return listing(engine._db, profile_id, max(1, min(int(limit), _MAX_LIMIT)))


@router.get("/api/overtaken-corrections")
def list_overtaken(request: Request, limit: int = 100, profile_id: str = ""):
    from superlocalmemory.server.routes import memories as m

    try:
        profile = m._routed_profile(profile_id)
        engine, target = m._authorize_review_read(request, profile)  # READ (G-M5)
        return {"success": True, "profile_id": target,
                "overtaken": overtaken_for(engine, target, limit)}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Overtaken correction list error")


@router.post("/api/overtaken-corrections/{case_id}/restore")
async def restore_overtaken(request: Request, case_id: str):
    from superlocalmemory.core.overtaken_cases import RestoreRefused, restore
    from superlocalmemory.server.routes import memories as m

    try:
        try:
            body = await request.json()
        except Exception:
            body = {}
        profile = m._routed_profile(body.get("profile_id") if isinstance(body, dict) else None)
        engine, target, ctx = m._authorize_memory_mutation(
            request, "update", case_id, run_pre_hook=False, profile=profile)
        try:
            with engine._db.transaction():
                result = restore(engine._db, case_id, profile_id=target,
                                 actor_id=str(ctx.get("agent_id") or ""))
        except RestoreRefused as exc:
            raise HTTPException(409, detail=f"{exc}. Nothing was changed.") from None
        from superlocalmemory.core.mutations import purge_profile_context_cache

        purge_profile_context_cache(engine, target)  # a pending case withholds its successor
        m._announce("memory.updated", {"case_id": case_id, "profile_id": target,
                                       "status": "restored"}, ctx["agent_id"])
        logger.info("overtaken correction case %s restored in %s", case_id[:12], target)
        return {"success": True, "profile_id": target, **result}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Overtaken correction restore error")


def _dismiss_filter(profile: str, body: DismissRequest):
    from superlocalmemory.core.dismissed_cases import DismissFilter, cutoff_for

    try:
        return DismissFilter(profile_id=profile, reasons=tuple(body.reasons),
                             before=cutoff_for(older_than_days=body.older_than_days,
                                               before=body.before))
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc)) from None


def _dismiss_apply(engine: Any, flt: Any, ctx: dict) -> dict:
    """Backup, move in chunks, purge the cache. Only called when cases are eligible."""
    import uuid
    from pathlib import Path

    from superlocalmemory.core.dismissed_cases import dismiss
    from superlocalmemory.core.mutations import purge_profile_context_cache
    from superlocalmemory.storage.save_repair_apply import backup_store

    copy = backup_store(Path(engine._db.db_path))   # the copy `slm ops repair-saves` takes
    result = dismiss(engine._db, flt, actor_id=str(ctx.get("agent_id") or ""),
                     operation_id=f"dismiss-{uuid.uuid4().hex}")
    purge_profile_context_cache(engine, flt.profile_id)  # machine cases made cached text a miss
    return {**result, "backup": str(copy)}


@router.post("/api/overtaken-corrections/dismiss")
def dismiss_corrections(request: Request, body: DismissRequest):
    """Dry run (a READ) or, with ``apply``, move the profile's pending machine cases
    to the overtaken table, restorable one by one. Never rejects a case."""
    from superlocalmemory.core.dismissed_cases import preview
    from superlocalmemory.server.routes import memories as m

    try:
        profile = m._routed_profile(body.profile_id)
        ctx: dict = {}
        if body.apply:
            engine, target, ctx = m._authorize_memory_mutation(
                request, "update", "corrections-dismiss", run_pre_hook=False, profile=profile)
        else:
            engine, target = m._authorize_review_read(request, profile)
        flt = _dismiss_filter(target, body)
        summary = preview(engine._db, flt)
        out = {"success": True, "profile_id": target, "dry_run": not body.apply, **summary}
        if body.apply and summary["eligible"]:
            out.update(_dismiss_apply(engine, flt, ctx))
            m._announce("memory.updated", {"profile_id": target, "status": "dismissed",
                                           "count": out["dismissed"]}, ctx["agent_id"])
        elif body.apply:
            out["dismissed"] = 0
        return out
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Correction dismiss error")


__all__ = ["DismissRequest", "overtaken_for", "router"]
