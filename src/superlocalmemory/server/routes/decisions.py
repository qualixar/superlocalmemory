# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Recent decisions by your agents, and the Undo for each (4.2.30, D-49).

``GET /api/decisions`` feeds the dashboard card "Recent decisions by your agents": who decided
what, through which channel and when (every promotion the instruction firewall allowed, every
dispute settled, every set of proposed changes applied or rolled back, every blind audit label),
newest first, with a flag for the ones that can still be undone. It needs the review-read role.
``POST /api/decisions/{id}/undo`` takes one back through the same door that made it (a promotion
is withdrawn, a set rolled back, a dispute reopened, a label retracted); it needs the
memory-update role like any write, and is itself recorded with who undid it and the channel.
Nothing here asks whether a person is calling: any caller may undo, and a caller from another
machine is governed by its role.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["decisions"])

_MAX_LIMIT = 200
_PREVIEW = 120


class UndoRequest(BaseModel):
    channel: str = Field(default="", max_length=16)
    by: str = Field(default="", max_length=80)
    profile_id: str | None = None


def _read(engine: Any):
    from superlocalmemory.truth.invariants import open_read_only

    return open_read_only(str(engine._db.db_path))


def _previews(conn: Any, found: list[dict]) -> None:
    """Add a one-line ``preview`` of the memory a promotion was about (read-only, cleaned)."""
    from superlocalmemory.trust.firewall import _clean

    ids = [d["subject_id"] for d in found if d["kind"] == "promotion"]
    if not ids:
        return
    marks = ",".join("?" * len(ids))
    try:
        texts = dict(conn.execute(
            f"SELECT fact_id, content FROM atomic_facts WHERE fact_id IN ({marks})",
            ids).fetchall())
    except Exception:  # noqa: BLE001 - a missing preview is never a reason to hide the decision
        texts = {}
    for d in found:
        if d["kind"] == "promotion" and d["subject_id"] in texts:
            d["preview"] = _clean(texts[d["subject_id"]], _PREVIEW)


@router.get("/api/decisions")
def list_decisions(request: Request, limit: int = 50, profile_id: str = ""):
    """The profile's recent decisions, newest first, with counts for Memory Health."""
    from superlocalmemory.core import decisions
    from superlocalmemory.server.routes import memories as m

    try:
        profile = m._routed_profile(profile_id)
        engine, target = m._authorize_review_read(request, profile)
        conn = _read(engine)
        try:
            found = decisions.recent(conn, target, max(1, min(int(limit), _MAX_LIMIT)))
            _previews(conn, found)
            counts = decisions.counts(conn, target)
        finally:
            conn.close()
        return {"success": True, "profile_id": target, "counts": counts, "decisions": found}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Decisions list error")


@router.post("/api/decisions/{decision_id}/undo")
def undo_decision(request: Request, decision_id: str, body: UndoRequest | None = None):
    """Take one decision back through the same door that made it. Recorded."""
    from superlocalmemory.core import decision_undo
    from superlocalmemory.core.mutations import purge_profile_context_cache
    from superlocalmemory.server import decision_door
    from superlocalmemory.server.routes import memories as m

    body = body or UndoRequest()
    try:
        profile = m._routed_profile(body.profile_id)
        engine, target, ctx = m._authorize_memory_mutation(
            request, "update", decision_id, run_pre_hook=False, profile=profile)
        who = decision_door.decider_for(request, str(ctx.get("agent_id") or ""), body.by)
        try:
            with engine._db.raw_connection() as conn:
                done = decision_undo.undo(conn, profile_id=target, decision_id=decision_id,
                                          decider=who)
        except LookupError:
            raise HTTPException(404, detail="No such decision") from None
        except decision_undo.UndoRefused as exc:
            raise HTTPException(409, detail=f"{exc} Nothing was changed.") from None
        purge_profile_context_cache(engine, target)
        m._announce("memory.updated", {"decision_id": decision_id, "profile_id": target,
                                       "status": "decision_undone"}, ctx["agent_id"])
        logger.info("decision %s undone in %s via %s", decision_id[:12], target, who.channel)
        return {"success": True, "profile_id": target, **done}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Decision undo error")
