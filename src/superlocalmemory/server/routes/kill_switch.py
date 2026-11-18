# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The kill switch over HTTP (D-23, D-49, blueprint §8.10.3 (k)).

``GET /api/flags/kill``: is everything new in 4.2.30 turned off, where that comes from, what it
does and does not do in plain words, and how many retired memories "Bring back memories that
4.2.30 retired" would restore. It needs the review-read role, like every other listing.

``POST /api/flags/kill`` ``{"on": true|false}`` turns the switch on or off. It needs the
memory-update role on the profile like any write: a caller on this computer is allowed, a caller
from another machine by its role (``rbac_enforce.require_permission``), and nothing asks whether a
person is calling.

``POST /api/flags/rollback-retirements`` ``{"dry_run": true|false}`` brings back every memory the
truth layer retired (``truth/retirement_rollback``). With ``dry_run`` it only counts. Otherwise it
is one recorded decision (who, channel, when) in the decision log, undoable from there. Works with
the switch on or off: it is the separate command the switch does not do. Ids and counts only.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["flags"])


class KillBody(BaseModel):
    on: bool


class RollbackBody(BaseModel):
    dry_run: bool = False
    by: str = Field(default="", max_length=80)
    channel: str = Field(default="", max_length=16)  # legacy: the channel comes from the connection
    profile_id: str | None = None


def _read(engine: Any):
    from superlocalmemory.truth.invariants import open_read_only

    return open_read_only(str(engine._db.db_path))


def _count(engine: Any, profile_id: str) -> int:
    from superlocalmemory.truth import retirement_rollback

    conn = _read(engine)
    try:
        return retirement_rollback.would_come_back(conn, profile_id)
    except Exception:  # noqa: BLE001 - a store before the ledger has nothing to bring back
        return 0
    finally:
        conn.close()


def _body(home: Any, engine: Any, profile_id: str) -> dict:
    from superlocalmemory.core import kill_switch

    return {"success": True, "kill_switch": kill_switch.view(home), "does": kill_switch.DOES_WORDS,
            "does_not": kill_switch.DOES_NOT_WORDS, "profile_id": profile_id,
            "would_come_back": _count(engine, profile_id)}


@router.get("/api/flags/kill")
def get_kill(request: Request, profile_id: str = ""):
    from superlocalmemory.infra.data_root import canonical_data_root
    from superlocalmemory.server.routes import memories as m

    try:
        engine, target = m._authorize_review_read(request, m._routed_profile(profile_id))
        return _body(canonical_data_root(), engine, target)
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Kill switch read error")


@router.post("/api/flags/kill")
def post_kill(request: Request, body: KillBody):
    from superlocalmemory.core import kill_switch
    from superlocalmemory.infra.data_root import canonical_data_root
    from superlocalmemory.server import decision_door
    from superlocalmemory.server.routes import memories as m

    try:
        engine, target, ctx = m._authorize_memory_mutation(
            request, "update", "kill-switch", run_pre_hook=False, profile=m._routed_profile(None))
        who = decision_door.decider_for(request, str(ctx.get("agent_id") or ""), "")
        home = canonical_data_root()
        state = kill_switch.set_state(home, body.on)
        with engine._db.raw_connection() as conn:  # D-49: recorded, listed, undoable
            kill_switch.record(conn, profile_id=target, on=body.on, decider=who)
        logger.info("kill switch %s by %s via %s", "on" if body.on else "off", who.actor, who.channel)
        return {**_body(home, engine, target), "kill_switch": state}
    except HTTPException:
        raise
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Kill switch change error")


@router.post("/api/flags/rollback-retirements")
def post_rollback(request: Request, body: RollbackBody | None = None):
    from superlocalmemory.core.mutations import purge_profile_context_cache
    from superlocalmemory.server import decision_door
    from superlocalmemory.server.routes import memories as m
    from superlocalmemory.truth import retirement_rollback

    body = body or RollbackBody()
    try:
        profile = m._routed_profile(body.profile_id)
        engine, target, ctx = m._authorize_memory_mutation(
            request, "update", "rollback-retirements", run_pre_hook=False, profile=profile)
        who = decision_door.decider_for(request, str(ctx.get("agent_id") or ""), body.by)
        with engine._db.raw_connection() as conn:
            done = retirement_rollback.run(conn, profile_id=target, decider=who,
                                           dry_run=body.dry_run)
        if not body.dry_run:
            purge_profile_context_cache(engine, target)
            m._announce("memory.updated", {"profile_id": target, "status": "retirements_rolled_back"},
                        ctx["agent_id"])
            logger.info("%d retired memories brought back in %s via %s", done["came_back"], target,
                        who.channel)
        return {"success": True, "profile_id": target, **done}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Bring back retired memories error")
