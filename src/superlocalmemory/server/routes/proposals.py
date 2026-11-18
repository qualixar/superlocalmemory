# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Proposed changes to memory: review, apply, reject and undo (WS-6, I-10).

``GET /api/proposals`` and ``GET /api/proposals/{set_id}`` feed the dashboard's
"Proposed changes" card and ``slm proposals``. ``POST .../apply``, ``.../reject``
and ``.../rollback`` are one decision by any caller (an agent, a script, a person on the
dashboard; D-49). They need the memory-update role on the profile, like any write; a caller
from another machine is governed by its role there. Who (an agent id, a client name,
``dashboard``), the channel and the time come from the connection and the credential
(``server/decision_door.py``), never from a parameter, and are recorded as the basis (I-10)
and in the decision log. An applied set is undone with ``rollback``. Nothing here applies on a
certificate; that is the compiler's own path behind ``compiler.auto_apply``.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["proposals"])

_MAX_LIMIT = 200


class DecisionRequest(BaseModel):
    #: Legacy: the channel comes from the connection, never from here; ``by`` is only a
    #: client name the caller offers (recorded as ``client:<name>``).
    channel: str = Field(default="", max_length=16)
    by: str = Field(default="", max_length=80)
    profile_id: str | None = None


def _read(engine: Any):
    from superlocalmemory.truth.invariants import open_read_only

    return open_read_only(str(engine._db.db_path))


def _refusal(exc: Exception) -> HTTPException:
    from superlocalmemory.compiler.proposals import BasisRefused, StateRefused

    if isinstance(exc, StateRefused):
        return HTTPException(409, detail=str(exc))
    if isinstance(exc, BasisRefused):
        return HTTPException(403, detail=str(exc))
    return HTTPException(422, detail=f"{exc}. Nothing was changed.")


@router.get("/api/proposals")
def list_proposals(request: Request, state: str = "proposed", limit: int = 50,
                   profile_id: str = ""):
    """Sets of proposed changes, newest first (``state=all`` for every state)."""
    from superlocalmemory.compiler import proposals
    from superlocalmemory.server.routes import memories as m

    try:
        profile = m._routed_profile(profile_id)
        engine, target = m._authorize_review_read(request, profile)
        states = None if state == "all" else [s for s in state.split(",") if s]
        conn = _read(engine)
        try:
            sets = proposals.list_sets(conn, target, states=states,
                                       limit=max(1, min(int(limit), _MAX_LIMIT)))
            count = proposals.count_pending(conn, target)
        finally:
            conn.close()
        return {"success": True, "profile_id": target, "count": count, "sets": sets}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Proposed changes list error")


@router.get("/api/proposals/{set_id}")
def show_proposal(request: Request, set_id: str, profile_id: str = ""):
    from superlocalmemory.compiler import proposals
    from superlocalmemory.server.routes import memories as m

    try:
        profile = m._routed_profile(profile_id)
        engine, target = m._authorize_review_read(request, profile)
        conn = _read(engine)
        try:
            found = proposals.get_set(conn, target, set_id)
        except LookupError:
            raise HTTPException(404, detail="No such proposed change") from None
        finally:
            conn.close()
        return {"success": True, "profile_id": target, "set": found}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Proposed change error")


def _decide(request: Request, set_id: str, body: DecisionRequest, action: str):
    from superlocalmemory.compiler import proposals
    from superlocalmemory.core.mutations import purge_profile_context_cache
    from superlocalmemory.server import decision_door
    from superlocalmemory.server.routes import memories as m

    try:
        profile = m._routed_profile(body.profile_id)
        engine, target, ctx = m._authorize_memory_mutation(
            request, "update", set_id, run_pre_hook=False, profile=profile)
        who = decision_door.decider_for(request, str(ctx.get("agent_id") or ""), body.by)
        actor = who.actor
        basis = proposals.Basis.decision(who.actor, who.channel)
        try:
            with engine._db.raw_connection() as conn:
                if action == "apply":
                    done = proposals.apply_set(conn, profile_id=target, set_id=set_id, basis=basis)
                elif action == "rollback":
                    done = proposals.rollback_set(conn, profile_id=target, set_id=set_id,
                                                  basis=basis)
                else:
                    done = proposals.reject_set(conn, profile_id=target, set_id=set_id,
                                                actor=actor, channel=who.channel)
        except LookupError:
            raise HTTPException(404, detail="No such proposed change") from None
        except proposals.ProposalError as exc:
            raise _refusal(exc) from None
        if action != "reject":
            purge_profile_context_cache(engine, target)
        m._announce("memory.updated", {"set_id": set_id, "profile_id": target,
                                       "status": f"proposal_{action}"}, ctx["agent_id"])
        logger.info("proposal set %s: %s in %s", set_id[:12], action, target)
        return {"success": True, "profile_id": target, "set": done}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Proposed change error")


@router.post("/api/proposals/{set_id}/apply")
def apply_proposal(request: Request, set_id: str, body: DecisionRequest | None = None):
    """Any caller applies a set; all of it or none of it. Recorded; undo with rollback."""
    return _decide(request, set_id, body or DecisionRequest(), "apply")


@router.post("/api/proposals/{set_id}/reject")
def reject_proposal(request: Request, set_id: str, body: DecisionRequest | None = None):
    """Any caller declines a set; no memory changes."""
    return _decide(request, set_id, body or DecisionRequest(), "reject")


@router.post("/api/proposals/{set_id}/rollback")
def rollback_proposal(request: Request, set_id: str, body: DecisionRequest | None = None):
    """Any caller undoes an applied set exactly."""
    return _decide(request, set_id, body or DecisionRequest(), "rollback")
