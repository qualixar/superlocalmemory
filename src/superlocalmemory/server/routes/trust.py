# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Memories waiting for a promotion (WS-5, the instruction firewall, D-49).

``GET /api/trust/pending`` feeds the dashboard's "Held back from instructions" card and
``slm trust pending``: memories that asked to be a pin, a standing rule, a core block
or part of the session start, and whose origin (a tool's output, a web page, an
agent's own words, anything saved before 4.2.30 and not already a pin or rule) is not
trusted enough to be one by itself (``trust/firewall.py``). ``GET /api/trust/status``
says whether the firewall is acting and the secret screen is on.

``POST /api/trust/promote/{fact_id}`` (also ``/approve/{fact_id}``), ``/approve-all`` and
``POST /api/trust/withdraw/{fact_id}`` are one explicit, recorded, reversible decision by
any caller: an agent, a script, or a person on the dashboard. They need the memory-update
role on the profile, exactly like any write; a caller from another machine is governed by
its role there. Nothing asks whether a person is behind the call: who (an agent id, a
client name, ``dashboard``), the channel and the time come from the connection and the
credential (``server/decision_door.py``), never from a parameter the caller sends, and
are recorded in ``trust_approvals`` and the decision log.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["trust"])

_MAX_LIMIT = 200


class ApproveRequest(BaseModel):
    #: Legacy fields: the channel is taken from the connection, never from here, and ``by`` is
    #: only a client name the caller offers (recorded as ``client:<name>``).
    channel: str = Field(default="", max_length=16)
    by: str = Field(default="", max_length=80)
    profile_id: str | None = None


def _read(engine: Any):
    from superlocalmemory.truth.invariants import open_read_only

    return open_read_only(str(engine._db.db_path))


@router.get("/api/trust/status")
def trust_status(request: Request, profile_id: str = ""):
    """Whether the firewall acts and the secret screen is on, with the counts."""
    from superlocalmemory.server.routes import memories as m
    from superlocalmemory.trust import firewall

    try:
        profile = m._routed_profile(profile_id)
        engine, target = m._authorize_review_read(request, profile)
        conn = _read(engine)
        try:
            counts = firewall.counts(conn, target)
            unclassified = firewall.pending(conn, target, 1)["unclassified"]
        finally:
            conn.close()
        from superlocalmemory.core.feature_flags import FeatureFlags
        from superlocalmemory.infra.data_root import canonical_data_root

        flags = FeatureFlags(canonical_data_root())
        return {"success": True, "profile_id": target, **counts, "unclassified": unclassified,
                "taint": flags.state("trust.taint").value,
                "secret_screen": flags.state("trust.secret_screen").value,
                "floor": firewall.FLOOR}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Trust status error")


@router.get("/api/trust/pending")
def trust_pending(request: Request, limit: int = 50, profile_id: str = ""):
    """Memories waiting for a promotion, newest first."""
    from superlocalmemory.server.routes import memories as m
    from superlocalmemory.trust import firewall

    try:
        profile = m._routed_profile(profile_id)
        engine, target = m._authorize_review_read(request, profile)
        conn = _read(engine)
        try:
            found = firewall.pending(conn, target, max(1, min(int(limit), _MAX_LIMIT)))
        finally:
            conn.close()
        return {"success": True, "profile_id": target, "acting": firewall.active(),
                "state": firewall.state(), **found}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Trust queue error")


def _decide(request: Request, fact_id: str | None, body: ApproveRequest, action: str):
    from superlocalmemory.core.mutations import purge_profile_context_cache
    from superlocalmemory.server import decision_door
    from superlocalmemory.server.routes import memories as m
    from superlocalmemory.trust import firewall

    try:
        profile = m._routed_profile(body.profile_id)
        engine, target, ctx = m._authorize_memory_mutation(
            request, "update", fact_id or "approve-all", run_pre_hook=False, profile=profile)
        who = decision_door.decider_for(request, str(ctx.get("agent_id") or ""), body.by)
        try:
            with engine._db.raw_connection() as conn:
                if fact_id is not None:
                    owner = conn.execute("SELECT 1 FROM atomic_facts WHERE fact_id = ? AND "
                                         "profile_id = ?", (fact_id, target)).fetchone()
                    if owner is None:
                        raise LookupError(fact_id)
                if action == "withdraw":
                    done = firewall.withdraw(conn, fact_id, by=who.actor, channel=who.channel)
                elif fact_id is None:
                    done = firewall.approve_all(conn, target, by=who.actor, channel=who.channel)
                else:
                    done = firewall.approve(conn, fact_id, by=who.actor, channel=who.channel)
        except LookupError:
            raise HTTPException(404, detail="No such memory, or nothing to withdraw") from None
        except ValueError as exc:
            raise HTTPException(422, detail=f"{exc}. Nothing was changed.") from None
        purge_profile_context_cache(engine, target)
        m._announce("memory.updated", {"fact_id": fact_id or "", "profile_id": target,
                                       "status": f"trust_{action}"}, ctx["agent_id"])
        logger.info("trust %s: %s in %s via %s", action, fact_id or "all", target, who.channel)
        return {"success": True, "profile_id": target, **done}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Trust decision error")


@router.post("/api/trust/promote/{fact_id}")
@router.post("/api/trust/approve/{fact_id}")
def promote_one(request: Request, fact_id: str, body: ApproveRequest | None = None):
    """Promote one memory on purpose: it becomes ``user_confirmed`` and may be an instruction."""
    return _decide(request, fact_id, body or ApproveRequest(), "promote")


@router.post("/api/trust/approve-all")
@router.post("/api/trust/promote-all")
def promote_everything(request: Request, body: ApproveRequest | None = None):
    """Promote everything the waiting list shows (one explicit call, each one recorded)."""
    return _decide(request, None, body or ApproveRequest(), "promote")


@router.post("/api/trust/withdraw/{fact_id}")
def withdraw_one(request: Request, fact_id: str, body: ApproveRequest | None = None):
    """Take a promotion back: the memory returns to its own class (and loses an applied pin)."""
    return _decide(request, fact_id, body or ApproveRequest(), "withdraw")
