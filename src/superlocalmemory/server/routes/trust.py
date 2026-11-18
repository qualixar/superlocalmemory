# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Memories waiting for a person's OK (WS-5, the instruction firewall, D-37).

``GET /api/trust/pending`` feeds the dashboard's "Waiting for your OK" card and
``slm trust pending``: memories that asked to be a pin, a standing rule, a core block
or part of the session start, and whose origin (a tool's output, a web page, an
agent's own words, anything saved before 4.2.30) is not trusted enough to be one
unasked (``trust/firewall.py``). ``GET /api/trust/status`` says whether the firewall
is acting and the secret screen is on.

``POST /api/trust/approve/{fact_id}`` and ``POST /api/trust/approve-all`` are a
person's answer. They need the memory-update role on the profile and refuse any
remote app, like the audit answers (``routes/truth.py``). On top of that the door must
be one a person uses: the dashboard (the install token its page carries), or the
command line **from an interactive terminal** (the client says whether it had one,
``agent_attribution.SHELL_HEADER``; an MCP tool call, a hook or a script says no
terminal and is refused). An agent can read the list and cannot approve. A terminal is
evidence, not proof (the audit routes have the same limit): the record keeps who, when
and through which door.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["trust"])

_MAX_LIMIT = 200
_WHO = re.compile(r"[^A-Za-z0-9_.@ -]")


class ApproveRequest(BaseModel):
    channel: str = Field(default="dashboard", max_length=16)
    by: str = Field(default="", max_length=80)
    profile_id: str | None = None


def _local_only(caller_view: str) -> None:
    """A person on this computer decides; never a remote app."""
    if (caller_view or "").strip():
        raise HTTPException(403, detail="Approvals are given on this computer only")


def _read(engine: Any):
    from superlocalmemory.truth.invariants import open_read_only

    return open_read_only(str(engine._db.db_path))


def _person_door(request: Request, channel: str, actor_id: str) -> None:
    """Refuse a door an agent could be using. See the module text for the limit."""
    from superlocalmemory.core.agent_attribution import SHELL_HEADER, SHELL_TTY, shell_signal

    if channel == "dashboard":
        if str(actor_id).startswith("daemon-capability:"):
            raise HTTPException(403, detail="The dashboard's door is the page itself; "
                                            "use `slm trust approve` at a terminal instead")
        return
    if channel == "cli":
        signal = shell_signal(request.headers.get(SHELL_HEADER, ""), actor_id)
        if signal != SHELL_TTY:
            raise HTTPException(403, detail="Approving asks a person at a terminal; it cannot "
                                            "be done by a script or an agent")
        return
    raise HTTPException(422, detail="channel must be 'dashboard' or 'cli'")


def _who(body: ApproveRequest) -> str:
    return " ".join(_WHO.sub("", body.by).split())[:80] or "local user"


@router.get("/api/trust/status")
def trust_status(request: Request, profile_id: str = "", caller_view: str = ""):
    """Whether the firewall acts and the secret screen is on, with the counts."""
    from superlocalmemory.server.routes import memories as m
    from superlocalmemory.trust import firewall

    _local_only(caller_view)
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
def trust_pending(request: Request, limit: int = 50, profile_id: str = "", caller_view: str = ""):
    """Memories waiting for a person's OK, newest first (read by agents too)."""
    from superlocalmemory.server.routes import memories as m
    from superlocalmemory.trust import firewall

    _local_only(caller_view)
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


def _approve(request: Request, fact_id: str | None, body: ApproveRequest, caller_view: str):
    from superlocalmemory.core.mutations import purge_profile_context_cache
    from superlocalmemory.server.routes import memories as m
    from superlocalmemory.trust import firewall

    _local_only(caller_view)
    try:
        profile = m._routed_profile(body.profile_id)
        engine, target, ctx = m._authorize_memory_mutation(
            request, "update", fact_id or "approve-all", run_pre_hook=False, profile=profile)
        _person_door(request, body.channel, str(ctx.get("agent_id") or ""))
        who = _who(body)
        try:
            with engine._db.raw_connection() as conn:
                if fact_id is None:
                    done = firewall.approve_all(conn, target, by=who, channel=body.channel)
                else:
                    owner = conn.execute("SELECT 1 FROM atomic_facts WHERE fact_id = ? AND "
                                         "profile_id = ?", (fact_id, target)).fetchone()
                    if owner is None:
                        raise LookupError(fact_id)
                    done = firewall.approve(conn, fact_id, by=who, channel=body.channel)
        except LookupError:
            raise HTTPException(404, detail="No such memory") from None
        except ValueError as exc:
            raise HTTPException(422, detail=f"{exc}. Nothing was changed.") from None
        purge_profile_context_cache(engine, target)
        m._announce("memory.updated", {"fact_id": fact_id or "", "profile_id": target,
                                       "status": "trust_approved"}, ctx["agent_id"])
        logger.info("trust approval: %s in %s via %s", fact_id or "all", target, body.channel)
        return {"success": True, "profile_id": target, **done}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Trust approval error")


@router.post("/api/trust/approve/{fact_id}")
def approve_one(request: Request, fact_id: str, body: ApproveRequest, caller_view: str = ""):
    """A person approves one memory: it becomes `user_confirmed` and may be an instruction."""
    return _approve(request, fact_id, body, caller_view)


@router.post("/api/trust/approve-all")
def approve_everything(request: Request, body: ApproveRequest, caller_view: str = ""):
    """A person approves everything the list shows (one answer for a store's old memories)."""
    return _approve(request, None, body, caller_view)
