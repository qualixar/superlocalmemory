# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""What needs a person's answer: open disputes and changed source files (D-37..D-39).

``GET /api/truth/attention`` feeds the dashboard's "Needs your answer" card;
``slm disputes`` and the MCP ``list_disputes`` tool read ``GET /api/disputes``.
``POST /api/disputes/{id}/settle`` is the one door for an answer, from an agent (MCP
``settle_dispute``), the CLI, a script or the dashboard (``truth/disputes.settle``), and
``POST /api/disputes/{id}/reopen`` takes it back. Any caller may decide (D-49); who (an agent
id, a client name, ``dashboard``), the channel and the time come from the connection and the
credential (``server/decision_door.py``), never from a parameter, and are recorded.
``GET /api/audit/pending`` and ``POST /api/audit/{id}/answer|retract`` are the blind audit
queue for any labeller (``core/audit_review``).
``GET /api/truth/facts/{fact_id}`` feeds the memory detail and
``GET /api/truth/checks`` the Health page.
``POST /api/changed-sources/{fact_id}/still-true`` records that someone looked at
a changed file and the memory still holds.

Authorized like correction review: reading needs the review-read role on the
profile, settling the memory-update role. A remote app sees and settles only
disputes whose memories it may see; another side it may not see is withheld.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["truth"])

_MAX_LIMIT = 200


class SettleRequest(BaseModel):
    keep: str = Field(min_length=1, max_length=128)
    #: Legacy fields: the channel comes from the connection, ``by`` is only a client name.
    channel: str = Field(default="", max_length=16)
    by: str = Field(default="", max_length=80)
    profile_id: str | None = None


class ReopenRequest(BaseModel):
    channel: str = Field(default="", max_length=16)
    by: str = Field(default="", max_length=80)
    profile_id: str | None = None


class StillTrueRequest(BaseModel):
    profile_id: str | None = None


def _hidden(engine: Any, profile: str, caller_view: str):
    from superlocalmemory.retrieval import remote_view

    view = remote_view.parse_view(caller_view)
    if not view:
        return lambda ids: set()
    return lambda ids: remote_view.hidden_among(view, engine._db, profile, ids)


def _read(engine: Any):
    from superlocalmemory.truth.invariants import open_read_only

    return open_read_only(str(engine._db.db_path))


def attention(engine: Any, profile: str, limit: int = 50, caller_view: str = "") -> dict:
    """Open disputes and changed or missing source files, for one profile."""
    from superlocalmemory.truth import disputes, leases

    hidden = _hidden(engine, profile, caller_view)
    conn = _read(engine)
    try:
        open_disputes = disputes.listing(conn, profile, limit, hidden=hidden)
        changed = leases.stale(conn, profile, limit)
    finally:
        conn.close()
    blocked = hidden([c["fact_id"] for c in changed]) if changed else set()
    changed = [c for c in changed if c["fact_id"] not in blocked]
    if caller_view:  # a remote app gets only disputes it can see both sides of
        open_disputes = [d for d in open_disputes if all(s["visible"] for s in d["sides"])]
    return {"disputes": open_disputes, "changed_sources": changed,
            "counts": {"disputes": len(open_disputes), "changed_sources": len(changed)}}


@router.get("/api/truth/attention")
def get_attention(request: Request, limit: int = 50, profile_id: str = "", caller_view: str = ""):
    from superlocalmemory.server.routes import memories as m

    try:
        profile = m._routed_profile(profile_id)
        engine, target = m._authorize_review_read(request, profile)
        return {"success": True, "profile_id": target,
                **attention(engine, target, max(1, min(int(limit), _MAX_LIMIT)), caller_view)}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Needs-your-answer list error")


@router.get("/api/disputes")
def list_disputes(request: Request, limit: int = 50, profile_id: str = "", caller_view: str = ""):
    from superlocalmemory.server.routes import memories as m

    try:
        profile = m._routed_profile(profile_id)
        engine, target = m._authorize_review_read(request, profile)
        found = attention(engine, target, max(1, min(int(limit), _MAX_LIMIT)), caller_view)
        return {"success": True, "profile_id": target, "disputes": found["disputes"]}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Dispute list error")


def _sides(engine: Any, dispute_id: str, profile: str) -> list[str]:
    conn = _read(engine)
    try:
        row = conn.execute(
            "SELECT ra.fact_id, rb.fact_id FROM slot_disputes d "
            "JOIN slot_revisions ra ON ra.revision_id = d.revision_a "
            "JOIN slot_revisions rb ON rb.revision_id = d.revision_b "
            "WHERE d.dispute_id = ? AND d.profile_id = ?", (dispute_id, profile)).fetchone()
    except Exception:  # noqa: BLE001 - no tables: not found below
        row = None
    finally:
        conn.close()
    return list(row) if row else []


@router.post("/api/disputes/{dispute_id}/settle")
def settle_dispute(request: Request, dispute_id: str, body: SettleRequest, caller_view: str = ""):
    from superlocalmemory.core.mutations import purge_profile_context_cache
    from superlocalmemory.server import decision_door
    from superlocalmemory.server.routes import memories as m
    from superlocalmemory.truth import disputes

    try:
        profile = m._routed_profile(body.profile_id)
        engine, target, ctx = m._authorize_memory_mutation(
            request, "update", dispute_id, run_pre_hook=False, profile=profile)
        who = decision_door.decider_for(request, str(ctx.get("agent_id") or ""), body.by)
        sides = _sides(engine, dispute_id, target)
        if not sides:
            raise HTTPException(404, detail="Dispute not found")
        m._refuse_if_hidden(caller_view, engine, target, sides, "Dispute not found")
        try:
            with engine._db.raw_connection() as conn:
                result = disputes.settle(conn, profile_id=target, dispute_id=dispute_id,
                                         keep=body.keep, channel=who.channel, actor=who.actor)
        except disputes.SettleRefused as exc:
            raise HTTPException(422, detail=f"{exc}. Nothing was changed.") from None
        except LookupError:
            raise HTTPException(404, detail="Dispute not found (it may be settled already)") from None
        if result["status"] == disputes.STALE:
            raise HTTPException(409, detail=result["message"])
        purge_profile_context_cache(engine, target)
        m._announce("memory.updated", {"dispute_id": dispute_id, "profile_id": target,
                                       "status": "settled"}, ctx["agent_id"])
        logger.info("dispute %s settled in %s (%s)", dispute_id[:12], target, result["resolution"])
        return {"success": True, "profile_id": target, **result}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Dispute settle error")


@router.post("/api/disputes/{dispute_id}/reopen")
def reopen_dispute(request: Request, dispute_id: str, body: ReopenRequest | None = None,
                   caller_view: str = ""):
    """Take a settled dispute's answer back: the retired memory returns, the question is open."""
    from superlocalmemory.core.mutations import purge_profile_context_cache
    from superlocalmemory.server import decision_door
    from superlocalmemory.server.routes import memories as m
    from superlocalmemory.truth import disputes

    body = body or ReopenRequest()
    try:
        profile = m._routed_profile(body.profile_id)
        engine, target, ctx = m._authorize_memory_mutation(
            request, "update", dispute_id, run_pre_hook=False, profile=profile)
        who = decision_door.decider_for(request, str(ctx.get("agent_id") or ""), body.by)
        sides = _sides(engine, dispute_id, target)
        if not sides:
            raise HTTPException(404, detail="No settled dispute of that id")
        m._refuse_if_hidden(caller_view, engine, target, sides, "Dispute not found")
        try:
            with engine._db.raw_connection() as conn:
                result = disputes.reopen(conn, profile_id=target, dispute_id=dispute_id,
                                         channel=who.channel, actor=who.actor)
        except disputes.ReopenRefused as exc:
            raise HTTPException(409, detail=f"{exc} Nothing was changed.") from None
        except LookupError:
            raise HTTPException(404, detail="No settled dispute of that id") from None
        purge_profile_context_cache(engine, target)
        m._announce("memory.updated", {"dispute_id": dispute_id, "profile_id": target,
                                       "status": "reopened"}, ctx["agent_id"])
        return {"success": True, "profile_id": target, **result}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Dispute reopen error")


@router.post("/api/changed-sources/{fact_id}/still-true")
def still_true(request: Request, fact_id: str, body: StillTrueRequest, caller_view: str = ""):
    from superlocalmemory.server.routes import memories as m
    from superlocalmemory.truth import leases

    try:
        profile = m._routed_profile(body.profile_id)
        engine, target, ctx = m._authorize_memory_mutation(
            request, "update", fact_id, run_pre_hook=False, profile=profile)
        m._refuse_if_hidden(caller_view, engine, target, [fact_id], "Memory not found")
        with engine._db.raw_connection() as conn:
            done = leases.mark_still_true(conn, profile_id=target, fact_id=fact_id)
        if not done:
            raise HTTPException(404, detail="No changed source file for this memory (or the file is gone)")
        m._announce("memory.updated", {"fact_id": fact_id, "profile_id": target,
                                       "status": "source_still_true"}, ctx["agent_id"])
        return {"success": True, "profile_id": target, "fact_id": fact_id, "leases": done}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Source check error")


@router.get("/api/truth/facts/{fact_id}")
def fact_truth(request: Request, fact_id: str, profile_id: str = "", caller_view: str = ""):
    """One memory's slots, values over time, open disputes and source file (dashboard detail)."""
    from superlocalmemory.server.routes import memories as m
    from superlocalmemory.truth.fact_truth import for_fact

    try:
        profile = m._routed_profile(profile_id)
        engine, target = m._authorize_review_read(request, profile)
        m._refuse_if_hidden(caller_view, engine, target, [fact_id], "Memory not found")
        conn = _read(engine)
        try:
            found = for_fact(conn, target, fact_id, hidden=_hidden(engine, target, caller_view))
        finally:
            conn.close()
        return {"success": True, "profile_id": target, **found}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Memory truth view error")


@router.get("/api/truth/checks")
def truth_checks(request: Request, profile_id: str = ""):
    """``slm doctor --truth`` for the Health page: the invariants, read-only, store-wide.

    Counts and ids only, never memory text. Run when someone asks (a button),
    not on every page load: on a large store it reads every slot.
    """
    from datetime import UTC, datetime

    from superlocalmemory.server.routes import memories as m
    from superlocalmemory.truth import invariants

    try:
        profile = m._routed_profile(profile_id)
        engine, target = m._authorize_review_read(request, profile)
        from superlocalmemory.core import judge_certificates as certs
        from superlocalmemory.core import shadow_escalation as shadow
        from superlocalmemory.trust import overview, signing_key

        conn = _read(engine)
        try:
            results = invariants.check(conn)
            certificates = certs.records(conn, target)
            second = [s.as_dict() for s in shadow.disagreements(conn, target)]
            # WS-5 (D-16): the daemon's own key (memory only, never the keychain), read-only.
            signature = overview.signature_state(conn, signing_key.current(), signing_key.status())
        finally:
            conn.close()
        audit_chain = overview.audit_state(engine._db.db_path.parent / "audit_chain.db",
                                           signing_key.current())
        return {"success": True, "profile_id": target,
                "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "results": [{"invariant": r.invariant, "title": r.title, "status": r.status,
                             "count": r.count} for r in results],
                "certificates": [{k: c[k] for k in ("tier", "model", "verdict", "population",
                                                   "regime", "n", "errors", "issued_at",
                                                   "voided_at", "void_reason")}
                                 for c in certificates],
                "second_opinions": second,
                "signatures": signature, "audit_chain": audit_chain}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Truth check error")


class AuditAnswer(BaseModel):
    label: str = Field(min_length=1, max_length=32)
    #: The labelling agent's own model id (D-49): its label counts toward a certificate cell
    #: only when this differs from the judged tier's model. Empty: recorded, never counted.
    model: str = Field(default="", max_length=80)
    #: Legacy fields: the channel comes from the connection, ``by`` is only a client name.
    channel: str = Field(default="", max_length=16)
    by: str = Field(default="", max_length=80)
    profile_id: str | None = None


class AuditRetract(BaseModel):
    channel: str = Field(default="", max_length=16)
    by: str = Field(default="", max_length=80)
    profile_id: str | None = None


@router.get("/api/audit/pending")
def audit_pending(request: Request, limit: int = 10, profile_id: str = "", model: str = ""):
    """Pairs drawn for a blind audit, without any judge's verdict or model (W2, R-16).

    ``model`` is the asking agent's own model id: pairs only that model judged are left out,
    since its label could not count (D-49)."""
    from superlocalmemory.core import audit_review
    from superlocalmemory.server.routes import memories as m

    try:
        profile = m._routed_profile(profile_id)
        engine, target = m._authorize_review_read(request, profile)
        conn = _read(engine)
        try:
            items = audit_review.pending(conn, target, limit, labeller_model=model)
            count = audit_review.pending_count(conn, target)
        finally:
            conn.close()
        return {"success": True, "profile_id": target, "count": count, "items": items}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Audit queue error")


@router.post("/api/audit/{item_id}/answer")
def audit_answer(request: Request, item_id: str, body: AuditAnswer):
    """Any caller's blind answer for one pair; written to every drawn receipt of it.

    The labeller kind is ``person`` only for the dashboard's own door, otherwise ``agent``
    (its ``model`` decides whether the label counts toward a certificate)."""
    from superlocalmemory.core import audit_review
    from superlocalmemory.server import decision_door
    from superlocalmemory.server.routes import memories as m

    try:
        profile = m._routed_profile(body.profile_id)
        engine, target, ctx = m._authorize_memory_mutation(
            request, "update", item_id, run_pre_hook=False, profile=profile)
        who = decision_door.decider_for(request, str(ctx.get("agent_id") or ""), body.by)
        kind = "person" if who.channel == "dashboard" else "agent"
        try:
            with engine._db.raw_connection() as conn:
                done = audit_review.answer(conn, profile_id=target, item_id=item_id,
                                           label=body.label, channel=who.channel,
                                           actor=who.actor, kind=kind, model=body.model)
        except LookupError:
            raise HTTPException(404, detail="Nothing waiting for that item") from None
        except ValueError as exc:
            raise HTTPException(422, detail=f"{exc}. Nothing was written.") from None
        return {"success": True, "profile_id": target, "item_id": item_id,
                "labeller_kind": kind, **done}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Audit answer error")


@router.post("/api/audit/{item_id}/retract")
def audit_retract(request: Request, item_id: str, body: AuditRetract | None = None):
    """Take a blind label back: its receipts wait for a label again."""
    from superlocalmemory.core import audit_review
    from superlocalmemory.server import decision_door
    from superlocalmemory.server.routes import memories as m

    body = body or AuditRetract()
    try:
        profile = m._routed_profile(body.profile_id)
        engine, target, ctx = m._authorize_memory_mutation(
            request, "update", item_id, run_pre_hook=False, profile=profile)
        who = decision_door.decider_for(request, str(ctx.get("agent_id") or ""), body.by)
        try:
            with engine._db.raw_connection() as conn:
                done = audit_review.retract(conn, profile_id=target, item_id=item_id,
                                            channel=who.channel, actor=who.actor)
        except LookupError:
            raise HTTPException(404, detail="No label of that item to retract") from None
        return {"success": True, "profile_id": target, "item_id": item_id, **done}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Audit retract error")
