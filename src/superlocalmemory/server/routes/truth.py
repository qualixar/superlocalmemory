# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""What needs a person's answer: open disputes and changed source files (D-37..D-39).

``GET /api/truth/attention`` feeds the dashboard's "Needs your answer" card;
``slm disputes`` and the MCP ``list_disputes`` tool read ``GET /api/disputes``.
``POST /api/disputes/{id}/settle`` is the one door for a person's answer, from an
agent (MCP ``settle_dispute``), the CLI or the dashboard (``truth/disputes.settle``).
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
    channel: str = Field(default="http", max_length=16)
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
    from superlocalmemory.server.routes import memories as m
    from superlocalmemory.truth import disputes

    try:
        profile = m._routed_profile(body.profile_id)
        engine, target, ctx = m._authorize_memory_mutation(
            request, "update", dispute_id, run_pre_hook=False, profile=profile)
        sides = _sides(engine, dispute_id, target)
        if not sides:
            raise HTTPException(404, detail="Dispute not found")
        m._refuse_if_hidden(caller_view, engine, target, sides, "Dispute not found")
        try:
            with engine._db.raw_connection() as conn:
                result = disputes.settle(conn, profile_id=target, dispute_id=dispute_id,
                                         keep=body.keep, channel=body.channel,
                                         actor=str(ctx.get("agent_id") or ""))
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
        conn = _read(engine)
        try:
            results = invariants.check(conn)
        finally:
            conn.close()
        return {"success": True, "profile_id": target,
                "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "results": [{"invariant": r.invariant, "title": r.title, "status": r.status,
                             "count": r.count} for r in results]}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Truth check error")


class AuditAnswer(BaseModel):
    label: str = Field(min_length=1, max_length=32)
    channel: str = Field(default="dashboard", max_length=16)
    profile_id: str | None = None


def _local_only(caller_view: str) -> None:
    """Audits are a person on this machine (core/audit_review.py): never a remote app."""
    if (caller_view or "").strip():
        raise HTTPException(403, detail="Audits are answered on this computer only")


@router.get("/api/audit/pending")
def audit_pending(request: Request, limit: int = 10, profile_id: str = "", caller_view: str = ""):
    """Pairs drawn for a blind audit, without any judge's verdict (W2, R-16)."""
    from superlocalmemory.core import audit_review
    from superlocalmemory.server.routes import memories as m

    _local_only(caller_view)
    try:
        profile = m._routed_profile(profile_id)
        engine, target = m._authorize_review_read(request, profile)
        conn = _read(engine)
        try:
            items = audit_review.pending(conn, target, limit)
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
def audit_answer(request: Request, item_id: str, body: AuditAnswer, caller_view: str = ""):
    """A person's blind answer for one pair; written to every drawn receipt of it."""
    from superlocalmemory.core import audit_review
    from superlocalmemory.server.routes import memories as m

    _local_only(caller_view)
    try:
        profile = m._routed_profile(body.profile_id)
        engine, target, ctx = m._authorize_memory_mutation(
            request, "update", item_id, run_pre_hook=False, profile=profile)
        try:
            with engine._db.raw_connection() as conn:
                written = audit_review.answer(conn, profile_id=target, item_id=item_id,
                                              label=body.label, channel=body.channel,
                                              actor=str(ctx.get("agent_id") or ""))
        except LookupError:
            raise HTTPException(404, detail="Nothing waiting for that item") from None
        except ValueError as exc:
            raise HTTPException(422, detail=f"{exc}. Nothing was written.") from None
        return {"success": True, "profile_id": target, "item_id": item_id, "labelled": written}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Audit answer error")
