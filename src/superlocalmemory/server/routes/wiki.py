# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The living wiki: compiled pages to read, and the Markdown projection (WS-6, D-17, D-18).

Reads (``GET /api/wiki/...``) need the review-read role and refuse any remote app: a page
quotes memory. ``POST /api/wiki/build`` projects the approved pages to the Markdown folder
(behind ``compiler.wiki``; in shadow it only says what it would write). ``POST
/api/wiki/rebuild`` and ``POST /api/wiki/sync`` are one decision by any caller (D-49): they
compile the pages again and apply them (recorded with who and the channel, undone with the
proposals rollback), and read edited Markdown back as proposals, so they need the
memory-update role; who and the channel come from the connection.
No request names a path: the folder is ``SLM_WIKI_DIR`` or ``<data folder>/wiki``, so the
daemon never writes where a caller says.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["wiki"])

_MAX_EDITS = 50


class WikiRequest(BaseModel):
    #: Legacy: the channel comes from the connection (``server/decision_door.py``, D-49).
    channel: str = Field(default="", max_length=16)
    by: str = Field(default="", max_length=80)
    profile_id: str | None = None


def _read(engine: Any):
    from superlocalmemory.truth.invariants import open_read_only

    return open_read_only(str(engine._db.db_path))


def _flag() -> str:
    from superlocalmemory.core.feature_flags import FeatureFlags
    from superlocalmemory.infra.data_root import canonical_data_root

    return FeatureFlags(canonical_data_root()).state("compiler.wiki").value


def _person(request: Request, body: WikiRequest, ctx: dict, target: str):
    """The basis of a caller's recorded decision: who and the channel, from the connection (I-10)."""
    from superlocalmemory.compiler import proposals
    from superlocalmemory.server import decision_door

    who = decision_door.decider_for(request, str(ctx.get("agent_id") or ""), body.by)
    basis = proposals.Basis.decision(who.actor, who.channel)
    try:
        proposals.check_basis(None, target, basis, [])
    except proposals.BasisRefused as exc:
        raise HTTPException(403, detail=str(exc)) from None
    return basis


@router.get("/api/wiki/pages")
def list_pages(request: Request, kind: str = "", profile_id: str = ""):
    """The compiled pages of a profile (titles and counts, no text) and the wiki's status."""
    from superlocalmemory.compiler import pages, wiki
    from superlocalmemory.server.routes import memories as m

    try:
        profile = m._routed_profile(profile_id)
        engine, target = m._authorize_review_read(request, profile)
        conn = _read(engine)
        try:
            listed = pages.list_pages(conn, target, kind=kind)
            status = wiki.status(conn, target, wiki.wiki_root())
        finally:
            conn.close()
        return {"success": True, "profile_id": target, "flag": _flag(), "status": status,
                "pages": listed}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Living wiki list error")


@router.get("/api/wiki/pages/{page_id}")
def show_page(request: Request, page_id: str, profile_id: str = ""):
    """One page with every sentence re-checked now, its citations and what it withheld."""
    from superlocalmemory.compiler import pages
    from superlocalmemory.server.routes import memories as m

    try:
        profile = m._routed_profile(profile_id)
        engine, target = m._authorize_review_read(request, profile)
        conn = _read(engine)
        try:
            found = pages.show_page(conn, target, page_id)
        finally:
            conn.close()
        if found is None:
            raise HTTPException(404, detail="No such page")
        return {"success": True, "profile_id": target, "page": found}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Living wiki page error")


@router.get("/api/wiki/diff")
def folder_diff(request: Request, profile_id: str = ""):
    """What a build would change in the folder, and which files hold edits waiting for sync."""
    from superlocalmemory.compiler import wiki
    from superlocalmemory.server.routes import memories as m

    try:
        profile = m._routed_profile(profile_id)
        engine, target = m._authorize_review_read(request, profile)
        conn = _read(engine)
        try:
            found = wiki.diff(conn, target, wiki.wiki_root())
        finally:
            conn.close()
        return {"success": True, "profile_id": target, "flag": _flag(), **found}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Living wiki diff error")


def _gate(request: Request, body: WikiRequest, *, person: bool):
    """Authorise a write: ``(engine, target, ctx, basis)``; ``basis`` only for a deciding route."""
    from superlocalmemory.server.routes import memories as m

    profile = m._routed_profile(body.profile_id)
    engine, target, ctx = m._authorize_memory_mutation(
        request, "update", "wiki", run_pre_hook=False, profile=profile)
    basis = None
    if person:
        basis = _person(request, body, ctx, target)
    return engine, target, ctx, basis


def _guard(exc: Exception, label: str):
    """The HTTP error for an exception from a write: the proposal rules' own words first."""
    from superlocalmemory.compiler.proposals import BasisRefused, ProposalError, StateRefused
    from superlocalmemory.server.routes import memories as m

    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, StateRefused):
        return HTTPException(409, detail=str(exc))
    if isinstance(exc, BasisRefused):
        return HTTPException(403, detail=str(exc))
    if isinstance(exc, ProposalError):
        return HTTPException(422, detail=f"{exc}. Nothing was changed.")
    return m._canonical_mutation_error(exc, label)


@router.post("/api/wiki/build")
def build_wiki(request: Request, body: WikiRequest):
    """Project the approved pages to the Markdown folder (shadow: report only; off: refused)."""
    from superlocalmemory.compiler import wiki

    try:
        engine, target, _ctx, _basis = _gate(request, body, person=False)
        state = _flag()
        if state == "off":
            raise HTTPException(409, detail="The Markdown wiki is off. Turn it on with: "
                                            "slm flags set compiler.wiki on")
        reader = _read(engine)
        try:
            with engine._db.raw_connection() as writer:
                report = wiki.build(reader, writer, target, wiki.wiki_root(),
                                    dry_run=state == "shadow")
        finally:
            reader.close()
        return {"success": True, "profile_id": target, "flag": state, **report}
    except HTTPException:
        raise
    except Exception as exc:
        raise _guard(exc, "Living wiki build error")


@router.post("/api/wiki/rebuild")
def rebuild_pages(request: Request, body: WikiRequest):
    """A caller compiles the pages again from memory and applies them in one recorded step (I-10);
    the Markdown folder follows when ``compiler.wiki`` is on."""
    from superlocalmemory.compiler import dream, proposals, wiki

    try:
        engine, target, _ctx, basis = _gate(request, body, person=True)
        reader = _read(engine)
        try:
            planned = dream.plan(reader, target, merges=False)
        finally:
            reader.close()
        out: dict[str, Any] = {"set_id": None, "pages": 0, "report": planned["report"]}
        if planned["items"]:
            with engine._db.raw_connection() as conn:
                set_id = proposals.create_set(conn, profile_id=target, origin="rebuild",
                                              title="Pages rebuilt",
                                              proposals=planned["items"])
                proposals.apply_set(conn, profile_id=target, set_id=set_id, basis=basis)
            out.update(set_id=set_id, pages=len(planned["items"]))
        files = None
        state = _flag()
        if state != "off":
            reader = _read(engine)
            try:
                with engine._db.raw_connection() as writer:
                    files = wiki.build(reader, writer, target, wiki.wiki_root(),
                                       dry_run=state == "shadow")
            finally:
                reader.close()
        return {"success": True, "profile_id": target, "flag": state, "files": files, **out}
    except HTTPException:
        raise
    except Exception as exc:
        raise _guard(exc, "Living wiki rebuild error")


@router.post("/api/wiki/sync")
def sync_wiki(request: Request, body: WikiRequest):
    """A caller reads the edited Markdown back: each edited block becomes a proposal, never a
    write. The words pass normal admission (secret screen, origin ``external_edit``)."""
    from superlocalmemory.compiler import wiki
    from superlocalmemory.core.mutations import update_fact_authorized
    from superlocalmemory.server.routes import memories as m

    try:
        engine, target, _ctx, _basis = _gate(request, body, person=True)
        state = _flag()
        if state == "off":
            raise HTTPException(409, detail="The Markdown wiki is off. Turn it on with: "
                                            "slm flags set compiler.wiki on")
        reader = _read(engine)
        try:
            plan = wiki.sync_plan(reader, target, wiki.wiki_root())
        finally:
            reader.close()
        edits = plan.pop("edits")[:_MAX_EDITS]
        summary = {"found": len(edits), "dry_run": state == "shadow", **plan}
        if state == "shadow" or not edits:
            return {"success": True, "profile_id": target, "flag": state, "queued": [],
                    "skipped": plan["skipped"], **{k: v for k, v in summary.items()
                                                   if k != "skipped"}}
        runtime = m._canonical_mutation_runtime(request)

        def admit(edit: dict) -> dict:
            _e, tgt, hook = m._authorize_memory_mutation(
                request, "update", edit["fact_id"], content_preview=edit["new_content"],
                run_pre_hook=False, profile=target)
            got = update_fact_authorized(  # embeds, may wait: this route runs off the loop
                engine, edit["fact_id"], edit["new_content"],
                trusted_actor_id=hook["agent_id"], source_agent_id="wiki",
                canonical_runtime=runtime, idempotency_key=edit["edit_key"], profile_id=tgt,
                origin_channel="external_edit")
            if not got.get("ok") or got.get("unchanged"):
                raise wiki.SkipEdit("the memory it edits is gone or already says this")
            return {"case_id": got["correction_case"]["case_id"],
                    "successor_fact_id": got["successor_fact_id"]}

        done = wiki.queue_edits(engine._db, target, edits, admit)
        summary.pop("skipped")
        return {"success": True, "profile_id": target, "flag": state, **summary,
                "queued": done["queued"], "skipped": plan["skipped"] + done["skipped"]}
    except HTTPException:
        raise
    except Exception as exc:
        raise _guard(exc, "Living wiki sync error")
