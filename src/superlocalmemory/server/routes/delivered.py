# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``GET /api/delivered/recent``: what SuperLocalMemory recently showed your agents, and why.

Feeds the dashboard's "Shown to your agents" card on the Health page (W3, D-41, §8.10.3).
Recent deliveries grouped by conversation, each memory with its reason in plain words
(``outcomes/delivered``), plus the two numbers from Memory Health: prompts the need gate
skipped and the memories the harm gate would withhold (shadow). Read-only on a ``mode=ro``
connection. Authorized like ``GET /api/outcomes/health``: the review-read role on the profile.
A remote app (``caller_view``) never sees a memory it may not see.

Run when someone asks (a button), not on every page load: the two gate numbers read the
receipt ledger.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(tags=["outcomes"])


@router.get("/api/delivered/recent")
def delivered_recent(request: Request, limit: int = 20, profile_id: str = "", caller_view: str = ""):
    from superlocalmemory.outcomes import delivered
    from superlocalmemory.server.routes import memories as m
    from superlocalmemory.server.routes.truth import _hidden
    from superlocalmemory.truth.invariants import open_read_only

    try:
        profile = m._routed_profile(profile_id)
        engine, target = m._authorize_review_read(request, profile)
        db_path = engine._db.db_path
        hidden = _hidden(engine, target, caller_view)
        conn = open_read_only(str(db_path))
        try:
            found = delivered.recent(conn, target, limit=limit, hidden=hidden, slm_home=db_path.parent)
            numbers = delivered.gates(conn, target, slm_home=db_path.parent, hidden=hidden)
        finally:
            conn.close()
        return {"success": True, "profile_id": target, **found, **numbers}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Shown-to-agents error")
