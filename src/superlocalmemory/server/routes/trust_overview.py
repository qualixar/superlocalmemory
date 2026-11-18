# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``GET /api/trust/overview``: the Trust card on the Health page (WS-5, §8.10.3 h).

Where memories came from, how many wait for a person's OK, whether any memory no longer
matches its signature, and whether the audit log is intact (``trust/overview``). Read-only:
a ``mode=ro`` connection, the audit log opened read-only, and the signature check with the key
the daemon already holds in memory (the route never asks the keychain). Counts only, never a
memory's text. Authorized like Memory Health: the review-read role on the profile. Run when
someone asks (a button): the audit log is walked end to end.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(tags=["trust"])


@router.get("/api/trust/overview")
def trust_overview(request: Request, profile_id: str = ""):
    from superlocalmemory.server.routes import memories as m
    from superlocalmemory.trust import overview, signing_key
    from superlocalmemory.truth.invariants import open_read_only

    try:
        profile = m._routed_profile(profile_id)
        engine, target = m._authorize_review_read(request, profile)
        db_path = engine._db.db_path
        conn = open_read_only(str(db_path))
        try:
            body = overview.build(conn, target, slm_home=db_path.parent,
                                  audit_path=db_path.parent / "audit_chain.db",
                                  signer=signing_key.current(), key_status=signing_key.status())
        finally:
            conn.close()
        return {"success": True, "profile_id": target, **body}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Trust overview error")
