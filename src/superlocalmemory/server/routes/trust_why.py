# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``GET /api/trust/why/{fact_id}``: everything that explains one memory (WS-5).

The same report as ``slm why`` (``trust/why.explain``): where the memory came from, whether the
instruction firewall holds it back, its signature, disputes and leases, derived work that
mentions it, and the last deliveries that showed it, each with its reason in words. Read-only
on a ``mode=ro`` connection. Authorized like the Memory Health and truth reads: the review-read
role on the profile; the memory must belong to that profile or it is a 404 (no hint that it
exists elsewhere). The signature is checked only with the key this daemon already holds
in memory; the route never asks the keychain.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(tags=["trust"])

_MAX_DELIVERIES = 50


@router.get("/api/trust/why/{fact_id}")
def trust_why(request: Request, fact_id: str, profile_id: str = "", deliveries: int = 10):
    from superlocalmemory.server.routes import memories as m
    from superlocalmemory.trust import signing_key, why
    from superlocalmemory.truth.invariants import open_read_only

    try:
        profile = m._routed_profile(profile_id)
        engine, target = m._authorize_review_read(request, profile)
        conn = open_read_only(str(engine._db.db_path))
        try:
            report = why.explain(conn, fact_id, profile_id=target, signer=signing_key.current(),
                                 deliveries=max(1, min(int(deliveries), _MAX_DELIVERIES)))
        finally:
            conn.close()
        if report is None:
            raise HTTPException(404, detail="Memory not found")
        return {"success": True, **report}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Why error")
