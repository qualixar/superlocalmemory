# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``GET /api/outcomes/health``: the Memory Health report for the dashboard (W3-5).

The same report as ``slm health --memory`` and the MCP ``health`` tool
(``outcomes.memory_report``): deliveries, use rates and randomised effects with
intervals, judge certificates and second opinions. Read-only on a ``mode=ro``
connection, ids and numbers only. Authorized like the truth checks: the
review-read role on the profile.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(tags=["outcomes"])


@router.get("/api/outcomes/health")
def outcomes_health(request: Request, profile_id: str = ""):
    from superlocalmemory.outcomes import memory_report
    from superlocalmemory.server.routes import memories as m
    from superlocalmemory.truth.invariants import open_read_only

    try:
        profile = m._routed_profile(profile_id)
        engine, target = m._authorize_review_read(request, profile)
        db_path = engine._db.db_path
        conn = open_read_only(str(db_path))
        try:
            report = memory_report.build(conn, target, slm_home=db_path.parent)
        finally:
            conn.close()
        return {"success": True, **report, "lines": memory_report.describe(report)}
    except HTTPException:
        raise
    except m._UnknownRoutedProfile as exc:
        return m._unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise m._canonical_mutation_error(exc, "Memory health error")
