# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``GET /api/outcomes/health``: the Memory Health report for the dashboard (W3-5).

The same report as ``slm health --memory`` and the MCP ``health`` tool
(``outcomes.memory_report``): deliveries, use rates and randomised effects with
intervals, judge certificates and second opinions. Read-only on a ``mode=ro``
connection, ids and numbers only. Authorized like the truth checks: the
review-read role on the profile.

``/api/outcomes/measurement`` (D-28): the notice about what SLM measures, the
state of each measure, and the one off switch (``outcomes.switch``). Reading is
open to the dashboard; the two POSTs need an issued credential (the install
token from loopback, an API key or the daemon capability) and the same
owner/admin role as the other installation-wide settings, so a remote caller or
a plain member cannot switch it.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, StrictBool

logger = logging.getLogger(__name__)
router = APIRouter(tags=["outcomes"])


class MeasurementChange(BaseModel):
    """Turn withholding and swapping on or off together; receipts are not touched here."""

    model_config = ConfigDict(extra="forbid")

    on: StrictBool


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


# ---------------------------------------------------------------------------
# /api/outcomes/measurement  (D-28)
# ---------------------------------------------------------------------------


def _require_credential(request: Request) -> None:
    """An issued credential, even from loopback (raises 403 without one)."""
    from superlocalmemory.server.write_identity import require_write_actor

    require_write_actor(request, getattr(request.app.state, "daemon_descriptor", None),
                        actor_kind="measurement")


def _require_manage(request: Request) -> None:
    from superlocalmemory.server.rbac_enforce import require_global_manage

    require_global_manage(request)


def _gate(request: Request) -> None:
    _require_credential(request)
    _require_manage(request)


def _measurement_body(extra: dict | None = None) -> dict:
    from superlocalmemory.infra.data_root import canonical_data_root
    from superlocalmemory.outcomes import disclosure, switch

    return {"success": True, **switch.status(canonical_data_root()),
            "notice": disclosure.text(), **(extra or {})}


@router.get("/api/outcomes/measurement")
def get_measurement():
    """The notice, each measure's state, and whether the notice is still pending."""
    try:
        return _measurement_body()
    except Exception:
        logger.exception("measurement status failed")
        raise HTTPException(500, detail="Could not read the measurement settings.")


@router.post("/api/outcomes/measurement")
def post_measurement(request: Request, body: MeasurementChange):
    """Turn withholding and swapping on or off (receipts stay as they are)."""
    _gate(request)
    from superlocalmemory.infra.data_root import canonical_data_root
    from superlocalmemory.outcomes import switch

    try:
        result = switch.set_measurement(canonical_data_root(), body.on, channel="dashboard")
        return _measurement_body({"changed": result["changed"]})
    except HTTPException:
        raise
    except Exception:
        logger.exception("measurement change failed")
        raise HTTPException(500, detail="Could not change the measurement settings.")


@router.post("/api/outcomes/measurement/acknowledge")
def post_measurement_acknowledge(request: Request):
    """A person has read the notice in the dashboard."""
    _gate(request)
    from superlocalmemory.infra.data_root import canonical_data_root
    from superlocalmemory.outcomes import disclosure

    try:
        disclosure.acknowledge(canonical_data_root(), "dashboard")
        return _measurement_body()
    except HTTPException:
        raise
    except Exception:
        logger.exception("measurement acknowledge failed")
        raise HTTPException(500, detail="Could not record that the notice was read.")
