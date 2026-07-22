# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""SuperLocalMemory V3 - Compliance Routes
 - AGPL-3.0-or-later

Routes: /api/compliance/status, /api/compliance/audit,
        /api/compliance/retention-policy
Uses V3 compliance modules: ABACEngine, AuditChain, RetentionEngine.
"""
import json
import logging
import sqlite3
from typing import Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from .helpers import get_active_profile, get_engine_lazy, MEMORY_DIR, DB_PATH
from superlocalmemory.server.route_mutations import authorize_route_mutation

logger = logging.getLogger("superlocalmemory.routes.compliance")
router = APIRouter()

# The engine's audit post-hooks write to <data_root>/audit_chain.db
# (core/engine_wiring.py). The route MUST read that same file — an earlier
# "audit.db" path pointed at a file nothing ever wrote, so the Compliance tab
# always showed zero events.
AUDIT_DB = MEMORY_DIR / "audit_chain.db"

# Feature detection
COMPLIANCE_AVAILABLE = False
try:
    from superlocalmemory.compliance.audit import AuditChain
    from superlocalmemory.compliance.retention import RetentionEngine
    from superlocalmemory.compliance.abac import ABACEngine
    from superlocalmemory.compliance.gdpr import GDPRCompliance
    COMPLIANCE_AVAILABLE = True
except ImportError:
    logger.info("V3 compliance engine not available")


@router.get("/api/compliance/status")
async def compliance_status():
    """Get compliance engine status for active profile."""
    if not COMPLIANCE_AVAILABLE:
        return {"available": False, "message": "Compliance engine not available"}

    try:
        profile = get_active_profile()

        # Audit events (hash-chained, from the engine's live audit_chain.db).
        # Scope the count + recent list to the active profile.
        audit_events_count = 0
        recent_audit_events = []
        try:
            audit = AuditChain(str(AUDIT_DB))
            recent_audit_events = audit.query(profile_id=profile, limit=30)
            # get_stats() is global; count this profile's rows for the header.
            audit_events_count = len(audit.query(profile_id=profile, limit=100000))
        except Exception as exc:
            logger.debug("audit chain: %s", exc)

        # Retention policies (scoped to the active profile)
        retention_policies = []
        try:
            conn = sqlite3.connect(str(DB_PATH))
            engine = RetentionEngine(conn)
            retention_policies = engine.list_rules(profile)
            conn.close()
        except Exception as exc:
            logger.debug("retention engine: %s", exc)

        # ABAC policies
        abac_policies_count = 0
        try:
            abac = ABACEngine()
            abac_policies_count = len(abac._policies)
        except Exception:
            pass

        return {
            "available": True,
            "active_profile": profile,
            "audit_events_count": audit_events_count,
            "recent_audit_events": recent_audit_events,
            "retention_policies": retention_policies,
            "abac_policies_count": abac_policies_count,
        }
    except Exception:
        logger.exception("compliance_status error")
        return {"available": False, "error": "Internal server error"}


@router.get("/api/compliance/audit")
async def query_audit_trail(
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
    event_type: Optional[str] = Query(default=None),
    since: Optional[str] = Query(default=None),
):
    """Query audit trail events with optional filters."""
    if not COMPLIANCE_AVAILABLE:
        return {"available": False, "error": "Compliance engine not available"}
    # The audit trail reveals operations, actors, and fact_ids. This GET is not
    # covered by the mutation middleware; the same-origin dashboard reads it via
    # a plain GET (no token header), so gate on the loopback-trusted mutation
    # boundary — local owner allowed, remote uncredentialed caller fails closed.
    from superlocalmemory.server.write_identity import require_http_mutation_actor
    require_http_mutation_actor(request, getattr(request.app.state, "daemon_descriptor", None),
                                actor_kind="audit-read")
    try:
        profile = get_active_profile()
        audit = AuditChain(str(AUDIT_DB))
        events = audit.query(
            profile_id=profile, operation=event_type, start_date=since,
            limit=limit,
        )
        chain_ok = True
        try:
            chain_ok = audit.verify_integrity()
        except Exception:
            pass
        return {
            "available": True, "events": events, "total": len(events),
            "chain_verified": chain_ok, "active_profile": profile,
            "filters": {"event_type": event_type, "since": since, "limit": limit},
        }
    except Exception:
        logger.exception("query_audit_trail error")
        return {"available": False, "error": "Internal server error"}


@router.post("/api/compliance/retention-policy")
async def create_retention_policy(data: dict):
    """Create a compliance retention policy.

    Body: {
        name: str,
        retention_days: int,
        category: str (maps to framework),
        action: "archive" | "tombstone" | "notify",
        applies_to: dict (optional)
    }
    """
    if not COMPLIANCE_AVAILABLE:
        return {"success": False, "error": "Compliance engine not available"}

    name = data.get('name')
    retention_days = data.get('retention_days')
    framework = data.get('category', 'custom')
    action = data.get('action')
    applies_to = data.get('applies_to', {})

    if not name or not isinstance(name, str):
        return {"success": False, "error": "name is required (string)"}
    if not isinstance(retention_days, int) or retention_days < 1:
        return {"success": False, "error": "retention_days must be a positive integer"}

    valid_actions = ("archive", "tombstone", "notify")
    if action not in valid_actions:
        return {"success": False, "error": f"action must be one of: {valid_actions}"}

    try:
        profile = get_active_profile()
        conn = sqlite3.connect(str(DB_PATH))
        engine = RetentionEngine(conn)

        rule_id = engine.create_rule(
            name=name, framework=framework,
            retention_days=retention_days, action=action,
            applies_to=applies_to, profile_id=profile,
        )
        conn.close()

        return {
            "success": True, "rule_id": rule_id,
            "active_profile": profile,
            "message": f"Retention policy '{name}' created ({retention_days}d, {action})",
        }
    except Exception:
        logger.exception("create_retention_policy error")
        return {"success": False, "error": "Internal server error"}


@router.delete("/api/compliance/retention-policy")
async def delete_retention_policy(name: str = Query(...)):
    """Delete a retention policy by name for the active profile."""
    if not COMPLIANCE_AVAILABLE:
        return {"success": False, "error": "Compliance engine not available"}
    try:
        profile = get_active_profile()
        conn = sqlite3.connect(str(DB_PATH))
        engine = RetentionEngine(conn)
        removed = engine.delete_rule(profile, name)
        conn.close()
        if not removed:
            return {"success": False, "error": f"Policy '{name}' not found"}
        return {"success": True, "active_profile": profile,
                "message": f"Retention policy '{name}' deleted"}
    except Exception:
        logger.exception("delete_retention_policy error")
        return {"success": False, "error": "Internal server error"}


@router.post("/api/compliance/retention/enforce")
async def enforce_retention():
    """Run all retention policies for the active profile now.

    Moves expired facts to their rule's terminal lifecycle zone (archive/
    tombstone) or counts them (notify). Soft-state only — never a raw delete.
    """
    if not COMPLIANCE_AVAILABLE:
        return {"success": False, "error": "Compliance engine not available"}
    try:
        profile = get_active_profile()
        conn = sqlite3.connect(str(DB_PATH))
        engine = RetentionEngine(conn)
        result = engine.enforce(profile)
        conn.close()
        return {"success": True, **result}
    except Exception:
        logger.exception("enforce_retention error")
        return {"success": False, "error": "Internal server error"}


# ── GDPR Art. 15/20 — Right to Access / Portability ─────────────────────────

@router.get("/api/compliance/gdpr/export")
async def gdpr_export(request: Request):
    """Export ALL data for the active profile (GDPR Art. 20 portability).

    Comprehensive 14-table export (memories, facts, entities, edges, trust,
    feedback, behavioral patterns, provenance, audit, …) as a downloadable
    JSON attachment. Read-only.
    """
    if not COMPLIANCE_AVAILABLE:
        return {"available": False, "error": "Compliance engine not available"}
    # A full-workspace data dump is an administrative action. Triggered by a
    # top-level navigation (a.href) that cannot carry a credential header, so
    # gate on the loopback-trusted mutation boundary (local owner allowed,
    # remote uncredentialed fails closed) AND require MANAGE — in company mode
    # a session cookie flows on navigation, so a non-admin user is still denied;
    # the machine owner keeps MANAGE.
    from superlocalmemory.server.write_identity import require_http_mutation_actor
    from superlocalmemory.server.rbac_enforce import require_manage
    require_http_mutation_actor(request, getattr(request.app.state, "daemon_descriptor", None),
                                actor_kind="gdpr-export")
    require_manage(request)
    try:
        engine = get_engine_lazy(request.app.state)
        if engine is None:
            return {"available": False, "error": "Engine not initialized"}
        profile = get_active_profile()
        data = GDPRCompliance(engine._db).export_profile_data(profile)
        filename = f"slm-export-{profile}.json"
        return JSONResponse(
            content=data,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception:
        logger.exception("gdpr_export error")
        return {"available": False, "error": "Internal server error"}


# ── GDPR Art. 17 — Right to Erasure ─────────────────────────────────────────

@router.post("/api/compliance/gdpr/erase")
async def gdpr_erase(request: Request, data: dict = {}):
    """Permanently erase ALL data for the active profile (GDPR Art. 17).

    IRREVERSIBLE. Requires an explicit ``confirm`` field in the body that
    exactly matches the active profile name — this is the guard against an
    accidental one-click wipe. The 'default' profile can never be erased
    (enforced in GDPRCompliance.forget_profile). Mutation-authorized.
    """
    if not COMPLIANCE_AVAILABLE:
        return {"success": False, "error": "Compliance engine not available"}
    try:
        engine = get_engine_lazy(request.app.state)
        if engine is None:
            return {"success": False, "error": "Engine not initialized"}
        profile = get_active_profile()
        confirm = (data or {}).get("confirm", "")
        if confirm != profile:
            return {
                "success": False,
                "error": (
                    "Confirmation required: send {\"confirm\": \"" + profile +
                    "\"} to erase this profile. This is irreversible."
                ),
            }
        if profile == "default":
            return {"success": False,
                    "error": "The 'default' profile cannot be erased."}
        # Irreversible erasure is admin-only (beyond mutation auth).
        from superlocalmemory.server.rbac_enforce import require_manage
        require_manage(request, profile=profile)
        authorization = authorize_route_mutation(
            request,
            operation="delete",
            source_agent_id="http-gdpr-erase",
            profile_id=profile,
        )
        result = GDPRCompliance(engine._db).forget_profile(profile)
        authorization.complete()
        return {"success": True, "active_profile": profile, **(result or {})}
    except Exception:
        logger.exception("gdpr_erase error")
        return {"success": False, "error": "Internal server error"}
