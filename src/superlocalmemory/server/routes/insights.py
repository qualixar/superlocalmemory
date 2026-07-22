# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
# Part of SuperLocalMemory v3.4.1 | https://qualixar.com

"""Quick Insight Actions — 5 one-click intelligence endpoints.

Actions: changed_this_week, opinions, contradictions, health, cross_project.
All queries use direct sqlite3 (Rule 06), parameterized SQL (Rule 11),
profile-scoped (Rule 01).
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Callable

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from superlocalmemory.server.routes.helpers import DB_PATH, get_active_profile

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v3/insights", tags=["insights"])


# ── Helper ────────────────────────────────────────────────────────

def _get_conn(profile: str = "") -> tuple[sqlite3.Connection | None, str]:
    """Open sqlite3 connection and resolve profile_id."""
    pid = profile or get_active_profile()
    if not DB_PATH.exists():
        return None, pid
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn, pid


# ── Action Handlers ───────────────────────────────────────────────

def _action_changed_this_week(
    conn: sqlite3.Connection, pid: str, limit: int = 50, days: int = 7,
) -> dict[str, Any]:
    """Facts created or modified in last N days."""
    items = []
    modified = []

    # New facts
    try:
        rows = conn.execute(
            "SELECT fact_id, content, fact_type, created_at, session_id, confidence "
            "FROM atomic_facts "
            "WHERE profile_id = ? AND created_at >= datetime('now', ?) "
            "ORDER BY created_at DESC LIMIT ?",
            (pid, f"-{days} days", limit),
        ).fetchall()
        items = [
            {
                "fact_id": dict(r)["fact_id"],
                "content": (dict(r).get("content") or "")[:200],
                "fact_type": dict(r).get("fact_type", ""),
                "created_at": dict(r).get("created_at", ""),
                "session_id": dict(r).get("session_id", ""),
                "confidence": round(float(dict(r).get("confidence", 0)), 3),
            }
            for r in rows
        ]
    except Exception as exc:
        logger.debug("changed_this_week new facts failed: %s", exc)

    # Modified facts (consolidation updates)
    try:
        rows = conn.execute(
            "SELECT cl.action_type, cl.new_fact_id, cl.existing_fact_id, "
            "cl.reason, cl.timestamp, af.content AS new_content "
            "FROM consolidation_log cl "
            "LEFT JOIN atomic_facts af ON af.fact_id = cl.new_fact_id "
            "WHERE cl.profile_id = ? AND cl.timestamp >= datetime('now', ?) "
            "AND cl.action_type IN ('update', 'supersede') "
            "ORDER BY cl.timestamp DESC LIMIT ?",
            (pid, f"-{days} days", limit),
        ).fetchall()
        modified = [
            {
                "action_type": dict(r).get("action_type", ""),
                "new_fact_id": dict(r).get("new_fact_id", ""),
                "existing_fact_id": dict(r).get("existing_fact_id", ""),
                "reason": (dict(r).get("reason") or "")[:200],
                "timestamp": dict(r).get("timestamp", ""),
                "new_content": (dict(r).get("new_content") or "")[:200],
            }
            for r in rows
        ]
    except Exception as exc:
        logger.debug("changed_this_week modifications failed: %s", exc)

    return {
        "action": "changed_this_week",
        "count": len(items),
        "items": items,
        "metadata": {
            "days": days,
            "modified_count": len(modified),
            "modifications": modified,
        },
    }


def _action_opinions(
    conn: sqlite3.Connection, pid: str, limit: int = 50, **_: Any,
) -> dict[str, Any]:
    """Opinion-type facts (user preferences, decisions, rationale)."""
    items = []
    try:
        rows = conn.execute(
            "SELECT fact_id, content, created_at, session_id, confidence "
            "FROM atomic_facts "
            "WHERE profile_id = ? AND fact_type = 'opinion' AND lifecycle = 'active' "
            "ORDER BY created_at DESC LIMIT ?",
            (pid, limit),
        ).fetchall()
        items = [
            {
                "fact_id": dict(r)["fact_id"],
                "content": (dict(r).get("content") or "")[:200],
                "created_at": dict(r).get("created_at", ""),
                "session_id": dict(r).get("session_id", ""),
                "confidence": round(float(dict(r).get("confidence", 0)), 3),
            }
            for r in rows
        ]
    except Exception as exc:
        logger.debug("opinions query failed: %s", exc)

    return {"action": "opinions", "count": len(items), "items": items}


def _action_contradictions(
    conn: sqlite3.Connection, pid: str, limit: int = 50, **_: Any,
) -> dict[str, Any]:
    """Contradiction edges from sheaf cohomology."""
    items = []
    try:
        rows = conn.execute(
            "SELECT ge.edge_id, ge.source_id, ge.target_id, ge.weight, ge.created_at, "
            "sf.content AS source_content, tf.content AS target_content "
            "FROM graph_edges ge "
            "LEFT JOIN atomic_facts sf ON sf.fact_id = ge.source_id "
            "LEFT JOIN atomic_facts tf ON tf.fact_id = ge.target_id "
            "WHERE ge.profile_id = ? AND ge.edge_type = 'contradiction' "
            "ORDER BY ge.weight DESC, ge.created_at DESC LIMIT ?",
            (pid, limit),
        ).fetchall()
        items = [
            {
                "edge_id": dict(r).get("edge_id", ""),
                "source_id": dict(r).get("source_id", ""),
                "target_id": dict(r).get("target_id", ""),
                "severity": round(float(dict(r).get("weight", 0)), 3),
                "source_content": (dict(r).get("source_content") or "")[:200],
                "target_content": (dict(r).get("target_content") or "")[:200],
                "created_at": dict(r).get("created_at", ""),
            }
            for r in rows
        ]
    except Exception as exc:
        logger.debug("contradictions query failed: %s", exc)

    return {"action": "contradictions", "count": len(items), "items": items}


def _action_health(
    conn: sqlite3.Connection, pid: str, **_: Any,
) -> dict[str, Any]:
    """Aggregate memory health: trust, retention, coverage, counts."""
    trust = {"high": 0, "medium": 0, "low": 0, "total": 0, "avg": 0.5}
    retention_zones: dict[str, Any] | None = None
    coverage: dict[str, int] = {}
    totals = {"facts": 0, "entities": 0, "edges": 0}
    community_count = 0

    # Trust distribution
    try:
        row = conn.execute(
            "SELECT "
            "COUNT(CASE WHEN trust_score >= 0.7 THEN 1 END) AS high_trust, "
            "COUNT(CASE WHEN trust_score >= 0.4 AND trust_score < 0.7 THEN 1 END) AS med_trust, "
            "COUNT(CASE WHEN trust_score < 0.4 THEN 1 END) AS low_trust, "
            "COUNT(*) AS total, ROUND(AVG(trust_score), 3) AS avg_trust "
            "FROM trust_scores WHERE profile_id = ?",
            (pid,),
        ).fetchone()
        if row:
            d = dict(row)
            trust = {
                "high": d.get("high_trust", 0),
                "medium": d.get("med_trust", 0),
                "low": d.get("low_trust", 0),
                "total": d.get("total", 0),
                "avg": float(d.get("avg_trust", 0) or 0.5),
            }
    except Exception:
        pass

    # Retention zones (v3.2+)
    try:
        rows = conn.execute(
            "SELECT lifecycle_zone, COUNT(*) AS cnt, "
            "ROUND(AVG(retention_score), 3) AS avg_retention "
            "FROM fact_retention WHERE profile_id = ? "
            "GROUP BY lifecycle_zone",
            (pid,),
        ).fetchall()
        retention_zones = {
            dict(r)["lifecycle_zone"]: {
                "count": dict(r)["cnt"],
                "avg_retention": float(dict(r).get("avg_retention", 0) or 0),
            }
            for r in rows
        }
    except Exception:
        pass  # Table may not exist in older DBs

    # Coverage by fact_type
    try:
        rows = conn.execute(
            "SELECT fact_type, COUNT(*) AS cnt "
            "FROM atomic_facts WHERE profile_id = ? AND lifecycle = 'active' "
            "GROUP BY fact_type",
            (pid,),
        ).fetchall()
        coverage = {dict(r)["fact_type"]: dict(r)["cnt"] for r in rows}
    except Exception:
        pass

    # Totals
    try:
        row = conn.execute(
            "SELECT "
            "(SELECT COUNT(*) FROM atomic_facts WHERE profile_id = ?) AS total_facts, "
            "(SELECT COUNT(*) FROM canonical_entities WHERE profile_id = ?) AS total_entities, "
            "(SELECT COUNT(*) FROM graph_edges WHERE profile_id = ?) AS total_edges",
            (pid, pid, pid),
        ).fetchone()
        if row:
            d = dict(row)
            totals = {
                "facts": d.get("total_facts", 0),
                "entities": d.get("total_entities", 0),
                "edges": d.get("total_edges", 0),
            }
    except Exception:
        pass

    # Community count
    try:
        row = conn.execute(
            "SELECT COUNT(DISTINCT community_id) AS cnt "
            "FROM fact_importance WHERE profile_id = ? AND community_id IS NOT NULL",
            (pid,),
        ).fetchone()
        if row:
            community_count = dict(row).get("cnt", 0)
    except Exception:
        pass

    return {
        "action": "health",
        "count": 1,
        "items": [{
            "trust": trust,
            "retention_zones": retention_zones,
            "coverage": coverage,
            "totals": totals,
            "community_count": community_count,
        }],
    }


def _action_cross_project(
    conn: sqlite3.Connection, pid: str, limit: int = 50, **_: Any,
) -> dict[str, Any]:
    """Entities spanning multiple sessions."""
    items = []
    try:
        rows = conn.execute(
            "SELECT ce.entity_id, ce.canonical_name, ce.entity_type, ce.fact_count, "
            "COUNT(DISTINCT af.session_id) AS session_count, "
            "GROUP_CONCAT(DISTINCT af.session_id) AS session_ids "
            "FROM canonical_entities ce "
            "JOIN atomic_facts af ON af.profile_id = ? AND EXISTS ("
            "  SELECT 1 FROM json_each(af.canonical_entities_json) je "
            "  WHERE je.value = ce.entity_id"
            ") "
            "WHERE ce.profile_id = ? "
            "GROUP BY ce.entity_id "
            "HAVING COUNT(DISTINCT af.session_id) > 1 "
            "ORDER BY session_count DESC, ce.fact_count DESC LIMIT ?",
            (pid, pid, limit),
        ).fetchall()
        items = [
            {
                "entity_id": dict(r).get("entity_id", ""),
                "canonical_name": dict(r).get("canonical_name", ""),
                "entity_type": dict(r).get("entity_type", ""),
                "fact_count": dict(r).get("fact_count", 0),
                "session_count": dict(r).get("session_count", 0),
                "session_ids": (dict(r).get("session_ids") or "").split(",")[:10],
            }
            for r in rows
        ]
    except Exception as exc:
        logger.debug("cross_project query failed: %s", exc)

    return {"action": "cross_project", "count": len(items), "items": items}


# ── Dispatch Map ──────────────────────────────────────────────────

ALLOWED_ACTIONS: dict[str, Callable] = {
    "changed_this_week": _action_changed_this_week,
    "opinions": _action_opinions,
    "contradictions": _action_contradictions,
    "health": _action_health,
    "cross_project": _action_cross_project,
}


# ── Endpoint ──────────────────────────────────────────────────────

@router.get("/{action_name}")
async def insight_action(
    action_name: str,
    profile: str = "",
    limit: int = Query(default=50, ge=1, le=200),
    days: int = Query(default=7, ge=1, le=90),
):
    """Run a quick insight action against the memory database."""
    handler = ALLOWED_ACTIONS.get(action_name)
    if not handler:
        valid = ", ".join(sorted(ALLOWED_ACTIONS.keys()))
        return JSONResponse(
            {"error": f"Unknown action: '{action_name}'. Valid: {valid}"},
            status_code=400,
        )

    conn, pid = _get_conn(profile)
    if conn is None:
        return {
            "action": action_name,
            "profile": pid,
            "count": 0,
            "items": [],
            "metadata": {"error": "Database not found"},
        }

    try:
        result = handler(conn, pid, limit=limit, days=days)
        result["profile"] = pid
        return result
    except Exception:
        logger.exception("Insight action %s failed", action_name)
        return JSONResponse(
            {"error": "Query failed"},
            status_code=500,
        )
    finally:
        conn.close()
