# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
# Part of SuperLocalMemory v3.4.1 | https://qualixar.com

"""Memory Timeline API — unified event stream from 3 sources.

Merges atomic_facts, temporal_events, and consolidation_log
into a single time-ordered event list for D3 timeline visualization.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from superlocalmemory.server.routes.helpers import DB_PATH, get_active_profile

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v3/timeline", tags=["timeline"])

VALID_RANGES = {"1d", "7d", "30d", "90d", "365d"}
# ``date`` is the dashboard's chronological view.  It does not need an extra
# enrichment pass because each returned event already carries its timestamp,
# but it is a first-class grouping selection and must not be rejected at the
# API boundary.
VALID_GROUP_BY = {"category", "community", "date"}
INTERNAL_CEILING = 5000


def _parse_range(range_str: str) -> str | None:
    """Parse range like '7d' into SQLite modifier '-7 days'."""
    m = re.match(r"^(\d+)d$", range_str)
    if not m or range_str not in VALID_RANGES:
        return None
    return f"-{m.group(1)} days"


@router.get("/")
async def get_timeline(
    range: str = "7d",
    group_by: str = "category",
    limit: int = Query(default=1000, ge=1, le=2000),
    offset: int = Query(default=0, ge=0),
    profile: str = "",
):
    """Get unified timeline events from all memory sources."""
    modifier = _parse_range(range)
    if modifier is None:
        return JSONResponse(
            {"error": f"Invalid range: '{range}'. Valid: {', '.join(sorted(VALID_RANGES))}"},
            status_code=400,
        )
    if group_by not in VALID_GROUP_BY:
        return JSONResponse(
            {"error": f"Invalid group_by: '{group_by}'. Valid: {', '.join(sorted(VALID_GROUP_BY))}"},
            status_code=400,
        )

    pid = profile or get_active_profile()
    if not DB_PATH.exists():
        return {"range": range, "group_by": group_by, "count": 0, "events": [], "total_available": 0, "offset": 0}

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    try:
        start_date = conn.execute("SELECT datetime('now', ?)", (modifier,)).fetchone()[0]
    except Exception:
        conn.close()
        return {"range": range, "group_by": group_by, "count": 0, "events": [], "total_available": 0, "offset": 0}

    events: list[dict] = []

    # 1. Atomic facts
    try:
        rows = conn.execute(
            "SELECT fact_id, content, fact_type, "
            "REPLACE(created_at, ' ', 'T') || 'Z' AS created_at, "
            "confidence, session_id "
            "FROM atomic_facts "
            "WHERE profile_id = ? AND created_at >= ? AND lifecycle = 'active' "
            "ORDER BY created_at DESC LIMIT ?",
            (pid, start_date, INTERNAL_CEILING),
        ).fetchall()
        for r in rows:
            d = dict(r)
            events.append({
                "id": d["fact_id"],
                "type": "fact_created",
                "timestamp": d["created_at"],
                "category": d.get("fact_type", "semantic"),
                "community_id": None,
                "content_preview": (d.get("content") or "")[:100],
                "trust_score": None,
                "lifecycle_zone": None,
                "retention_score": None,
                "session_id": d.get("session_id", ""),
                "source": "atomic_facts",
            })
    except Exception as exc:
        logger.debug("Timeline atomic_facts query failed: %s", exc)

    # 2. Temporal events
    try:
        rows = conn.execute(
            "SELECT event_id, entity_id, fact_id, "
            "observation_date, referenced_date, interval_start, description, "
            "REPLACE(COALESCE(observation_date, referenced_date, interval_start), ' ', 'T') || 'Z' "
            "AS event_date "
            "FROM temporal_events "
            "WHERE profile_id = ? "
            "AND COALESCE(observation_date, referenced_date, interval_start) >= ? "
            "ORDER BY event_date DESC LIMIT ?",
            (pid, start_date, INTERNAL_CEILING),
        ).fetchall()
        for r in rows:
            d = dict(r)
            events.append({
                "id": d.get("event_id", ""),
                "type": "temporal_event",
                "timestamp": d.get("event_date", ""),
                "category": "temporal",
                "community_id": None,
                "content_preview": (d.get("description") or "")[:100],
                "trust_score": None,
                "lifecycle_zone": None,
                "retention_score": None,
                "session_id": None,
                "source": "temporal_events",
            })
    except Exception as exc:
        logger.debug("Timeline temporal_events query failed: %s", exc)

    # 3. Consolidation log
    try:
        rows = conn.execute(
            "SELECT action_id, action_type, new_fact_id, existing_fact_id, reason, "
            "REPLACE(timestamp, ' ', 'T') || 'Z' AS timestamp "
            "FROM consolidation_log "
            "WHERE profile_id = ? AND timestamp >= ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (pid, start_date, INTERNAL_CEILING),
        ).fetchall()
        for r in rows:
            d = dict(r)
            preview = f"{d.get('action_type', '')}: {(d.get('reason') or '')[:80]}"
            events.append({
                "id": d.get("action_id", ""),
                "type": "consolidation",
                "timestamp": d.get("timestamp", ""),
                "category": "consolidation",
                "community_id": None,
                "content_preview": preview[:100],
                "trust_score": None,
                "lifecycle_zone": None,
                "retention_score": None,
                "session_id": None,
                "source": "consolidation_log",
            })
    except Exception as exc:
        logger.debug("Timeline consolidation_log query failed: %s", exc)

    # Sort merged events by timestamp desc
    events.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    total_available = len(events)

    # Paginate
    events = events[offset:offset + limit]

    # Enrich with community_id if group_by=community
    fact_ids = [e["id"] for e in events if e["source"] == "atomic_facts"]
    if group_by == "community" and fact_ids:
        _enrich_communities(conn, pid, fact_ids, events)

    # Enrich with trust scores + retention
    if fact_ids:
        _enrich_trust(conn, pid, fact_ids, events)
        _enrich_retention(conn, pid, fact_ids, events)

    conn.close()

    return {
        "range": range,
        "group_by": group_by,
        "total_available": total_available,
        "count": len(events),
        "offset": offset,
        "events": events,
    }


def _enrich_communities(
    conn: sqlite3.Connection, pid: str, fact_ids: list[str], events: list[dict],
) -> None:
    """Add community_id from fact_importance."""
    try:
        placeholders = ",".join("?" * len(fact_ids))
        sql = (
            "SELECT fact_id, community_id, pagerank_score "
            "FROM fact_importance WHERE profile_id = ? AND fact_id IN ("
            + placeholders + ")"
        )
        rows = conn.execute(sql, (pid, *fact_ids)).fetchall()
        comm_map = {dict(r)["fact_id"]: dict(r).get("community_id") for r in rows}
        for e in events:
            if e["id"] in comm_map:
                e["community_id"] = comm_map[e["id"]]
    except Exception:
        pass


def _enrich_trust(
    conn: sqlite3.Connection, pid: str, fact_ids: list[str], events: list[dict],
) -> None:
    """Add trust_score from trust_scores table."""
    try:
        placeholders = ",".join("?" * len(fact_ids))
        sql = (
            "SELECT target_id AS fact_id, trust_score "
            "FROM trust_scores WHERE profile_id = ? AND target_type = 'fact' "
            "AND target_id IN (" + placeholders + ")"
        )
        rows = conn.execute(sql, (pid, *fact_ids)).fetchall()
        trust_map = {dict(r)["fact_id"]: round(float(dict(r).get("trust_score", 0)), 3) for r in rows}
        for e in events:
            if e["id"] in trust_map:
                e["trust_score"] = trust_map[e["id"]]
    except Exception:
        pass


def _enrich_retention(
    conn: sqlite3.Connection, pid: str, fact_ids: list[str], events: list[dict],
) -> None:
    """Add lifecycle_zone + retention_score from fact_retention."""
    try:
        placeholders = ",".join("?" * len(fact_ids))
        sql = (
            "SELECT fact_id, lifecycle_zone, retention_score "
            "FROM fact_retention WHERE profile_id = ? AND fact_id IN ("
            + placeholders + ")"
        )
        rows = conn.execute(sql, (pid, *fact_ids)).fetchall()
        ret_map = {dict(r)["fact_id"]: dict(r) for r in rows}
        for e in events:
            d = ret_map.get(e["id"])
            if d:
                e["lifecycle_zone"] = d.get("lifecycle_zone")
                e["retention_score"] = round(float(d.get("retention_score", 0)), 3)
    except Exception:
        pass
