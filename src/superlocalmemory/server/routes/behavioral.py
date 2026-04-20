# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""SuperLocalMemory V3 - Behavioral Routes
 - AGPL-3.0-or-later

Routes: /api/behavioral/status, /api/behavioral/report-outcome
Uses V3 learning.behavioral.BehavioralPatternStore and learning.outcomes.OutcomeTracker.
"""
import json
import logging

from fastapi import APIRouter

from .helpers import get_active_profile, MEMORY_DIR

logger = logging.getLogger("superlocalmemory.routes.behavioral")
router = APIRouter()

LEARNING_DB = MEMORY_DIR / "learning.db"

# Feature detection
BEHAVIORAL_AVAILABLE = False
try:
    from superlocalmemory.learning.behavioral import BehavioralPatternStore
    from superlocalmemory.learning.outcomes import OutcomeTracker
    BEHAVIORAL_AVAILABLE = True
except ImportError as e:
    logger.warning("V3 behavioral engine import failed: %s", e)


@router.get("/api/behavioral/status")
async def behavioral_status():
    """Get behavioral learning status for active profile."""
    if not BEHAVIORAL_AVAILABLE:
        return {"available": False, "message": "Behavioral engine not available"}

    try:
        profile = get_active_profile()
        db_path = str(LEARNING_DB)

        # Outcomes
        total_outcomes = 0
        outcome_breakdown = {"success": 0, "failure": 0, "partial": 0}
        recent_outcomes = []
        try:
            tracker = OutcomeTracker(db_path)
            all_outcomes = tracker.get_outcomes(profile_id=profile, limit=50)
            total_outcomes = len(all_outcomes)
            for o in all_outcomes:
                key = o.outcome if hasattr(o, 'outcome') else str(o)
                if key in outcome_breakdown:
                    outcome_breakdown[key] += 1
            recent_outcomes = [
                {"outcome": o.outcome, "action_type": o.action_type,
                 "timestamp": o.timestamp}
                for o in all_outcomes[:20]
                if hasattr(o, 'outcome')
            ]
        except Exception as exc:
            logger.debug("outcome tracker: %s", exc)

        # Patterns
        patterns = []
        cross_project_transfers = 0
        try:
            store = BehavioralPatternStore(db_path)
            patterns = store.get_patterns(profile_id=profile)
            # Count patterns spanning multiple projects
            cross_project_transfers = len([
                p for p in patterns
                if isinstance(p, dict) and p.get("project_count", 1) > 1
            ])
        except Exception as exc:
            logger.warning("pattern store error: %s", exc)

        return {
            "available": True,
            "active_profile": profile,
            "total_outcomes": total_outcomes,
            "outcome_breakdown": outcome_breakdown,
            "patterns": patterns,
            "cross_project_transfers": cross_project_transfers,
            "recent_outcomes": recent_outcomes,
            "stats": {
                "success_count": outcome_breakdown.get("success", 0),
                "failure_count": outcome_breakdown.get("failure", 0),
                "partial_count": outcome_breakdown.get("partial", 0),
                "patterns_count": len(patterns),
            },
        }
    except Exception as e:
        logger.error("behavioral_status error: %s", e)
        return {"available": False, "error": str(e)}


@router.post("/api/behavioral/report-outcome")
async def report_outcome(data: dict):
    """Record an explicit dashboard-reported outcome.

    Body: {
        memory_ids: [str, ...],
        outcome: "success" | "failure" | "partial",
        action_type: str (optional),
        context: str (optional)
    }

    S9-DASH-02: previously this handler passed a path string to
    ``OutcomeTracker(db)`` (which expects a DatabaseManager) and also
    targeted ``learning.db`` — but ``action_outcomes`` lives in
    ``memory.db`` (M006). Both failures were silent. This rewrite
    writes directly to ``action_outcomes`` with a reward label derived
    from ``outcome``:
      success=1.0, failure=0.0, partial=0.5
    """
    memory_ids = data.get('memory_ids')
    outcome = data.get('outcome')
    action_type = data.get('action_type', 'other')
    context_note = data.get('context', '')

    if not memory_ids or not isinstance(memory_ids, list):
        return {"success": False, "error": "memory_ids must be a non-empty list"}

    valid_outcomes = ("success", "failure", "partial")
    if outcome not in valid_outcomes:
        return {"success": False, "error": f"outcome must be one of: {valid_outcomes}"}

    import sqlite3
    import time
    import uuid
    from datetime import datetime, timezone

    reward_map = {"success": 1.0, "failure": 0.0, "partial": 0.5}
    reward = reward_map[outcome]
    now_iso = datetime.now(timezone.utc).isoformat()
    outcome_id = str(uuid.uuid4())
    memory_db_path = MEMORY_DIR / "memory.db"

    try:
        profile = get_active_profile()
        context_dict = {
            "note": context_note,
            "action_type": action_type,
            "source": "dashboard_report_outcome",
        }
        conn = sqlite3.connect(str(memory_db_path), timeout=5.0)
        try:
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute(
                "INSERT INTO action_outcomes "
                "(outcome_id, profile_id, query, fact_ids_json, outcome, "
                " context_json, timestamp, reward, settled, settled_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)",
                (
                    outcome_id, profile, "",
                    json.dumps(memory_ids),
                    outcome,
                    json.dumps(context_dict),
                    now_iso, reward, now_iso,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        return {
            "success": True, "outcome_id": outcome_id,
            "active_profile": profile,
            "reward": reward,
            "message": (
                f"Recorded {outcome} outcome for {len(memory_ids)} "
                f"memories (reward={reward})"
            ),
        }
    except Exception as e:
        logger.error("report_outcome error: %s", e)
        return {"success": False, "error": str(e)}


# --------------------------------------------------------------------------
# v3.4.7: Behavioral Assertions API (for dashboard + external consumers)
# --------------------------------------------------------------------------

@router.get("/api/behavioral/assertions")
async def get_assertions(min_confidence: float = 0.0, category: str = "", limit: int = 50):
    """Get learned behavioral assertions for dashboard display."""
    try:
        import sqlite3 as _sqlite3
        profile = get_active_profile()
        conn = _sqlite3.connect(str(MEMORY_DIR / "memory.db"))
        conn.row_factory = _sqlite3.Row

        query = (
            "SELECT id, trigger_condition, action, category, confidence, "
            "evidence_count, reinforcement_count, contradiction_count, "
            "project_path, source, created_at, updated_at "
            "FROM behavioral_assertions "
            "WHERE profile_id = ? AND confidence >= ?"
        )
        params: list = [profile, min_confidence]
        if category:
            query += " AND category = ?"
            params.append(category)
        query += " ORDER BY confidence DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, tuple(params)).fetchall()
        conn.close()

        assertions = [dict(r) for r in rows]
        return {
            "assertions": assertions,
            "count": len(assertions),
            "active_profile": profile,
        }
    except Exception as e:
        logger.debug("get_assertions error: %s", e)
        return {"assertions": [], "count": 0, "error": str(e)}


@router.get("/api/behavioral/tool-events")
async def get_tool_events(tool_name: str = "", limit: int = 100):
    """Get recent tool events for dashboard display."""
    try:
        import sqlite3 as _sqlite3
        profile = get_active_profile()
        limit = min(int(limit), 1000)
        conn = _sqlite3.connect(str(MEMORY_DIR / "memory.db"))
        conn.row_factory = _sqlite3.Row

        query = (
            "SELECT id, tool_name, event_type, input_summary, output_summary, "
            "duration_ms, created_at FROM tool_events "
            "WHERE profile_id = ?"
        )
        params: list = [profile]
        if tool_name:
            query += " AND tool_name = ?"
            params.append(tool_name)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        try:
            rows = conn.execute(query, tuple(params)).fetchall()
            events = [dict(r) for r in rows]
            return {"events": events, "count": len(events)}
        finally:
            conn.close()
    except Exception as e:
        logger.debug("get_tool_events error: %s", e)
        return {"events": [], "count": 0, "error": str(e)}


@router.get("/api/behavioral/soft-prompts")
async def get_soft_prompts():
    """Get active soft prompt templates for dashboard display."""
    try:
        import sqlite3 as _sqlite3
        conn = _sqlite3.connect(str(MEMORY_DIR / "memory.db"))
        conn.row_factory = _sqlite3.Row
        rows = conn.execute(
            "SELECT prompt_id, category, content, confidence, effectiveness, "
            "token_count, active, version, created_at "
            "FROM soft_prompt_templates WHERE active = 1 ORDER BY category"
        ).fetchall()
        conn.close()
        return {"prompts": [dict(zip(
            ["prompt_id", "category", "content", "confidence", "effectiveness",
             "token_count", "active", "version", "created_at"], r
        )) for r in rows], "count": len(rows)}
    except Exception as e:
        logger.debug("get_soft_prompts error: %s", e)
        return {"prompts": [], "count": 0, "error": str(e)}


@router.post("/api/v3/tool-event")
async def log_tool_event_api(data: dict):
    """Log a tool event via HTTP (called by PostToolUse hook).

    Body (v3.4.10 enriched):
    {
        "tool_name": "Skill",
        "event_type": "complete",
        "input_summary": "{\"skill\": \"superpowers:brainstorming\"}",
        "output_summary": "{\"success\": true}",
        "session_id": "abc123",
        "project_path": "/path/to/project"
    }

    All fields except tool_name are optional for backward compatibility.
    Lightweight — no LLM, just an INSERT.
    """
    try:
        import sqlite3 as _sqlite3
        from datetime import datetime, timezone
        import os

        tool_name = data.get("tool_name", "unknown")
        event_type = data.get("event_type", "complete")
        input_summary = data.get("input_summary", "")
        output_summary = data.get("output_summary", "")
        session_id = data.get("session_id") or os.environ.get("CLAUDE_SESSION_ID", "hook")
        project_path = data.get("project_path", "")
        now = datetime.now(timezone.utc).isoformat()
        profile = get_active_profile()

        # Truncate to prevent oversized payloads (defense in depth)
        input_summary = str(input_summary)[:500] if input_summary else ""
        output_summary = str(output_summary)[:500] if output_summary else ""

        conn = _sqlite3.connect(str(MEMORY_DIR / "memory.db"))
        try:
            conn.execute(
                "INSERT INTO tool_events "
                "(session_id, profile_id, project_path, tool_name, event_type, "
                " input_summary, output_summary, duration_ms, metadata, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 0, '{}', ?)",
                (session_id, profile, project_path, tool_name, event_type,
                 input_summary, output_summary, now),
            )
            conn.commit()
        finally:
            conn.close()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}
