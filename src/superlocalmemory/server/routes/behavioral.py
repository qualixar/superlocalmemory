# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""SuperLocalMemory V3 - Behavioral Routes
 - AGPL-3.0-or-later

Routes: /api/behavioral/status, /api/behavioral/report-outcome
Uses V3 learning.behavioral.BehavioralPatternStore and direct telemetry reads.
"""
import json
import logging
import sqlite3
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from .helpers import MEMORY_DIR, get_active_profile

logger = logging.getLogger("superlocalmemory.routes.behavioral")
router = APIRouter()

_RECENT_OUTCOMES_LIMIT = 20
_REWARD_TIMELINE_DAYS = 182
_MAX_OUTCOME_FACT_IDS = 100
_MAX_FACT_ID_LENGTH = 200


class ReportOutcomeRequest(BaseModel):
    """Bounded explicit outcome payload accepted from dashboard clients."""

    model_config = ConfigDict(extra="forbid")

    memory_ids: list[StrictStr] = Field(
        min_length=1,
        max_length=_MAX_OUTCOME_FACT_IDS,
    )
    outcome: Literal["success", "failure", "partial"]
    action_type: StrictStr = Field(default="other", max_length=80)
    context: StrictStr = Field(default="", max_length=1000)

    @field_validator("memory_ids")
    @classmethod
    def normalize_fact_ids(cls, value: list[str]) -> list[str]:
        """Strip and de-duplicate fact IDs while preserving request order."""
        deduplicated: list[str] = []
        seen: set[str] = set()
        for raw_fact_id in value:
            fact_id = raw_fact_id.strip()
            if not fact_id:
                raise ValueError("memory_ids must not contain blank fact IDs")
            if len(fact_id) > _MAX_FACT_ID_LENGTH:
                raise ValueError(
                    f"memory_ids entries must be at most "
                    f"{_MAX_FACT_ID_LENGTH} characters"
                )
            if fact_id not in seen:
                seen.add(fact_id)
                deduplicated.append(fact_id)
        return deduplicated


def _require_read(request: Request) -> None:
    from superlocalmemory.access.rbac import Permission
    from superlocalmemory.server.rbac_enforce import require_permission

    require_permission(request, Permission.READ, profile=get_active_profile())


def _require_write(request: Request) -> None:
    from superlocalmemory.access.rbac import Permission
    from superlocalmemory.server.rbac_enforce import require_permission

    require_permission(request, Permission.WRITE, profile=get_active_profile())


def _authorize_outcome_write(request: Request) -> None:
    """Run authorization before FastAPI validates the request body."""
    _require_write(request)
    request.state.outcome_write_authorized = True


def _validate_profile_fact_ids(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    fact_ids: list[str],
) -> None:
    """Reject missing or foreign-profile facts before an outcome is stored."""
    placeholders = ",".join("?" for _ in fact_ids)
    rows = conn.execute(
        "SELECT fact_id FROM atomic_facts "
        f"WHERE profile_id = ? AND fact_id IN ({placeholders})",
        (profile_id, *fact_ids),
    ).fetchall()
    if {str(row[0]) for row in rows} != set(fact_ids):
        raise HTTPException(
            status_code=422,
            detail="Every memory_id must identify a fact in the active profile",
        )


# Feature detection
BEHAVIORAL_AVAILABLE = False
try:
    from superlocalmemory.learning.behavioral import BehavioralPatternStore
    BEHAVIORAL_AVAILABLE = True
except ImportError as e:
    logger.warning("V3 behavioral engine import failed: %s", e)


def _memory_db_path() -> Path:
    """Resolve at read time so tests and profile-scoped routes stay aligned."""
    return MEMORY_DIR / "memory.db"


def _learning_db_path() -> Path:
    return MEMORY_DIR / "learning.db"


def _is_cross_project_pattern(pattern: dict) -> bool:
    metadata = pattern.get("metadata")
    return (
        isinstance(metadata, dict)
        and bool(str(metadata.get("transferred_from") or "").strip())
    )


def _load_action_outcomes(profile_id: str) -> dict:
    """Read bounded explicit/finalized outcome telemetry from ``memory.db``.

    ``OutcomeTracker`` requires a ``DatabaseManager`` and the outcome table
    belongs to ``memory.db``.  This read-only query deliberately does not
    infer outcomes from recall hits, which are exposure signals rather than
    evidence that the returned memory helped.
    """
    empty = {
        "total": 0,
        "breakdown": {"success": 0, "failure": 0, "partial": 0},
        "recent": [],
        "reward": _empty_reward_telemetry(),
    }
    db_path = _memory_db_path()
    if not db_path.exists():
        return empty
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=1.0)
        conn.row_factory = sqlite3.Row
        try:
            columns = {
                str(row["name"])
                for row in conn.execute(
                    "PRAGMA table_info(action_outcomes)",
                ).fetchall()
            }
            rows = conn.execute(
                "SELECT outcome, COUNT(*) AS count FROM action_outcomes "
                "WHERE profile_id = ? "
                "AND outcome IN ('success', 'failure', 'partial') "
                "GROUP BY outcome",
                (profile_id,),
            ).fetchall()
            recent = conn.execute(
                "SELECT outcome, context_json, timestamp FROM action_outcomes "
                "WHERE profile_id = ? "
                "AND outcome IN ('success', 'failure', 'partial') "
                "ORDER BY timestamp DESC LIMIT ?",
                (profile_id, _RECENT_OUTCOMES_LIMIT),
            ).fetchall()
            reward = (
                _query_reward_telemetry(conn, profile_id, columns)
                if {"reward", "settled"}.issubset(columns)
                else _empty_reward_telemetry()
            )
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.debug("action_outcomes telemetry unavailable: %s", exc)
        return empty
    breakdown = {"success": 0, "failure": 0, "partial": 0}
    for row in rows:
        if row["outcome"] in breakdown:
            breakdown[row["outcome"]] = int(row["count"] or 0)
    return {
        "total": sum(breakdown.values()),
        "breakdown": breakdown,
        "recent": [_outcome_preview(row) for row in recent],
        "reward": reward,
    }


def _empty_reward_telemetry() -> dict:
    return {
        "count": 0,
        "average": None,
        "distribution": {"positive": 0, "neutral": 0, "negative": 0},
        "timeline": [],
        "source": "memory.db:action_outcomes.reward",
        "window_days": _REWARD_TIMELINE_DAYS,
    }


def _query_reward_telemetry(
    conn: sqlite3.Connection,
    profile_id: str,
    columns: set[str],
) -> dict:
    """Aggregate numeric settled labels without materializing reward rows."""
    settled_time = (
        "COALESCE(settled_at, timestamp)"
        if "settled_at" in columns
        else "timestamp"
    )
    aggregate = conn.execute(
        "SELECT COUNT(*) AS count, AVG(reward) AS average, "
        "SUM(CASE WHEN reward > 0.6 THEN 1 ELSE 0 END) AS positive, "
        "SUM(CASE WHEN reward < 0.4 THEN 1 ELSE 0 END) AS negative, "
        "SUM(CASE WHEN reward >= 0.4 AND reward <= 0.6 "
        "THEN 1 ELSE 0 END) AS neutral "
        "FROM action_outcomes WHERE profile_id = ? AND settled = 1 "
        "AND reward IS NOT NULL AND typeof(reward) IN ('integer', 'real')",
        (profile_id,),
    ).fetchone()
    timeline = conn.execute(
        "WITH reward_days AS ("
        f" SELECT substr({settled_time}, 1, 10) AS day,"
        "        reward"
        " FROM action_outcomes"
        " WHERE profile_id = ? AND settled = 1 AND reward IS NOT NULL"
        "   AND typeof(reward) IN ('integer', 'real')"
        f"), latest AS (SELECT MAX(day) AS day FROM reward_days)"
        " SELECT reward_days.day AS date, COUNT(*) AS count,"
        "        AVG(reward_days.reward) AS average"
        " FROM reward_days, latest"
        " WHERE reward_days.day >= date(latest.day, ?)"
        " GROUP BY reward_days.day ORDER BY reward_days.day ASC LIMIT ?",
        (
            profile_id,
            f"-{_REWARD_TIMELINE_DAYS - 1} days",
            _REWARD_TIMELINE_DAYS,
        ),
    ).fetchall()
    count = int(aggregate["count"] or 0)
    return {
        "count": count,
        "average": (
            round(float(aggregate["average"]), 4) if count else None
        ),
        "distribution": {
            "positive": int(aggregate["positive"] or 0),
            "neutral": int(aggregate["neutral"] or 0),
            "negative": int(aggregate["negative"] or 0),
        },
        "timeline": [
            {
                "date": str(row["date"]),
                "count": int(row["count"] or 0),
                "average": round(float(row["average"] or 0.0), 4),
            }
            for row in timeline
            if row["date"]
        ],
        "source": "memory.db:action_outcomes.reward",
        "window_days": _REWARD_TIMELINE_DAYS,
    }


def _outcome_preview(row: sqlite3.Row) -> dict:
    """Return a safe, structured summary without exposing free-form notes."""
    try:
        context = json.loads(str(row["context_json"] or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        context = {}
    action_type = context.get("action_type", "other")
    return {
        "outcome": str(row["outcome"] or "partial"),
        "action_type": str(action_type)[:80],
        "timestamp": str(row["timestamp"] or ""),
        "source": "memory.db:action_outcomes",
    }


@router.get("/api/behavioral/status")
def behavioral_status():
    """Get behavioral learning status for active profile."""
    if not BEHAVIORAL_AVAILABLE:
        return {"available": False, "message": "Behavioral engine not available"}

    try:
        profile = get_active_profile()
        outcome_data = _load_action_outcomes(profile)
        total_outcomes = outcome_data["total"]
        outcome_breakdown = outcome_data["breakdown"]
        recent_outcomes = outcome_data["recent"]
        reward_telemetry = outcome_data["reward"]

        # Patterns
        patterns = []
        cross_project_patterns = []
        cross_project_transfers = 0
        try:
            store = BehavioralPatternStore(str(_learning_db_path()))
            patterns = store.get_patterns(profile_id=profile)
            cross_project_patterns = [
                p for p in patterns
                if isinstance(p, dict) and _is_cross_project_pattern(p)
            ]
            cross_project_transfers = len(cross_project_patterns)
        except Exception as exc:
            logger.warning("pattern store error: %s", exc)

        return {
            "available": True,
            "active_profile": profile,
            "total_outcomes": total_outcomes,
            "outcome_breakdown": outcome_breakdown,
            "outcomes_source": "memory.db:action_outcomes",
            "outcomes_are_finalized": True,
            "outcomes_provenance": "explicit_reports_or_finalized_signals",
            "patterns": patterns,
            "cross_project_transfers": cross_project_transfers,
            "cross_project_patterns": cross_project_patterns,
            "recent_outcomes": recent_outcomes,
            "reward_telemetry": reward_telemetry,
            "stats": {
                "success_count": outcome_breakdown.get("success", 0),
                "failure_count": outcome_breakdown.get("failure", 0),
                "partial_count": outcome_breakdown.get("partial", 0),
                "patterns_count": len(patterns),
            },
        }
    except Exception:
        logger.exception("behavioral_status error")
        return {"available": False, "error": "Internal server error"}


@router.post(
    "/api/behavioral/report-outcome",
    dependencies=[Depends(_authorize_outcome_write)],
)
def report_outcome(request: Request, data: ReportOutcomeRequest):
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
    if not getattr(request.state, "outcome_write_authorized", False):
        _require_write(request)
    if isinstance(data, dict):
        # Preserve the long-standing direct-call API while applying the same
        # constrained model used by FastAPI at the HTTP boundary.
        data = ReportOutcomeRequest.model_validate(data)
    memory_ids = data.memory_ids
    outcome = data.outcome
    action_type = data.action_type
    context_note = data.context

    import sqlite3
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
            conn.execute("BEGIN IMMEDIATE")
            _validate_profile_fact_ids(
                conn,
                profile_id=profile,
                fact_ids=memory_ids,
            )
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

        try:
            from superlocalmemory.learning.source_quality import (
                update_source_quality_for_reward,
            )
            update_source_quality_for_reward(
                memory_db_path=memory_db_path,
                learning_db_path=_learning_db_path(),
                profile_id=profile,
                outcome_id=outcome_id,
                fact_ids=[str(memory_id) for memory_id in memory_ids],
                reward=reward,
            )
        except Exception as exc:  # noqa: BLE001 - outcome write already committed
            logger.debug("source-quality explicit outcome feed skipped: %s", exc)

        return {
            "success": True, "outcome_id": outcome_id,
            "active_profile": profile,
            "reward": reward,
            "message": (
                f"Recorded {outcome} outcome for {len(memory_ids)} "
                f"memories (reward={reward})"
            ),
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("report_outcome error")
        return {"success": False, "error": "Internal server error"}


# --------------------------------------------------------------------------
# v3.4.7: Behavioral Assertions API (for dashboard + external consumers)
# --------------------------------------------------------------------------

@router.get("/api/behavioral/assertions")
def get_assertions(
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    category: str = Query(default="", max_length=100),
    limit: int = Query(default=50, ge=1, le=1000),
):
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

        # F8 fix: use try/finally so conn.close() is guaranteed even when
        # conn.execute() raises (e.g. SQLITE_BUSY under dashboard burst load).
        try:
            rows = conn.execute(query, tuple(params)).fetchall()
        finally:
            conn.close()

        assertions = [dict(r) for r in rows]
        return {
            "assertions": assertions,
            "count": len(assertions),
            "active_profile": profile,
        }
    except Exception:
        logger.exception("get_assertions error")
        return {"assertions": [], "count": 0, "error": "Internal server error"}


@router.get("/api/behavioral/tool-events")
def get_tool_events(
    tool_name: str = "",
    limit: int = Query(default=100, ge=1, le=1000),
):
    """Get recent tool events for dashboard display."""
    try:
        import sqlite3 as _sqlite3
        profile = get_active_profile()
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
    except Exception:
        logger.exception("get_tool_events error")
        return {"events": [], "count": 0, "error": "Internal server error"}


@router.get("/api/behavioral/soft-prompts")
def get_soft_prompts(request: Request):
    """Get active soft prompt templates for dashboard display."""
    _require_read(request)
    try:
        import sqlite3 as _sqlite3
        profile = get_active_profile()
        conn = _sqlite3.connect(str(MEMORY_DIR / "memory.db"))
        conn.row_factory = _sqlite3.Row
        rows = conn.execute(
            "SELECT prompt_id, category, content, confidence, effectiveness, "
            "token_count, active, version, created_at "
            "FROM soft_prompt_templates "
            "WHERE profile_id = ? AND active = 1 "
            "ORDER BY category, prompt_id",
            (profile,),
        ).fetchall()
        conn.close()
        return {"prompts": [dict(zip(
            ["prompt_id", "category", "content", "confidence", "effectiveness",
             "token_count", "active", "version", "created_at"], r
        )) for r in rows], "count": len(rows)}
    except Exception:
        logger.exception("get_soft_prompts error")
        return {"prompts": [], "count": 0, "error": "Internal server error"}


@router.post("/api/v3/tool-event")
def log_tool_event_api(request: Request, data: dict):
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
    _require_write(request)
    try:
        import os
        import sqlite3 as _sqlite3
        from datetime import datetime, timezone

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
    except Exception:
        logger.exception("behavioral route error")
        return {"ok": False, "error": "Internal server error"}
