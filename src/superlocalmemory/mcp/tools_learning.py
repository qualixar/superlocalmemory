# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SLM v3.4.7 "The Learning Brain" — Learning MCP Tools.

Two-way learning tools:
  - log_tool_event: Passive tool usage telemetry
  - get_assertions: Retrieve behavioral assertions
  - reinforce_assertion: Explicitly reinforce a learned pattern
  - contradict_assertion: Mark a learned pattern as wrong

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from mcp.types import ToolAnnotations

from superlocalmemory.core.admission import admits
from superlocalmemory.core.operation_request import OperationKind
from superlocalmemory.mcp.shared import authorize_mcp_mutation

logger = logging.getLogger(__name__)

_MAX_SUMMARY_LEN = 500  # Truncate input/output summaries


def settle_pending_session_outcomes(
    memory_db_path: str | Path,
    *,
    profile_id: str,
    session_id: str,
    evidence_only: bool = False,
) -> dict[str, int]:
    """Finalize real pending recalls for one host-owned session.

    This is deliberately a narrow adapter around ``EngagementRewardModel``:
    it selects only pending rows belonging to the supplied active profile and
    exact host session, then delegates every state transition and reward
    calculation to the established finalizer.  No feedback is inferred here.
    Repeating the call is a no-op because settled rows are not selected.
    """
    session_id = session_id.strip()
    if not session_id:
        return {"selected": 0, "settled": 0}

    path = Path(memory_db_path)
    try:
        with sqlite3.connect(str(path), timeout=2.0) as conn:
            rows = conn.execute(
                "SELECT outcome_id, signals_json FROM pending_outcomes "
                "WHERE profile_id=? AND session_id=? AND status='pending'",
                (profile_id, session_id),
            ).fetchall()
    except sqlite3.Error as exc:
        raise RuntimeError("pending outcome lookup failed") from exc

    if evidence_only:
        from superlocalmemory.learning.reward import _carries_evidence

        rows = [
            row for row in rows
            if _carries_evidence(json.loads(row[1] or "{}"))
        ]

    if not rows:
        return {"selected": 0, "settled": 0}

    from superlocalmemory.learning.reward import EngagementRewardModel

    model = EngagementRewardModel(memory_db_path=str(path))
    try:
        for outcome_id, _signals_json in rows:
            # Preserve the established keyword-only finalization contract.
            model.finalize_outcome(outcome_id=outcome_id)
    finally:
        model.close()

    outcome_ids = [row[0] for row in rows]
    placeholders = ",".join("?" for _ in outcome_ids)
    try:
        with sqlite3.connect(str(path), timeout=2.0) as conn:
            settled = conn.execute(
                "SELECT COUNT(*) FROM pending_outcomes "
                "WHERE profile_id=? AND session_id=? AND status='settled' "
                f"AND outcome_id IN ({placeholders})",
                (profile_id, session_id, *outcome_ids),
            ).fetchone()[0]
    except sqlite3.Error as exc:
        raise RuntimeError("pending outcome settlement verification failed") from exc
    return {"selected": len(outcome_ids), "settled": int(settled)}


def register_learning_tools(server, get_engine: Callable) -> None:
    """Register learning MCP tools for two-way intelligence."""

    @server.tool()
    @admits(OperationKind.REMEMBER)
    async def log_tool_event(
        tool_name: str,
        event_type: str = "invoke",
        input_summary: str = "",
        output_summary: str = "",
        duration_ms: int = 0,
        metadata: str = "{}",
        session_id: str = "",
        agent_id: str = "",
        project_path: str = "",
    ) -> dict:
        """Log a tool usage event for behavioral learning.

        Passive telemetry — low overhead, no LLM calls.
        Events feed into the behavioral assertion mining pipeline
        during consolidation. Accessible via MCP, CLI, and hooks.

        Args:
            tool_name: Name of the tool (e.g. "Read", "Edit", "Bash")
            event_type: One of 'invoke', 'complete', 'error', 'correction'
            input_summary: Truncated input (max 500 chars, auto-scrubbed)
            output_summary: Truncated output (max 500 chars, auto-scrubbed)
            duration_ms: Execution time in milliseconds
            metadata: JSON string with additional context
        """
        engine = get_engine()
        now = datetime.now(timezone.utc).isoformat()
        from superlocalmemory.mcp.session_binding import resolve_session_id
        effective_session_id = resolve_session_id(
            session_id, agent_id=agent_id or "mcp_client", allow_agent_fallback=True,
        )
        effective_project_path = project_path or (
            os.environ.get("CLAUDE_PROJECT_DIR")
            or os.environ.get("PROJECT_PATH")
            or os.getcwd()
        )

        # Scrub and truncate
        input_clean = _scrub(input_summary[:_MAX_SUMMARY_LEN])
        output_clean = _scrub(output_summary[:_MAX_SUMMARY_LEN])

        try:
            authorization = authorize_mcp_mutation(
                engine,
                "update",
                mutation_source="mcp-log-tool-event",
                profile_id=engine.profile_id,
                content_preview=tool_name,
            )
            engine._db.execute(
                "INSERT INTO tool_events "
                "(session_id, profile_id, project_path, tool_name, event_type, "
                " input_summary, output_summary, duration_ms, metadata, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (effective_session_id, engine.profile_id, effective_project_path, tool_name,
                 event_type, input_clean, output_clean, duration_ms, metadata, now),
            )
            authorization.complete()
            return {"success": True, "tool": tool_name, "event": event_type,
                    "session_id": effective_session_id}
        except Exception as exc:
            logger.debug("log_tool_event failed: %s", exc)
            return {"success": False, "error": str(exc)}

    @server.tool()
    @admits(OperationKind.REMEMBER)
    async def settle_session_outcomes(
        session_id: str, agent_id: str = "", finalize: bool = False,
    ) -> dict:
        """Settle pending recall outcomes for one exact host session.

        Native hosts call this after each turn and at finalization. Per-turn
        calls settle only recalls that have actual evidence; finalization also
        clears evidence-free pending rows. The operation never treats a hook
        boundary itself as feedback. ``session_id`` must be explicit.
        """
        if not session_id or not session_id.strip():
            return {"success": False, "error": "session_id is required"}
        engine = get_engine()
        try:
            authorization = authorize_mcp_mutation(
                engine,
                "update",
                mutation_source="mcp-settle-session-outcomes",
                profile_id=engine.profile_id,
                content_preview=agent_id[:100],
            )
            from superlocalmemory.hooks._outcome_common import memory_db_path

            result = await asyncio.to_thread(
                settle_pending_session_outcomes,
                memory_db_path(),
                profile_id=engine.profile_id,
                session_id=session_id,
                evidence_only=not finalize,
            )
            authorization.complete()
            return {
                "success": True,
                "session_id": session_id.strip(),
                "agent_id": agent_id.strip(),
                **result,
            }
        except Exception as exc:
            logger.debug("settle_session_outcomes failed: %s", exc)
            return {"success": False, "error": str(exc)}

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    async def get_assertions(
        min_confidence: float = 0.0,
        category: str = "",
        project_path: str = "",
        limit: int = 50,
    ) -> dict:
        """Get learned behavioral assertions.

        Behavioral assertions are patterns SLM discovered from your usage:
        trigger conditions paired with recommended actions, each with a
        confidence score that evolves over time.

        Args:
            min_confidence: Minimum confidence threshold (0.0-1.0)
            category: Filter by category (workflow, code_style, tool_preference, communication)
            project_path: Filter by project (empty = all including global)
            limit: Maximum results
        """
        engine = get_engine()
        try:
            query = (
                "SELECT id, trigger_condition, action, category, confidence, "
                "evidence_count, reinforcement_count, contradiction_count, "
                "project_path, source, created_at, updated_at "
                "FROM behavioral_assertions "
                "WHERE profile_id = ? AND confidence >= ?"
            )
            params: list = [engine.profile_id, min_confidence]

            if category:
                query += " AND category = ?"
                params.append(category)
            if project_path:
                query += " AND (project_path = ? OR project_path = '')"
                params.append(project_path)

            query += " ORDER BY confidence DESC LIMIT ?"
            params.append(limit)

            rows = engine._db.execute(query, tuple(params))
            assertions = [dict(r) for r in rows]
            return {
                "assertions": assertions,
                "count": len(assertions),
                "min_confidence": min_confidence,
            }
        except Exception as exc:
            logger.debug("get_assertions failed: %s", exc)
            return {"assertions": [], "count": 0, "error": str(exc)}

    @server.tool()
    @admits(OperationKind.CORRECT)
    async def reinforce_assertion(assertion_id: str) -> dict:
        """Reinforce a behavioral assertion (increase confidence).

        Call this when an assertion's recommendation was helpful.
        Confidence increases via Bayesian update toward 1.0.

        Args:
            assertion_id: The assertion ID to reinforce
        """
        engine = get_engine()
        try:
            authorization = authorize_mcp_mutation(
                engine,
                "update",
                mutation_source="mcp-reinforce-assertion",
                profile_id=engine.profile_id,
                fact_id=assertion_id,
            )
            result = _update_assertion_confidence(
                engine._db, assertion_id, reinforce=True,
                profile_id=engine.profile_id,
            )
            if result.get("success"):
                authorization.complete()
            return result
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    @server.tool()
    @admits(OperationKind.FORGET)
    async def contradict_assertion(assertion_id: str) -> dict:
        """Contradict a behavioral assertion (decrease confidence).

        Call this when an assertion's recommendation was wrong.
        Confidence decays by 30%. Assertions below 0.2 are auto-deleted.

        Args:
            assertion_id: The assertion ID to contradict
        """
        engine = get_engine()
        try:
            authorization = authorize_mcp_mutation(
                engine,
                "delete",
                mutation_source="mcp-contradict-assertion",
                profile_id=engine.profile_id,
                fact_id=assertion_id,
            )
            result = _update_assertion_confidence(
                engine._db, assertion_id, reinforce=False,
                profile_id=engine.profile_id,
            )
            if result.get("success"):
                authorization.complete()
            return result
        except Exception as exc:
            return {"success": False, "error": str(exc)}


def _update_assertion_confidence(
    db, assertion_id: str, reinforce: bool, profile_id: str,
) -> dict:
    """Bayesian confidence update for behavioral assertions, scoped to one profile."""
    now = datetime.now(timezone.utc).isoformat()
    try:
        row = db.execute(
            "SELECT confidence, reinforcement_count, contradiction_count "
            "FROM behavioral_assertions WHERE id = ? AND profile_id = ?",
            (assertion_id, profile_id),
        )
        rows = list(row)
        if not rows:
            return {"success": False, "error": "assertion not found"}

        r = dict(rows[0])
        old_conf = r["confidence"]

        if reinforce:
            new_conf = old_conf + (1.0 - old_conf) * 0.15  # Bayesian nudge toward 1.0
            db.execute(
                "UPDATE behavioral_assertions SET confidence = ?, "
                "reinforcement_count = reinforcement_count + 1, "
                "last_reinforced_at = ?, updated_at = ? "
                "WHERE id = ? AND profile_id = ?",
                (round(new_conf, 4), now, now, assertion_id, profile_id),
            )
        else:
            new_conf = old_conf * 0.7  # 30% decay
            db.execute(
                "UPDATE behavioral_assertions SET confidence = ?, "
                "contradiction_count = contradiction_count + 1, "
                "last_contradicted_at = ?, updated_at = ? "
                "WHERE id = ? AND profile_id = ?",
                (round(new_conf, 4), now, now, assertion_id, profile_id),
            )
            # Auto-delete if confidence drops below 0.2
            if new_conf < 0.2:
                db.execute(
                    "DELETE FROM behavioral_assertions WHERE id = ? AND profile_id = ?",
                    (assertion_id, profile_id),
                )
                return {
                    "success": True, "action": "deleted",
                    "reason": f"confidence dropped to {new_conf:.3f} (below 0.2 threshold)",
                }

        return {
            "success": True,
            "action": "reinforced" if reinforce else "contradicted",
            "old_confidence": round(old_conf, 4),
            "new_confidence": round(new_conf, 4),
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def _scrub(text: str) -> str:
    """Remove potential secrets from telemetry text."""
    import re
    # Remove API keys, tokens, passwords
    text = re.sub(r'\b(sk-|pk-|api[_-]?key[_-]?)[A-Za-z0-9_-]{10,}\b', '[REDACTED]', text)
    text = re.sub(r'\b[A-Za-z0-9+/]{40,}={0,2}\b', '[REDACTED]', text)  # Base64 strings
    text = re.sub(r'password\s*[=:]\s*\S+', 'password=[REDACTED]', text, flags=re.IGNORECASE)
    return text
