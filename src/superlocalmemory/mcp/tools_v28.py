# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — V2.8 Ported MCP Tools (6 tools).

report_outcome, get_lifecycle_status, set_retention_policy,
compact_memories, get_behavioral_patterns, audit_trail.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
from typing import Callable


from superlocalmemory.mcp.tool_hints import ADDS, DESTROYS, DESTROYS_ONCE, READ_ONLY
from superlocalmemory.core.admission import admits
from superlocalmemory.core.operation_request import OperationKind
from superlocalmemory.mcp.request_profile import tool_profile
from superlocalmemory.mcp.shared import authorize_mcp_mutation

logger = logging.getLogger(__name__)


def register_v28_tools(server, get_engine: Callable) -> None:
    """Register 6 V2.8-ported tools on *server*."""

    # ------------------------------------------------------------------
    # 1. report_outcome
    # ------------------------------------------------------------------
    @server.tool(title="Report Recall Outcome", annotations=ADDS)
    @admits(OperationKind.REMEMBER)
    async def report_outcome(
        memory_ids: str,
        outcome: str,
        context: str = "",
        recall_query_id: str = "",
        profile_id: str = "",
        agent_id: str = "",
        report_id: str = "",
    ) -> dict:
        """Report outcome of using recalled memories.

        Feeds into the adaptive learning loop to improve future retrieval.
        Valid outcomes: success, failure, partial.

        Args:
            memory_ids: Comma-separated list of fact/memory IDs.
            outcome: One of 'success', 'failure', 'partial'.
            context: Optional freetext context about the outcome.
            recall_query_id: The ``query_id`` that came back with the recall
                this report is about. Passing it ties the report to that exact
                answer; leaving it out falls back to matching on which memories
                overlap, within a time window.
            profile_id: The profile the recall was made in (empty = the active
                one).
            agent_id: Optional name of the reporting agent (provenance).
            report_id: Optional idempotency key; re-sending a report with the
                same key records nothing new.

        Returns ``reward`` (success 1.0 / partial 0.5 / failure 0.0),
        ``duplicate`` when the same report was already recorded, and
        ``attributed`` when it is tied to one recall by ``recall_query_id``.
        """
        try:
            engine = get_engine()
            pid, refused = tool_profile(engine, profile_id)
            if refused:
                return refused
            authorization = authorize_mcp_mutation(
                engine,
                "update",
                mutation_source="mcp-report-outcome",
                profile_id=pid,
            )
            from superlocalmemory.learning.outcomes import VALID_OUTCOMES

            if outcome not in VALID_OUTCOMES:
                # Answered here rather than as a stack trace: the caller is an
                # assistant that can correct itself if told what is allowed.
                return {
                    "success": False,
                    "error": (f"outcome must be one of {sorted(VALID_OUTCOMES)}, not {outcome!r}"),
                }
            ids = [mid.strip() for mid in memory_ids.split(",") if mid.strip()]
            # A remote app reports on memories it can see; the rest are left out, and if
            # none are left the answer is the one for an id that is not there.
            from superlocalmemory.mcp.remote_visibility import unseen_ids

            unseen = unseen_ids(engine._db, pid, ids)
            if unseen:
                ids = [mid for mid in ids if mid not in unseen]
            if not ids:
                return {"success": False, "error": "Memory not found."}
            from pathlib import Path as _Path

            from superlocalmemory.learning.outcome_events import (
                OutcomeReport,
                UnknownMemories,
                write_outcome,
            )

            # The one writer every surface shares (audit BRN-05): reward,
            # settlement fields, attribution, provenance and idempotence.
            try:
                written = write_outcome(
                    OutcomeReport(
                        profile_id=pid,
                        fact_ids=tuple(ids),
                        outcome=outcome,
                        surface="mcp",
                        recall_query_id=recall_query_id or "",
                        agent_id=agent_id or "",
                        note=context or "",
                        idempotency_key=report_id or "",
                        query="[mcp_feedback]",
                    ),
                    memory_db_path=_Path(engine._db.db_path),
                    learning_db_path=_Path(engine._config.base_dir) / "learning.db",
                )
            except UnknownMemories:
                return {"success": False, "error": "Memory not found."}
            authorization.complete()
            return {
                "success": True,
                "outcome_id": written.outcome_id,
                "outcome": outcome,
                "reward": written.reward,
                "duplicate": not written.created,
                "attributed": written.attributed,
            }
        except Exception as exc:
            logger.exception("report_outcome failed")
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # 2. get_lifecycle_status
    # ------------------------------------------------------------------
    @server.tool(title="Get Lifecycle Status", annotations=READ_ONLY)
    async def get_lifecycle_status(limit: int = 50, profile_id: str = "") -> dict:
        """Get lifecycle state distribution for stored memories.

        Shows counts per lifecycle state (active, warm, cold, archived)
        and the most recent facts in each state.

        Args:
            limit: Maximum facts to inspect (default 50).
            profile_id: The profile to inspect (empty = the active one).
        """
        try:
            engine = get_engine()
            pid, refused = tool_profile(engine, profile_id)
            if refused:
                return refused
            from superlocalmemory.mcp.remote_visibility import visible_facts

            everything = engine._db.get_all_facts(pid)
            facts: list = []
            step = max(int(limit), 1)
            # The view check runs on a page at a time, not on every fact of the profile.
            for start in range(0, len(everything), step):
                facts += visible_facts(engine._db, pid, everything[start : start + step])
                if len(facts) >= limit:
                    break
            facts = facts[:limit]
            states: dict[str, list[dict]] = {
                "active": [],
                "warm": [],
                "cold": [],
                "archived": [],
            }
            for f in facts:
                state = f.lifecycle.value
                if state in states and len(states[state]) < 10:
                    states[state].append(
                        {
                            "fact_id": f.fact_id,
                            "content": f.content[:80],
                            "access_count": f.access_count,
                            "created_at": f.created_at,
                        }
                    )
            counts = {k: len(v) for k, v in states.items()}
            return {"success": True, "counts": counts, "samples": states}
        except Exception as exc:
            logger.exception("get_lifecycle_status failed")
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # 3. set_retention_policy
    # ------------------------------------------------------------------
    @server.tool(title="Set Retention Policy", annotations=DESTROYS_ONCE)
    @admits(OperationKind.CONSOLIDATE)
    async def set_retention_policy(
        cold_after_days: int = 30,
        archive_after_days: int = 90,
    ) -> dict:
        """Set memory retention policy thresholds.

        Controls when memories transition from active to warm, cold,
        and archived states. Coupled with Langevin dynamics when available.

        Args:
            cold_after_days: Days of inactivity before cold state (default 30).
            archive_after_days: Days before archival (default 90).
        """
        try:
            engine = get_engine()
            pid, refused = tool_profile(engine, "")
            if refused:
                return refused
            authorization = authorize_mcp_mutation(
                engine,
                "update",
                mutation_source="mcp-retention-policy",
                profile_id=pid,
            )
            from superlocalmemory.learning.lifecycle_policy import set_configured_policy

            set_configured_policy(engine._db, pid, cold_after_days, archive_after_days)
            authorization.complete()
            return {
                "success": True,
                "profile_id": pid,
                "cold_after_days": cold_after_days,
                "archive_after_days": archive_after_days,
            }
        except Exception as exc:
            logger.exception("set_retention_policy failed")
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # 4. compact_memories
    # ------------------------------------------------------------------
    @server.tool(title="Compact Memories", annotations=DESTROYS)
    @admits(OperationKind.CONSOLIDATE)
    async def compact_memories(dry_run: bool = True) -> dict:
        """Compact memory store by archiving cold/stale facts.

        Transitions eligible memories from cold to archived state.
        Run with dry_run=True first to preview changes.

        Args:
            dry_run: If True, only preview without making changes (default True).
        """
        try:
            engine = get_engine()
            pid = engine.profile_id
            from superlocalmemory.learning.lifecycle_policy import LifecycleManager

            mgr = LifecycleManager(engine._db)
            facts = engine._db.get_all_facts(pid)
            candidates = []
            for f in facts:
                new_state = mgr.get_lifecycle_state(f)
                if new_state != f.lifecycle:
                    candidates.append(
                        {
                            "fact_id": f.fact_id,
                            "current": f.lifecycle.value,
                            "proposed": new_state.value,
                        }
                    )
            if not dry_run:
                authorization = authorize_mcp_mutation(
                    engine,
                    "update",
                    mutation_source="mcp-compact-memories",
                    profile_id=pid,
                )
                from superlocalmemory.storage.retention_lifecycle import compact_facts

                candidates = compact_facts(engine._db, candidates, pid)
                authorization.complete()
            return {
                "success": True,
                "dry_run": dry_run,
                "candidates": len(candidates),
                "transitions": candidates[:20],
            }
        except Exception as exc:
            logger.exception("compact_memories failed")
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # 5. get_behavioral_patterns
    # ------------------------------------------------------------------
    @server.tool(title="Get Behavioral Patterns", annotations=READ_ONLY)
    async def get_behavioral_patterns(limit: int = 20, profile_id: str = "") -> dict:
        """Get detected behavioral patterns for the active profile (or ``profile_id``).

        Returns patterns such as topic interests, refinement habits,
        and time-of-day usage with confidence scores.

        Args:
            limit: Maximum patterns to return (default 20).
        """
        try:
            engine = get_engine()
            pid, refused = tool_profile(engine, profile_id)
            if refused:
                return refused
            from superlocalmemory.learning.behavioral import BehavioralPatternStore

            store = BehavioralPatternStore(engine._db.db_path)
            patterns = store.get_patterns(pid, limit=limit)
            summary = store.get_summary(pid)
            return {
                "success": True,
                "patterns": patterns,
                "summary": summary,
                "count": len(patterns),
            }
        except Exception as exc:
            logger.exception("get_behavioral_patterns failed")
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # 6. audit_trail
    # ------------------------------------------------------------------
    @server.tool(title="Get Audit Trail", annotations=READ_ONLY)
    async def audit_trail(limit: int = 50) -> dict:
        """Get compliance audit trail for the active profile.

        Returns logged operations (store, retrieve, delete, export)
        for GDPR and EU AI Act compliance reporting.

        Args:
            limit: Maximum audit entries to return (default 50).
        """
        try:
            engine = get_engine()
            from superlocalmemory.compliance.gdpr import GDPRCompliance

            gdpr = GDPRCompliance(engine._db)
            entries = gdpr.get_audit_trail(engine.profile_id, limit=limit)
            return {
                "success": True,
                "entries": entries,
                "count": len(entries),
            }
        except Exception as exc:
            logger.exception("audit_trail failed")
            return {"success": False, "error": str(exc)}
