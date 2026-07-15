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

from superlocalmemory.mcp.shared import authorize_mcp_mutation

logger = logging.getLogger(__name__)


def register_v28_tools(server, get_engine: Callable) -> None:
    """Register 6 V2.8-ported tools on *server*."""

    # ------------------------------------------------------------------
    # 1. report_outcome
    # ------------------------------------------------------------------
    @server.tool()
    async def report_outcome(
        memory_ids: str,
        outcome: str,
        context: str = "",
    ) -> dict:
        """Report outcome of using recalled memories.

        Feeds into the adaptive learning loop to improve future retrieval.
        Valid outcomes: success, failure, partial.

        Args:
            memory_ids: Comma-separated list of fact/memory IDs.
            outcome: One of 'success', 'failure', 'partial'.
            context: Optional freetext context about the outcome.
        """
        try:
            engine = get_engine()
            authorization = authorize_mcp_mutation(
                engine,
                "update",
                mutation_source="mcp-report-outcome",
            )
            from superlocalmemory.learning.outcomes import OutcomeTracker
            tracker = OutcomeTracker(engine._db)
            ids = [mid.strip() for mid in memory_ids.split(",") if mid.strip()]
            ctx = {"note": context} if context else None
            ao = tracker.record_outcome(
                query="[mcp_feedback]",
                fact_ids=ids,
                outcome=outcome,
                profile_id=engine.profile_id,
                context=ctx,
            )

            # v3.4.7: Bridge outcomes → learning signals for two-way learning.
            # Previously, outcomes were stored but never created learning signals.
            try:
                from superlocalmemory.learning.feedback import FeedbackCollector
                from superlocalmemory.infra.data_root import state_path
                learning_db = state_path("learning.db")
                collector = FeedbackCollector(learning_db)
                signal_map = {"success": ("user_positive", 1.0),
                              "failure": ("user_negative", 0.0),
                              "partial": ("user_correction", 0.5)}
                sig_type, sig_val = signal_map.get(outcome, ("user_correction", 0.5))
                for fid in ids:
                    collector.record_explicit(
                        profile_id=engine.profile_id,
                        fact_id=fid,
                        signal_type=sig_type,
                        value=sig_val,
                    )
            except Exception as exc2:
                logger.debug("Outcome→signal bridge: %s", exc2)

            authorization.complete()
            return {"success": True, "outcome_id": ao.outcome_id, "outcome": outcome}
        except Exception as exc:
            logger.exception("report_outcome failed")
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # 2. get_lifecycle_status
    # ------------------------------------------------------------------
    @server.tool()
    async def get_lifecycle_status(limit: int = 50) -> dict:
        """Get lifecycle state distribution for stored memories.

        Shows counts per lifecycle state (active, warm, cold, archived)
        and the most recent facts in each state.

        Args:
            limit: Maximum facts to inspect (default 50).
        """
        try:
            engine = get_engine()
            pid = engine.profile_id
            facts = engine._db.get_all_facts(pid)[:limit]
            states: dict[str, list[dict]] = {
                "active": [], "warm": [], "cold": [], "archived": [],
            }
            for f in facts:
                state = f.lifecycle.value
                if state in states and len(states[state]) < 10:
                    states[state].append({
                        "fact_id": f.fact_id,
                        "content": f.content[:80],
                        "access_count": f.access_count,
                        "created_at": f.created_at,
                    })
            counts = {k: len(v) for k, v in states.items()}
            return {"success": True, "counts": counts, "samples": states}
        except Exception as exc:
            logger.exception("get_lifecycle_status failed")
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # 3. set_retention_policy
    # ------------------------------------------------------------------
    @server.tool()
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
            authorization = authorize_mcp_mutation(
                engine,
                "update",
                mutation_source="mcp-retention-policy",
            )
            engine._db.set_config("retention_cold_days", str(cold_after_days))
            engine._db.set_config("retention_archive_days", str(archive_after_days))
            authorization.complete()
            return {
                "success": True,
                "cold_after_days": cold_after_days,
                "archive_after_days": archive_after_days,
            }
        except Exception as exc:
            logger.exception("set_retention_policy failed")
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # 4. compact_memories
    # ------------------------------------------------------------------
    @server.tool()
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
            from superlocalmemory.compliance.lifecycle import LifecycleManager
            mgr = LifecycleManager(engine._db)
            facts = engine._db.get_all_facts(pid)
            candidates = []
            for f in facts:
                new_state = mgr.get_lifecycle_state(f)
                if new_state != f.lifecycle:
                    candidates.append({
                        "fact_id": f.fact_id,
                        "current": f.lifecycle.value,
                        "proposed": new_state.value,
                    })
            if not dry_run:
                authorization = authorize_mcp_mutation(
                    engine,
                    "update",
                    mutation_source="mcp-compact-memories",
                    profile_id=pid,
                )
                for c in candidates:
                    engine._db.update_fact(
                        c["fact_id"], {"lifecycle": c["proposed"]},
                    )
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
    @server.tool()
    async def get_behavioral_patterns(limit: int = 20) -> dict:
        """Get detected behavioral patterns for the active profile.

        Returns patterns such as topic interests, refinement habits,
        and time-of-day usage with confidence scores.

        Args:
            limit: Maximum patterns to return (default 20).
        """
        try:
            engine = get_engine()
            from superlocalmemory.learning.behavioral import BehavioralPatternStore
            store = BehavioralPatternStore(engine._db.db_path)
            patterns = store.get_patterns(engine.profile_id, limit=limit)
            summary = store.get_summary(engine.profile_id)
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
    @server.tool()
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
