# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — MCP Resources (6 resources).

slm://recent, slm://stats, slm://clusters, slm://identity,
slm://learning, slm://engagement.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import json
import logging
from typing import Callable

logger = logging.getLogger(__name__)


def register_resources(server, get_engine: Callable) -> None:
    """Register 7 MCP resources on *server*."""

    # ------------------------------------------------------------------
    # 0. slm://context — Active Memory auto-injection
    # ------------------------------------------------------------------
    @server.resource("slm://context")
    async def session_context() -> str:
        """Active session context — auto-injected on MCP connect.

        Returns the most relevant memories for the current session:
        recent decisions, active patterns, and project context.
        AI tools read this automatically on connection to get instant context.
        """
        try:
            from superlocalmemory.hooks.auto_recall import AutoRecall
            from superlocalmemory.mcp._pool_adapter import pool_recall
            # Pass engine so the soft-prompt bridge (engine->AutoInvoker) fires on
            # this real MCP session-context surface; recall_fn still drives recall.
            # Fail-soft: if the engine is unavailable, recall via pool_recall alone.
            try:
                _eng = get_engine()
            except Exception:
                _eng = None
            auto = AutoRecall(
                recall_fn=pool_recall,
                engine=_eng,
                config={"enabled": True, "max_memories_injected": 10, "relevance_threshold": 0.3},
            )
            context = auto.get_session_context(query="recent decisions and important context")
            if not context:
                return "No session context available yet. Use 'remember' to store memories."
            return context
        except Exception as exc:
            return f"Context unavailable: {exc}"

    # ------------------------------------------------------------------
    # 1. slm://recent
    # ------------------------------------------------------------------
    @server.resource("slm://recent")
    async def recent_memories() -> str:
        """Recent memories summary.

        Returns a plain-text summary of the 20 most recent facts
        stored in the active profile.
        """
        try:
            engine = get_engine()
            pid = engine.profile_id
            facts = engine._db.get_all_facts(pid)[:20]
            if not facts:
                return "No memories stored yet."
            lines = [f"Recent memories for profile '{pid}' ({len(facts)} shown):"]
            for f in facts:
                preview = f.content[:100].replace("\n", " ")
                lines.append(
                    f"  [{f.fact_type.value}] {preview} "
                    f"(id={f.fact_id}, {f.created_at[:10]})"
                )
            return "\n".join(lines)
        except Exception as exc:
            return f"Error loading recent memories: {exc}"

    # ------------------------------------------------------------------
    # 2. slm://stats
    # ------------------------------------------------------------------
    @server.resource("slm://stats")
    async def stats() -> str:
        """Memory system statistics.

        Returns a text summary of fact counts, entity counts,
        edge counts, and storage breakdown by type.
        """
        try:
            engine = get_engine()
            pid = engine.profile_id
            fact_count = engine._db.get_fact_count(pid)

            entity_rows = engine._db.execute(
                "SELECT COUNT(*) AS c FROM canonical_entities WHERE profile_id = ?",
                (pid,),
            )
            entity_count = int(dict(entity_rows[0])["c"]) if entity_rows else 0

            edge_rows = engine._db.execute(
                "SELECT COUNT(*) AS c FROM graph_edges WHERE profile_id = ?",
                (pid,),
            )
            edge_count = int(dict(edge_rows[0])["c"]) if edge_rows else 0

            scene_rows = engine._db.execute(
                "SELECT COUNT(*) AS c FROM memory_scenes WHERE profile_id = ?",
                (pid,),
            )
            scene_count = int(dict(scene_rows[0])["c"]) if scene_rows else 0

            lines = [
                f"SuperLocalMemory V3 Statistics (profile: {pid})",
                f"  Mode: {engine._config.mode.value.upper()}",
                f"  Facts: {fact_count}",
                f"  Entities: {entity_count}",
                f"  Edges: {edge_count}",
                f"  Scenes: {scene_count}",
            ]
            return "\n".join(lines)
        except Exception as exc:
            return f"Error loading stats: {exc}"

    # ------------------------------------------------------------------
    # 3. slm://clusters
    # ------------------------------------------------------------------
    @server.resource("slm://clusters")
    async def clusters() -> str:
        """Memory scene clusters.

        Returns a summary of memory scenes (topic clusters)
        with their themes and fact counts.
        """
        try:
            engine = get_engine()
            pid = engine.profile_id
            scenes = engine._db.get_all_scenes(pid)
            if not scenes:
                return "No memory scenes/clusters formed yet."
            lines = [f"Memory Scenes ({len(scenes)} clusters):"]
            for s in scenes[:30]:
                theme = s.theme[:80] if s.theme else "(no theme)"
                lines.append(
                    f"  [{s.scene_id[:8]}] {theme} "
                    f"({len(s.fact_ids)} facts, {len(s.entity_ids)} entities)"
                )
            return "\n".join(lines)
        except Exception as exc:
            return f"Error loading clusters: {exc}"

    # ------------------------------------------------------------------
    # 4. slm://identity
    # ------------------------------------------------------------------
    @server.resource("slm://identity")
    async def coding_identity() -> str:
        """Active profile identity and coding context.

        Returns profile name, personality description, mode, and
        top entities with fact counts for context-aware assistance.
        """
        try:
            engine = get_engine()
            pid = engine.profile_id
            mode = engine._config.mode.value.upper()

            # Top entities by fact count
            entity_rows = engine._db.execute(
                "SELECT canonical_name, fact_count FROM canonical_entities "
                "WHERE profile_id = ? ORDER BY fact_count DESC LIMIT 10",
                (pid,),
            )
            entities = [
                f"{dict(r)['canonical_name']} ({dict(r)['fact_count']} facts)"
                for r in entity_rows
            ]

            lines = [
                f"Profile: {pid}",
                f"Mode: {mode}",
                f"Top entities: {', '.join(entities) if entities else 'none yet'}",
            ]
            return "\n".join(lines)
        except Exception as exc:
            return f"Error loading identity: {exc}"

    # ------------------------------------------------------------------
    # 5. slm://learning
    # ------------------------------------------------------------------
    @server.resource("slm://learning")
    async def learning_status() -> str:
        """Learning system status.

        Returns adaptive learning health, pattern summary, and
        outcome tracking statistics.
        """
        try:
            engine = get_engine()
            pid = engine.profile_id

            # Behavioral patterns summary
            try:
                from superlocalmemory.learning.behavioral import BehavioralPatternStore
                store = BehavioralPatternStore(engine._db.db_path)
                summary = store.get_summary(pid)
            except Exception as exc:
                logger.warning("learning_status behavioral summary failed: %s", exc)
                summary = {}

            # Outcome stats
            try:
                outcome_rows = engine._db.execute(
                    "SELECT outcome, COUNT(*) AS c FROM action_outcomes "
                    "WHERE profile_id = ? GROUP BY outcome",
                    (pid,),
                )
                outcomes = {dict(r)["outcome"]: dict(r)["c"] for r in outcome_rows}
            except Exception as exc:
                logger.warning("learning_status outcome query failed: %s", exc)
                outcomes = {}

            lines = [
                f"Learning Status (profile: {pid})",
                f"  Behavioral patterns: {json.dumps(summary) if summary else 'none detected'}",
                f"  Outcome tracking: {json.dumps(outcomes) if outcomes else 'no outcomes recorded'}",
            ]
            return "\n".join(lines)
        except Exception as exc:
            return f"Error loading learning status: {exc}"

    # ------------------------------------------------------------------
    # 6. slm://engagement
    # ------------------------------------------------------------------
    @server.resource("slm://engagement")
    async def engagement() -> str:
        """Engagement metrics.

        Returns memory usage activity: recent stores, recalls,
        and top-accessed facts.
        """
        try:
            engine = get_engine()
            pid = engine.profile_id

            # Recent activity via audit trail
            try:
                audit_rows = engine._db.execute(
                    "SELECT action, COUNT(*) AS c FROM compliance_audit "
                    "WHERE profile_id = ? GROUP BY action",
                    (pid,),
                )
                activity = {dict(r)["action"]: dict(r)["c"] for r in audit_rows}
            except Exception as exc:
                logger.warning("engagement audit-trail query failed: %s", exc)
                activity = {}

            # Top-accessed facts
            top_rows = engine._db.execute(
                "SELECT fact_id, content, access_count FROM atomic_facts "
                "WHERE profile_id = ? AND access_count > 0 "
                "ORDER BY access_count DESC LIMIT 5",
                (pid,),
            )
            top_facts = [
                f"{dict(r)['content'][:60]}... (accessed {dict(r)['access_count']}x)"
                for r in top_rows
            ]

            lines = [
                f"Engagement (profile: {pid})",
                f"  Activity: {json.dumps(activity) if activity else 'no activity'}",
                f"  Top facts: {'; '.join(top_facts) if top_facts else 'none accessed'}",
            ]
            return "\n".join(lines)
        except Exception as exc:
            return f"Error loading engagement: {exc}"
