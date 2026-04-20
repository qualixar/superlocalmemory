# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Core MCP Tools (13 tools).

remember, recall, search, fetch, list_recent, get_status, build_graph,
switch_profile, backup_status, memory_used, get_learned_patterns,
correct_pattern, get_attribution.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

_DB_PATH = str(Path.home() / ".superlocalmemory" / "memory.db")


def _emit_event(event_type: str, payload: dict | None = None,
                source_agent: str = "mcp_client") -> None:
    """Emit an event to the EventBus (best-effort, never raises)."""
    try:
        from superlocalmemory.infra.event_bus import EventBus
        bus = EventBus.get_instance(_DB_PATH)
        bus.emit(event_type, payload=payload, source_agent=source_agent,
                 source_protocol="mcp")
    except Exception:
        pass


def _record_recall_hits(
    get_engine: Callable,
    query: str,
    results: list[dict],
    *,
    query_id: str = "",
    fact_ids_candidates: list[str] | None = None,
) -> None:
    """Record honest shown-state signals (LLD-02 §4.9).

    v3.4.22: No more fake positives. For every candidate we enqueue a
    ``shown`` / ``not_shown`` flip based on whether it was returned in the
    top-K presented to the user. Outcome/reward arrives in v3.4.22 via the
    action-outcomes pipeline.

    Non-blocking: all work funnels through ``signals.enqueue_shown_flip``
    (module-level queue + background drain). Failures are swallowed —
    signal quality is never load-bearing on recall correctness.
    """
    try:
        from pathlib import Path
        from superlocalmemory.learning.signals import (
            LearningSignals,
            enqueue_shown_flip,
        )

        engine = get_engine()
        pid = engine.profile_id
        slm_dir = Path.home() / ".superlocalmemory"

        shown_ids = [r.get("fact_id", "") for r in results[:10]
                     if r.get("fact_id")]
        candidates = (fact_ids_candidates
                      if fact_ids_candidates is not None
                      else shown_ids)
        if not candidates:
            return

        # Shown-flip enqueue per §4.9. No synthetic positives.
        shown_set = set(shown_ids)
        if query_id:
            for fid in candidates:
                enqueue_shown_flip(query_id, fid, shown=(fid in shown_set))

        # Legacy zero-cost signals — unchanged (co-retrieval + confidence).
        try:
            signals = LearningSignals(slm_dir / "learning.db")
            signals.record_co_retrieval(pid, shown_ids)
        except Exception:
            pass
        try:
            mem_db = str(slm_dir / "memory.db")
            for fid in shown_ids[:5]:
                LearningSignals.boost_confidence(mem_db, fid)
        except Exception:
            pass
    except Exception:
        pass


def register_core_tools(server, get_engine: Callable) -> None:
    """Register the 13 core MCP tools on *server*."""

    @server.tool()
    async def remember(
        content: str, tags: str = "", project: str = "",
        importance: int = 5, session_id: str = "",
        agent_id: str = "mcp_client",
    ) -> dict:
        """Store content to memory with intelligent indexing.

        Extracts atomic facts, resolves entities, builds graph edges,
        and indexes for 4-channel retrieval.
        """
        import asyncio
        try:
            # V3.3.27: Store-first pattern — write to pending.db immediately
            # (<100ms), then process through full pipeline in background.
            # This eliminates the 30-40s blocking that Mode B users experience.
            # Pending memories are auto-processed on next engine.initialize()
            # or by the daemon's background loop.
            from superlocalmemory.cli.pending_store import store_pending, mark_done

            pending_id = store_pending(content, tags=tags, metadata={
                "project": project,
                "importance": importance,
                "agent_id": agent_id,
                "session_id": session_id,
            })

            # Fire-and-forget: process in background thread
            async def _process_in_background():
                try:
                    from superlocalmemory.core.worker_pool import WorkerPool
                    pool = WorkerPool.shared()
                    result = await asyncio.to_thread(
                        pool.store, content, metadata={
                            "tags": tags, "project": project,
                            "importance": importance, "agent_id": agent_id,
                            "session_id": session_id,
                        },
                    )
                    if result.get("ok"):
                        mark_done(pending_id)
                        _emit_event("memory.created", {
                            "content_preview": content[:80],
                            "agent_id": agent_id,
                            "fact_count": result.get("count", 0),
                        }, source_agent=agent_id)
                except Exception as _bg_exc:
                    logger.warning(
                        "Background store failed (pending_id=%s): %s",
                        pending_id, _bg_exc,
                    )

            asyncio.create_task(_process_in_background())

            return {
                "success": True,
                "fact_ids": [f"pending:{pending_id}"],
                "count": 1,
                "pending": True,
                "message": "Stored to pending — processing in background.",
            }
        except Exception as exc:
            logger.exception("remember failed")
            return {"success": False, "error": str(exc)}

    @server.tool()
    async def recall(
        query: str, limit: int = 10, agent_id: str = "mcp_client",
        session_id: str = "",
    ) -> dict:
        """Search memories by semantic query with 4-channel retrieval, RRF fusion, and reranking.

        S9-DASH-02: optional ``session_id`` threads through to the
        engine's outcome-queue so PostToolUse / Stop hooks can attach
        engagement signals to this recall. Claude Code should pass its
        ``CLAUDE_SESSION_ID``. Omitting it degrades to "no closed-loop
        learning for this recall" — the recall itself always works.
        """
        import asyncio
        try:
            from superlocalmemory.core.worker_pool import WorkerPool
            pool = WorkerPool.shared()
            # S9-DASH-10: priority for session_id, so engagement
            # signals land on the right pending_outcome:
            #   1. Explicit ``session_id`` tool-call argument.
            #   2. ``SLM_SESSION_ID`` / ``CLAUDE_SESSION_ID`` env var.
            #   3. Most-recent-active Claude session from the hook
            #      registry (last 60s). This catches the common case
            #      where Claude Code's hooks ran the UserPromptSubmit
            #      hook right before invoking the MCP tool.
            #   4. Stable per-agent fallback ``mcp:<agent_id>`` — the
            #      Stop hook will NOT match this, so the reaper
            #      settles it at neutral 0.5.
            effective_sid = session_id
            if not effective_sid:
                import os as _os
                effective_sid = (
                    _os.environ.get("SLM_SESSION_ID")
                    or _os.environ.get("CLAUDE_SESSION_ID")
                    or ""
                )
            if not effective_sid:
                try:
                    from superlocalmemory.hooks.session_registry import (
                        lookup_by_parent,
                        most_recent_active,
                    )
                    # Parent-PID lookup is collision-free across multiple
                    # parallel Claude sessions (each MCP server's parent
                    # is the IDE that spawned it).
                    effective_sid = (
                        lookup_by_parent(within_seconds=60)
                        or most_recent_active(
                            agent_type="claude", within_seconds=60,
                        )
                        or ""
                    )
                except Exception:
                    pass
            if not effective_sid:
                effective_sid = f"mcp:{agent_id}"
            # V3.3.19: Run in thread pool to avoid blocking MCP event loop
            result = await asyncio.to_thread(
                pool.recall, query, limit=limit, session_id=effective_sid,
            )
            if result.get("ok"):
                # Record implicit feedback: every returned result is a recall_hit
                try:
                    _record_recall_hits(get_engine, query, result.get("results", []))
                except Exception:
                    pass  # Feedback is non-critical, never block recall
                _emit_event("memory.recalled", {
                    "query": query[:80],
                    "result_count": result.get("result_count", 0),
                    "query_type": result.get("query_type", "unknown"),
                    "agent_id": agent_id,
                }, source_agent=agent_id)
                return {
                    "success": True,
                    "results": result.get("results", []),
                    "count": result.get("result_count", 0),
                    "query_type": result.get("query_type", "unknown"),
                    "channel_weights": result.get("channel_weights", {}),
                    "retrieval_time_ms": result.get("retrieval_time_ms", 0),
                }
            return {"success": False, "error": result.get("error", "Recall failed")}
        except Exception as exc:
            logger.exception("recall failed")
            return {"success": False, "error": str(exc)}

    @server.tool()
    async def search(query: str, limit: int = 10, profile_id: str = "") -> dict:
        """Full-text search across memories using FTS5 with BM25 ranking."""
        try:
            engine = get_engine()
            pid = profile_id or engine.profile_id
            facts = engine._db.search_facts_fts(query, pid, limit=limit)
            items = []
            for f in facts:
                items.append({
                    "fact_id": f.fact_id,
                    "content": f.content,
                    "fact_type": f.fact_type.value,
                    "confidence": round(f.confidence, 3),
                    "date": f.observation_date,
                })
            return {"success": True, "results": items, "count": len(items)}
        except Exception as exc:
            logger.exception("search failed")
            return {"success": False, "error": str(exc)}

    @server.tool()
    async def fetch(fact_ids: str) -> dict:
        """Fetch full details for specific fact IDs (comma-separated)."""
        try:
            engine = get_engine()
            ids = [fid.strip() for fid in fact_ids.split(",") if fid.strip()]
            facts = engine._db.get_facts_by_ids(ids, engine.profile_id)
            items = []
            for f in facts:
                items.append({
                    "fact_id": f.fact_id,
                    "content": f.content,
                    "fact_type": f.fact_type.value,
                    "entities": f.canonical_entities,
                    "confidence": round(f.confidence, 3),
                    "importance": round(f.importance, 3),
                    "observation_date": f.observation_date,
                    "referenced_date": f.referenced_date,
                    "lifecycle": f.lifecycle.value,
                    "access_count": f.access_count,
                })
            return {"success": True, "results": items, "count": len(items)}
        except Exception as exc:
            logger.exception("fetch failed")
            return {"success": False, "error": str(exc)}

    @server.tool()
    async def list_recent(limit: int = 20, profile_id: str = "") -> dict:
        """List most recently stored memories, newest first."""
        try:
            engine = get_engine()
            pid = profile_id or engine.profile_id
            facts = engine._db.get_all_facts(pid)[:limit]
            items = []
            for f in facts:
                items.append({
                    "fact_id": f.fact_id,
                    "content": f.content[:120],
                    "fact_type": f.fact_type.value,
                    "created_at": f.created_at,
                    "session_id": f.session_id,
                })
            return {"success": True, "results": items, "count": len(items)}
        except Exception as exc:
            logger.exception("list_recent failed")
            return {"success": False, "error": str(exc)}

    @server.tool()
    async def get_status() -> dict:
        """Get memory system status: fact count, entity count, mode, profile, db size."""
        try:
            engine = get_engine()
            pid = engine.profile_id
            fact_count = engine._db.get_fact_count(pid)
            entities = engine._db.execute(
                "SELECT COUNT(*) AS c FROM canonical_entities WHERE profile_id = ?",
                (pid,),
            )
            entity_count = int(dict(entities[0])["c"]) if entities else 0
            edges = engine._db.execute(
                "SELECT COUNT(*) AS c FROM graph_edges WHERE profile_id = ?",
                (pid,),
            )
            edge_count = int(dict(edges[0])["c"]) if edges else 0

            import os
            db_size_mb = 0.0
            db_path = engine._db.db_path
            if db_path.exists():
                db_size_mb = round(os.path.getsize(db_path) / (1024 * 1024), 2)

            return {
                "success": True,
                "mode": engine._config.mode.value,
                "profile": pid,
                "fact_count": fact_count,
                "entity_count": entity_count,
                "edge_count": edge_count,
                "db_size_mb": db_size_mb,
            }
        except Exception as exc:
            logger.exception("get_status failed")
            return {"success": False, "error": str(exc)}

    @server.tool()
    async def build_graph(profile_id: str = "") -> dict:
        """Rebuild knowledge graph edges for all facts in the active profile."""
        try:
            engine = get_engine()
            pid = profile_id or engine.profile_id
            facts = engine._db.get_all_facts(pid)
            edge_count = 0
            for fact in facts:
                if engine._graph_builder:
                    engine._graph_builder.build_edges(fact, pid)
                    edge_count += 1
            return {
                "success": True,
                "facts_processed": len(facts),
                "edges_built": edge_count,
            }
        except Exception as exc:
            logger.exception("build_graph failed")
            return {"success": False, "error": str(exc)}

    @server.tool()
    async def switch_profile(profile_id: str) -> dict:
        """Switch the active memory profile. All operations scope to this profile."""
        try:
            engine = get_engine()
            old = engine.profile_id
            engine.profile_id = profile_id

            # Persist to both config stores so CLI and Dashboard stay in sync
            try:
                from superlocalmemory.server.routes.helpers import (
                    ensure_profile_in_db, set_active_profile_everywhere,
                )
                ensure_profile_in_db(profile_id)
                set_active_profile_everywhere(profile_id)
            except ImportError:
                # Dashboard not installed — profile switch still works for MCP/CLI
                logger.debug("Dashboard routes not available, profile set in engine only")

            return {
                "success": True,
                "previous_profile": old,
                "current_profile": profile_id,
            }
        except Exception as exc:
            logger.exception("switch_profile failed")
            return {"success": False, "error": str(exc)}

    @server.tool()
    async def backup_status() -> dict:
        """Get backup system status, last backup time, and available backup files."""
        try:
            engine = get_engine()
            from superlocalmemory.infra.backup import BackupManager
            bm = BackupManager(engine._config.base_dir, engine._db.db_path)
            return {"success": True, **bm.get_status()}
        except Exception as exc:
            logger.exception("backup_status failed")
            return {"success": False, "error": str(exc)}

    @server.tool()
    async def memory_used() -> dict:
        """Get memory usage breakdown by fact type and lifecycle state."""
        try:
            engine = get_engine()
            pid = engine.profile_id
            facts = engine._db.get_all_facts(pid)
            by_type: dict[str, int] = {}
            by_lifecycle: dict[str, int] = {}
            for f in facts:
                by_type[f.fact_type.value] = by_type.get(f.fact_type.value, 0) + 1
                by_lifecycle[f.lifecycle.value] = (
                    by_lifecycle.get(f.lifecycle.value, 0) + 1
                )
            return {
                "success": True,
                "total_facts": len(facts),
                "by_type": by_type,
                "by_lifecycle": by_lifecycle,
                "profile": pid,
            }
        except Exception as exc:
            logger.exception("memory_used failed")
            return {"success": False, "error": str(exc)}

    @server.tool()
    async def get_learned_patterns(pattern_type: str = "", limit: int = 20) -> dict:
        """Get learned behavioral patterns (interests, refinements, archival habits)."""
        try:
            engine = get_engine()
            from superlocalmemory.learning.behavioral import BehavioralPatternStore
            store = BehavioralPatternStore(engine._db.db_path)
            ptype = pattern_type if pattern_type else None
            patterns = store.get_patterns(
                engine.profile_id, pattern_type=ptype, limit=limit,
            )
            return {"success": True, "patterns": patterns, "count": len(patterns)}
        except Exception as exc:
            logger.exception("get_learned_patterns failed")
            return {"success": False, "error": str(exc)}

    @server.tool()
    async def correct_pattern(pattern_id: str, correction: str) -> dict:
        """Correct or annotate a learned behavioral pattern to improve retrieval."""
        try:
            engine = get_engine()
            from superlocalmemory.learning.behavioral import BehavioralPatternStore
            store = BehavioralPatternStore(engine._db.db_path)
            store.record(
                engine.profile_id,
                pattern_type="correction",
                pattern_key=pattern_id,
                metadata={"correction": correction},
            )
            return {"success": True, "pattern_id": pattern_id}
        except Exception as exc:
            logger.exception("correct_pattern failed")
            return {"success": False, "error": str(exc)}

    @server.tool()
    async def delete_memory(fact_id: str, agent_id: str = "mcp_client") -> dict:
        """Delete a specific memory by exact fact ID.

        Security note: This is a destructive operation. All deletions are
        logged with the calling agent_id for audit trail. Use get_status or
        list_recent to find fact_ids before deleting.

        Args:
            fact_id: Exact fact ID to delete (from recall or list_recent results).
            agent_id: Identifier of the calling agent (logged for audit).
        """
        try:
            from superlocalmemory.core.worker_pool import WorkerPool
            pool = WorkerPool.shared()
            result = pool._send({
                "cmd": "delete_memory",
                "fact_id": fact_id,
                "agent_id": agent_id,
            })
            if result.get("ok"):
                logger.info("Memory deleted: %s by agent: %s", fact_id[:16], agent_id)
                _emit_event("memory.deleted", {
                    "fact_id": fact_id,
                    "agent_id": agent_id,
                }, source_agent=agent_id)
                return {"success": True, "deleted": fact_id, "agent_id": agent_id}
            return {"success": False, "error": result.get("error", "Delete failed")}
        except Exception as exc:
            logger.exception("delete_memory failed")
            return {"success": False, "error": str(exc)}

    @server.tool()
    async def update_memory(
        fact_id: str, content: str, agent_id: str = "mcp_client",
    ) -> dict:
        """Update the content of a specific memory by exact fact ID.

        Security note: All updates are logged with the calling agent_id.
        The fact_id must belong to the active profile.

        Args:
            fact_id: Exact fact ID to update.
            content: New content for the memory (cannot be empty).
            agent_id: Identifier of the calling agent (logged for audit).
        """
        try:
            if not content or not content.strip():
                return {"success": False, "error": "content cannot be empty"}
            from superlocalmemory.core.worker_pool import WorkerPool
            pool = WorkerPool.shared()
            result = pool._send({
                "cmd": "update_memory",
                "fact_id": fact_id,
                "content": content.strip(),
                "agent_id": agent_id,
            })
            if result.get("ok"):
                logger.info("Memory updated: %s by agent: %s", fact_id[:16], agent_id)
                return {"success": True, "fact_id": fact_id, "content": content.strip()}
            return {"success": False, "error": result.get("error", "Update failed")}
        except Exception as exc:
            logger.exception("update_memory failed")
            return {"success": False, "error": str(exc)}

    @server.tool()
    async def get_attribution() -> dict:
        """Get system attribution: author, version, license, and provenance metadata."""
        return {
            "success": True,
            "product": "SuperLocalMemory V3",
            "author": "Varun Pratap Bhardwaj",
            "organization": "Qualixar",
            "license": "Elastic-2.0",
            "urls": {
                "product": "https://superlocalmemory.com",
                "author": "https://varunpratap.com",
                "organization": "https://qualixar.com",
            },
        }


# -- Helpers ------------------------------------------------------------------

def _format_results(results) -> list[dict]:
    """Convert RetrievalResult list to serialisable dicts."""
    items: list[dict] = []
    for r in results:
        items.append({
            "fact_id": r.fact.fact_id,
            "content": r.fact.content,
            "score": round(r.score, 3),
            "confidence": round(r.confidence, 3),
            "trust_score": round(r.trust_score, 3),
            "fact_type": r.fact.fact_type.value,
            "channel_scores": {
                k: round(v, 3) for k, v in r.channel_scores.items()
            },
        })
    return items
