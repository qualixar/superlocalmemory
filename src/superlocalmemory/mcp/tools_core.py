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

import hashlib
import logging
from typing import Callable

from mcp.types import ToolAnnotations

from superlocalmemory.core.admission import admits
from superlocalmemory.core.config import CANONICAL_RECALL_LIMIT
from superlocalmemory.core.operation_request import OperationKind
from superlocalmemory.infra.data_root import state_path
from superlocalmemory.mcp.shared import authorize_mcp_mutation

logger = logging.getLogger(__name__)


async def _runtime_profile(get_engine: Callable, explicit: str = "") -> str:
    """Resolve an MCP default profile from daemon runtime truth."""
    if explicit:
        return explicit
    import asyncio

    try:
        from superlocalmemory.cli.daemon import daemon_request, is_daemon_running

        if await asyncio.to_thread(is_daemon_running):
            status = await asyncio.to_thread(daemon_request, "GET", "/status")
            if isinstance(status, dict) and status.get("profile"):
                return str(status["profile"])
            raise RuntimeError("resident daemon did not report its active profile")
    except RuntimeError:
        raise
    except Exception as exc:
        logger.debug("daemon profile resolution failed: %s", exc)
    return str(get_engine().profile_id)

def _emit_event(event_type: str, payload: dict | None = None,
                source_agent: str = "mcp_client") -> None:
    """Emit an event to the EventBus (best-effort, never raises)."""
    try:
        from superlocalmemory.infra.event_bus import EventBus
        bus = EventBus.get_instance(str(state_path("memory.db")))
        bus.emit(event_type, payload=payload, source_agent=source_agent,
                 source_protocol="mcp")
    except Exception:
        pass


def register_core_tools(server, get_engine: Callable) -> None:
    """Register the 13 core MCP tools on *server*."""

    @server.tool()
    @admits(OperationKind.REMEMBER)
    async def remember(
        content: str, tags: str = "", project: str = "",
        importance: int = 5, session_id: str = "",
        agent_id: str = "mcp_client",
        scope: str | None = None,
        shared_with: str = "",
        idempotency_key: str = "",
    ) -> dict:
        """Store content to memory with intelligent indexing.

        Extracts atomic facts, resolves entities, builds graph edges,
        and indexes for hybrid retrieval with graph-aware enhancement.

        Multi-scope: ``scope`` sets visibility (personal/shared/global).
        ``shared_with`` is a comma-separated list of profile_ids for
        shared scope.
        """
        # v3.6.10: resolve "mcp_client" sentinel → URL path (HTTP) or env var (stdio)
        if agent_id == "mcp_client":
            from superlocalmemory.mcp.agent_context import get_current_agent_id
            agent_id = get_current_agent_id()
        meta = {
            "project": project,
            "importance": importance,
            "agent_id": agent_id,
            "session_id": session_id,
        }
        effective_idempotency_key = idempotency_key
        if not effective_idempotency_key:
            # Derive a stable key before the first attempt so every retry of
            # the same logical call carries the same key.  When a session token
            # is present it is included in the material to keep per-session
            # stores separate.  Without a session token the key is derived from
            # the remaining call parameters so repeated observations with the
            # same content, agent, and scope are deduplicated across retries.
            if session_id:
                material = (
                    f"{agent_id}\0{session_id}\0{scope or ''}\0{shared_with}\0{content}"
                )
                effective_idempotency_key = "mcp:" + hashlib.sha256(
                    material.encode("utf-8")
                ).hexdigest()
            else:
                material = f"{agent_id}\0{scope or ''}\0{shared_with}\0{content}"
                effective_idempotency_key = "mcp:req:" + hashlib.sha256(
                    material.encode("utf-8")
                ).hexdigest()
        # Parse shared_with from comma-separated string
        _shared_list = [s.strip() for s in shared_with.split(",") if s.strip()] if shared_with else None
        # v3.5.5 WRITE-THROUGH: route through the daemon's /remember, which does
        # a synchronous verbatim insert (memory is keyword/BM25-recallable the
        # instant this returns) and enqueues async enrichment. This closes the
        # recall window so a parallel/next agent finds memories saved seconds ago.
        # Falls back to the capability-owned worker only if the daemon is
        # unreachable. Raw pending.db writes are legacy replay input only.
        daemon_owned = False
        try:
            import asyncio as _asyncio

            from superlocalmemory.cli.daemon import daemon_request, is_daemon_running
            # is_daemon_running() and daemon_request() both use blocking urllib
            # against the same uvicorn server — run in threads so the MCP
            # event loop stays unblocked (#34 class bug).
            daemon_owned = await _asyncio.to_thread(is_daemon_running)
            if daemon_owned:
                # A positively identified daemon owns this database. Never
                # spawn a WorkerPool writer after a transient daemon failure:
                # that creates the competing writers which SQLite WAL cannot
                # support. Retry the canonical path, then return an explicit
                # retryable result to the MCP client.
                for attempt in range(3):
                    resp = await _asyncio.to_thread(daemon_request, "POST", "/remember", {
                        "content": content, "tags": tags, "metadata": meta,
                        "scope": scope, "shared_with": _shared_list,
                        "session_id": session_id,
                        "idempotency_key": effective_idempotency_key or None,
                    })
                    if resp and (resp.get("fact_ids") is not None or resp.get("ok")):
                        fids = resp.get("fact_ids") or []
                        materialization_state = resp.get("materialization_state")
                        if materialization_state is None:
                            materialization_state = (
                                "complete" if resp.get("status") == "stored" else "queryable"
                            )
                        pending = materialization_state != "complete"
                        return {
                            "success": True,
                            "fact_ids": fids,
                            "count": int(resp.get("count", len(fids))),
                            "pending": pending,
                            "pending_id": resp.get("pending_id") if pending else None,
                            "operation_id": resp.get("operation_id"),
                            "materialization_state": materialization_state,
                            "message": (
                                "Stored through canonical daemon ingestion."
                                if not pending
                                else "Queryable now; canonical enrichment is still running."
                            ),
                        }
                    if attempt < 2:
                        await _asyncio.sleep(0.05 * (attempt + 1))
                return {
                    "success": False,
                    "code": "DAEMON_UNAVAILABLE",
                    "retryable": True,
                    "error": (
                        "DAEMON_UNAVAILABLE: owned daemon is unavailable; retry later."
                    ),
                }
        except Exception as dexc:
            logger.debug("MCP remember via daemon failed, pending fallback: %s", dexc)
            if daemon_owned:
                return {
                    "success": False,
                    "code": "DAEMON_UNAVAILABLE",
                    "retryable": True,
                    "error": (
                        "DAEMON_UNAVAILABLE: owned daemon is unavailable; retry later."
                    ),
                }

        try:
            import asyncio as _asyncio

            from superlocalmemory.mcp._daemon_proxy import choose_pool

            worker_meta = {
                **meta,
                "tags": tags,
                "scope": scope or "personal",
                "shared_with": _shared_list or [],
                "idempotency_key": (
                    effective_idempotency_key
                    or "mcp:" + hashlib.sha256(content.encode("utf-8")).hexdigest()
                ),
            }

            def _store_via_daemon_pool():
                pool = choose_pool()
                return pool.store(content, worker_meta)

            stored = await _asyncio.to_thread(_store_via_daemon_pool)
            if not isinstance(stored, dict) or not stored.get("ok"):
                if isinstance(stored, dict) and stored.get("code") == "DAEMON_UNAVAILABLE":
                    return {
                        "success": False,
                        "code": "DAEMON_UNAVAILABLE",
                        "retryable": True,
                        "error": stored.get(
                            "error",
                            "DAEMON_UNAVAILABLE: owned daemon is unavailable; retry later.",
                        ),
                    }
                return {
                    "success": False,
                    "code": "DAEMON_UNAVAILABLE",
                    "retryable": True,
                    "error": "DAEMON_UNAVAILABLE: owned daemon is unavailable; retry later.",
                }
            fact_ids = list(stored.get("fact_ids") or [])
            materialization_state = str(
                stored.get("materialization_state") or "complete"
            )
            allowed_states = {"queryable", "enriching", "complete"}
            if materialization_state not in allowed_states:
                raise RuntimeError(
                    "canonical worker returned invalid materialization state: "
                    f"{materialization_state}"
                )
            pending = materialization_state != "complete"
            operation_id = stored.get("operation_id")
            pending_id = stored.get("pending_id")
            if pending and pending_id is None:
                pending_id = operation_id
            return {
                "success": True,
                "fact_ids": fact_ids,
                "count": int(stored.get("count", len(fact_ids))),
                "pending": pending,
                "pending_id": pending_id if pending else None,
                "operation_id": operation_id,
                "materialization_state": materialization_state,
                "message": (
                    "Stored through canonical local ingestion."
                    if not pending
                    else "Queryable now; canonical enrichment is still running."
                ),
            }
        except Exception:
            logger.exception("remember failed")
            return {
                "success": False,
                "code": "DAEMON_UNAVAILABLE",
                "retryable": True,
                "error": "DAEMON_UNAVAILABLE: owned daemon is unavailable; retry later.",
            }

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    async def recall(
        query: str, limit: int = CANONICAL_RECALL_LIMIT, agent_id: str = "mcp_client",
        session_id: str = "", fast: bool | None = None,
        include_global: bool | None = None,
        include_shared: bool | None = None,
        window: str = "",
        as_of: str | None = None,
    ) -> dict:
        """Search memories through hybrid retrieval, RRF fusion, and reranking.

        Fast local retrieval (six channels + reranker) returns in ~1-2s. This
        tool does NOT run an internal LLM reformulation round — YOU (the calling
        model) are the reasoner. Drive refinement using the confidence signals
        in the response:
          • ``no_confident_match: true`` → nothing cleared the evidence floor.
            Do NOT invent a memory. Rewrite the query into 1-3 more specific
            sub-queries (split multi-hop questions; try entity names, synonyms,
            or a broader phrasing) and call ``recall`` again before concluding
            the information is unknown.
          • ``answer_confidence`` low / ``abstained: true`` → the top hit is
            weak. Re-query with a sharper phrasing, or widen with
            ``include_shared=true`` / ``include_global=true`` if appropriate.
          • Confident match → use it directly; no second call needed.
        One extra targeted recall is cheap and beats a wrong "not found".

        Optional ``session_id`` threads through to the
        engine's outcome-queue so PostToolUse / Stop hooks can attach
        engagement signals to this recall. Claude Code should pass its
        ``CLAUDE_SESSION_ID``. Omitting it degrades to "no closed-loop
        learning for this recall" — the recall itself always works.

        Multi-scope: ``include_global`` / ``include_shared`` control which
        scopes participate in retrieval. Leave them unset (``None``) to use the
        configured default — shared memory is OPT-IN, so by default recall
        returns only this profile's own facts. Pass ``True`` to opt in per call.

        Time window: optional ``window`` restricts results to a event-time
        range. Accepts a relative span (``"24h"``, ``"7d"``, ``"30d"``,
        ``"1y"``) or an explicit range (``"2026-07-01..2026-07-31"``). Empty =
        no time filter.

        Point-in-time: optional ``as_of`` (ISO-8601 string, e.g.
        ``"2026-01-01T00:00:00+00:00"``) pins recall to a temporal snapshot;
        omit or pass ``None`` for current-state recall.
        """
        # v3.6.10: resolve "mcp_client" sentinel → URL path (HTTP) or env var (stdio)
        if agent_id == "mcp_client":
            from superlocalmemory.mcp.agent_context import get_current_agent_id
            agent_id = get_current_agent_id()
        import asyncio
        try:
            from superlocalmemory.mcp._daemon_proxy import choose_pool
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
            # Resolve the daemon proxy inside the worker too. ``choose_pool``
            # verifies daemon ownership through a synchronous /health request;
            # when this tool is served by the daemon's mounted HTTP MCP app,
            # resolving it on Uvicorn's event-loop thread makes that loop wait
            # on its own health response forever. Stdio did not exhibit this
            # because its MCP process is external to the daemon.
            #
            # V3.4.26: WorkerPool now concurrent — parallel calls no longer
            # block behind a single threading.Lock. See worker_pool.py.
            def _recall_via_daemon_pool():
                pool = choose_pool()
                return pool.recall(
                    query, limit=limit, session_id=effective_sid,
                    fast=fast, include_global=include_global,
                    include_shared=include_shared, window=window or None,
                    as_of=as_of,
                )

            result = await asyncio.to_thread(
                _recall_via_daemon_pool,
            )
            if result.get("ok"):
                return {
                    "success": True,
                    "results": result.get("results", []),
                    "count": result.get("result_count", 0),
                    "query_type": result.get("query_type", "unknown"),
                    "channel_weights": result.get("channel_weights", {}),
                    "retrieval_time_ms": result.get("retrieval_time_ms", 0),
                    # v3.6.6: surface evidence-floor signal to MCP clients.
                    "no_confident_match": result.get("no_confident_match", False),
                    "score_contract_version": result.get("score_contract_version", "2"),
                    "calibration_status": result.get("calibration_status", "uncalibrated"),
                    "calibration_id": result.get("calibration_id"),
                    "answer_confidence": result.get("answer_confidence"),
                    "abstained": result.get("abstained", False),
                    "abstention_reason": result.get("abstention_reason"),
                }
            return {"success": False, "error": result.get("error", "Recall failed")}
        except Exception as exc:
            logger.exception("recall failed")
            return {"success": False, "error": str(exc)}

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    async def search(query: str, limit: int = 10) -> dict:
        """Full-text search across memories using FTS5 with BM25 ranking."""
        try:
            engine = get_engine()
            pid = await _runtime_profile(get_engine)
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

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    async def fetch(fact_ids: str) -> dict:
        """Fetch full details for specific fact IDs (comma-separated)."""
        try:
            engine = get_engine()
            ids = [fid.strip() for fid in fact_ids.split(",") if fid.strip()]
            pid = await _runtime_profile(get_engine)
            facts = engine._db.get_facts_by_ids(ids, pid)
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

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    async def list_recent(limit: int = 20) -> dict:
        """List most recently stored memories, newest first."""
        try:
            engine = get_engine()
            pid = await _runtime_profile(get_engine)
            # v3.6.12 (search-2): push the limit into the query — was loading the
            # ENTIRE facts table (deserializing every 768-float embedding) just
            # to return the top N. get_all_facts preserves created_at DESC order.
            facts = engine._db.get_all_facts(pid, limit=limit)
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

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    async def get_status() -> dict:
        """Get memory system status: fact count, entity count, mode, profile, db size."""
        try:
            import asyncio
            import os

            from superlocalmemory.cli.daemon import (
                daemon_request,
                is_daemon_running,
            )

            if await asyncio.to_thread(is_daemon_running):
                daemon_status = await asyncio.to_thread(
                    daemon_request,
                    "GET",
                    "/status",
                )
                if isinstance(daemon_status, dict) and daemon_status.get("profile"):
                    return {
                        "success": True,
                        "mode": daemon_status.get("mode", "unknown"),
                        "provider": daemon_status.get("provider", "none"),
                        "profile": daemon_status["profile"],
                        "base_dir": daemon_status.get("base_dir", ""),
                        "db_path": daemon_status.get("db_path", ""),
                        "db_size_mb": float(daemon_status.get("db_size_mb", 0.0)),
                        "fact_count": int(daemon_status.get("fact_count", 0)),
                        "entity_count": int(daemon_status.get("entity_count", 0)),
                        "edge_count": int(daemon_status.get("edge_count", 0)),
                        "profile_generation": int(
                            daemon_status.get("profile_generation", 0)
                        ),
                    }

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

            db_size_mb = 0.0
            db_path = engine._db.db_path
            if db_path.exists():
                db_size_mb = round(os.path.getsize(db_path) / (1024 * 1024), 2)

            # WP-02 D8: additive canonical key set — provider/base_dir/db_path added.
            # All pre-existing keys are preserved (zero removals).
            cfg = engine._config
            return {
                "success": True,
                "mode": cfg.mode.value,
                "provider": cfg.llm.provider or "none",
                "profile": pid,
                "base_dir": str(cfg.base_dir),
                "db_path": str(db_path),
                "db_size_mb": db_size_mb,
                "fact_count": fact_count,
                "entity_count": entity_count,
                "edge_count": edge_count,
                "profile_generation": 0,
            }
        except Exception as exc:
            logger.exception("get_status failed")
            return {"success": False, "error": str(exc)}

    @server.tool()
    async def build_graph() -> dict:
        """Rebuild knowledge graph edges for all facts in the active profile."""
        try:
            engine = get_engine()
            pid = await _runtime_profile(get_engine)
            authorization = authorize_mcp_mutation(
                engine,
                "update",
                mutation_source="mcp-build-memory-graph",
                profile_id=pid,
            )
            facts = engine._db.get_all_facts(pid)
            edge_count = 0
            for fact in facts:
                if engine._graph_builder:
                    engine._graph_builder.build_edges(fact, pid)
                    edge_count += 1
            authorization.complete()
            return {
                "success": True,
                "facts_processed": len(facts),
                "edges_built": edge_count,
            }
        except Exception as exc:
            logger.exception("build_graph failed")
            return {"success": False, "error": str(exc)}

    @server.tool()
    @admits(OperationKind.PROFILE_SWITCH)
    async def switch_profile(profile_id: str) -> dict:
        """Switch the active memory profile. All operations scope to this profile."""
        try:
            import asyncio

            engine = get_engine()
            old = engine.profile_id
            authorization = authorize_mcp_mutation(
                engine,
                "update",
                mutation_source="mcp-switch-profile",
                profile_id=profile_id,
                content_preview=f"{old} -> {profile_id}",
            )
            from superlocalmemory.cli.daemon import (
                daemon_request,
                is_daemon_running,
            )

            generation = 0
            # Only the profile_id explicitly confirmed by this process
            # (daemon-acknowledged + locally-validated, or locally
            # validated directly) is ever synced into engine state.
            confirmed_profile_id = None
            if await asyncio.to_thread(is_daemon_running):
                result = await asyncio.to_thread(
                    daemon_request,
                    "POST",
                    f"/api/profiles/{profile_id}/switch",
                )
                if not result or not result.get("success"):
                    return {
                        "success": False,
                        "error": "resident daemon rejected the profile switch",
                    }
                acknowledged = str(result.get("active_profile", ""))
                if not acknowledged or acknowledged != profile_id:
                    return {
                        "success": False,
                        "error": "resident daemon acknowledged a different profile",
                    }
                # Local consistency guard (SEC-H-01): the daemon's HTTP
                # acknowledgement alone is not sufficient — this MCP
                # process must also confirm the profile exists in its
                # own local DB handle before syncing local state to it.
                # Mirrors the existence check the no-daemon branch already
                # performs below.
                local_rows = engine._db.execute(
                    "SELECT 1 FROM profiles WHERE profile_id = ?",
                    (profile_id,),
                )
                if not local_rows:
                    return {
                        "success": False,
                        "error": (
                            f"resident daemon acknowledged profile "
                            f"'{acknowledged}' but it does not exist in "
                            f"this process's local profile store"
                        ),
                    }
                generation = int(result.get("generation", 0))
                # Sync target is the DAEMON-CONFIRMED value, never the raw
                # caller-supplied profile_id, even though they are equal
                # here by construction (checked above).
                confirmed_profile_id = acknowledged
            else:
                rows = engine._db.execute(
                    "SELECT 1 FROM profiles WHERE profile_id = ?",
                    (profile_id,),
                )
                if not rows:
                    return {
                        "success": False,
                        "error": f"Profile '{profile_id}' does not exist.",
                    }
                from superlocalmemory.server.profile_runtime import (
                    persist_active_profile,
                )

                persistence = persist_active_profile(profile_id)
                try:
                    engine.profile_id = profile_id
                    engine._config.active_profile = profile_id
                except BaseException:
                    engine.profile_id = old
                    engine._config.active_profile = old
                    persistence.rollback()
                    raise
                confirmed_profile_id = profile_id

            if not confirmed_profile_id:
                # Defensive: should be unreachable — every path above
                # either returns an error or sets confirmed_profile_id.
                return {
                    "success": False,
                    "error": "profile switch could not be confirmed",
                }

            # Synchronize this MCP process only after confirmation
            # (daemon-acknowledged + locally-validated, or directly
            # locally-validated in the no-daemon branch above).
            engine.profile_id = confirmed_profile_id
            engine._config.active_profile = confirmed_profile_id

            # v3.6.12 (search-3): recall/delete run in a separate worker
            # subprocess that caches its engine (and profile_id) at init. Recycle
            # it so the NEXT recall uses the new profile instead of the stale one.
            try:
                from superlocalmemory.core.worker_pool import WorkerPool
                WorkerPool.shared().shutdown()
            except Exception:
                logger.debug("worker-pool recycle on profile switch skipped")

            authorization.complete()
            return {
                "success": True,
                "previous_profile": old,
                "current_profile": profile_id,
                "generation": generation,
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
            bm = BackupManager(
                db_path=engine._db.db_path,
                base_dir=engine._config.base_dir,
            )
            return {"success": True, **bm.get_status()}
        except Exception as exc:
            logger.exception("backup_status failed")
            return {"success": False, "error": str(exc)}

    @server.tool()
    async def memory_used() -> dict:
        """Get memory usage breakdown by fact type and lifecycle state."""
        try:
            engine = get_engine()
            pid = await _runtime_profile(get_engine)
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
            pid = await _runtime_profile(get_engine)
            from superlocalmemory.learning.behavioral import BehavioralPatternStore
            store = BehavioralPatternStore(engine._db.db_path)
            ptype = pattern_type if pattern_type else None
            patterns = store.get_patterns(
                pid, pattern_type=ptype, limit=limit,
            )
            return {"success": True, "patterns": patterns, "count": len(patterns)}
        except Exception as exc:
            logger.exception("get_learned_patterns failed")
            return {"success": False, "error": str(exc)}

    @server.tool()
    @admits(OperationKind.CORRECT)
    async def correct_pattern(pattern_id: str, correction: str) -> dict:
        """Correct or annotate a learned behavioral pattern to improve retrieval."""
        try:
            engine = get_engine()
            pid = await _runtime_profile(get_engine)
            authorization = authorize_mcp_mutation(
                engine,
                "update",
                mutation_source="mcp-correct-pattern",
                profile_id=pid,
                fact_id=pattern_id,
                content_preview=correction,
            )
            from superlocalmemory.learning.behavioral import BehavioralPatternStore
            store = BehavioralPatternStore(engine._db.db_path)
            store.record(
                pid,
                pattern_type="correction",
                pattern_key=pattern_id,
                metadata={"correction": correction},
            )
            authorization.complete()
            return {"success": True, "pattern_id": pattern_id}
        except Exception as exc:
            logger.exception("correct_pattern failed")
            return {"success": False, "error": str(exc)}

    @server.tool(annotations=ToolAnnotations(destructiveHint=True))
    @admits(OperationKind.FORGET)
    async def delete_memory(fact_id: str, agent_id: str = "mcp_client") -> dict:
        """Delete a specific memory by exact fact ID.

        Security note: This is a destructive operation. All deletions are
        logged with the calling agent_id for audit trail. Use get_status or
        list_recent to find fact_ids before deleting.

        Args:
            fact_id: Exact fact ID to delete (from recall or list_recent results).
            agent_id: Identifier of the calling agent (logged for audit).
        """
        # v3.6.10: resolve "mcp_client" sentinel → URL path (HTTP) or env var (stdio)
        if agent_id == "mcp_client":
            from superlocalmemory.mcp.agent_context import get_current_agent_id
            agent_id = get_current_agent_id()
        try:
            import asyncio
            import urllib.parse

            from superlocalmemory.cli.daemon import (
                daemon_request,
                is_daemon_running,
            )

            if await asyncio.to_thread(is_daemon_running):
                path = "/api/memories/" + urllib.parse.quote(fact_id, safe="")
                result = await asyncio.to_thread(
                    daemon_request, "DELETE", path,
                )
                if isinstance(result, dict) and result.get("success"):
                    _emit_event("memory.deleted", {
                        "fact_id": fact_id,
                        "agent_id": agent_id,
                    }, source_agent=agent_id)
                    return {
                        "success": True, "deleted": fact_id,
                        "agent_id": agent_id,
                    }
                return {
                    "success": False,
                    "retryable": True,
                    "error": "resident daemon rejected the delete operation",
                }

            from superlocalmemory.core.worker_pool import WorkerPool
            pool = WorkerPool.shared()
            result = pool._send({
                "cmd": "delete_memory",
                "fact_id": fact_id,
                # Informational IDE/client label only.  The worker derives its
                # authorization actor from the private local capability.
                "source_agent_id": agent_id,
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

    @server.tool(annotations=ToolAnnotations(idempotentHint=True))
    @admits(OperationKind.CORRECT)
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
        # v3.6.10: resolve "mcp_client" sentinel → URL path (HTTP) or env var (stdio)
        if agent_id == "mcp_client":
            from superlocalmemory.mcp.agent_context import get_current_agent_id
            agent_id = get_current_agent_id()
        try:
            if not content or not content.strip():
                return {"success": False, "error": "content cannot be empty"}
            import asyncio
            import urllib.parse

            from superlocalmemory.cli.daemon import (
                daemon_request,
                is_daemon_running,
            )

            if await asyncio.to_thread(is_daemon_running):
                path = "/api/memories/" + urllib.parse.quote(fact_id, safe="")
                result = await asyncio.to_thread(
                    daemon_request,
                    "PATCH",
                    path,
                    {"content": content.strip()},
                )
                if isinstance(result, dict) and result.get("success"):
                    return {
                        "success": True, "fact_id": fact_id,
                        "content": content.strip(),
                    }
                return {
                    "success": False,
                    "retryable": True,
                    "error": "resident daemon rejected the update operation",
                }

            from superlocalmemory.core.worker_pool import WorkerPool
            pool = WorkerPool.shared()
            result = pool._send({
                "cmd": "update_memory",
                "fact_id": fact_id,
                "content": content.strip(),
                "source_agent_id": agent_id,
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
            "license": "AGPL-3.0-or-later",
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
            "relevance_score": round(
                getattr(r, "relevance_score", r.score) or 0.0, 3
            ),
            "ranking_score": getattr(r, "ranking_score", None),
            "memory_confidence": round(
                getattr(r, "memory_confidence", r.confidence) or 0.0, 3
            ),
            "rank_position": int(getattr(r, "rank_position", 0) or 0),
            "trust_score": round(r.trust_score, 3),
            "fact_type": r.fact.fact_type.value,
            "channel_scores": {
                k: round(v, 3) for k, v in r.channel_scores.items()
            },
        })
    return items
