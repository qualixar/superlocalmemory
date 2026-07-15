# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3.1 — Active Memory MCP Tools.

session_init    — Auto-recall project context at session start.
observe         — Monitor conversation for auto-capture (decisions/bugs/prefs).
report_feedback — Record explicit feedback on recall results for learning.

These tools transform SLM from a passive database into an active
intelligence layer that learns and improves over time.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import sqlite3
import uuid
from typing import TYPE_CHECKING, Callable

from superlocalmemory.infra.data_root import canonical_data_root, state_path

if TYPE_CHECKING:
    from superlocalmemory.mcp._pool_adapter import PoolRecallResponse

logger = logging.getLogger(__name__)


def _sqlite_emergency_recall(
    query: str, limit: int, profile_id: str = "default",
    max_age_days: int = 30,
) -> "PoolRecallResponse":
    """Emergency fallback: direct SQLite FTS5 BM25 when daemon is unreachable.

    Uses the same ``atomic_facts_fts`` virtual table the daemon uses, with
    native BM25 ranking via ``ORDER BY fts.rank``. This is the Mem0 / Letta
    industry pattern — multi-process safe via SQLite WAL mode.

    Quality degraded vs full 6-channel (no semantic, no entity graph, no
    temporal/spreading-activation/Hopfield) but still provides real BM25
    math + age gate. Returns ``degraded_mode=True`` via the caller's flag.

    Used ONLY when Tier-1 (full daemon recall) fails completely. Normal
    path is full 6-channel; this is the fire-alarm.
    """
    import re

    from superlocalmemory.mcp._pool_adapter import PoolFact, PoolRecallItem, PoolRecallResponse
    try:
        # FTS5 MATCH syntax: tokenize the query, drop special characters
        # that confuse the parser (/, :, ., etc), and join with OR for
        # broadest matching. Wrap each term in quotes to escape any
        # remaining special-meaning chars.
        tokens = re.findall(r"[A-Za-z0-9]+", query)
        tokens = [t for t in tokens if len(t) >= 2]
        if not tokens:
            return PoolRecallResponse()
        safe_query = " OR ".join(f'"{t}"' for t in tokens)
        age_clause = (
            f"AND f.created_at >= datetime('now', '-{int(max_age_days)} days') "
            if max_age_days > 0 else ""
        )
        conn = sqlite3.connect(str(state_path("memory.db")), timeout=5.0)
        try:
            rows = conn.execute(
                f"""SELECT f.fact_id, f.content, f.memory_id, f.created_at,
                           fts.rank AS bm25_rank
                    FROM atomic_facts_fts AS fts
                    JOIN atomic_facts AS f ON f.fact_id = fts.fact_id
                    WHERE fts.atomic_facts_fts MATCH ?
                      AND f.profile_id = ?
                      {age_clause}
                    ORDER BY fts.rank
                    LIMIT ?""",
                (safe_query, profile_id, limit * 2),
            ).fetchall()
        finally:
            conn.close()
        # FTS5 rank is negative (lower = better). Normalize to [0.3, 0.9].
        if not rows:
            return PoolRecallResponse()
        ranks = [r[4] for r in rows]
        rmin, rmax = min(ranks), max(ranks)
        rng = (rmax - rmin) or 1.0
        items = [
            PoolRecallItem(
                fact=PoolFact(
                    fact_id=r[0] or "", content=r[1] or "",
                    memory_id=r[2] or "", created_at=r[3] or "",
                ),
                score=round(0.3 + 0.6 * (1.0 - (r[4] - rmin) / rng), 3),
            )
            for r in rows
        ]
        logger.warning(
            "session_init: EMERGENCY FTS5 fallback (%d results). "
            "Daemon unreachable — semantic/graph channels disabled.", len(items),
        )
        return PoolRecallResponse(results=items[:limit])
    except Exception as exc:
        logger.warning("Emergency FTS5 fallback failed: %s", exc)
        return PoolRecallResponse()


def _get_agent_id(default: str = "mcp_client") -> str:
    """Resolve the calling agent's ID for attribution.

    Priority chain (v3.6.10+):
    1. ContextVar set by HTTP URL path (/mcp/{agent_id}) — HTTP transport.
    2. SLM_AGENT_ID env var — stdio transport per-process identity.
    3. Provided default (legacy "mcp_client").
    """
    from superlocalmemory.mcp.agent_context import get_current_agent_id
    resolved = get_current_agent_id(env_fallback=True)
    return resolved if resolved != "mcp_client" else default


def _emit_event(event_type: str, payload: dict | None = None,
                source_agent: str | None = None) -> None:  # V3.3.12: see also mcp/shared.py
    """Emit an event to the EventBus (best-effort, never raises).

    Dashboard visibility is load-bearing per the v3.4.26 user contract,
    so we log on failure rather than silently dropping the signal.
    """
    resolved_agent = source_agent if source_agent is not None else _get_agent_id()
    try:
        from superlocalmemory.infra.event_bus import EventBus
        bus = EventBus.get_instance(str(state_path("memory.db")))
        bus.emit(event_type, payload=payload, source_agent=resolved_agent,
                 source_protocol="mcp")
    except Exception as exc:
        logger.warning("event emit failed: type=%s err=%s", event_type, exc)


def _register_agent(agent_id: str, profile_id: str) -> None:
    """Register an agent in the AgentRegistry (best-effort)."""
    try:
        from superlocalmemory.core.registry import AgentRegistry
        registry_path = canonical_data_root() / "agents.json"
        registry = AgentRegistry(persist_path=registry_path)
        registry.register_agent(agent_id, profile_id)
    except Exception as exc:
        logger.warning(
            "agent registry write failed: agent=%s err=%s", agent_id, exc,
        )


def register_active_tools(server, get_engine: Callable) -> None:
    """Register 3 active memory tools on *server*."""

    # ------------------------------------------------------------------
    # 1. session_init — Auto-recall project context at session start
    # ------------------------------------------------------------------
    @server.tool()
    async def session_init(
        project_path: str = "",
        query: str = "",
        max_results: int = 10,
        max_age_days: int = 30,
    ) -> dict:
        """Initialize session with relevant memory context.

        Call this ONCE at the start of every session. Returns:
        - Recent decisions and patterns for this project
        - Top relevant memories based on project path or query
        - Learning status (signal count, ranking phase)

        The AI should call this automatically before any other work.

        Parameters:
            project_path: Working directory path. Used to build the search query
                when no explicit query is provided.
            query: Override the search query. If omitted, derived from project_path
                or falls back to "recent important decisions".
            max_results: Maximum memories to return (default: 10).
            max_age_days: Suppress memories older than this many days unless their
                relevance score is ≥ 0.70 (architectural decisions that remain
                permanently relevant still surface). Default: 30.
                Set to 0 to disable the age gate entirely.

        Scoring: Uses 6-channel fusion (semantic + BM25 + entity_graph + temporal +
        spreading_activation + hopfield) with Ebbinghaus exponential recency decay
        and FSRS stability strengthening by access frequency.
        """
        try:
            from superlocalmemory.hooks.rules_engine import RulesEngine
            from superlocalmemory.mcp._pool_adapter import pool_recall

            engine = get_engine()
            rules = RulesEngine()

            if not rules.should_recall("session_start"):
                return {
                    "success": True,
                    "context": "",
                    "memories": [],
                    "message": "Auto-recall disabled",
                }

            recall_config = rules.get_recall_config()
            relevance_threshold = recall_config.get("relevance_threshold", 0.3)
            if query:
                search_query = query
            elif project_path:
                search_query = f"project context {project_path}"
            else:
                search_query = "recent important decisions"

            # 2-tier recall (industry pattern: Hindsight / Zep / Supermemory):
            # PRIMARY: full 6-channel via daemon (semantic + BM25 + entity + temporal
            #          + Hopfield + spreading-activation, Fisher-Rao fusion, FSRS decay).
            #          Fast because Ollama embed model is kept warm (keep_alive=-1
            #          + eager pre-warm at daemon boot).
            # EMERGENCY: direct FTS5 BM25 (Mem0 / Letta pattern). Used ONLY when
            #            daemon is completely unreachable. Returns degraded_mode=True.
            from superlocalmemory.mcp._pool_adapter import PoolError
            degraded_mode = False
            try:
                # v3.6.9-audit: pool_recall uses blocking urllib under the hood
                # (DaemonPoolProxy.recall → urllib.urlopen). Must run in a
                # thread so the async MCP event loop is not stalled — same
                # fix class as #34 mesh tools deadlock.
                response = await asyncio.to_thread(
                    pool_recall, search_query, limit=max_results, fast=False,
                )
            except (PoolError, Exception) as exc:
                logger.warning(
                    "session_init: daemon recall failed (%s) — using FTS5 emergency fallback. "
                    "Memory system is in DEGRADED MODE: semantic/graph channels unavailable.",
                    exc,
                )
                response = _sqlite_emergency_recall(
                    search_query, max_results,
                    profile_id=engine.profile_id,
                    max_age_days=max_age_days,
                )
                degraded_mode = True

            # Age gate: suppress stale memories at session start.
            # Memories older than max_age_days are excluded unless their score
            # exceeds 0.7 (high-relevance architectural decisions always surface).
            # max_age_days=0 disables the gate entirely.
            from datetime import UTC
            from datetime import datetime as _dt
            _now = _dt.now(UTC)

            def _age_days(created_at_str: str) -> float:
                if not created_at_str:
                    return 0.0
                try:
                    created = _dt.fromisoformat(
                        created_at_str.replace("Z", "+00:00")
                    )
                    return max(0.0, (_now - created).total_seconds() / 86400.0)
                except (ValueError, TypeError):
                    return 0.0

            relevant = [
                r for r in response.results
                if r.score >= relevance_threshold
                and (
                    max_age_days <= 0
                    or _age_days(r.fact.created_at) <= max_age_days
                    or r.score >= 0.7
                )
            ]

            # Build both return shapes from one recall. Calling recall twice
            # doubles session startup latency and can return duplicate snippets.

            # v3.4.65: use shared injection formatter for full-fidelity context.
            from superlocalmemory.core.injection import (
                InjectableMemory,
                clamp_content,
                is_low_quality,
                render_context,
                sanitize_untrusted_content,
            )

            pid = engine.profile_id

            # Merge pinned facts (Q3: Core Memory explicit pins).
            # Pinned facts surface even if the query didn't retrieve them.
            try:
                pinned_facts = engine.db.get_pinned(pid)
            except Exception:
                pinned_facts = []
            pinned_seen = set()

            cfg_inj = getattr(getattr(engine, "config", None), "injection", None)
            # Defend against MagicMock / non-config objects in tests.
            try:
                from superlocalmemory.core.config import InjectionConfig
                if not isinstance(cfg_inj, InjectionConfig):
                    cfg_inj = None
            except Exception:
                cfg_inj = None

            inj_mems: list[InjectableMemory] = []
            # Pinned facts first (they always head the core block).
            for pf in pinned_facts[:20]:  # safety cap
                inj_mems.append(InjectableMemory(
                    content=pf.content,
                    score=0.0,
                    fact_id=pf.fact_id,
                    importance=getattr(pf, "importance", 0.0) or 0.0,
                    access_count=getattr(pf, "access_count", 0) or 0,
                    pinned=True,
                    source_type="pinned-fact",
                ))
                pinned_seen.add(pf.fact_id)

            # Then recall results (skip duplicates of pinned).
            for r in relevant[:max_results]:
                if r.fact.fact_id in pinned_seen:
                    continue
                inj_mems.append(InjectableMemory(
                    content=r.fact.content,
                    score=round(r.score, 3),
                    fact_id=r.fact.fact_id,
                    importance=getattr(r.fact, "importance", 0.0) or 0.0,
                    access_count=getattr(r.fact, "access_count", 0) or 0,
                    source_type="recall",
                ))

            mode_str = str(getattr(engine, "mode", "B")).upper()
            try:
                context = render_context(
                    inj_mems, mode=mode_str, cfg=cfg_inj, wrap=True,
                )
            except Exception:
                # Fail closed: never serialize retrieved content through a
                # weaker ad-hoc path when the mandatory renderer fails.
                context = ""

            # GAP-FIX (v3.4.65 delivery-lead): the memories[] array is part of
            # the MCP response Claude Code ingests — it MUST be bounded too, not
            # just the rendered `context` string. Previously full unclamped
            # content shipped here (one fact was 131K chars → ~124K-token
            # response, defeating the whole token budget). Clamp each content
            # to per_memory_max_tokens, drop junk, and honour max_results.
            memories = [
                {
                    "fact_id": m.fact_id,
                    "content": clamp_content(
                        sanitize_untrusted_content(m.content), cfg_inj,
                    ),
                    "score": m.score,
                    "is_core": m.is_core,
                    "untrusted": True,
                    "source_type": m.source_type,
                }
                for m in inj_mems[:max_results]
                if not is_low_quality(m.content)
            ]

            # Get learning status
            feedback_count = 0
            try:
                feedback_count = engine._adaptive_learner.get_feedback_count(pid)
            except Exception as exc:
                # Feedback count is a Dash-Core signal; a silent zero
                # masks wiring bugs. Log so operators see the failure.
                logger.warning(
                    "session_init feedback_count read failed: %s", exc,
                )

            # v3.6.9 (#35): generate a stable session_id so clients can pass it
            # to remember() and close_session() for proper session aggregation.
            session_id = (
                f"slm-{datetime.datetime.now(datetime.timezone.utc):%Y%m%d}"
                f"-{uuid.uuid4().hex[:8]}"
            )

            # Register agent + emit event (v3.4.39: SLM_AGENT_ID env support)
            agent_id = _get_agent_id()
            _register_agent(agent_id, pid)
            _emit_event("agent.connected", {
                "agent_id": agent_id,
                "project_path": project_path,
                "memory_count": len(memories),
            })

            return {
                "success": True,
                "session_id": session_id,
                "context": context,
                "memories": memories[:max_results],
                "memory_count": len(memories),
                "core_memory": [m["content"] for m in memories if m.get("is_core")],
                "degraded_mode": degraded_mode,
                "retrieval_mode": (
                    "emergency_fts5_bm25"
                    if degraded_mode
                    else "hybrid_candidate_fusion"
                ),
                "learning": {
                    "feedback_signals": feedback_count,
                    "phase": 1 if feedback_count < 50 else (2 if feedback_count < 200 else 3),
                    "status": (
                        "collecting"
                        if feedback_count < 50
                        else "learning" if feedback_count < 200 else "trained"
                    ),
                },
            }
        except Exception as exc:
            logger.exception("session_init failed")
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # 2. observe — Auto-capture decisions/bugs/preferences
    # ------------------------------------------------------------------
    @server.tool()
    async def observe(
        content: str,
        agent_id: str | None = None,
    ) -> dict:
        """Observe conversation content for automatic memory capture.

        Send conversation snippets here. The system evaluates whether
        the content contains decisions, bug fixes, or preferences worth
        storing. If so, it auto-captures them with classification metadata.

        Call this after making decisions, fixing bugs, or expressing preferences.
        The system will NOT store low-confidence or irrelevant content.

        v3.4.39: ``agent_id`` now defaults to the ``SLM_AGENT_ID`` env var
        (set by each MCP client's config) so observations carry proper
        per-agent attribution.
        """
        if agent_id is None:
            agent_id = _get_agent_id()
        try:
            from superlocalmemory.hooks.auto_capture import AutoCapture
            from superlocalmemory.hooks.rules_engine import RulesEngine
            from superlocalmemory.mcp._pool_adapter import pool_store

            rules = RulesEngine()

            auto = AutoCapture(
                store_fn=pool_store,
                config=rules.get_capture_config(),
            )

            decision = auto.evaluate(content)

            if not decision.capture:
                return {
                    "captured": False,
                    "reason": decision.reason,
                    "category": decision.category,
                    "confidence": round(decision.confidence, 3),
                }

            # Check rules engine for category-level permission
            if not rules.should_capture(decision.category, decision.confidence):
                return {
                    "captured": False,
                    "reason": f"Category '{decision.category}' disabled in rules",
                    "category": decision.category,
                    "confidence": round(decision.confidence, 3),
                }

            # Auto-store via engine.
            # pool_store uses blocking urllib (DaemonPoolProxy) — run in
            # thread so the MCP event loop stays unblocked (#34 class).
            stored = await asyncio.to_thread(
                auto.capture,
                content,
                category=decision.category,
                metadata={"agent_id": agent_id, "source": "auto-observe"},
            )

            if stored:
                _emit_event("memory.created", {
                    "agent_id": agent_id,
                    "category": decision.category,
                    "content_preview": content[:80],
                    "source": "auto-observe",
                }, source_agent=agent_id)

            return {
                "captured": stored,
                "category": decision.category,
                "confidence": round(decision.confidence, 3),
                "reason": decision.reason,
            }
        except Exception as exc:
            logger.exception("observe failed")
            return {"captured": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # 3. report_feedback — Explicit feedback for learning
    # ------------------------------------------------------------------
    @server.tool()
    async def report_feedback(
        fact_id: str,
        feedback: str = "relevant",
        query: str = "",
    ) -> dict:
        """Report whether a recalled memory was useful.

        feedback: "relevant" (memory was helpful), "irrelevant" (not useful),
                  "partial" (somewhat relevant).

        This feedback trains the adaptive ranker to return better results
        over time. The more feedback, the smarter the system gets.
        """
        try:
            engine = get_engine()
            pid = engine.profile_id

            if feedback not in ("relevant", "irrelevant", "partial"):
                return {
                    "success": False,
                    "error": (
                        f"Invalid feedback: {feedback}. "
                        "Use relevant/irrelevant/partial"
                    ),
                }

            record = engine._adaptive_learner.record_feedback(
                query=query,
                fact_id=fact_id,
                feedback_type=feedback,
                profile_id=pid,
            )

            count = engine._adaptive_learner.get_feedback_count(pid)

            _emit_event("pattern.learned", {
                "fact_id": fact_id,
                "feedback": feedback,
                "total_signals": count,
                "phase": 1 if count < 50 else (2 if count < 200 else 3),
            })

            return {
                "success": True,
                "feedback_id": record.feedback_id,
                "total_signals": count,
                "phase": 1 if count < 50 else (2 if count < 200 else 3),
                "message": f"Feedback recorded. {count} total signals."
                + (" Phase 2 unlocked!" if count == 50 else "")
                + (" Phase 3 (ML) unlocked!" if count == 200 else ""),
            }
        except Exception as exc:
            logger.exception("report_feedback failed")
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # close_session — V3.3.12: Expose session closure via MCP
    # ------------------------------------------------------------------

    @server.tool()
    async def close_session(session_id: str = "") -> dict:
        """Close the current session and create temporal summary events.

        Aggregates facts from the session into per-entity temporal summaries,
        enabling temporal queries like "What happened in session X?"

        Args:
            session_id: Session to close. Defaults to the most recent session.
        """
        try:
            engine = get_engine()
            sid = session_id or getattr(engine, '_last_session_id', '')
            # v3.6.9 (#35): _last_session_id was never assigned — fall back to
            # querying the DB for the most recent session_id instead of silently
            # returning summary_events_created: 0.
            if not sid:
                try:
                    db = getattr(engine, '_db', None) or getattr(engine, 'db', None)
                    if db and hasattr(db, 'execute'):
                        rows = db.execute(
                            "SELECT session_id FROM memories "
                            "WHERE session_id != '' ORDER BY created_at DESC LIMIT 1",
                            ()
                        )
                        if rows:
                            sid = str(rows[0][0])
                except Exception:
                    pass
            if not sid:
                return {"success": False, "error": "No session_id provided or found"}
            count = engine.close_session(sid)
            return {
                "success": True,
                "session_id": sid,
                "summary_events_created": count,
            }
        except Exception as exc:
            logger.exception("close_session failed")
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # core_memory — v3.4.65: explicit Core Memory pin management
    # ------------------------------------------------------------------

    @server.tool()
    async def core_memory(
        action: str,
        fact_id: str = "",
        profile_id: str = "default",
    ) -> dict:
        """Manage the explicit Core Memory pin set (v3.4.65).

        - pin:   mark a fact as always-injected
        - unpin: clear the pin
        - list:  return currently pinned facts
        """
        try:
            engine = get_engine()
            db = engine.db
            pid = profile_id or engine.profile_id

            if action == "pin":
                if not fact_id:
                    return {"success": False, "error": "fact_id required for pin"}
                db.set_pinned(fact_id, True)
                return {"success": True, "action": "pin", "fact_id": fact_id}

            if action == "unpin":
                if not fact_id:
                    return {"success": False, "error": "fact_id required for unpin"}
                db.set_pinned(fact_id, False)
                return {"success": True, "action": "unpin", "fact_id": fact_id}

            if action == "list":
                pinned = db.get_pinned(pid)
                cfg_inj = getattr(getattr(engine, "config", None), "injection", None)
                max_tok = getattr(cfg_inj, "per_memory_max_tokens", 600) if cfg_inj else 600
                return {
                    "success": True,
                    "pinned": [
                        {
                            "fact_id": f.fact_id,
                            "content": f.content[: max_tok * 4],
                            "importance": getattr(f, "importance", 0.0),
                        }
                        for f in pinned
                    ],
                    "count": len(pinned),
                }

            return {"success": False, "error": f"unknown action: {action}"}

        except Exception as exc:
            logger.exception("core_memory failed")
            return {"success": False, "error": str(exc)}
