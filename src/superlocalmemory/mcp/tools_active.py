# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

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
import hashlib
import logging
from typing import TYPE_CHECKING, Callable

from mcp.types import ToolAnnotations

from superlocalmemory.core.admission import admits
from superlocalmemory.core.operation_request import OperationKind
from superlocalmemory.infra.data_root import state_path
from superlocalmemory.mcp.shared import authorize_mcp_mutation
from superlocalmemory.storage.database import (
    current_fact_clause_for_connection,
)
from superlocalmemory.storage.read_connection import ReadConnectionFactory

if TYPE_CHECKING:
    from superlocalmemory.mcp._pool_adapter import PoolRecallResponse

logger = logging.getLogger(__name__)


def _answer_check_prefix(response: object) -> str:
    """One plain line to prepend to ``session_init``'s ``context`` string.

    ``context`` is the part many hosts inject verbatim; the sibling JSON
    fields (``abstained``, ``abstention_reason``, ...) already pass through
    the return dict unchanged, but a host reading only ``context`` saw no
    sign the judge had run at all. "" when no judge is configured
    (``calibration_status == "uncalibrated"``) — byte-identical context to
    before this existed, which is the overwhelming majority of installs.
    The same function words the line on every auto-injection surface.
    """
    from superlocalmemory.core.answer_check_notice import answer_check_line

    return answer_check_line(response)


def _sqlite_emergency_recall(
    query: str,
    limit: int,
    profile_id: str = "default",
    max_age_days: int = 30,
) -> "PoolRecallResponse":
    """Emergency fallback: direct SQLite FTS5 BM25 when daemon is unreachable.

    Uses the same ``atomic_facts_fts`` virtual table the daemon uses, with
    native BM25 ranking via ``ORDER BY fts.rank``. This is the Mem0 / Letta
    industry pattern — multi-process safe via SQLite WAL mode.

    Quality degraded vs the full recall path (no semantic, no entity graph, no
    temporal/spreading-activation/Hopfield) but still provides real BM25
    math + age gate. Returns ``degraded_mode=True`` via the caller's flag.

    Used ONLY when Tier-1 (full daemon recall) fails completely. Normal
    path is the full five-producer fusion + entity-graph enhancement;
    this is the fire-alarm.
    """
    import re

    from superlocalmemory.mcp._pool_adapter import PoolFact, PoolRecallItem, PoolRecallResponse

    try:
        from superlocalmemory.storage.fts_terms import version_match_phrases

        tokens = re.findall(r"[A-Za-z0-9]+", query)
        tokens = [t for t in tokens if len(t) >= 2]
        terms = [*(f'"{t}"' for t in tokens), *version_match_phrases(query)]
        if not terms:
            return PoolRecallResponse()
        safe_query = " OR ".join(terms)
        age_clause = (
            f"AND f.created_at >= datetime('now', '-{int(max_age_days)} days') "
            if max_age_days > 0
            else ""
        )
        memory_db = state_path("memory.db").resolve()
        conn = ReadConnectionFactory(memory_db, timeout_ms=250).open()
        try:
            rows = conn.execute(
                f"""SELECT f.fact_id, f.content, f.memory_id, f.created_at, f.profile_id,
                           fts.rank AS bm25_rank
                    FROM atomic_facts_fts AS fts
                    JOIN atomic_facts AS f ON f.fact_id = fts.fact_id
                    WHERE fts.atomic_facts_fts MATCH ?
                      AND f.profile_id = ?
                      {age_clause}
                      {current_fact_clause_for_connection(conn, "f", exclude_non_answer=True)}
                    ORDER BY fts.rank
                    LIMIT ?""",
                (safe_query, profile_id, limit * 2),
            ).fetchall()
            from superlocalmemory.retrieval.fallback_fidelity import admit_fallback
            # Transfer the read lease: a stalled check owns it until it exits.
            owned_conn, conn = conn, None
            rows, fidelity = admit_fallback(
                [dict(r) for r in rows], owned_conn, profile_id, close_db=True)
        finally:
            if conn is not None:
                conn.close()
        from types import MappingProxyType

        from superlocalmemory.server.recall_serializer import recall_response_metadata
        metadata = MappingProxyType(recall_response_metadata(fidelity))
        # FTS5 rank is negative (lower = better). Normalize to [0.3, 0.9].
        if not rows:
            return PoolRecallResponse(metadata=metadata)
        ranks = [r["bm25_rank"] for r in rows]
        rmin, rmax = min(ranks), max(ranks)
        rng = (rmax - rmin) or 1.0
        items = [
            PoolRecallItem(
                fact=PoolFact(
                    fact_id=r["fact_id"] or "", content=r["content"] or "",
                    memory_id=r["memory_id"] or "", created_at=r["created_at"] or "",
                ),
                score=round(0.3 + 0.6 * (1.0 - (r["bm25_rank"] - rmin) / rng), 3),
            )
            for r in rows
        ]
        logger.warning(
            "session_init: EMERGENCY FTS5 fallback (%d results). "
            "Daemon unreachable — semantic/graph channels disabled.",
            len(items),
        )
        return PoolRecallResponse(results=items[:limit], metadata=metadata)
    except Exception as exc:
        logger.warning("Emergency FTS5 fallback failed: %s", exc)
        return PoolRecallResponse()


def _emergency_or_nothing(
    query: str, limit: int, profile_id: str, max_age_days: int
) -> "PoolRecallResponse":
    """The direct-database fallback, for callers on this computer only.

    It reads memories without the visibility rules recall applies, so a caller
    on another computer gets an empty answer instead (its call came through the
    running service, which is the path that applies those rules).
    """
    from superlocalmemory.mcp.remote_caller import current_remote_key_id

    if current_remote_key_id() is not None:
        from superlocalmemory.mcp._pool_adapter import PoolRecallResponse

        return PoolRecallResponse()
    return _sqlite_emergency_recall(query, limit, profile_id=profile_id, max_age_days=max_age_days)


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


def _emit_event(
    event_type: str, payload: dict | None = None, source_agent: str | None = None
) -> None:  # V3.3.12: see also mcp/shared.py
    """Emit an event to the EventBus (best-effort, never raises).

    Dashboard visibility is load-bearing per the v3.4.26 user contract,
    so we log on failure rather than silently dropping the signal.
    """
    resolved_agent = source_agent if source_agent is not None else _get_agent_id()
    try:
        from superlocalmemory.infra.event_bus import EventBus

        bus = EventBus.get_instance(str(state_path("memory.db")))
        bus.emit(event_type, payload=payload, source_agent=resolved_agent, source_protocol="mcp")
    except Exception as exc:
        logger.warning("event emit failed: type=%s err=%s", event_type, exc)


_FEEDBACK_SIGNAL_MAP: dict[str, tuple[str, float]] = {
    "relevant": ("user_positive", 1.0),
    "irrelevant": ("user_negative", 0.0),
    "partial": ("user_correction", 0.5),
}


def _learning_db_path():
    """Resolve the canonical learning.db path."""
    return state_path("learning.db")


def _phase_thresholds() -> tuple[int, int]:
    """Return the (phase 2, phase 3) signal thresholds.

    Sourced from ``learning.ranker`` so the MCP surface can never report a
    different phase than the one recall actually applies. Falls back to the
    documented defaults only if the learning package is unavailable.
    """
    try:
        from superlocalmemory.learning.ranker import (
            PHASE_2_THRESHOLD,
            PHASE_3_THRESHOLD,
        )

        return PHASE_2_THRESHOLD, PHASE_3_THRESHOLD
    except Exception:  # pragma: no cover — learning extras absent
        return 50, 200


_PHASE_2_THRESHOLD, _PHASE_3_THRESHOLD = _phase_thresholds()


def _phase_for_signal_count(count: int) -> int:
    """Map a canonical signal count onto the adaptive ranking phase."""
    if count < _PHASE_2_THRESHOLD:
        return 1
    return 2 if count < _PHASE_3_THRESHOLD else 3


def _record_canonical_feedback(
    *,
    profile_id: str,
    fact_id: str,
    feedback: str,
    query: str = "",
    channel: str = "explicit",
) -> bool:
    """Write explicit feedback to learning.db. Returns True on success.

    True means the ``learning_signals`` row that every phase counter reads
    actually landed — not merely that some row was written somewhere. The
    outcome is RETURNED rather than swallowed so the caller can tell the user
    the truth about whether the write was durable.
    """
    signal_type, value = _FEEDBACK_SIGNAL_MAP.get(
        feedback,
        ("user_correction", 0.5),
    )
    try:
        from superlocalmemory.learning.feedback import FeedbackCollector

        collector = FeedbackCollector(_learning_db_path())
        write = collector.record_explicit_event(
            profile_id=profile_id,
            fact_id=fact_id,
            signal_type=signal_type,
            value=value,
            query=query,
            channel=channel,
        )
        return write.canonical
    except Exception as exc:
        logger.warning(
            "canonical feedback write failed (fact_id=%s): %s",
            fact_id,
            exc,
        )
        return False


def _learning_evidence(engine, profile_id: str) -> dict:
    """``status`` + evidence fields for session_init's ``learning`` block."""
    try:
        from pathlib import Path as _Path

        from superlocalmemory.learning.learning_state import learning_readiness

        state = learning_readiness(
            _Path(engine._config.base_dir) / "learning.db",
            _Path(engine._db.db_path),
            profile_id,
        )
    except Exception as exc:  # noqa: BLE001 — session_init must not fail on it
        logger.debug("learning evidence unavailable: %s", exc)
        return {"status": "unknown", "learning_demonstrated": False}
    return {
        "status": state["status"],
        "learning_demonstrated": state["learning_demonstrated"],
        "labelled_outcomes": state["labelled_outcomes"],
        "settled_from_outcomes": state["settled_from_outcomes"],
        "active_model": bool(state["active_model"]),
    }


def _canonical_feedback_count(profile_id: str) -> int | None:
    """Count the store that gates the adaptive phases.

    Returns None when the store cannot be read. The caller must NOT substitute
    a count from a different table: before issue #106 an unreadable learning.db
    silently fell back to ``feedback_records`` in memory.db — a table no
    consumer reads — so the user watched a fabricated counter climb toward a
    threshold that nothing was measuring, while the durable write did nothing.
    """
    try:
        from superlocalmemory.learning.feedback import FeedbackCollector

        return FeedbackCollector(
            _learning_db_path(),
        ).get_signal_count(profile_id)
    except Exception as exc:
        logger.warning("canonical feedback count failed: %s", exc)
        return None


_SCHEDULED_HORIZON_DAYS = 14
_SCHEDULED_LIMIT = 5


def _upcoming_scheduled_facts(
    engine, now: datetime.datetime, profile_id: str | None = None
) -> list[dict]:
    """Read the current scheduled facts for session startup."""
    from superlocalmemory.mcp.session_schedule import _upcoming_scheduled_facts as scheduled

    return scheduled(engine, now, profile_id)


def _owns_fact(db, profile_id: str, fact_id: str) -> bool:
    """True when ``fact_id`` is a fact of the profile that this caller may act on.

    A remote app is told "not found" for a memory its recall hides, as for one that is not there.
    """
    from superlocalmemory.mcp.remote_visibility import unseen_ids

    if not db.execute(
        "SELECT 1 AS one FROM atomic_facts WHERE fact_id = ? AND profile_id = ?",
        (fact_id, profile_id),
    ):
        return False
    return fact_id not in unseen_ids(db, profile_id, [fact_id])


def _soft_prompt_for(engine, profile_id: str) -> str:
    """The behavioral soft prompt for ``profile_id`` ("" = the engine's own)."""
    invoker = getattr(engine, "_auto_invoker", None)
    if not profile_id:
        getter = getattr(invoker, "_get_soft_prompt_text", None)
        return getter() if callable(getter) else ""
    injector = getattr(invoker, "_prompt_injector", None)
    if injector is None:
        return ""
    return injector.get_injection_context(profile_id) or ""


def register_active_tools(server, get_engine: Callable) -> None:
    """Register 3 active memory tools on *server*."""

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    @admits(OperationKind.RECALL)
    async def session_init(
        project_path: str = "",
        query: str = "",
        max_results: int = 10,
        max_age_days: int = 30,
        session_id: str = "",
        agent_id: str = "",
        profile_id: str = "",
    ) -> dict:
        """Load project context, confirmed rules, pins and session learning."""
        from superlocalmemory.mcp.session_start import session_init as assemble

        return await assemble(
            get_engine,
            project_path=project_path,
            query=query,
            max_results=max_results,
            max_age_days=max_age_days,
            profile_id=profile_id,
            session_id=session_id,
            agent_id=agent_id,
        )

    @server.tool()
    @admits(OperationKind.REMEMBER)
    async def observe(
        content: str,
        agent_id: str | None = None,
        session_id: str = "",
        profile_id: str = "",
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

        ``profile_id`` captures into another profile (empty = the active one);
        such a capture is saved ``personal`` to that profile, never shared,
        and the active profile is not moved.
        """
        if agent_id is None:
            agent_id = _get_agent_id()
        try:
            from superlocalmemory.mcp.request_profile import requested_profile, tool_profile

            named = requested_profile(profile_id)
            if named:
                _pid, refused = tool_profile(get_engine(), named)
                if refused:
                    return {"captured": False, **refused}
            from superlocalmemory.hooks.auto_capture import AutoCapture
            from superlocalmemory.hooks.rules_engine import RulesEngine
            from superlocalmemory.mcp._pool_adapter import pool_store

            rules = RulesEngine(config_path=state_path("config.json"))

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

            if not rules.should_capture(decision.category, decision.confidence):
                return {
                    "captured": False,
                    "reason": f"Category '{decision.category}' disabled in rules",
                    "category": decision.category,
                    "confidence": round(decision.confidence, 3),
                }

            from superlocalmemory.mcp.session_binding import resolve_session_id

            effective_session_id = resolve_session_id(
                session_id,
                agent_id=agent_id,
                allow_agent_fallback=False,
            )
            capture_meta = {
                "agent_id": agent_id,
                "session_id": effective_session_id,
                "source": "auto-observe",
            }
            if named:
                capture_meta = {**capture_meta, "profile_id": named, "scope": "personal"}
            stored = await asyncio.to_thread(
                auto.capture,
                content,
                category=decision.category,
                metadata=capture_meta,
            )

            if stored:
                from superlocalmemory.core.config import SLMConfig
                from superlocalmemory.memory_core import (
                    effective_pii_redaction,
                    prepare_user_text,
                )

                _config = SLMConfig.load()
                if effective_pii_redaction(_config):
                    _config.pii_redaction = True

                _emit_event(
                    "memory.captured",
                    {
                        "agent_id": agent_id,
                        "category": decision.category,
                        "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                        "content_preview": prepare_user_text(
                            _config,
                            content,
                        ).text[:80],
                        "source": "auto-observe",
                    },
                    source_agent=agent_id,
                )

            return {
                "captured": stored,
                "category": decision.category,
                "confidence": round(decision.confidence, 3),
                "reason": decision.reason,
                "session_id": effective_session_id,
            }
        except Exception as exc:
            logger.exception("observe failed")
            return {"captured": False, "error": str(exc)}

    @server.tool()
    @admits(OperationKind.REMEMBER)
    async def report_feedback(
        fact_id: str,
        feedback: str = "relevant",
        query: str = "",
        profile_id: str = "",
        recall_query_id: str = "",
    ) -> dict:
        """Report whether a recalled memory was useful.

        feedback: "relevant" (memory was helpful), "irrelevant" (not useful),
                  "partial" (somewhat relevant).

        This feedback trains the adaptive ranker to return better results
        over time. The more feedback, the smarter the system gets.
        ``profile_id``: the profile the memory was recalled from (empty = the
        active one).
        ``recall_query_id``: the ``query_id`` recall returned. With it, the
        judgement is also stored as an outcome of that exact recall (the same
        writer ``report_outcome`` uses), so it settles that recall's bandit
        play and becomes a training label for it.
        """
        try:
            from superlocalmemory.mcp.request_profile import tool_profile

            engine = get_engine()
            pid, refused = tool_profile(engine, profile_id)
            if refused:
                return refused

            if feedback not in ("relevant", "irrelevant", "partial"):
                return {
                    "success": False,
                    "error": (f"Invalid feedback: {feedback}. Use relevant/irrelevant/partial"),
                }

            from superlocalmemory.mcp.remote_visibility import unseen_ids

            if fact_id in unseen_ids(engine._db, pid, [fact_id]):
                return {"success": False, "error": "Memory not found."}

            authorization = authorize_mcp_mutation(
                engine,
                "update",
                mutation_source="mcp-recall-feedback",
                profile_id=pid,
                fact_id=fact_id,
                content_preview=feedback,
            )
            record = engine._adaptive_learner.record_feedback(
                query=query,
                fact_id=fact_id,
                feedback_type=feedback,
                profile_id=pid,
            )

            canonical_recorded = _record_canonical_feedback(
                profile_id=pid,
                fact_id=fact_id,
                feedback=feedback,
                query=query,
            )

            count = _canonical_feedback_count(pid)
            if recall_query_id:
                from superlocalmemory.learning.outcome_events import (
                    record_feedback_outcome,
                )

                record_feedback_outcome(engine, pid, fact_id, feedback, recall_query_id)
            authorization.complete()

            if not canonical_recorded or count is None:
                return {
                    "success": False,
                    "durable": False,
                    "feedback_id": record.feedback_id,
                    "total_signals": count,
                    "error": (
                        "Feedback was accepted but could not be written to "
                        "the canonical learning store (learning.db), so it "
                        "will not influence ranking. Run 'slm doctor' to "
                        "diagnose learning.db."
                    ),
                }

            phase = _phase_for_signal_count(count)
            _emit_event(
                "pattern.learned",
                {
                    "fact_id": fact_id,
                    "feedback": feedback,
                    "total_signals": count,
                    "phase": phase,
                },
            )

            result = {
                "success": True,
                "durable": True,
                "feedback_id": record.feedback_id,
                "total_signals": count,
                "phase": phase,
                "message": f"Feedback recorded. {count} total signals."
                + (" Phase 2 unlocked!" if count == _PHASE_2_THRESHOLD else "")
                + (" Phase 3 (ML) unlocked!" if count == _PHASE_3_THRESHOLD else ""),
            }
            return result
        except Exception as exc:
            logger.exception("report_feedback failed")
            return {"success": False, "error": str(exc)}

    @server.tool()
    @admits(OperationKind.CONSOLIDATE)
    async def close_session(session_id: str = "", profile_id: str = "") -> dict:
        """Close the current session and create temporal summary events.

        Aggregates facts from the session into per-entity temporal summaries,
        enabling temporal queries like "What happened in session X?"

        Args:
            session_id: Session to close. Defaults to the most recent session.
            profile_id: The profile the session belongs to (empty = the
                active one). The active profile is not moved.
        """
        try:
            from superlocalmemory.mcp.request_profile import requested_profile, tool_profile

            engine = get_engine()
            named = requested_profile(profile_id)
            pid, refused = tool_profile(engine, named)
            if refused:
                return refused
            sid = session_id or ("" if named else getattr(engine, "_last_session_id", ""))
            if not sid:
                try:
                    db = getattr(engine, "_db", None) or getattr(engine, "db", None)
                    if db and hasattr(db, "execute"):
                        rows = db.execute(
                            "SELECT session_id FROM memories "
                            "WHERE profile_id = ? AND session_id != '' "
                            "ORDER BY created_at DESC LIMIT 1",
                            (pid,),
                        )
                        if rows:
                            sid = str(rows[0][0])
                except Exception:
                    pass
            if not sid:
                return {"success": False, "error": "No session_id provided or found"}
            authorization = authorize_mcp_mutation(
                engine,
                "update",
                mutation_source="mcp-session-close",
                profile_id=pid,
                content_preview=sid,
            )
            count = await asyncio.to_thread(
                (lambda: engine.close_session(sid, profile_id=named))
                if named else (lambda: engine.close_session(sid))
            )
            authorization.complete()
            return {
                "success": True,
                "session_id": sid,
                "summary_events_created": count,
            }
        except Exception as exc:
            logger.exception("close_session failed")
            return {"success": False, "error": str(exc)}

    @server.tool()
    @admits(OperationKind.CORRECT)
    async def core_memory(
        action: str,
        fact_id: str = "",
        profile_id: str = "",
    ) -> dict:
        """Manage the explicit Core Memory pin set (v3.4.65).

        - pin:   mark a fact as always-injected
        - unpin: clear the pin
        - list:  return currently pinned facts

        ``profile_id``: the profile whose pins these are (empty = the active
        one). A pin or unpin changes only a fact that profile owns.
        """
        try:
            from superlocalmemory.mcp.request_profile import tool_profile

            engine = get_engine()
            db = engine.db
            pid, refused = tool_profile(engine, profile_id)
            if refused:
                return refused

            if action == "pin":
                if not fact_id:
                    return {"success": False, "error": "fact_id required for pin"}
                if not _owns_fact(db, pid, fact_id):
                    return {"success": False, "error": f"Memory {fact_id} not found"}
                authorization = authorize_mcp_mutation(
                    engine,
                    "update",
                    mutation_source="mcp-core-memory-pin",
                    profile_id=pid,
                    fact_id=fact_id,
                )
                db.set_pinned(fact_id, True)
                authorization.complete()
                return {"success": True, "action": "pin", "fact_id": fact_id}

            if action == "unpin":
                if not fact_id:
                    return {"success": False, "error": "fact_id required for unpin"}
                if not _owns_fact(db, pid, fact_id):
                    return {"success": False, "error": f"Memory {fact_id} not found"}
                authorization = authorize_mcp_mutation(
                    engine,
                    "update",
                    mutation_source="mcp-core-memory-unpin",
                    profile_id=pid,
                    fact_id=fact_id,
                )
                db.set_pinned(fact_id, False)
                authorization.complete()
                return {"success": True, "action": "unpin", "fact_id": fact_id}

            if action == "list":
                from superlocalmemory.mcp.remote_visibility import visible_facts

                pinned = visible_facts(db, pid, db.get_pinned(pid))
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
