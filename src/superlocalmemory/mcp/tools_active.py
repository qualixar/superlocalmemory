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
import hashlib
import logging
import uuid
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
    query: str, limit: int, profile_id: str = "default",
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
        # FTS5 MATCH syntax: tokenize the query, drop special characters
        # that confuse the parser (/, :, ., etc), and join with OR for
        # broadest matching. Wrap each term in quotes to escape any
        # remaining special-meaning chars.
        from superlocalmemory.storage.fts_terms import version_match_phrases

        tokens = re.findall(r"[A-Za-z0-9]+", query)
        tokens = [t for t in tokens if len(t) >= 2]
        # Dotted versions are matched whole too (storage.fts_terms); the
        # length filter above would otherwise drop the "4" and "1" of 4.1.20.
        terms = [*(f'"{t}"' for t in tokens), *version_match_phrases(query)]
        if not terms:
            return PoolRecallResponse()
        safe_query = " OR ".join(terms)
        age_clause = (
            f"AND f.created_at >= datetime('now', '-{int(max_age_days)} days') "
            if max_age_days > 0 else ""
        )
        memory_db = state_path("memory.db").resolve()
        conn = ReadConnectionFactory(memory_db, timeout_ms=250).open()
        try:
            rows = conn.execute(
                f"""SELECT f.fact_id, f.content, f.memory_id, f.created_at,
                           fts.rank AS bm25_rank
                    FROM atomic_facts_fts AS fts
                    JOIN atomic_facts AS f ON f.fact_id = fts.fact_id
                    WHERE fts.atomic_facts_fts MATCH ?
                      AND f.profile_id = ?
                      {age_clause}
                      {current_fact_clause_for_connection(conn, "f")}
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


def _emergency_or_nothing(query: str, limit: int, profile_id: str, max_age_days: int) -> "PoolRecallResponse":
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


# ---------------------------------------------------------------------------
# Canonical learning-store feedback (issues #102, #106)
#
# learning.db is the single store every learning consumer reads. Within it the
# canonical tables are ``learning_signals`` + ``learning_features``: the phase
# gate (recall_pipeline), the dashboard Living Brain panel, the ranker-phase
# card, and the retrainer all resolve their phase from ``learning_signals``.
# ``learning_feedback`` is the pre-v3.4.22 table that legacy_migration copies
# forward into it.
#
# Recall itself is deliberately read-only and must never open a writer, so an
# explicit feedback command is the only durable writer in the design. These
# helpers are that writer, and — per issue #106 — they report the SAME number
# the gate and the dashboard use. There is deliberately no fall back to a
# different store's count: a cross-store fallback is what let a total write
# failure still return "success" beside a plausibly incrementing counter.
# ---------------------------------------------------------------------------

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
    *, profile_id: str, fact_id: str, feedback: str, query: str = "",
    channel: str = "explicit",
) -> bool:
    """Write explicit feedback to learning.db. Returns True on success.

    True means the ``learning_signals`` row that every phase counter reads
    actually landed — not merely that some row was written somewhere. The
    outcome is RETURNED rather than swallowed so the caller can tell the user
    the truth about whether the write was durable.
    """
    signal_type, value = _FEEDBACK_SIGNAL_MAP.get(
        feedback, ("user_correction", 0.5),
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
            "canonical feedback write failed (fact_id=%s): %s", fact_id, exc,
        )
        return False


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


# How far ahead a session looks for scheduled facts, and how many it shows.
# A session preamble is not a calendar: a long horizon or a large cap turns a
# useful heads-up into a wall of text nobody reads.
_SCHEDULED_HORIZON_DAYS = 14
_SCHEDULED_LIMIT = 5


def _upcoming_scheduled_facts(engine, now: datetime.datetime,
                              profile_id: str | None = None) -> list[dict]:
    """Facts scheduled from today through the horizon, soonest first.

    Bounded and index-backed, because this runs on every session start. Returns
    an empty list on any failure: a session must still open when this query
    cannot answer, so the caller omits the surface entirely rather than showing
    an empty section that reads like a defect.

    Selects on the stored type of the fact, which is a classification recorded at
    write time. It is unrelated to the similarly named retrieval channel despite
    sharing the word.
    """
    try:
        db = getattr(engine, "_db", None) or getattr(engine, "db", None)
        if db is None or not hasattr(db, "execute"):
            return []
        # Both 'YYYY-MM-DD' and full timestamps compare correctly as text,
        # because ISO-8601 orders lexicographically. The upper bound is
        # exclusive, so the horizon day itself is included.
        start = now.date().isoformat()
        end = (now + datetime.timedelta(days=_SCHEDULED_HORIZON_DAYS + 1)).date().isoformat()
        # Resolved the way retrieval/scope_policy.py resolves it: `db` here is
        # duck-typed on `.execute` alone, so a caller may hand us an object
        # that is not a DatabaseManager. Reaching for the attribute directly
        # raised inside this function's `except`, which swallowed the entire
        # prospective surface rather than the filter -- silently, and only a
        # test double noticed. Fall back to scope-only rather than to nothing.
        current = ""
        clause_fn = getattr(type(db), "current_fact_clause", None)
        if callable(clause_fn):
            try:
                current = clause_fn(db)
            except Exception:  # noqa: BLE001 -- fall back to scope-only
                logger.warning("scheduled surface could not resolve its filter")
                current = ""
        rows = db.execute(
            "SELECT fact_id, content, referenced_date"
            " FROM atomic_facts"
            " WHERE profile_id = ?"
            "   AND fact_type = 'prospective'"
            "   AND referenced_date IS NOT NULL"
            "   AND referenced_date >= ?"
            "   AND referenced_date < ?"
            f"   {current}"
            " ORDER BY referenced_date ASC"
            f" LIMIT {_SCHEDULED_LIMIT}",
            (profile_id or engine.profile_id, start, end),
        )
        return [
            {"fact_id": r["fact_id"], "content": r["content"],
             "scheduled_at": r["referenced_date"]}
            for r in rows
        ]
    except Exception as exc:
        logger.warning("scheduled-fact surface failed: %s", exc)
        return []


def _owns_fact(db, profile_id: str, fact_id: str) -> bool:
    return bool(db.execute(
        "SELECT 1 AS one FROM atomic_facts WHERE fact_id = ? AND profile_id = ?",
        (fact_id, profile_id)))


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

    # ------------------------------------------------------------------
    # 1. session_init — Auto-recall project context at session start
    # ------------------------------------------------------------------
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
        """Initialize session with relevant memory context.

        Call this ONCE at the start of every session. Returns:
        - Recent decisions and patterns for this project
        - Top relevant memories based on project path or query
        - Learning status (signal count, ranking phase)

        The AI should call this automatically before any other work.

        Parameters:
            project_path: Working directory path. Memories saved under this
                project rank above others of similar relevance (nothing is
                removed), and its name builds the search query when no
                explicit query is provided.
            query: Override the search query. If omitted, derived from project_path
                or falls back to "recent important decisions".
            max_results: Maximum memories to return (default: 10).
            max_age_days: Suppress memories older than this many days unless their
                relevance score is ≥ 0.70 (architectural decisions that remain
                permanently relevant still surface). Default: 30.
                Set to 0 to disable the age gate entirely.
            profile_id: Load another profile's context (empty = the active
                one). The active profile is not moved.
        """
        try:
            from superlocalmemory.core.answer_check_scope import skip_answer_check
            from superlocalmemory.hooks.rules_engine import RulesEngine
            from superlocalmemory.mcp._pool_adapter import pool_recall
            from superlocalmemory.mcp._recall_metadata import forward_recall_metadata

            from superlocalmemory.mcp.request_profile import requested_profile, tool_profile

            engine = get_engine()
            named = requested_profile(profile_id)
            pid, refused = tool_profile(engine, named)
            if refused:
                return refused
            rules = RulesEngine(config_path=state_path("config.json"))

            if not rules.should_recall("session_start"):
                return {
                    "success": True,
                    "context": "",
                    "memories": [],
                    "message": "Auto-recall disabled",
                }

            recall_config = rules.get_recall_config()
            relevance_threshold = recall_config.get("relevance_threshold", 0.3)
            # #150: the project is used as a project, not as search words. Its
            # memories rank above others of similar relevance (a bounded
            # preference that removes nothing), and a derived query names the
            # project rather than every directory in its path.
            from superlocalmemory.core.project_identity import session_context_query

            search_query = query or session_context_query(project_path)
            _project_kwargs = (
                {"prefer_project": project_path.strip()} if (project_path or "").strip() else {}
            )
            if named:
                _project_kwargs = {**_project_kwargs, "profile_id": named}

            # 2-tier recall (industry pattern: Hindsight / Zep / Supermemory):
            # PRIMARY: full recall via daemon — five candidate producers (semantic
            #          + BM25 + temporal + Hopfield + spreading-activation) into RRF
            #          fusion, then entity-graph post-fusion enhancement, FSRS decay.
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
                # S-M2: loading a session's context is not a question — with
                # or without an explicit query it is a topic, not something
                # to ask whether the memories answer. Never judged, so the
                # project path and the top memories are never sent to the
                # online check and never billed. ``to_thread`` carries the
                # marker; the proxy turns it into ``answer_check=skip``.
                with skip_answer_check():
                    response = await asyncio.to_thread(
                        pool_recall, search_query, limit=max_results, fast=None,
                        **_project_kwargs,
                    )
            except (PoolError, Exception) as exc:
                logger.warning(
                    "session_init: daemon recall failed (%s) — using FTS5 emergency fallback. "
                    "Memory system is in DEGRADED MODE: semantic/graph channels unavailable.",
                    exc,
                )
                response = _emergency_or_nothing(search_query, max_results, pid, max_age_days)
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

            # Standing rules and active decisions you or your agents confirmed
            # (kind declared on save, or set by you) follow the pins, so every
            # session starts knowing them. Suggested kinds are never used here.
            try:
                from superlocalmemory.core import standing_rules

                if standing_rules.enabled(getattr(engine, "config", None)):
                    for sf in standing_rules.standing_facts(
                            engine.db, pid, frozenset(pinned_seen)):
                        if sf.fact_id in pinned_seen:
                            continue
                        inj_mems.append(InjectableMemory(
                            content=sf.content,
                            score=0.0,
                            fact_id=sf.fact_id,
                            importance=sf.importance,
                            access_count=sf.access_count,
                            pinned=True,
                            source_type=f"standing-{sf.kind}",
                        ))
                        pinned_seen.add(sf.fact_id)
            except Exception:  # noqa: BLE001 - session start never fails on this
                pass

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

            # Live soft-prompt injection (Phase 5): prepend the profile's behavioral
            # soft prompt so it reaches the session_init context agents actually
            # consume (same engine->AutoInvoker bridge AutoRecall uses). Fail-soft.
            try:
                _soft_prompt = _soft_prompt_for(engine, named)
                if _soft_prompt:
                    context = f"{_soft_prompt}\n\n{context}" if context else _soft_prompt
            except Exception as exc:
                logger.warning("session_init soft-prompt injection failed: %s", exc)

            # Answer-check prefix: say up front when the judge found the
            # candidates insufficient (or confidently sufficient), since the
            # agent may act on `context` alone without reading the sibling
            # `abstained` / `abstention_reason` fields below.
            _answer_check = _answer_check_prefix(response)
            if _answer_check:
                context = f"{_answer_check}\n\n{context}" if context else _answer_check

            # GAP-FIX (v3.4.65 delivery-lead): the memories[] array is part of
            # the MCP response Claude Code ingests — it MUST be bounded too, not
            # just the rendered `context` string. Previously full unclamped
            # content shipped here (one fact was 131K chars → ~124K-token
            # response, defeating the whole token budget). Clamp each content
            # to per_memory_max_tokens, drop junk, and honour max_results.
            contract_by_id = {
                r.fact.fact_id: r for r in relevant[:max_results]
            }
            memories = []
            for position, m in enumerate(inj_mems[:max_results], start=1):
                if is_low_quality(m.content):
                    continue
                result_contract = contract_by_id.get(m.fact_id)
                relevance = (
                    getattr(result_contract, "relevance_score", m.score)
                    if result_contract is not None else m.score
                )
                memory_confidence = (
                    getattr(result_contract, "memory_confidence", None)
                    if result_contract is not None else None
                )
                memories.append({
                    "fact_id": m.fact_id,
                    "content": clamp_content(
                        sanitize_untrusted_content(m.content), cfg_inj,
                    ),
                    "score": m.score,
                    "relevance_score": relevance,
                    "ranking_score": (
                        getattr(result_contract, "ranking_score", None)
                        if result_contract is not None else None
                    ),
                    "confidence": memory_confidence,
                    "memory_confidence": memory_confidence,
                    "rank_position": (
                        getattr(result_contract, "rank_position", position)
                        if result_contract is not None else position
                    ),
                    "is_core": m.is_core,
                    "untrusted": True,
                    "source_type": m.source_type,
                })

            # Learning status — issue #106: read the SAME canonical counter
            # that report_feedback reports, the recall gate applies, and the
            # dashboard displays. This used to read ``feedback_records`` in
            # memory.db, so session_init and report_feedback returned two
            # different "signal" totals for one profile in the same session.
            # A silent zero masks wiring bugs, so a failed read is logged.
            feedback_count = _canonical_feedback_count(pid)
            if feedback_count is None:
                logger.warning(
                    "session_init canonical signal count unavailable for "
                    "profile %s; reporting 0", pid,
                )
                feedback_count = 0

            # A gateway can serve concurrent host conversations.  An explicit
            # host id therefore always wins; generating one is retained only
            # for older clients that cannot supply lifecycle identity.
            effective_session_id = session_id.strip() or (
                f"slm-{datetime.datetime.now(datetime.timezone.utc):%Y%m%d}"
                f"-{uuid.uuid4().hex[:8]}"
            )
            effective_agent_id = agent_id.strip() or _get_agent_id()
            # Backward-compatible default for legacy close_session() callers;
            # native hosts must pass their explicit id when sessions overlap.
            # A session opened for a named profile is not recorded here: this
            # shared value belongs to the active profile's callers.
            if not named:
                engine._last_session_id = effective_session_id

            _upcoming_events = _upcoming_scheduled_facts(engine, _now, pid)

            return {
                "success": True,
                "session_id": effective_session_id,
                "agent_id": effective_agent_id,
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
                # M-10: the recall's metadata in full (who chose the order,
                # abandoned channels, temporal frame, fields added later).
                # The explicit fields below keep their session_init meaning.
                **forward_recall_metadata(response),
                "score_contract_version": getattr(
                    response, "score_contract_version", "2"
                ),
                "calibration_status": getattr(
                    response, "calibration_status", "uncalibrated"
                ),
                "calibration_id": getattr(response, "calibration_id", None),
                "answer_confidence": getattr(response, "answer_confidence", None),
                "abstained": getattr(response, "abstained", not bool(relevant)),
                "abstention_reason": getattr(response, "abstention_reason", None),
                "learning": {
                    "feedback_signals": feedback_count,
                    "phase": _phase_for_signal_count(feedback_count),
                    "status": (
                        "collecting"
                        if feedback_count < _PHASE_2_THRESHOLD
                        else "learning"
                        if feedback_count < _PHASE_3_THRESHOLD
                        else "trained"
                    ),
                },
                # Scheduled-event surface: present only when facts exist in the
                # 14-day window. Absent means the window is empty, not an error.
                **( {"upcoming_events": _upcoming_events}
                    if _upcoming_events else {} ),
            }
        except Exception as exc:
            logger.exception("session_init failed")
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # 2. observe — Auto-capture decisions/bugs/preferences
    # ------------------------------------------------------------------
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
            from superlocalmemory.mcp.session_binding import resolve_session_id
            effective_session_id = resolve_session_id(
                session_id, agent_id=agent_id, allow_agent_fallback=False,
            )
            capture_meta = {"agent_id": agent_id, "session_id": effective_session_id,
                            "source": "auto-observe"}
            if named:
                # DaemonPoolProxy.store sends both as /remember request fields.
                capture_meta = {**capture_meta, "profile_id": named, "scope": "personal"}
            stored = await asyncio.to_thread(
                auto.capture,
                content,
                category=decision.category,
                metadata=capture_meta,
            )

            if stored:
                # The hash is the stored memory's source hash, so an erasure can
                # find this event and blank its preview (core/erasure_scrub.py).
                from superlocalmemory.core.config import SLMConfig
                from superlocalmemory.memory_core import (
                    effective_pii_redaction,
                    prepare_user_text,
                )

                _config = SLMConfig.load()
                # A plain loaded config does not carry the deployment policy.
                if effective_pii_redaction(_config):
                    _config.pii_redaction = True

                _emit_event("memory.captured", {
                    "agent_id": agent_id,
                    "category": decision.category,
                    "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    "content_preview": prepare_user_text(
                        _config, content,
                    ).text[:80],
                    "source": "auto-observe",
                }, source_agent=agent_id)

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

    # ------------------------------------------------------------------
    # 3. report_feedback — Explicit feedback for learning
    # ------------------------------------------------------------------
    @server.tool()
    @admits(OperationKind.REMEMBER)
    async def report_feedback(
        fact_id: str,
        feedback: str = "relevant",
        query: str = "",
        profile_id: str = "",
    ) -> dict:
        """Report whether a recalled memory was useful.

        feedback: "relevant" (memory was helpful), "irrelevant" (not useful),
                  "partial" (somewhat relevant).

        This feedback trains the adaptive ranker to return better results
        over time. The more feedback, the smarter the system gets.
        ``profile_id``: the profile the memory was recalled from (empty = the
        active one).
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
                    "error": (
                        f"Invalid feedback: {feedback}. "
                        "Use relevant/irrelevant/partial"
                    ),
                }

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

            # The AdaptiveLearner write above lands in ``feedback_records`` in
            # memory.db — a table whose only readers are AdaptiveLearner's own
            # count and its train(), which nothing in the running system
            # calls. It is kept so existing data and GDPR erasure stay intact,
            # but it is NOT the learning write and its count is NOT reported.
            #
            # The canonical store is learning.db's ``learning_signals`` (+ the
            # paired ``learning_features`` row). Writing there is what makes
            # feedback do work: the recall phase gate, the dashboard Living
            # Brain panel, the ranker-phase card, and the retrainer all read
            # it. Recall stays read-only by design, so this explicit path is
            # the only durable writer.
            canonical_recorded = _record_canonical_feedback(
                profile_id=pid,
                fact_id=fact_id,
                feedback=feedback,
                query=query,
            )

            # issue #106: report the count from the store that ACTUALLY gates
            # the phases, and report NOTHING when it cannot be read. The old
            # fallback to ``feedback_records`` is what made a total write
            # failure indistinguishable from success: the response carried a
            # plausible, incrementing ``total_signals`` sourced from a table
            # nothing consumes, so the caller had no way to notice that
            # learning.db was never touched.
            count = _canonical_feedback_count(pid)
            authorization.complete()

            if not canonical_recorded or count is None:
                # Never claim a durable learning write that did not happen.
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
            _emit_event("pattern.learned", {
                "fact_id": fact_id,
                "feedback": feedback,
                "total_signals": count,
                "phase": phase,
            })

            result = {
                "success": True,
                "durable": True,
                "feedback_id": record.feedback_id,
                "total_signals": count,
                "phase": phase,
                "message": f"Feedback recorded. {count} total signals."
                + (" Phase 2 unlocked!"
                   if count == _PHASE_2_THRESHOLD else "")
                + (" Phase 3 (ML) unlocked!"
                   if count == _PHASE_3_THRESHOLD else ""),
            }
            return result
        except Exception as exc:
            logger.exception("report_feedback failed")
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # close_session — V3.3.12: Expose session closure via MCP
    # ------------------------------------------------------------------

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
            # The remembered last session belongs to the active profile's
            # callers, so a named profile falls back to its own latest one.
            sid = session_id or ("" if named else getattr(engine, '_last_session_id', ''))
            # v3.6.9 (#35): _last_session_id was never assigned — fall back to
            # querying the DB for the most recent session_id instead of silently
            # returning summary_events_created: 0.
            if not sid:
                try:
                    db = getattr(engine, '_db', None) or getattr(engine, 'db', None)
                    if db and hasattr(db, 'execute'):
                        rows = db.execute(
                            "SELECT session_id FROM memories "
                            "WHERE profile_id = ? AND session_id != '' "
                            "ORDER BY created_at DESC LIMIT 1",
                            (pid,)
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
            count = (engine.close_session(sid, profile_id=named) if named
                     else engine.close_session(sid))
            authorization.complete()
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
            # Isolation: the tenant is the active profile, or the profile this
            # call names (it must exist). A fact another profile owns is "not
            # found": before 4.1.21 a pin could reach any fact by id.
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
