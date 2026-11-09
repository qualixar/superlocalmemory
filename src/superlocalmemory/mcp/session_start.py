# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""Project-scoped session startup, with bounded context assembly stages."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime

from superlocalmemory.core.injection import (
    InjectableMemory,
    clamp_content,
    is_low_quality,
    render_context,
    sanitize_untrusted_content,
)
from superlocalmemory.mcp import tools_active as shared


async def _recall(query, project_path, named, pid, max_results, max_age_days):
    from superlocalmemory.core.answer_check_scope import context_load
    from superlocalmemory.mcp._pool_adapter import pool_recall

    kwargs = {"prefer_project": project_path.strip()} if project_path.strip() else {}
    if named:
        kwargs["profile_id"] = named
    try:
        with context_load():
            response = await asyncio.to_thread(
                pool_recall, query, limit=max_results, fast=None, **kwargs
            )
        return response, False
    except Exception as exc:
        shared.logger.warning("session_init recall failed (%s); using emergency FTS5", exc)
        return shared._emergency_or_nothing(query, max_results, pid, max_age_days), True


def _relevant(response, threshold, max_age_days):
    now = datetime.now(UTC)

    def age(value):
        if not value:
            return 0.0
        try:
            created = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
            return max(0.0, (now - created).total_seconds() / 86400.0)
        except (ValueError, TypeError):
            return 0.0

    return [
        r
        for r in response.results
        if r.score >= threshold
        and (max_age_days <= 0 or age(r.fact.created_at) <= max_age_days or r.score >= 0.7)
    ]


def _recall_relevance(response) -> dict[str, float]:
    """Memory id -> best score this session's recall gave any of its facts:
    which standing rules matter to this session, at no extra cost."""
    scores: dict[str, float] = {}
    for result in getattr(response, "results", None) or []:
        memory_id = str(getattr(result.fact, "memory_id", "") or "")
        if memory_id:
            scores[memory_id] = max(scores.get(memory_id, 0.0), float(result.score or 0.0))
    return scores


def _priority_facts(engine, pid, project_path, response=None):
    from superlocalmemory.core import standing_rules
    from superlocalmemory.mcp import remote_visibility

    try:
        pins = engine.db.get_pinned(pid)
    except Exception:
        pins = []
    standing = []
    try:
        if standing_rules.enabled(getattr(engine, "config", None)):
            standing = list(
                standing_rules.standing_facts(
                    engine.db, pid, frozenset(f.fact_id for f in pins), project_path=project_path,
                    relevance=_recall_relevance(response),
                )
            )
    except Exception as exc:
        shared.logger.debug("session standing rules unavailable: %s", exc)
    upcoming = shared._upcoming_scheduled_facts(engine, datetime.now(UTC), pid)
    if remote_visibility.current_view():
        hidden = remote_visibility.hidden_fact_ids(
            engine._db,
            pid,
            [f.fact_id for f in pins]
            + [f.fact_id for f in standing]
            + [e["fact_id"] for e in upcoming],
        )
        pins = [f for f in pins if f.fact_id not in hidden]
        standing = [f for f in standing if f.fact_id not in hidden]
        upcoming = [e for e in upcoming if e["fact_id"] not in hidden]
    return pins, standing, upcoming


def _jev_ordered(engine, query, relevant):
    from superlocalmemory.core.session_jev import jev_order

    retrieval = getattr(getattr(engine, "config", None), "retrieval", None)
    return jev_order(query, relevant, lambda result: result.fact.content, retrieval)


def _injectable(pins, standing, relevant, max_results):
    memories, seen = [], set()
    for fact in pins[:20]:
        memories.append(_item(fact, pinned=True, source="pinned-fact"))
        seen.add(fact.fact_id)
    for fact in standing:
        if fact.fact_id not in seen:
            memories.append(_item(fact, pinned=True, source=f"standing-{fact.kind}"))
            seen.add(fact.fact_id)
    for result in relevant[:max_results]:
        if result.fact.fact_id not in seen:
            memories.append(_item(result.fact, score=round(result.score, 3), source="recall",
                                  tier=getattr(result, "tier", "") or ""))
    return memories


def _item(fact, *, pinned=False, source, score=0.0, tier=""):
    return InjectableMemory(
        content=fact.content,
        score=score,
        fact_id=fact.fact_id,
        importance=getattr(fact, "importance", 0.0) or 0.0,
        access_count=getattr(fact, "access_count", 0) or 0,
        pinned=pinned,
        source_type=source,
        support_tier=str(tier),
    )


def _context(engine, named, response, memories, cfg):
    try:
        context = render_context(
            memories, mode=str(getattr(engine, "mode", "B")).upper(), cfg=cfg, wrap=True
        )
    except Exception:
        context = ""
    try:
        soft = shared._soft_prompt_for(engine, named)
        if soft:
            context = f"{soft}\n\n{context}" if context else soft
    except Exception as exc:
        shared.logger.warning("session_init soft-prompt injection failed: %s", exc)
    check = shared._answer_check_prefix(response)
    return f"{check}\n\n{context}" if check and context else check or context


def _memory_rows(memories, relevant, max_results, cfg):
    """Rows for what the context carries. Pins and standing rules have caps of
    their own; 4.2.14: they no longer use up the recall rows' ``max_results``
    (with ten rules, recall's relevant memories used to get no row at all)."""
    contracts = {r.fact.fact_id: r for r in relevant[:max_results]}
    priority = [m for m in memories if m.source_type != "recall"]
    recalled = [m for m in memories if m.source_type == "recall"][:max_results]
    rows = []
    for position, memory in enumerate(priority + recalled, start=1):
        if is_low_quality(memory.content):
            continue
        contract = contracts.get(memory.fact_id)
        confidence = getattr(contract, "memory_confidence", None)
        rows.append(
            {
                "fact_id": memory.fact_id,
                "content": clamp_content(sanitize_untrusted_content(memory.content), cfg),
                "score": memory.score,
                "relevance_score": getattr(contract, "relevance_score", memory.score),
                "ranking_score": getattr(contract, "ranking_score", None),
                "confidence": confidence,
                "memory_confidence": confidence,
                "rank_position": getattr(contract, "rank_position", position),
                "is_core": memory.is_core,
                "untrusted": True,
                "source_type": memory.source_type,
            }
        )
    return rows


def _feedback_count(pid):
    count = shared._canonical_feedback_count(pid)
    if count is None:
        shared.logger.warning("session_init signal count unavailable for profile %s", pid)
    return count or 0


def _envelope(
    engine,
    pid,
    named,
    session_id,
    agent_id,
    context,
    memories,
    response,
    degraded,
    relevant,
    upcoming,
):
    from superlocalmemory.mcp._recall_metadata import forward_recall_metadata

    count = _feedback_count(pid)
    sid = session_id.strip() or f"slm-{datetime.now(UTC):%Y%m%d}-{uuid.uuid4().hex[:8]}"
    if not named:
        engine._last_session_id = sid
    contract = {
        key: getattr(response, key, default)
        for key, default in (
            ("score_contract_version", "2"),
            ("calibration_status", "uncalibrated"),
            ("calibration_id", None),
            ("answer_confidence", None),
            ("abstained", not bool(relevant)),
            ("abstention_reason", None),
        )
    }
    return {
        "success": True,
        "session_id": sid,
        "agent_id": agent_id.strip() or shared._get_agent_id(),
        "context": context,
        "memories": memories,
        "memory_count": len(memories),
        "core_memory": [m["content"] for m in memories if m.get("is_core")],
        "degraded_mode": degraded,
        "retrieval_mode": "emergency_fts5_bm25" if degraded else "hybrid_candidate_fusion",
        **forward_recall_metadata(response),
        **contract,
        "learning": {
            "feedback_signals": count,
            "phase": shared._phase_for_signal_count(count),
            **shared._learning_evidence(engine, pid),
        },
        **({"upcoming_events": upcoming} if upcoming else {}),
    }


async def session_init(
    get_engine,
    project_path="",
    query="",
    max_results=10,
    max_age_days=30,
    session_id="",
    agent_id="",
    profile_id="",
):
    from superlocalmemory.core.project_identity import session_context_query
    from superlocalmemory.mcp.tools_mesh import bind_local_project

    try:
        engine = get_engine()
        named, pid, rules, refused = _start_inputs(engine, profile_id)
        if refused:
            return refused
        response, degraded = await _recall(
            query or session_context_query(project_path),
            project_path,
            named,
            pid,
            max_results,
            max_age_days,
        )
        threshold = rules.get_recall_config().get("relevance_threshold", 0.3)
        relevant = _relevant(response, threshold, max_age_days)
        # 4.2.17: Jev orders them when the person chose Jev reordering
        # (core/session_jev: order only, 1.5 s budget, never hides anything).
        relevant = await asyncio.to_thread(
            _jev_ordered, engine, query or session_context_query(project_path), relevant)
        pins, standing, upcoming = _priority_facts(engine, pid, project_path, response)
        context, rows = _render_session(
            engine, named, response, pins, standing, relevant, max_results
        )
        await asyncio.to_thread(bind_local_project, project_path)
        return _envelope(
            engine,
            pid,
            named,
            session_id,
            agent_id,
            context,
            rows,
            response,
            degraded,
            relevant,
            upcoming,
        )
    except Exception as exc:
        shared.logger.exception("session_init failed")
        return {"success": False, "error": str(exc)}


def _start_inputs(engine, profile_id):
    from superlocalmemory.hooks.rules_engine import RulesEngine
    from superlocalmemory.mcp.request_profile import requested_profile, tool_profile

    named = requested_profile(profile_id)
    pid, refused = tool_profile(engine, named)
    rules = RulesEngine(config_path=shared.state_path("config.json"))
    if not refused and not rules.should_recall("session_start"):
        refused = {
            "success": True,
            "context": "",
            "memories": [],
            "message": "Auto-recall disabled",
        }
    return named, pid, rules, refused


def _render_session(engine, named, response, pins, standing, relevant, limit):
    from superlocalmemory.core.config import InjectionConfig

    cfg = getattr(getattr(engine, "config", None), "injection", None)
    cfg = cfg if isinstance(cfg, InjectionConfig) else None
    memories = _injectable(pins, standing, relevant, limit)
    return _context(engine, named, response, memories, cfg), _memory_rows(
        memories, relevant, limit, cfg
    )
