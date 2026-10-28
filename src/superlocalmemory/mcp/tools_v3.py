# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — V3-Only MCP Tools (5 tools).

set_mode, get_mode, health, consistency_check, recall_trace.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
from typing import Any, Callable


from superlocalmemory.mcp.tool_hints import READ_ONLY, SETS
from superlocalmemory.core.admission import admits
from superlocalmemory.core.operation_request import OperationKind
from superlocalmemory.mcp.shared import authorize_mcp_mutation

logger = logging.getLogger(__name__)


def register_v3_tools(server, get_engine: Callable) -> None:
    """Register 5 V3-exclusive tools on *server*."""

    # ------------------------------------------------------------------
    # 0. get_version (so IDEs can check compatibility)
    # ------------------------------------------------------------------
    @server.tool(title="Get Version", annotations=READ_ONLY)
    async def get_version() -> dict:
        """Get SuperLocalMemory version, Python version, and platform info."""
        try:
            from importlib.metadata import version as _pkg_version
            slm_ver = _pkg_version("superlocalmemory")
        except Exception:
            slm_ver = "unknown"
        import platform
        import sys as _sys
        return {
            "success": True,
            "version": slm_ver,
            "python": _sys.version.split()[0],
            "platform": platform.system(),
            "arch": platform.machine(),
        }

    # ------------------------------------------------------------------
    # 1. set_mode
    # ------------------------------------------------------------------
    @server.tool(title="Set Operating Mode", annotations=SETS)
    @admits(OperationKind.MODE_CHANGE)
    async def set_mode(mode: str) -> dict:
        """Switch operating mode (a, b, or c).

        Modes are named by what you get, not by a vendor:
        Mode A (Local Guardian): Nothing leaves this device. No AI language
                model runs. Fastest and most private.
        Mode B (Smart Local): All data stays on this device. Uses a model
                running on this machine to improve recall quality — Ollama
                by default, but any local OpenAI-compatible server works.
        Mode C (Full Power): Uses your own endpoint, or a cloud AI provider
                (OpenAI, Anthropic, …), for best recall quality. Queries
                leave this device; a key is required for a cloud provider,
                but not for a keyless custom endpoint.
        In any mode, the optional online answer check (off unless the user
        turns it on and agrees) sends each recall's question and top
        memories to the chosen provider; the returned description says so
        while it is on.

        Resets the engine to apply the new mode configuration.

        Args:
            mode: Target mode - 'a', 'b', or 'c'.
        """
        try:
            mode_lower = mode.strip().lower()
            if mode_lower not in ("a", "b", "c"):
                return {
                    "success": False,
                    "error": f"Invalid mode '{mode}'. Use 'a', 'b', or 'c'.",
                }
            engine = get_engine()
            authorization = authorize_mcp_mutation(
                engine,
                "update",
                mutation_source="mcp-set-mode",
                profile_id=engine.profile_id,
                content_preview=mode_lower,
            )
            from superlocalmemory.core.config import SLMConfig
            from superlocalmemory.core.egress_notice import online_check_on
            from superlocalmemory.mcp.server import reset_engine

            # Use switch_mode() — the correct load-then-patch path that preserves
            # all user-tuned config blocks (forgetting, injection, retrieval, scope,
            # math, channel_weights, …). for_mode() resets them to hardcoded defaults.
            old_config = SLMConfig.load()
            config = SLMConfig.switch_mode(mode_lower)

            # A mode file naming another model: every engine stays on the live one
            # and the daemon re-indexes in the background (core/embedding_live.py).
            from superlocalmemory.core.embedding_reindex import (
                pending_switch_message,
                space_changed,
            )
            needs_reindex = space_changed(old_config.embedding, config.embedding)

            reset_engine()
            authorization.complete()

            return {
                "success": True,
                "mode": mode_lower,
                "description": _mode_description(
                    mode_lower, config.retrieval, _config_locality(config)),
                "online_answer_check": online_check_on(config.retrieval),
                "needs_reindex": needs_reindex,
                "message": (pending_switch_message(old_config.embedding, config.embedding)
                            if needs_reindex else ""),
            }
        except Exception as exc:
            logger.exception("set_mode failed")
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # 2. get_mode
    # ------------------------------------------------------------------
    @server.tool(title="Get Operating Mode", annotations=READ_ONLY)
    async def get_mode() -> dict:
        """Get current operating mode and its capabilities.

        Returns mode identifier, description, and feature flags
        (LLM availability, cross-encoder, agentic retrieval).
        """
        try:
            from superlocalmemory.core.egress_notice import online_check_on

            engine = get_engine()
            m = engine._config.mode.value
            retrieval = _current_retrieval(engine)
            caps = {
                "llm_available": engine._llm is not None,
                "cross_encoder": engine._config.retrieval.reranker_enabled(),
                "agentic_rounds": engine._config.retrieval.agentic_max_rounds,
                "sheaf_at_encoding": engine._config.math.sheaf_at_encoding,
            }
            return {
                "success": True,
                "mode": m,
                "description": _mode_description(
                    m, retrieval, _config_locality(engine._config)),
                "online_answer_check": online_check_on(retrieval),
                "capabilities": caps,
            }
        except Exception as exc:
            logger.exception("get_mode failed")
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # 3. health
    # ------------------------------------------------------------------
    @server.tool(title="Check System Health", annotations=READ_ONLY)
    async def health(profile_id: str = "") -> dict:
        """Get system health including math layer status.

        Reports on Fisher-Rao, Sheaf consistency, and Langevin dynamics
        health. Also includes database integrity and component status.
        ``profile_id`` counts another profile's memories (empty = the active
        one).
        """
        try:
            from superlocalmemory.mcp.request_profile import tool_profile

            engine = get_engine()
            pid, refused = tool_profile(engine, profile_id)
            if refused:
                return refused

            status: dict = {
                "success": True,
                "mode": engine._config.mode.value,
                "profile": pid,
                "components": {},
            }

            # Database health
            fact_count = engine._db.get_fact_count(pid)
            status["components"]["database"] = {
                "status": "ok",
                "fact_count": fact_count,
            }

            # Embedding service — distinguish real embedder from V3.5.9 daemon proxy
            _emb = engine._embedder
            from superlocalmemory.core.mcp_embedder_proxy import McpEmbedderProxy
            _is_proxy = isinstance(_emb, McpEmbedderProxy)
            status["components"]["embedder"] = {
                "status": "ok" if _emb else "unavailable",
                "source": "daemon_proxy" if _is_proxy else "local",
                "model": engine._config.embedding.model_name,
            }

            # Recall's answer check for this text model: the same fact and advice
            # the dashboard card and slm doctor show (core/support_fit_guard).
            from superlocalmemory.core.support_fit_guard import answer_check_status
            status["components"]["answer_check"] = answer_check_status(engine._config.embedding)

            # LLM
            status["components"]["llm"] = {
                "status": "ok" if engine._llm else "disabled",
                "provider": engine._config.llm.provider or "none",
            }

            # Fisher-Rao (math layer 1)
            fisher_facts = 0
            if fact_count > 0:
                rows = engine._db.execute(
                    "SELECT COUNT(*) AS c FROM atomic_facts "
                    "WHERE profile_id = ? AND fisher_mean IS NOT NULL",
                    (pid,),
                )
                fisher_facts = int(dict(rows[0])["c"]) if rows else 0
            status["components"]["fisher_rao"] = {
                "status": "ok" if fisher_facts > 0 else "no_data",
                "indexed_facts": fisher_facts,
                "temperature": engine._config.math.fisher_temperature,
            }

            # Sheaf consistency (math layer 2)
            status["components"]["sheaf"] = {
                "status": "active" if engine._sheaf_checker else "disabled",
                "threshold": engine._config.math.sheaf_contradiction_threshold,
            }

            # Langevin dynamics (math layer 3)
            langevin_facts = 0
            if fact_count > 0:
                rows = engine._db.execute(
                    "SELECT COUNT(*) AS c FROM atomic_facts "
                    "WHERE profile_id = ? AND langevin_position IS NOT NULL",
                    (pid,),
                )
                langevin_facts = int(dict(rows[0])["c"]) if rows else 0
            status["components"]["langevin"] = {
                "status": "ok" if langevin_facts > 0 else "no_data",
                "positioned_facts": langevin_facts,
            }

            return status
        except Exception as exc:
            logger.exception("health failed")
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # 4. consistency_check
    # ------------------------------------------------------------------
    @server.tool(title="Check Memory Consistency", annotations=READ_ONLY)
    async def consistency_check(limit: int = 100) -> dict:
        """Run sheaf consistency check on stored memories.

        Detects contradictions between facts using algebraic topology
        (sheaf cohomology). Returns pairs of contradicting facts with
        severity scores.

        Args:
            limit: Maximum facts to check (default 100).
        """
        try:
            engine = get_engine()
            pid = engine.profile_id

            if not engine._sheaf_checker:
                return {
                    "success": True,
                    "contradictions": [],
                    "note": "Sheaf checker is disabled in current configuration.",
                }

            facts = engine._db.get_all_facts(pid)[:limit]
            all_contradictions: list[dict] = []
            errors_count = 0
            for fact in facts:
                if not fact.embedding or not fact.canonical_entities:
                    continue
                try:
                    contradictions = engine._sheaf_checker.check_consistency(
                        fact, pid,
                    )
                    for c in contradictions:
                        all_contradictions.append({
                            "fact_a": fact.fact_id,
                            "fact_b": c.fact_id_b,
                            "severity": round(c.severity, 3),
                            "content_a": fact.content[:80],
                        })
                except Exception:
                    errors_count += 1
                    continue

            return {
                "success": True,
                "facts_checked": len(facts),
                "facts_errored": errors_count,
                "contradictions": all_contradictions[:50],
                "total_contradictions": len(all_contradictions),
            }
        except Exception as exc:
            logger.exception("consistency_check failed")
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # 5. recall_trace
    # ------------------------------------------------------------------
    @server.tool(title="Trace Recall Scores", annotations=READ_ONLY)
    async def recall_trace(
        query: str,
        limit: int = 10,
        as_of: str | None = None,
        known_as_of: str | None = None,
        valid_at: str | None = None,
        include_unknown: bool = False,
        project: str = "",
        prefer_project: str = "",
        profile_id: str = "",
        tags: "str | list[str]" = "",
        tags_match: str = "all",
        project_strict: bool = False,
    ) -> dict:
        """Recall with per-channel score breakdown.

        Like recall, but returns detailed channel-by-channel scores
        for debugging retrieval quality.

        Args:
            query: Natural-language search query.
            limit: Maximum results (default 10).
            as_of: Optional ISO 8601 UTC datetime for point-in-time recall
                (e.g. "2024-01-01T00:00:00Z"). Omit for current-state recall.
            project / prefer_project: exactly as ``recall`` takes them, so the
                order explained here is the order recall returns for the same
                arguments. A preferred memory's evidence names
                ``same_project``; ``project_scope`` says what was applied.
            profile_id: recall another profile (empty = the active one), as
                ``recall`` takes it; the active profile is not moved.
            tags / tags_match: exactly as ``recall`` takes them (4.1.22);
                ``tag_scope`` says what the filter did.
        """
        try:
            import asyncio

            from superlocalmemory.mcp._daemon_proxy import choose_pool
            from superlocalmemory.mcp._recall_metadata import forward_recall_metadata
            from superlocalmemory.retrieval.temporal_utils import (
                normalize_as_of,
                normalize_strict_boundary,
            )

            # Normalize at MCP boundary before forwarding.
            _as_of: str | None = None
            if as_of:
                _as_of = normalize_as_of(as_of)
                if _as_of is None:
                    return {"success": False, "error": "invalid_as_of"}
            try:
                _known_as_of = normalize_strict_boundary(known_as_of, "known_as_of")
                _valid_at = normalize_strict_boundary(valid_at, "valid_at")
            except ValueError as exc:
                return {"success": False, "error": str(exc)}

            # choose_pool().recall uses blocking urllib; run off the event loop
            # so recall_trace doesn't stall the MCP server for other tools.
            raw = await asyncio.to_thread(
                lambda: choose_pool().recall(
                    query=query, limit=limit, as_of=_as_of,
                    known_as_of=_known_as_of, valid_at=_valid_at,
                    include_unknown=include_unknown,
                    # #150: sent only when set, as recall does, so a trace of a
                    # project recall explains that recall and not another one.
                    **{k: v.strip() for k, v in (("project", project),
                                                 ("prefer_project", prefer_project),
                                                 ("profile_id", profile_id))
                       if (v or "").strip()},
                    # 4.1.22: sent only when set, same as the facets above.
                    **({"tags": tags if isinstance(tags, list) else tags.strip()}
                       if (tags if isinstance(tags, list) else (tags or "").strip())
                       else {}),
                    **({"tags_match": tags_match.strip()}
                       if (tags_match or "").strip().lower() == "any" else {}),
                    **({"project_strict": True}
                       if project_strict and (project or "").strip() else {}),
                    evidence=True,
                )
            )
            if ((profile_id or "").strip() and isinstance(raw, dict)
                    and raw.get("ok") is False and raw.get("code")):
                # For a named profile a refusal (no such profile) is an
                # answer, never an empty recall.
                return {"success": False, "code": raw["code"],
                        "retryable": bool(raw.get("retryable", False)),
                        "error": raw.get("error", "")}
            items = raw.get("results", []) if isinstance(raw, dict) else []
            results = []
            for item in items[:limit]:
                results.append({
                    "fact_id": item.get("fact_id", ""),
                    "content": item.get("content", ""),
                    "score": round(float(item.get("score") or 0.0), 4),
                    "relevance_score": round(
                        float(item.get("relevance_score", item.get("score")) or 0.0), 4
                    ),
                    "match_confidence": item.get("match_confidence"),
                    "confidence_basis": item.get("confidence_basis", "unmeasured"),
                    "ranking_score": item.get("ranking_score"),
                    "confidence": round(float(item.get("confidence", 0.0)), 3),
                    "memory_confidence": round(
                        float(item.get("memory_confidence", item.get("confidence", 0.0))), 3
                    ),
                    "rank_position": int(item.get("rank_position", 0)),
                    "trust_score": round(float(item.get("trust_score", 0.0)), 3),
                    "channel_scores": item.get("channel_scores", {}) or {},
                    "evidence_chain": item.get("evidence_chain", []) or [],
                    "fact_type": item.get("fact_type", ""),
                    "lifecycle": item.get("lifecycle", ""),
                    "access_count": int(item.get("access_count", 0)),
                    # 4.1.25: what each signal says about this result
                    # (retrieval/evidence_features.py); {} from an older daemon.
                    "evidence": item.get("evidence", {}) or {},
                })
            return {
                "success": True,
                "results": results,
                "count": len(results),
                "query_type": raw.get("query_type", "") if isinstance(raw, dict) else "",
                "channel_weights": raw.get("channel_weights", {}) if isinstance(raw, dict) else {},
                "total_candidates": raw.get("total_candidates", 0) if isinstance(raw, dict) else 0,
                "retrieval_time_ms": round(
                    float(raw.get("retrieval_time_ms", 0.0)) if isinstance(raw, dict) else 0.0, 1),
                # M-10: the HTTP envelope's metadata in full (and any field
                # added to it later), not a hand-picked subset.
                **forward_recall_metadata(raw if isinstance(raw, dict) else {}),
                "query_evidence": (raw.get("query_evidence", {}) or {})
                if isinstance(raw, dict) else {},
            }
        except Exception as exc:
            logger.exception("recall_trace failed")
            return {"success": False, "error": str(exc)}


# -- Helpers ------------------------------------------------------------------

def _current_retrieval(engine: Any) -> Any:
    """The saved answer-check settings (what the dashboard last wrote), else
    the engine's own copy."""
    try:
        from superlocalmemory.core.config import SLMConfig

        return SLMConfig.load().retrieval
    except Exception:  # noqa: BLE001 — a status call never fails on this
        return getattr(getattr(engine, "_config", None), "retrieval", None)


def _mode_description(mode: str, retrieval: Any = None, locality: Any = None) -> str:
    """Human-readable capability description for a mode (never a legal claim).

    While the online answer check is on, "nothing leaves this device" is not
    true in any mode, so the description says what does leave instead. The same
    goes for a local mode whose embedding endpoint or model is not on this
    computer (#303): ``locality`` is the effective one from
    ``core.data_locality`` and replaces the claim with what it found.
    """
    from superlocalmemory.core.egress_notice import online_check_notice

    notice = online_check_notice(retrieval) if retrieval is not None else ""
    text = _base_mode_description(mode)
    if locality is not None and not locality.stays_local and locality.remote_parts:
        from superlocalmemory.core.data_locality import remote_parts_sentence

        text = text.replace("nothing leaves this device", "not everything stays on this device")
        text = f"{text} {remote_parts_sentence(locality)}"
    if notice and mode in ("a", "b"):
        text = text.replace("nothing leaves this device", "nothing else leaves this device")
        return f"{text} {notice}"
    return f"{text} {notice}" if notice else text


def _config_locality(config: Any) -> Any:
    """The effective data locality of a loaded config (``core.data_locality``)."""
    from superlocalmemory.core.data_locality import effective_data_locality

    return effective_data_locality(
        config.mode, getattr(config, "embedding", None), getattr(config, "llm", None),
    )


def _base_mode_description(mode: str) -> str:
    """Capability blurb for a mode, read from the one place every surface
    shares (#112) — CLI help, the setup wizard, and this MCP tool must never
    describe a mode differently from one another.
    """
    from superlocalmemory.core.modes import mode_short_name, mode_tagline
    from superlocalmemory.storage.models import Mode as _Mode

    try:
        m = _Mode(mode)
    except ValueError:
        return "Unknown mode"
    return f"{mode_short_name(m)} — {mode_tagline(m)}"
