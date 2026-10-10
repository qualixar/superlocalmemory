# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The answer ``/recall`` gives when full recall ran past the daemon budget.

Moved out of ``server/unified_daemon.py`` (4.1.20) and fixed in two ways:

1. It could not find anything a person would actually ask. It matched the WHOLE
   question as one substring (``content LIKE '%When is the Halcyon migration
   window?%'``), so a natural question never matched a stored sentence, even
   one containing every word of it. On a fresh install — the moment this path
   fired most — a memory saved seconds earlier with a "queryable" receipt came
   back as "No confident match". It now matches the question's words and ranks
   rows by how many of them they contain, with the exact phrase first.

2. It said nothing about what it skipped. ``channel_status`` was ``{}`` and
   ``incomplete_channels`` was ``[]`` — the shape of a COMPLETE recall that
   found nothing. Every channel was abandoned here, and the envelope now says
   so channel by channel.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from superlocalmemory.retrieval import channel_status as chstat

logger = logging.getLogger(__name__)

__all__ = [
    "abandoned_channel_metadata", "fallback_terms", "keyword_match_sql",
    "recall_keyword_fallback",
]

# Bounded: each term is one LIKE in the WHERE and one in the ORDER BY.
_MAX_TERMS = 8
_MIN_TERM_LEN = 3
_STOPWORDS = frozenset({
    "the", "and", "for", "are", "was", "were", "what", "when", "where",
    "which", "who", "whom", "why", "how", "did", "does", "with", "that",
    "this", "from", "have", "has", "had", "you", "your", "our", "about",
    "into", "there", "their", "can", "could", "would", "should", "any",
    "all", "tell", "know", "remember", "recall",
})


def fallback_terms(query: str) -> list[str]:
    """The exact phrase first, then the question's content words, deduped."""
    phrase = (query or "").strip()
    terms: list[str] = [phrase] if phrase else []
    seen = {phrase.lower()}
    for word in re.findall(r"\w+", phrase.lower()):
        if len(word) < _MIN_TERM_LEN or word in _STOPWORDS or word in seen:
            continue
        seen.add(word)
        terms.append(word)
        if len(terms) >= _MAX_TERMS:
            break
    return terms


def _like(term: str) -> str:
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def keyword_match_sql(
    query: str, column: str = "af.content",
) -> tuple[str, list[str], str, list[str]] | None:
    """(where, where_params, order_by, order_params) for a word-based match.

    Shared by every over-budget surface (daemon ``/recall`` and the dashboard
    ``/api/search``) so the CLI, MCP and dashboard find the same rows. A row
    matches when it contains the phrase or any content word; rows are ranked by
    how many they contain, the exact phrase counting double. ``None`` when the
    query has nothing to match on.
    """
    terms = fallback_terms(query)
    if not terms:
        return None
    likes = [_like(t) for t in terms]
    match = f"{column} LIKE ? ESCAPE '\\'"
    where = "(" + " OR ".join(match for _ in likes) + ")"
    order = "(" + " + ".join(
        ("2 * " if i == 0 else "") + f"({match})" for i in range(len(likes))
    ) + ") DESC"
    return where, list(likes), order, list(likes)


def abandoned_channel_metadata() -> dict:
    """Every channel reported as abandoned: nothing from full recall ran."""
    status = {name: chstat.TIMEOUT for name in chstat.CHANNEL_NAMES}
    return {"channel_status": status, "incomplete_channels": sorted(status)}


def _fetch_candidates(db, pid: str, query: str, pool_limit: int) -> list[dict]:
    sql = keyword_match_sql(query)
    if sql is None:
        return []
    where, where_params, order, order_params = sql
    # The "af" alias matters: current_fact_clause's temporal-validity check is
    # a correlated subquery against fact_temporal_validity, which has its own
    # fact_id/profile_id columns. Unqualified, SQLite resolves the outer
    # reference against the subquery's OWN table and excludes every row once
    # any one fact in the profile was ever superseded.
    current_clause = db.current_fact_clause("af")
    rows = db.execute(
        "SELECT af.fact_id AS fact_id, af.content AS content, "
        "af.confidence AS confidence FROM atomic_facts AS af "
        f"WHERE af.profile_id = ? AND {where} {current_clause} "
        f"ORDER BY {order}, af.confidence DESC LIMIT ?",
        (pid, *where_params, *order_params, pool_limit),
    )
    return [dict(r) for r in rows]


def _apply_facets(engine, db, pid, candidates, facets
                  ) -> tuple[list[dict], str | None, dict | None, dict | None]:
    """Narrow like full recall does (``retrieval.project_scope.narrow``). A
    facet that cannot be verified keeps NOTHING rather than silently serving
    the unfiltered pool; ``project`` falls back and reports it, exactly as in
    full recall. This path has no scores, so ``prefer_project`` reorders
    nothing here - its report says how many it would have preferred.

    4.1.22: ``tags`` is folded into ``narrow``'s own
    ``matching_fact_ids`` call (it never falls back, so needs no special
    casing there); the 4th return value is its ``tag_scope`` report, built
    from the same ``scoped.kept`` count — no extra narrowing pass."""
    try:
        from superlocalmemory.core.kind_query import engine_display_min_confidence
        from superlocalmemory.retrieval.project_scope import narrow

        scoped = narrow(
            db, [c["fact_id"] for c in candidates], pid, facets,
            display_min_confidence=engine_display_min_confidence(engine),
        )
        keep = set(scoped.kept)
        tag_scope = None
        if getattr(facets, "tags", None):
            from superlocalmemory.retrieval.tag_scope import build_report
            tag_scope = build_report(db, pid, facets, len(scoped.kept))
        return ([c for c in candidates if c["fact_id"] in keep], None, scoped.report,
                tag_scope)
    except Exception as exc:  # noqa: BLE001 - must cost results, never skip
        return [], type(exc).__name__, None, None


def recall_keyword_fallback(
    engine, query: str, limit: int, *, profile_id: str | None = None,
    profile: str | None = None, profile_generation: int | None = None,
    facets: Any = None,
) -> dict:
    """Profile-scoped keyword recall for a recall that exceeded its budget.

    Honours the same hard filters as full recall (quarantine, caller
    replacement via ``current_fact_clause``, and project / agent / about / kind
    facets), echoes the SERVED namespace, and reports every channel as
    abandoned so the answer can never be read as complete.
    """
    results: list[dict] = []
    has_facets = facets is not None and not facets.empty
    facet_filter_error: str | None = None
    project_scope: dict | None = None
    tag_scope: dict | None = None
    try:
        db = engine._db
        pid = profile_id or engine.profile_id
        from superlocalmemory.retrieval.kind_filter import overfetch_limit

        pool_limit = overfetch_limit(limit) if has_facets else limit
        candidates = _fetch_candidates(db, pid, query, pool_limit)
        from superlocalmemory.retrieval import visibility

        if not visibility.is_empty():  # a remote caller: never what it may not see
            hidden = visibility.hidden_among(db, pid, [d.get("fact_id") for d in candidates])
            candidates = [d for d in candidates if d.get("fact_id") not in hidden]
        if has_facets:
            candidates, facet_filter_error, project_scope, tag_scope = _apply_facets(
                engine, db, pid, candidates, facets)
        for pos, d in enumerate(candidates[:limit], start=1):
            results.append({
                "fact_id": d.get("fact_id"),
                "content": (d.get("content") or "")[:2400],
                "score": None, "relevance_score": None, "ranking_score": None,
                "confidence": d.get("confidence"),
                "rank_position": pos,
            })
    except Exception as exc:
        logger.warning("recall keyword fallback failed (non-fatal): %s", exc)
        if has_facets and facet_filter_error is None:
            facet_filter_error = type(exc).__name__
    from superlocalmemory.server.recall_serializer import recall_response_metadata
    from superlocalmemory.storage.models import RecallResponse

    # Nothing from full recall reached this answer: every channel was
    # abandoned when the budget ran out. Saying "{}" here made a degraded
    # answer look like a complete one that found nothing.
    abandoned = abandoned_channel_metadata()
    contract = recall_response_metadata(RecallResponse(
        query=query,
        incomplete_channels=tuple(abandoned["incomplete_channels"]),
        channel_status=abandoned["channel_status"],
        project_scope=project_scope,
        tag_scope=tag_scope,
    ))
    return {
        **contract,
        "ok": True,
        "query": query,
        "query_type": "text_search",
        "retrieval_mode": "degraded_lexical",
        "degraded_reason": "recall_budget_exceeded",
        "facet_filter_error": facet_filter_error,
        "profile": profile if profile else engine.profile_id,
        "profile_generation": profile_generation,
        "result_count": len(results),
        "results": results,
        "count": len(results),
        # Never judged, so never "confident" — the results are still shown.
        "no_confident_match": True,
        "answerability": "unjudged",
        "answerability_reason": "budget_exhausted" if results else "no_results",
        "retrieval_time_ms": 0,
    }
