# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 — Recall Serializer (v3.6.6)

"""Recall output budget and source_content discipline helpers (v3.6.6).

F-2: Per-fact content clamp + total budget stubs.
F-3: source_content preview + template firewall.

THE single shared serialization chokepoint. Every surface that turns a
RecallResponse into transport dicts goes through ``serialize_recall_response``
so MCP, CLI, the daemon HTTP route, the in-process queue adapter, and the
WorkerPool fallback all return byte-for-byte identical output (parity across
surfaces AND modes A/B). The evidence floor lives upstream in
RetrievalEngine.recall (also shared); this layer owns presentation only.

Pure functions — no side effects, no DB access. Stdlib-only at import
(hooks import chain must stay light).
"""

from __future__ import annotations

import re
from typing import Any

from superlocalmemory.core.config import CANONICAL_RECALL_LIMIT
from superlocalmemory.retrieval.temporal_frame import relative_age, temporal_frame


# ---------------------------------------------------------------------------
# F-2: Per-fact content clamp
# ---------------------------------------------------------------------------

def clamp_fact_content(
    content: str,
    max_chars: int = 2400,
) -> tuple[str, bool]:
    """Clamp a single fact's content to max_chars.

    Strategy: head 70% + "\\n…[truncated N chars]…\\n" + tail 30%.
    The tail is kept because session-close facts put OPEN ITEMS at the end.

    Returns:
        (clamped_content, was_truncated)
    """
    if not content or len(content) <= max_chars:
        return content, False

    head_len = int(max_chars * 0.70)
    tail_len = max_chars - head_len
    dropped = len(content) - max_chars
    marker = f"\n…[truncated {dropped} chars]…\n"

    result = content[:head_len] + marker + content[-tail_len:]
    return result, True


def apply_recall_budget(
    results: list[dict],
    per_fact_max: int = 2400,
    total_max: int = 12000,
    full: bool = False,
) -> list[dict]:
    """Apply per-fact clamp and total budget to a list of result dicts.

    Args:
        results: List of result dicts (must have at minimum 'fact_id',
                 'score', 'content' keys).
        per_fact_max: Maximum chars for a single fact's content.
        total_max: Maximum total content chars before remaining results
                   become stubs.
        full: If True, bypasses all clamping (escape hatch for tools/CLI
              that need full content — additive backward-compat param).

    Returns:
        New list of result dicts with potentially clamped/stubbed content.
        Mutates nothing — returns new dicts.
    """
    if not results:
        return []

    if full:
        # full=True: return everything as-is, no clamping, no stubs
        return [dict(r) for r in results]

    out: list[dict] = []
    cumulative_chars = 0

    for r in results:
        content = r.get("content", "") or ""

        # Check if we're already over total budget
        if cumulative_chars >= total_max:
            # Emit stub: fact_id, score, first 120 chars + "…"
            stub_content = content[:120] + ("…" if len(content) > 120 else "")
            stub = {k: v for k, v in r.items() if k not in ("content",)}
            stub["content"] = stub_content
            stub["stub"] = True
            out.append(stub)
            continue

        # Per-fact clamp
        clamped, was_truncated = clamp_fact_content(content, max_chars=per_fact_max)
        new_r = dict(r)
        new_r["content"] = clamped
        if was_truncated:
            new_r["truncated"] = True

        cumulative_chars += len(clamped)
        out.append(new_r)

    return out


# ---------------------------------------------------------------------------
# F-3: source_content discipline
# ---------------------------------------------------------------------------

def apply_source_content_discipline(
    result: dict,
    include_source: bool = False,
) -> dict:
    """Apply source_content discipline to a single result dict.

    Default behavior:
      - Trim source_content to ≤ 280 chars
      - Drop entirely if it matches prompt-template patterns

    include_source=True:
      - Returns full source_content (unless it's a template, always dropped)

    Returns a new dict — never mutates input.
    """
    from superlocalmemory.core.injection import is_prompt_template

    if "source_content" not in result:
        return dict(result)

    src = result.get("source_content") or ""

    # Template firewall: drop regardless of include_source
    if src and is_prompt_template(src):
        new_r = dict(result)
        new_r["source_content"] = ""
        return new_r

    # Empty source: return unchanged
    if not src:
        return dict(result)

    if include_source:
        return dict(result)

    # Default: preview ≤ 280 chars
    new_r = dict(result)
    new_r["source_content"] = src[:280]
    return new_r


def media_object(source: dict) -> dict | None:
    """The ``media`` block for a result whose memory came from a picture or a page."""
    kind = {"media": "image", "document": "page"}.get(source.get("type"))
    if kind is None:
        return None
    media_id = source.get("media_id")
    return {
        "media_id": media_id,
        "kind": kind,
        "thumbnail_uri": f"slm://media/{media_id}/thumb" if media_id else None,
        "page": source.get("page"),
        "document_id": source.get("document_id"),
        "citation": source.get("citation") or "",
    }


# ---------------------------------------------------------------------------
# THE shared chokepoint: RecallResponse -> transport dicts (all surfaces)
# ---------------------------------------------------------------------------

from superlocalmemory.storage.memory_kinds import kind_fields  # noqa: E402


def serialize_recall_response(
    response: Any,
    *,
    limit: int = CANONICAL_RECALL_LIMIT,
    memory_map: dict[str, str] | None = None,
    per_fact_max: int = 2400,
    total_max: int = 12000,
    full: bool = False,
    include_source: bool = False,
    include_marker: bool = False,
    display_min_confidence: float | None = None,
    source_map: dict[str, dict] | None = None,
) -> tuple[list[dict], bool]:
    """Convert a RecallResponse into budgeted, source-disciplined dicts.

    This is the ONE function every recall surface calls (daemon HTTP route,
    in-process queue adapter, CLI direct-fallback, WorkerPool). Guarantees
    identical output regardless of surface or mode.

    Args:
        response:       A RecallResponse (engine result objects in .results).
        limit:          Max results to serialize.
        memory_map:     fact.memory_id -> source memory content (optional).
        per_fact_max:   Per-fact content char cap (config-driven).
        total_max:      Total content char budget before stubs (config-driven).
        full:           Bypass clamping/stubs (additive escape hatch).
        include_source: Return full source_content (else ≤280-char preview).
        include_marker: Emit each result's HMAC usage marker. See below.
        source_map:     memory_id -> ``_slm_source`` for the pictures and pages
            among the results (``retrieval.media_channel.memory_sources``).
            Those results get a ``media`` block; ``None`` changes nothing.
        display_min_confidence: The confidence below which a model-suggested
            kind is shown as its legacy/untyped fallback instead of a
            suggestion (``storage.memory_kinds.kind_fields``). ``None`` (the
            default) keeps that function's own 0.20 default — a caller with a
            live ``SLMConfig`` should pass
            ``core.kind_query.engine_display_min_confidence(engine)`` so a
            recall result agrees with what ``slm list`` and MCP show for the
            SAME fact (4.1.21 #16: this used to be silently hard-coded here
            regardless of what was configured).

    Returns:
        (results, no_confident_match) — results is a list of dicts; the bool
        is the evidence-floor signal lifted from the response (additive).

    THE MARKER, AND WHY IT WAS MISSING
    ----------------------------------
    ``run_recall`` sets ``result.marker`` on every result — an HMAC of the
    fact id, computed on the hot path already. Until 4.0.8 **no serialiser
    ever read it**, so the value was computed and discarded on every recall.

    That one omission broke the entire closed learning loop. The
    ``post_tool_outcome`` hook settles an outcome by finding a validated
    ``slm:fact:<id>:<hmac8>`` marker in a later tool response; with markers
    never reaching the agent it found nothing, every outcome settled at the
    formula's 0.5 base, and the consequences were visible all the way out to
    the dashboard: 162 outcomes at the default label, all 294 source-quality
    observations at exactly 0.5, therefore ``alpha == beta`` for all 18
    sources and "no quality signal has settled", and 165 bandit arms with 4
    plays between them.

    Off by default, and gated by the caller on ``session_id``. A marker costs
    roughly 33 characters of the agent's context per result, and it can only
    buy a signal when a ``pending_outcomes`` row exists to settle — which
    happens only for session-bearing recalls. Spending context on an ad-hoc
    recall that could never learn from it is pure waste.
    """
    memory_map = memory_map or {}
    # T-inject: one shared "now" so every result's age label is consistent.
    from datetime import datetime as _dt, timezone as _tz
    _now = _dt.now(_tz.utc)
    # None keeps kind_fields' own default; a caller that passed a value means it.
    _kind_kwargs = ({} if display_min_confidence is None
                    else {"display_min_confidence": display_min_confidence})
    raw: list[dict] = []
    for r in (response.results or [])[:limit]:
        fact = r.fact
        _created = getattr(fact, "created_at", "") or ""
        fact_type = getattr(fact, "fact_type", None)
        lifecycle = getattr(fact, "lifecycle", None)
        entry = {
            "fact_id": fact.fact_id,
            "memory_id": fact.memory_id,
            "content": fact.content or "",
            "source_content": memory_map.get(fact.memory_id, "") or "",
            "score": round(r.score, 4),
            "relevance_score": round(
                getattr(r, "relevance_score", r.score) or 0.0, 4
            ),
            "confidence": round(getattr(r, "confidence", 0.0), 4),
            "memory_confidence": round(
                getattr(r, "memory_confidence", r.confidence) or 0.0, 4
            ),
            "ranking_score": (
                round(r.ranking_score, 6)
                if getattr(r, "ranking_score", None) is not None
                else None
            ),
            "rank_position": int(getattr(r, "rank_position", 0) or 0),
            "trust_score": round(getattr(r, "trust_score", 0.0), 4),
            "channel_scores": {
                k: round(v, 4) for k, v in (getattr(r, "channel_scores", None) or {}).items()
            },
            "fact_type": fact_type.value
                if fact_type is not None and hasattr(fact_type, "value")
                else (getattr(fact, "fact_type", "") or ""),
            "lifecycle": lifecycle.value
                if lifecycle is not None and hasattr(lifecycle, "value")
                else (lifecycle or ""),
            "access_count": getattr(fact, "access_count", 0),
            "created_at": _created,
            # T-inject: human-relative age so consumers (and the LLM) can
            # weigh recency without doing date math. "" when undated.
            "age_label": relative_age(_created, _now),
            "evidence_chain": list(getattr(r, "evidence_chain", []) or []),
            # The memory's kind, the same five fields on every surface.
            **kind_fields(fact, **_kind_kwargs),
        }
        # Only when asked, and only when the engine actually produced one —
        # an empty key would be indistinguishable from a marker that failed
        # to compute, and the hook validates before trusting anything anyway.
        block = media_object(source_map[fact.memory_id]) if source_map and fact.memory_id in source_map else None
        if block is not None:
            entry["media"] = block
        if include_marker:
            marker = getattr(r, "marker", "") or ""
            if marker:
                entry["marker"] = marker
        raw.append(entry)

    # F-3 source discipline, then F-2 budget — order matters (discipline first
    # so the template firewall runs before any preview slicing).
    disciplined = [apply_source_content_discipline(d, include_source=include_source) for d in raw]
    budgeted = apply_recall_budget(
        disciplined, per_fact_max=per_fact_max, total_max=total_max, full=full,
    )
    no_confident_match = bool(getattr(response, "no_confident_match", False))
    return budgeted, no_confident_match


def recall_response_metadata(response: Any) -> dict:
    """Return Score Contract v2 response metadata for transport envelopes."""
    # T-inject: a one-line temporal frame anchoring the result set to "now"
    # and its age span, so time-blind LLMs get an explicit recency signal.
    _timestamps = [
        getattr(getattr(r, "fact", None), "created_at", "") or ""
        for r in (getattr(response, "results", None) or [])
    ]
    from superlocalmemory.retrieval.answer_check_status import (
        ANSWER_CHECK_DETAILS,
        answer_check_note,
    )
    _check_status = getattr(response, "answer_check_status", "skipped") or "skipped"
    _check_detail = getattr(response, "answer_check_detail", "") or ""
    if _check_detail not in ANSWER_CHECK_DETAILS:
        _check_detail = ""
    from superlocalmemory.retrieval.answerability import of_response as _answerable

    _answerability = _answerable(response)
    return {
        "score_contract_version": getattr(response, "score_contract_version", "2"),
        "calibration_status": getattr(response, "calibration_status", "uncalibrated"),
        "calibration_id": getattr(response, "calibration_id", None),
        # The name of this answer. A caller that reports back how the answer
        # went can quote it, and the report then joins to this exact recall
        # instead of being matched by overlapping memory ids.
        "query_id": getattr(response, "query_id", "") or "",
        "answer_confidence": getattr(response, "answer_confidence", None),
        "abstained": bool(getattr(response, "abstained", False)),
        "abstention_reason": getattr(response, "abstention_reason", None),
        "temporal_frame": temporal_frame(_timestamps),
        # Q2b: thematic community summary (pure pass-through; computed upstream
        # in the engine where DB access is available). None on most recalls.
        "thematic_context": getattr(response, "community_context", None),
        # Channels abandoned at the hang guard, so their candidates are absent
        # from this answer. Empty on a healthy recall, which is the normal case.
        # Non-empty is the one situation in which asking the same question twice
        # may legitimately give different answers, so it has to travel with the
        # response rather than living only in a server log — otherwise a caller
        # comparing two runs has no way to tell an incomplete answer from a
        # changed one. A list, because JSON has no tuple.
        "incomplete_channels": list(
            getattr(response, "incomplete_channels", ()) or ()
        ),
        # What became of every channel. Travels with the answer for the same
        # reason as the field above: a caller comparing two runs, or an
        # operator looking at a thin result set, otherwise cannot tell a store
        # with nothing to say from a retrieval path that is partly down.
        "channel_status": dict(getattr(response, "channel_status", {}) or {}),
        # Who chose the final order. "jev_listwise" when the online answer
        # check reordered it, which is not perfectly repeatable — so, like the
        # two fields above, it travels with the answer. ``local_reranker_status``
        # keeps what the local reranking step did; empty when nothing replaced it.
        "reranker_status": getattr(response, "reranker_status", "not_configured")
        or "not_configured",
        "local_reranker_status": getattr(response, "local_reranker_status", "") or "",
        # What became of the answer check on this recall: judged / off /
        # skipped / busy / warming / unavailable. A verdict that is absent
        # because the check was busy or still loading is not "nothing
        # answers", so — like the fields above — it travels with the answer.
        "answer_check_status": _check_status,
        # 4.1.20: a recall that was not checked must never read like one that
        # was (``abstained`` is False on both). ``answer_check_ran`` says which;
        # ``answer_check_reason`` is the closed-set why (or "reused" when the
        # verdict repeats an identical earlier check); ``answer_check_note`` is
        # the sentence to show a person, "" when there is nothing to explain.
        "answer_check_ran": _check_status == "judged",
        "answer_check_reason": _check_detail,
        "answer_check_note": answer_check_note(_check_status, _check_detail),
        # 4.1.22: supported / unsupported / unjudged, and why (retrieval/
        # answerability). ``abstained: false`` never means "checked"; this does.
        "answerability": _answerability[0],
        "answerability_reason": _answerability[1],
        # 4.1.21 (#150): what the recall's project did. ``filter.applied``
        # False means ``project`` matched nothing found for the question and
        # the results are NOT narrowed to it; ``note`` says so in words. None
        # when the recall named no project.
        "project_scope": getattr(response, "project_scope", None),
        # 4.1.22: what the recall's ``tags`` filter did. ``applied`` is
        # always True for tags (a hard filter that never falls back);
        # ``reason``/``note`` are only present when ``matched`` is 0. None
        # when the recall named no tags.
        "tag_scope": getattr(response, "tag_scope", None),
    }
