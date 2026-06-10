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


# ---------------------------------------------------------------------------
# THE shared chokepoint: RecallResponse -> transport dicts (all surfaces)
# ---------------------------------------------------------------------------

def serialize_recall_response(
    response: Any,
    *,
    limit: int = 10,
    memory_map: dict[str, str] | None = None,
    per_fact_max: int = 2400,
    total_max: int = 12000,
    full: bool = False,
    include_source: bool = False,
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

    Returns:
        (results, no_confident_match) — results is a list of dicts; the bool
        is the evidence-floor signal lifted from the response (additive).
    """
    memory_map = memory_map or {}
    raw: list[dict] = []
    for r in (response.results or [])[:limit]:
        fact = r.fact
        fact_type = getattr(fact, "fact_type", None)
        lifecycle = getattr(fact, "lifecycle", None)
        raw.append({
            "fact_id": fact.fact_id,
            "memory_id": fact.memory_id,
            "content": fact.content or "",
            "source_content": memory_map.get(fact.memory_id, "") or "",
            "score": round(r.score, 4),
            "confidence": round(getattr(r, "confidence", 0.0), 4),
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
            "created_at": getattr(fact, "created_at", "") or "",
            "evidence_chain": list(getattr(r, "evidence_chain", []) or []),
        })

    # F-3 source discipline, then F-2 budget — order matters (discipline first
    # so the template firewall runs before any preview slicing).
    disciplined = [apply_source_content_discipline(d, include_source=include_source) for d in raw]
    budgeted = apply_recall_budget(
        disciplined, per_fact_max=per_fact_max, total_max=total_max, full=full,
    )
    no_confident_match = bool(getattr(response, "no_confident_match", False))
    return budgeted, no_confident_match
