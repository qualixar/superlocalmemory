# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Pictures that say nothing in words are ranked by what they show, not by text.

The text reranker and the answer check read words. A picture saved with no
words and no readable text has only a label for a memory, so asking a text model
about it would score the label. Such a candidate keeps the place fusion gave it,
and the answer check reports it as not judged.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

_EPSILON = 1e-6


def _labels() -> tuple[str, ...]:
    from superlocalmemory.media.labels import LABELS

    return LABELS


def strip_labels(text: str, labels: Iterable[str] | None = None) -> str:
    """``text`` without the label lines saving a picture added to it."""
    if not text or "[" not in text:
        return text or ""
    drop = {s.strip() for s in (labels if labels is not None else _labels())}
    return "\n".join(ln for ln in text.split("\n") if ln.strip() not in drop).strip()


def is_label_only(text: str, labels: Iterable[str] | None = None) -> bool:
    """True for a memory whose only words are saving labels."""
    return bool(text and text.strip()) and not strip_labels(text, labels).strip()


def is_wordless_picture(text: str) -> bool:
    """A picture memory whose picture had no readable text (it carries the no-text label)."""
    from superlocalmemory.media.labels import NO_TEXT

    return bool(text) and any(ln.strip() == NO_TEXT for ln in text.split("\n"))


def wordless_picture_evidence(scores: dict[str, float], min_media: float) -> bool:
    """What counts for a wordless picture: the picture itself, or the words of the person's note.

    Its other text channels matched the saving label or a file name, not what the picture shows.
    """
    return scores.get("media", 0.0) >= min_media or scores.get("bm25", 0.0) > 0.0


def all_unreadable(results: Sequence[Any]) -> bool:
    """Every result is a picture with no words of its own."""
    return bool(results) and all(
        is_label_only(getattr(getattr(r, "fact", None), "content", "") or "") for r in results)


def neutral_ids(fused: Sequence[Any], fact_map: dict[str, Any]) -> frozenset[str]:
    """Candidates found only by the picture channel and holding no words of their own."""
    if not any("media" in (fr.channel_scores or {}) for fr in fused):
        return frozenset()
    out = set()
    for fr in fused:
        cs = fr.channel_scores or {}
        fact = fact_map.get(fr.fact_id)
        if (cs.get("media", 0.0) > 0.0 and fact is not None
                and not any(v > 0.0 for k, v in cs.items() if k != "media")
                and not strip_labels(getattr(fact, "content", "") or "").strip()):
            out.add(fr.fact_id)
    return frozenset(out)


def restore_ranks(fused: Sequence[Any], reranked: Sequence[Any], neutral: frozenset[str]) -> list[Any]:
    """The reranked order with each neutral candidate back at its fused position.

    A neutral candidate's score is set between its new neighbours so a later
    sort by score cannot move it.
    """
    if not neutral:
        return list(reranked)
    movers = iter([fr for fr in reranked if fr.fact_id not in neutral])
    placed = [fr if fr.fact_id in neutral else next(movers) for fr in fused]
    from superlocalmemory.retrieval.fusion import FusionResult

    for i, fr in enumerate(placed):
        if fr.fact_id not in neutral:
            continue
        before = placed[i - 1].fused_score if i else None
        after = next((p.fused_score for p in placed[i + 1:] if p.fact_id not in neutral), None)
        if before is None:
            score = (after or 0.0) + _EPSILON
        elif after is None:
            score = max(0.0, before - _EPSILON)
        else:
            score = (before + after) / 2.0
        placed[i] = FusionResult(fact_id=fr.fact_id, fused_score=score,
                                 channel_ranks=fr.channel_ranks, channel_scores=fr.channel_scores)
    return placed


__all__ = ["all_unreadable", "is_label_only", "neutral_ids", "restore_ranks", "strip_labels"]
