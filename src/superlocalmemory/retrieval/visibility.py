# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The one place where recall hides what a caller may not see.

A request sets a ``VisibilityContext`` around its recall. Recall applies it at
three points, all on the recall thread (channel worker threads never see it):
right after correction admission, inside the fact loader, and on the final cut.
Candidates that arrive after the channels (profile, supplements, bridges, scenes,
graph boosts, channel-diversity promotions) are therefore all covered, and a
hidden fact never reaches the reranker, the sufficiency judge or the answer check.

The default context hides nothing. Every filter then returns its argument
untouched and asks the database nothing.
"""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

_IN_CHUNK = 500


@dataclass(frozen=True)
class VisibilityContext:
    """``hidden_fact_ids`` are never shown; ``hide_media`` also hides pictures and pages."""

    hidden_fact_ids: frozenset[str] = frozenset()
    hide_media: bool = False


_EMPTY = VisibilityContext()
_current: contextvars.ContextVar[VisibilityContext] = contextvars.ContextVar(
    "slm_recall_visibility", default=_EMPTY)


def current() -> VisibilityContext:
    return _current.get()


def is_empty() -> bool:
    ctx = _current.get()
    return not ctx.hidden_fact_ids and not ctx.hide_media


def hides_media() -> bool:
    return _current.get().hide_media


@contextlib.contextmanager
def use(ctx: VisibilityContext) -> Iterator[VisibilityContext]:
    token = _current.set(ctx)
    try:
        yield ctx
    finally:
        _current.reset(token)


def _media_hidden_memories(db: Any, memory_ids: Iterable[str]) -> set[str]:
    from superlocalmemory.retrieval.media_channel import memory_sources

    return set(memory_sources(db, list(memory_ids)))


def _memory_of_facts(db: Any, profile_id: str, fact_ids: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for i in range(0, len(fact_ids), _IN_CHUNK):
        part = list(fact_ids[i:i + _IN_CHUNK])
        rows = db.execute(
            "SELECT fact_id, memory_id FROM atomic_facts WHERE profile_id = ? AND fact_id IN ("
            + ",".join("?" * len(part)) + ")", (profile_id, *part))
        out.update((r["fact_id"], r["memory_id"]) for r in rows)
    return out


def drop_hidden_results(fused: list, db: Any, profile_id: str) -> list:
    """``fused`` without the hidden facts. The same list when nothing is hidden."""
    if is_empty():
        return fused
    ctx = current()
    hidden = {fr.fact_id for fr in fused if fr.fact_id in ctx.hidden_fact_ids}
    if ctx.hide_media:
        rest = [fr.fact_id for fr in fused if fr.fact_id not in hidden]
        memory_of = _memory_of_facts(db, profile_id, rest) if rest else {}
        media = _media_hidden_memories(db, memory_of.values())
        hidden |= {fid for fid, mem in memory_of.items() if mem in media}
    return [fr for fr in fused if fr.fact_id not in hidden] if hidden else fused


def drop_hidden_facts(facts: dict, db: Any) -> dict:
    """The loaded facts without the hidden ones. The same dict when nothing is hidden."""
    if is_empty() or not facts:
        return facts
    ctx = current()
    hidden = {fid for fid in facts if fid in ctx.hidden_fact_ids}
    if ctx.hide_media:
        media = _media_hidden_memories(db, {f.memory_id for f in facts.values()})
        hidden |= {fid for fid, f in facts.items() if f.memory_id in media}
    return {fid: f for fid, f in facts.items() if fid not in hidden} if hidden else facts


def keep_loaded(final_top: list, facts: dict) -> list:
    """The final cut without entries whose fact is hidden or was not loaded."""
    if is_empty():
        return final_top
    return [fr for fr in final_top if fr.fact_id in facts]


__all__ = ["VisibilityContext", "current", "drop_hidden_facts", "drop_hidden_results",
           "hides_media", "is_empty", "keep_loaded", "use"]
