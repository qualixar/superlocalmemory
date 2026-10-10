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
import json
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

logger = logging.getLogger(__name__)
_IN_CHUNK = 500


@dataclass(frozen=True)
class VisibilityContext:
    """``hidden_fact_ids`` are never shown; ``hide_media`` also hides pictures and pages;
    ``hide_sources`` also hides everything that came from a connected folder.

    ``vetted_media``, when not ``None``, lets only the pictures and pages it names show
    (see :func:`media_token`); every other picture or page is hidden."""

    hidden_fact_ids: frozenset[str] = frozenset()
    hide_media: bool = False
    hide_sources: bool = False
    vetted_media: frozenset[str] | None = None


_EMPTY = VisibilityContext()
_BAD = object()  # metadata that could not be read
_current: contextvars.ContextVar[VisibilityContext] = contextvars.ContextVar(
    "slm_recall_visibility", default=_EMPTY)


def current() -> VisibilityContext:
    return _current.get()


def is_empty() -> bool:
    ctx = _current.get()
    return (not ctx.hidden_fact_ids and not ctx.hide_media and not ctx.hide_sources
            and ctx.vetted_media is None)


def media_token(source: Any) -> str:
    """The name a picture, page or document memory goes by in ``vetted_media``.

    ``m:<media id>`` for a picture, ``p:<document id>:<page>`` for a page and
    ``d:<document id>`` for a memory of the whole document. ``""`` when the marker names none.
    """
    if not isinstance(source, dict):
        return ""
    if source.get("media_id"):
        return f"m:{source['media_id']}"
    if source.get("document_id"):
        page = source.get("page")
        return f"p:{source['document_id']}:{page}" if page is not None else f"d:{source['document_id']}"
    return ""


def hides_media() -> bool:
    return _current.get().hide_media


@contextlib.contextmanager
def use(ctx: VisibilityContext) -> Iterator[VisibilityContext]:
    token = _current.set(ctx)
    try:
        yield ctx
    finally:
        _current.reset(token)


def _is_hidden_source(source: Any, ctx: VisibilityContext) -> bool:
    if not isinstance(source, dict):
        return False
    from superlocalmemory.retrieval.media_channel import MEDIA_SOURCE_TYPES

    if ctx.hide_media and source.get("type") in MEDIA_SOURCE_TYPES:
        return True
    if ctx.vetted_media is not None and (
            source.get("type") in MEDIA_SOURCE_TYPES or source.get("media_id")):
        if media_token(source) not in ctx.vetted_media:
            return True
    return ctx.hide_sources and (source.get("type") == "folder" or source.get("origin") == "folder")


def _hidden_source_memories(db: Any, memory_ids: Iterable[str], ctx: VisibilityContext) -> set[str]:
    """The memories the context hides by where they came from (pictures, pages, folders).
    Always asks, whatever the feature switches say; metadata that cannot be read counts as hidden."""
    ids = list(dict.fromkeys(i for i in memory_ids if i))
    found: set[str] = set()
    for i in range(0, len(ids), _IN_CHUNK):
        part = ids[i:i + _IN_CHUNK]
        rows = db.execute(
            "SELECT memory_id, metadata_json FROM memories WHERE memory_id IN ("
            + ",".join("?" * len(part)) + ")", tuple(part))
        parsed: list[tuple[Any, Any]] = []
        for row in rows:
            try:
                parsed.append((row, (json.loads(row["metadata_json"] or "{}") or {}).get("_slm_source")))
            except (ValueError, AttributeError):
                parsed.append((row, _BAD))
        prime = getattr(ctx.vetted_media, "prime", None)
        if callable(prime):  # ask the media store about these candidates only, in one go
            prime(media_token(src) for _, src in parsed if src is not _BAD)
        for row, source in parsed:
            hidden = True if source is _BAD else _is_hidden_source(source, ctx)
            if hidden:
                found.add(row["memory_id"])
    return found


def _memory_of_facts(db: Any, profile_id: str, fact_ids: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for i in range(0, len(fact_ids), _IN_CHUNK):
        part = list(fact_ids[i:i + _IN_CHUNK])
        rows = db.execute(
            "SELECT fact_id, memory_id FROM atomic_facts WHERE profile_id = ? AND fact_id IN ("
            + ",".join("?" * len(part)) + ")", (profile_id, *part))
        out.update((r["fact_id"], r["memory_id"]) for r in rows)
    return out


def _hidden_source_facts(db: Any, profile_id: str, fact_ids: Sequence[str],
                         ctx: VisibilityContext) -> set[str]:
    memory_of = _memory_of_facts(db, profile_id, fact_ids) if fact_ids else {}
    hidden = _hidden_source_memories(db, memory_of.values(), ctx)
    return {fid for fid, mem in memory_of.items() if mem in hidden}


def _asks_sources(ctx: VisibilityContext) -> bool:
    return ctx.hide_media or ctx.hide_sources or ctx.vetted_media is not None


def drop_hidden_results(fused: list, db: Any, profile_id: str) -> list:
    """``fused`` without the hidden facts. The same list when nothing is hidden.
    If the media lookup fails, nothing is shown."""
    if is_empty():
        return fused
    ctx = current()
    hidden = {fr.fact_id for fr in fused if fr.fact_id in ctx.hidden_fact_ids}
    if _asks_sources(ctx):
        try:
            hidden |= _hidden_source_facts(
                db, profile_id, [fr.fact_id for fr in fused if fr.fact_id not in hidden], ctx)
        except Exception as exc:  # noqa: BLE001 - fail closed
            logger.warning("visibility lookup failed, hiding all (%s)", type(exc).__name__)
            return []
    return [fr for fr in fused if fr.fact_id not in hidden] if hidden else fused


def drop_hidden_facts(facts: dict, db: Any) -> dict:
    """The loaded facts without the hidden ones. The same dict when nothing is hidden.
    If the media lookup fails, nothing is shown."""
    if is_empty() or not facts:
        return facts
    ctx = current()
    hidden = {fid for fid in facts if fid in ctx.hidden_fact_ids}
    if _asks_sources(ctx):
        try:
            media = _hidden_source_memories(db, {f.memory_id for f in facts.values()}, ctx)
        except Exception as exc:  # noqa: BLE001 - fail closed
            logger.warning("visibility lookup failed, hiding all (%s)", type(exc).__name__)
            return {}
        hidden |= {fid for fid, f in facts.items() if f.memory_id in media}
    return {fid: f for fid, f in facts.items() if fid not in hidden} if hidden else facts


def hidden_among(db: Any, profile_id: str, fact_ids: Sequence[str]) -> set[str]:
    """Which of ``fact_ids`` the current context hides (all of them if the lookup fails)."""
    if is_empty():
        return set()
    ctx = current()
    hidden = {fid for fid in fact_ids if fid in ctx.hidden_fact_ids}
    if _asks_sources(ctx):
        try:
            rest = [fid for fid in fact_ids if fid not in hidden]
            hidden |= _hidden_source_facts(db, profile_id, rest, ctx)
        except Exception as exc:  # noqa: BLE001 - fail closed
            logger.warning("visibility lookup failed, hiding all (%s)", type(exc).__name__)
            return set(fact_ids)
    return hidden


def keep_loaded(final_top: list, facts: dict) -> list:
    """The final cut without entries whose fact is hidden or was not loaded."""
    if is_empty():
        return final_top
    return [fr for fr in final_top if fr.fact_id in facts]


__all__ = ["VisibilityContext", "current", "drop_hidden_facts", "drop_hidden_results",
           "hidden_among", "hides_media", "is_empty", "keep_loaded", "media_token", "use"]
