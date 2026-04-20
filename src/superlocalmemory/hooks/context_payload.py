# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-05 §3

"""Shared content builder — single source for every adapter body.

LLD reference: ``.backup/active-brain/lld/LLD-05-cross-platform-adapters.md``
Section 3 (Content Builder). One builder → five formatters. Every string is
passed through ``redact_secrets`` before entering the dataclass, so no
adapter ever writes an unredacted secret.

Hard rule A9: secret redaction applied to payload before write.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from superlocalmemory.core.security_primitives import redact_secrets


VERSION = "3.4.22"
DEFAULT_TOP_K = 10
DEFAULT_DECISIONS_K = 5
DEFAULT_MEMORIES_K = 10

# A RecallFn takes (query, limit, profile_id) and returns a list of memory
# dicts with at least {"text": str, "score": float}. Adapters inject the
# real recall engine at construction time; tests inject a fake.
RecallFn = Callable[[str, int, str], list[dict]]


@dataclass(frozen=True, slots=True)
class ContextPayload:
    """Normalised, redacted context ready for any adapter to format.

    All strings are post-redaction. Topics and entities are ranked tuples to
    keep the structure immutable and deterministically serialisable.
    """

    profile_id: str
    topics: tuple[tuple[str, float], ...]
    entities: tuple[tuple[str, int], ...]
    recent_decisions: tuple[str, ...]
    project_memories: tuple[str, ...]
    generated_at: str
    version: str = VERSION


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact_str(s: str) -> str:
    return redact_secrets(s) if isinstance(s, str) else ""


def _redact_seq(items: Iterable[str], limit: int) -> tuple[str, ...]:
    cleaned: list[str] = []
    for item in items:
        if not isinstance(item, str) or not item:
            continue
        cleaned.append(_redact_str(item))
        if len(cleaned) >= limit:
            break
    return tuple(cleaned)


def _recall_topics(
    recall_fn: RecallFn, profile_id: str, scope: str, limit: int,
) -> tuple[tuple[str, float], ...]:
    query = "topics" if scope == "global" else "project topics"
    try:
        results = recall_fn(query, limit, profile_id) or []
    except Exception:
        return ()
    topics: list[tuple[str, float]] = []
    for row in results:
        if not isinstance(row, dict):
            continue
        name = row.get("name") or row.get("text") or ""
        if not isinstance(name, str) or not name:
            continue
        strength = float(row.get("score", row.get("strength", 0.0)) or 0.0)
        topics.append((_redact_str(name), strength))
        if len(topics) >= limit:
            break
    return tuple(topics)


def _recall_entities(
    recall_fn: RecallFn, profile_id: str, scope: str, limit: int,
) -> tuple[tuple[str, int], ...]:
    query = "entities" if scope == "global" else "project entities"
    try:
        results = recall_fn(query, limit, profile_id) or []
    except Exception:
        return ()
    entities: list[tuple[str, int]] = []
    for row in results:
        if not isinstance(row, dict):
            continue
        name = row.get("name") or row.get("text") or ""
        if not isinstance(name, str) or not name:
            continue
        mentions = int(row.get("mentions", row.get("count", 0)) or 0)
        entities.append((_redact_str(name), mentions))
        if len(entities) >= limit:
            break
    return tuple(entities)


def _recall_decisions(
    recall_fn: RecallFn, profile_id: str, limit: int,
) -> tuple[str, ...]:
    try:
        rows = recall_fn("recent decisions", limit, profile_id) or []
    except Exception:
        return ()
    texts = (row.get("text", "") for row in rows if isinstance(row, dict))
    return _redact_seq(texts, limit)


def _recall_memories(
    recall_fn: RecallFn, profile_id: str, scope: str, limit: int,
) -> tuple[str, ...]:
    query = "project memories" if scope == "project" else "memories"
    try:
        rows = recall_fn(query, limit, profile_id) or []
    except Exception:
        return ()
    texts = (row.get("text", "") for row in rows if isinstance(row, dict))
    return _redact_seq(texts, limit)


def build_payload(
    profile_id: str,
    scope: str,
    cwd: Path,
    *,
    recall_fn: RecallFn,
    top_k: int = DEFAULT_TOP_K,
    decisions_k: int = DEFAULT_DECISIONS_K,
    memories_k: int = DEFAULT_MEMORIES_K,
    now_fn: Callable[[], str] | None = None,
) -> ContextPayload:
    """Build a redacted, ranked context payload.

    ``scope`` ∈ {"project", "global"}. ``cwd`` is informational for the
    recall engine (engine-specific signals can key off it); the builder
    itself is a pure transform. ``recall_fn`` is injected — adapters wire
    it to the real engine, tests wire a fake.
    """
    if scope not in ("project", "global"):
        raise ValueError(f"scope must be 'project' or 'global', got {scope!r}")

    topics = _recall_topics(recall_fn, profile_id, scope, top_k)
    entities = _recall_entities(recall_fn, profile_id, scope, top_k)
    decisions = _recall_decisions(recall_fn, profile_id, decisions_k)
    memories = _recall_memories(recall_fn, profile_id, scope, memories_k)

    # Late-bind ``now_fn`` so monkeypatching ``_now_iso`` at module scope
    # still controls the timestamp — crucial for deterministic content-hash
    # tests across sync attempts.
    ts_fn = now_fn if now_fn is not None else _now_iso

    return ContextPayload(
        profile_id=profile_id,
        topics=topics,
        entities=entities,
        recent_decisions=decisions,
        project_memories=memories,
        generated_at=ts_fn(),
        version=VERSION,
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_topics(payload: ContextPayload, limit: int = DEFAULT_TOP_K) -> str:
    if not payload.topics:
        return "_(none yet)_"
    lines = []
    for name, strength in payload.topics[:limit]:
        lines.append(f"- {name} ({strength:.2f})")
    return "\n".join(lines)


def format_entities(
    payload: ContextPayload, limit: int = DEFAULT_TOP_K,
) -> str:
    if not payload.entities:
        return "_(none yet)_"
    lines = []
    for name, mentions in payload.entities[:limit]:
        lines.append(f"- {name} ({mentions})")
    return "\n".join(lines)


def format_decisions(
    payload: ContextPayload, limit: int = DEFAULT_DECISIONS_K,
) -> str:
    if not payload.recent_decisions:
        return "_(none yet)_"
    return "\n".join(f"- {d}" for d in payload.recent_decisions[:limit])


def format_memories(
    payload: ContextPayload, limit: int = DEFAULT_MEMORIES_K,
) -> str:
    if not payload.project_memories:
        return "_(none yet)_"
    return "\n".join(f"- {m}" for m in payload.project_memories[:limit])


def truncate_payload_for_cap(
    payload: ContextPayload, *, hard_cap: int, render: Callable[[ContextPayload], bytes],
) -> bytes:
    """Repeatedly drop sections until ``render(payload)`` fits ``hard_cap``.

    Truncation order (LLD-05 §4.3): project_memories → recent_decisions →
    entities → topics (topics are kept if at all possible).
    """
    rendered = render(payload)
    if len(rendered) <= hard_cap:
        return rendered

    # 1. trim project_memories
    p = _with_memories(payload, ())
    rendered = render(p)
    if len(rendered) <= hard_cap:
        return rendered

    # 2. trim recent_decisions
    p = _with_decisions(p, ())
    rendered = render(p)
    if len(rendered) <= hard_cap:
        return rendered

    # 3. trim entities
    p = _with_entities(p, ())
    rendered = render(p)
    if len(rendered) <= hard_cap:
        return rendered

    # 4. (as last resort) trim topics too
    p = _with_topics(p, ())
    rendered = render(p)
    return rendered  # caller applies truncate_to_cap safety net


def _with_memories(p: ContextPayload,
                   memories: tuple[str, ...]) -> ContextPayload:
    return ContextPayload(
        profile_id=p.profile_id, topics=p.topics, entities=p.entities,
        recent_decisions=p.recent_decisions, project_memories=memories,
        generated_at=p.generated_at, version=p.version,
    )


def _with_decisions(p: ContextPayload,
                    decisions: tuple[str, ...]) -> ContextPayload:
    return ContextPayload(
        profile_id=p.profile_id, topics=p.topics, entities=p.entities,
        recent_decisions=decisions, project_memories=p.project_memories,
        generated_at=p.generated_at, version=p.version,
    )


def _with_entities(p: ContextPayload,
                   entities: tuple[tuple[str, int], ...]) -> ContextPayload:
    return ContextPayload(
        profile_id=p.profile_id, topics=p.topics, entities=entities,
        recent_decisions=p.recent_decisions, project_memories=p.project_memories,
        generated_at=p.generated_at, version=p.version,
    )


def _with_topics(p: ContextPayload,
                 topics: tuple[tuple[str, float], ...]) -> ContextPayload:
    return ContextPayload(
        profile_id=p.profile_id, topics=topics, entities=p.entities,
        recent_decisions=p.recent_decisions, project_memories=p.project_memories,
        generated_at=p.generated_at, version=p.version,
    )


__all__ = (
    "ContextPayload",
    "DEFAULT_DECISIONS_K",
    "DEFAULT_MEMORIES_K",
    "DEFAULT_TOP_K",
    "RecallFn",
    "VERSION",
    "build_payload",
    "format_decisions",
    "format_entities",
    "format_memories",
    "format_topics",
    "truncate_payload_for_cap",
)
