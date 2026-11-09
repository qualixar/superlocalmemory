# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-05 §3

"""Shared content builder — single source for every adapter body.

passed through ``redact_for_hosted_judge`` before entering the dataclass, so no
adapter ever writes an unredacted secret.

Hard rule A9: secret redaction applied to payload before write.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from superlocalmemory import __version__
from superlocalmemory.retrieval.hosted_redaction import redact_for_hosted_judge

VERSION = __version__
DEFAULT_TOP_K = 10
DEFAULT_DECISIONS_K = 5
DEFAULT_MEMORIES_K = 10

# A RecallFn takes (query, limit, profile_id) and returns a list of memory
# dicts with at least {"text": str, "score": float}. Adapters inject the
# real recall engine at construction time; tests inject a fake. When the
# builder knows the project (4.1.21, #150) it also passes ``project=<name>``
# as a keyword: recall's project filter, which falls back to unfiltered
# results when the project has no saved memories.
RecallFn = Callable[..., list[dict]]


def _call(recall_fn: RecallFn, query: str, limit: int, profile_id: str,
          project: str | None) -> list:
    if project:
        return recall_fn(query, limit, profile_id, project=project) or []
    return recall_fn(query, limit, profile_id) or []


def _query(section: str, scope: str, project: str | None) -> str:
    """The query for one section. With a known project it names the project
    (#150): the canned global words ("project topics", "entities") matched
    whatever memories mention those words, so cross-project trivia was baked
    into the file. Without one, the queries are exactly as before."""
    if scope == "project" and project:
        return f"{project} {section}"
    if section == "topics":
        return "topics" if scope == "global" else "project topics"
    if section == "entities":
        return "entities" if scope == "global" else "project entities"
    if section == "memories":
        return "project memories" if scope == "project" else "memories"
    return section


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
    # These files are injected into an agent's context and can be committed
    # with a repository: the strong screen, with no tail of any credential.
    return redact_for_hosted_judge(s) if isinstance(s, str) else ""


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
    project: str | None = None,
) -> tuple[tuple[str, float], ...]:
    try:
        results = _call(recall_fn, _query("topics", scope, project), limit,
                        profile_id, project if scope == "project" else None)
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
    project: str | None = None,
) -> tuple[tuple[str, int], ...]:
    try:
        results = _call(recall_fn, _query("entities", scope, project), limit,
                        profile_id, project if scope == "project" else None)
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
    project: str | None = None,
) -> tuple[str, ...]:
    try:
        query = f"{project} recent decisions" if project else "recent decisions"
        rows = _call(recall_fn, query, limit, profile_id, project)
    except Exception:
        return ()
    texts = (row.get("text", "") for row in rows if isinstance(row, dict))
    return _redact_seq(texts, limit)


def _recall_memories(
    recall_fn: RecallFn, profile_id: str, scope: str, limit: int,
    project: str | None = None,
) -> tuple[str, ...]:
    try:
        rows = _call(recall_fn, _query("memories", scope, project), limit,
                     profile_id, project if scope == "project" else None)
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
    project: str | None = None,
) -> ContextPayload:
    """Build a redacted, ranked context payload.

    ``scope`` ∈ {"project", "global"}. ``cwd`` is informational for the
    recall engine (engine-specific signals can key off it); the builder
    itself is a pure transform. ``recall_fn`` is injected — adapters wire
    it to the real engine, tests wire a fake.
    """
    if scope not in ("project", "global"):
        raise ValueError(f"scope must be 'project' or 'global', got {scope!r}")

    # ``project`` (a name or a path): only for the project scope, and only
    # when it names one. Omitted, every query is what it always was.
    from superlocalmemory.core.project_identity import project_name

    name = project_name(project) if scope == "project" else None
    topics = _recall_topics(recall_fn, profile_id, scope, top_k, name)
    entities = _recall_entities(recall_fn, profile_id, scope, top_k, name)
    decisions = _recall_decisions(recall_fn, profile_id, decisions_k, name)
    memories = _recall_memories(recall_fn, profile_id, scope, memories_k, name)

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
