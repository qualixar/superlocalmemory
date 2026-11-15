# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Where each fact came from, recorded once, when it is admitted (blueprint D-21).

Origin cannot be reconstructed later, so 4.2.30 records it from the first day
even though nothing acts on it yet (taint and the origin lattice, M-15, come in
WS-5). What is stored is the raw evidence, not a trust class: the channel the
fact arrived by, how its text relates to its input (``transform``), and the
signals in hand (source type, actor, agent attribution, parent ids). The class
is computed from these later, so a change to the lattice never needs data
nobody kept.

How it is wired: an entry point declares its origin with ``declare(...)`` (a
context variable), and the database's two fact writers call ``record`` beside
the knowledge-time anchor every write path already reaches. A fact stored
with no declaration gets channel ``unknown``: coverage is complete by
construction, and a save path that forgets to declare shows up as unknown
instead of as a silent gap. The first origin recorded for a fact stands.
Recording never blocks a save.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Iterator, Mapping

logger = logging.getLogger(__name__)

#: How a fact arrived. ``remember_http`` is the one route MCP agents, the CLI,
#: the dashboard and hooks share; the signals beside it tell them apart later.
CHANNELS = frozenset({
    "remember_http", "hook_file_change", "hook_session_summary", "observe", "cli_observe",
    "ingest_adapter", "import", "folder", "document", "media", "python_api", "mcp_offline",
    "legacy_replay", "restore", "loop_ledger", "derived", "correction", "repair", "bundle",
    "unknown"})
#: How the stored text relates to what came in.
TRANSFORMS = frozenset({
    "verbatim", "excerpt", "split", "rule_extracted", "model_rewritten", "system_built",
    "copied", "unknown"})

_BY_SOURCE_TYPE: Mapping[str, str] = {
    "http-observe": "observe", "cli-observe": "cli_observe", "http-ingest": "ingest_adapter",
    "http-import": "import", "folder": "folder", "document": "document", "media": "media",
    "python-api": "python_api", "python-api-prebuilt": "python_api",
    "mcp-offline-worker": "mcp_offline", "legacy-pending": "legacy_replay",
    "restore-reimport": "restore",
}
_ADAPTER_TYPES = frozenset({"gmail", "calendar", "transcript"})


@dataclass(frozen=True)
class Origin:
    channel: str
    transform: str
    source_type: str | None = None
    actor_id: str | None = None
    agent_id: str | None = None
    agent_id_source: str | None = None
    speaker: str | None = None
    role: str | None = None
    parent_memory_id: str | None = None
    parent_fact_id: str | None = None
    #: For a batch of derived facts: the transform of each one (from its text).
    per_fact: Any = None

    def transform_of(self, fact: Any) -> str:
        if self.per_fact is None:
            return self.transform
        try:
            got = self.per_fact(fact)
        except Exception:  # noqa: BLE001 - an unknown transform, never a lost save
            return "unknown"
        return got if got in TRANSFORMS else "unknown"

    def __post_init__(self) -> None:
        if self.channel not in CHANNELS:
            raise ValueError(f"unknown origin channel: {self.channel}")
        if self.transform not in TRANSFORMS:
            raise ValueError(f"unknown origin transform: {self.transform}")


UNKNOWN = Origin("unknown", "unknown")
_CURRENT: ContextVar[Origin | None] = ContextVar("slm_fact_origin", default=None)


@contextmanager
def declare(origin: Origin) -> Iterator[Origin]:
    """Facts stored inside this block are recorded with ``origin``."""
    token = _CURRENT.set(origin)
    try:
        yield origin
    finally:
        _CURRENT.reset(token)


def current() -> Origin | None:
    return _CURRENT.get()


def _tags(metadata: Mapping[str, Any]) -> set[str]:
    raw = metadata.get("tags")
    if isinstance(raw, str):
        return {t.strip() for t in raw.split(",") if t.strip()}
    if isinstance(raw, (list, tuple)):
        return {str(t).strip() for t in raw}
    return set()


def channel_for(source_type: str, metadata: Mapping[str, Any]) -> str:
    """The channel a request names with certainty; ``remember_http`` otherwise."""
    if source_type in _BY_SOURCE_TYPE:
        return _BY_SOURCE_TYPE[source_type]
    if source_type in _ADAPTER_TYPES:
        return "ingest_adapter"
    if source_type == "http":
        tags = _tags(metadata)
        if "hook-file-change" in tags:
            return "hook_file_change"
        if "session-end" in tags:
            return "hook_session_summary"
        return "remember_http"
    return "unknown"


def for_request(request: Any, metadata: Mapping[str, Any], *, excerpt: bool) -> Origin:
    """The origin of the fact ``write_queryable`` writes for one admitted request."""
    channel = channel_for(str(request.source_type), metadata)
    built = channel in {"hook_file_change", "hook_session_summary"}
    return Origin(
        channel=channel,
        transform="system_built" if built else ("excerpt" if excerpt else "verbatim"),
        source_type=str(request.source_type), actor_id=request.trusted_actor_id or None,
        agent_id=str(metadata.get("agent_id") or "") or None,
        agent_id_source=str(metadata.get("agent_id_source") or "") or None,
        speaker=request.speaker or None, role=request.role or None)


def derived_from(content: str, *, source_type: str, metadata: Mapping[str, Any],
                 actor_id: str | None, model_mode: str) -> Origin:
    """Facts derived from one memory in the materializer.

    The channel is the memory's own; each fact's transform comes from its text:
    a fact found word for word in the memory is a ``split`` (or the memory
    itself, ``verbatim``), anything else was rewritten, by a model in modes
    b and c, by rules in mode a.
    """
    source = content.strip()
    rewritten = "model_rewritten" if model_mode.lower() in {"b", "c"} else "rule_extracted"

    def per_fact(fact: Any) -> str:
        text = str(getattr(fact, "content", "") or "").strip()
        if text and text == source:
            return "verbatim"
        return "split" if text and text in source else rewritten

    channel = channel_for(str(source_type), metadata)
    return Origin(channel=channel if channel != "unknown" else "derived", transform="unknown",
                  source_type=str(source_type), actor_id=actor_id,
                  agent_id=str(metadata.get("agent_id") or "") or None,
                  agent_id_source=str(metadata.get("agent_id_source") or "") or None,
                  per_fact=per_fact)


_COLUMNS = ("fact_id", "profile_id", "channel", "transform", "source_type", "actor_id",
            "agent_id", "agent_id_source", "speaker", "role", "parent_memory_id",
            "parent_fact_id", "recorded_at")


def record(db: Any, fact: Any) -> None:
    """Record ``fact``'s origin (the declared one, else unknown). Never raises."""
    if not getattr(db, "_fact_origin_ready", False):
        try:
            rows = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                              "AND name='fact_origin'", ())
        except Exception:  # noqa: BLE001 - never block a save
            return
        if not rows:
            return  # the deferred M074 has not run yet
        db._fact_origin_ready = True  # only a positive answer is cached
    try:
        origin = current() or UNKNOWN
        values = (fact.fact_id, fact.profile_id, origin.channel, origin.transform_of(fact),
                  origin.source_type, origin.actor_id, origin.agent_id, origin.agent_id_source,
                  origin.speaker, origin.role,
                  origin.parent_memory_id or getattr(fact, "memory_id", None),
                  origin.parent_fact_id, datetime.now(UTC).isoformat())
        db.execute(f"INSERT OR IGNORE INTO fact_origin ({', '.join(_COLUMNS)}) "  # noqa: S608
                   f"VALUES ({', '.join('?' * len(_COLUMNS))})", values)
    except Exception as exc:  # noqa: BLE001 - origin is evidence, never a reason to lose a save
        logger.warning("fact origin not recorded for %s (%s)", fact.fact_id, type(exc).__name__)


__all__ = ["CHANNELS", "Origin", "TRANSFORMS", "UNKNOWN", "channel_for", "current", "declare",
           "derived_from", "for_request", "record"]
