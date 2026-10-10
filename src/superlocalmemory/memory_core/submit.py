# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Save a memory built from several parts, each prepared by where it came from.

A part a person typed is stored as written; a part SLM derived (text read out
of an image) always has credentials stripped. Personal-data redaction, when it
is on, covers every part, the metadata and the dedup key. Callers outside the
server use this instead of building a write request themselves.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

from superlocalmemory.memory_core.save_scope import resolve_scope
from superlocalmemory.memory_core.save_path import (
    ContentOrigin,
    effective_pii_redaction,
    prepare_for_save,
    prepare_key,
    prepare_metadata,
)

logger = logging.getLogger(__name__)

#: The same bounds the HTTP remember route gives the writer.
DEADLINE_MS = 2_000
ACCEPT_AFTER_MS = 1_200


@dataclass(frozen=True)
class SaveRequest:
    #: Parts in order; the caller puts any separators inside the part text.
    segments: tuple[tuple[str, ContentOrigin], ...]
    profile_id: str
    source_type: str
    trusted_actor_id: str
    tags: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    trusted_metadata: Mapping[str, Any] = field(default_factory=dict)
    idempotency_key: str = ""
    session_date: str = ""
    #: ``None`` takes the configured default scope, as typed text does.
    scope: str | None = None
    shared_with: tuple[str, ...] = ()


@dataclass(frozen=True)
class SaveReceipt:
    status: str
    memory_id: str | None
    fact_ids: tuple[str, ...]
    operation_id: str
    pii_count: int
    secret_count: int


def _prepare_segments(request: SaveRequest, redact: bool) -> tuple[str, int, int]:
    parts: list[str] = []
    pii = secrets = 0
    for text, origin in request.segments:
        prepared = prepare_for_save(text, origin=origin, pii_redaction=redact)
        parts.append(prepared.text)
        pii += prepared.pii_count
        secrets += prepared.secret_count
    return "".join(parts), pii, secrets


def _metadata(request: SaveRequest, redact: bool) -> tuple[dict, int]:
    from superlocalmemory.core.metadata_guard import strip_reserved_metadata

    meta: dict[str, Any] = {}
    if request.tags:
        meta["tags"] = request.tags
    meta.update(strip_reserved_metadata(dict(request.metadata)))
    meta, count = prepare_metadata(meta, pii_redaction=redact)
    meta.update(request.trusted_metadata)  # server-set ids only; nothing personal
    return meta, count


def _memory_id(runtime: Any, payload: Mapping[str, Any], fact_ids: tuple[str, ...]) -> str | None:
    """The memory the facts belong to: from the receipt, else looked up from the first fact."""
    if payload.get("memory_id"):
        return str(payload["memory_id"])
    # Private on purpose: the writer's receipt carries no memory_id, so the fact row is the only link.
    db = getattr(runtime, "_db", None)
    if db is None or not fact_ids:
        return None
    try:
        rows = db.execute("SELECT memory_id FROM atomic_facts WHERE fact_id = ?", (fact_ids[0],))
        return str(rows[0]["memory_id"]) if rows else None
    except Exception:  # noqa: BLE001 - the link is best effort; the save already happened
        logger.warning("could not look up the memory of a saved fact")
        return None


def submit_memory(
    runtime: Any, request: SaveRequest, *, config: object | None,
    deadline_ms: int = DEADLINE_MS, accept_after_ms: int = ACCEPT_AFTER_MS,
) -> SaveReceipt:
    """Prepare every part and hand one request to the writer; returns its receipt."""
    from superlocalmemory.storage.admission_journal import Actor, RememberRequest

    redact = effective_pii_redaction(config)
    joined, pii, secrets = _prepare_segments(request, redact)
    if not joined.strip():
        raise ValueError("there is nothing to save")
    meta, meta_pii = _metadata(request, redact)
    scope = resolve_scope(config, request.scope)
    # The parts are already prepared; this identity pass (user text, redaction off)
    # gives the writer one prepared value without changing a character.
    final = prepare_for_save(joined, origin=ContentOrigin.USER_TEXT, pii_redaction=False)
    key = prepare_key(request.idempotency_key, pii_redaction=redact) if request.idempotency_key else uuid.uuid4().hex
    admission = RememberRequest(
        content=final.text, profile_id=request.profile_id, source_type=request.source_type,
        idempotency_key=key, metadata=meta, scope=scope,
        shared_with=tuple(request.shared_with),
        trusted_actor_id=request.trusted_actor_id, session_date=request.session_date,
    )
    actor = Actor(principal_id=request.trusted_actor_id,
                  allowed_profiles=frozenset({request.profile_id}),
                  allowed_scopes=frozenset({scope}))
    receipt = runtime.remember(admission, actor, deadline_ms=deadline_ms, accept_after_ms=accept_after_ms)
    payload = dict(receipt.payload)
    fact_ids = tuple(str(f) for f in payload.get("fact_ids") or ())
    return SaveReceipt(
        status=str(payload.get("status") or ""),
        memory_id=_memory_id(runtime, payload, fact_ids) if payload.get("status") != "accepted" else None,
        fact_ids=fact_ids, operation_id=str(payload.get("operation_id") or ""),
        pii_count=pii + meta_pii, secret_count=secrets,
    )
