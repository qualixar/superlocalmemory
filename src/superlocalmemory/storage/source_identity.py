# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""Deduplicate source observations without losing visibility or metadata.

Admission runs inside the coordinator's transaction. Equal text alone is
not identity: owner, scope, project, audience and speaker all matter.
Conflicting metadata is retained in a history alongside the latest value.
"""

from __future__ import annotations

import json

from superlocalmemory.storage.json_columns import is_damaged, load_json_column


def merge_metadata(old: dict, new: dict, dates=()) -> dict:
    merged = dict(old)
    history = {key: list(values) for key, values in old.get("metadata_history", {}).items()}
    for key, value in new.items():
        previous = old.get(key)
        if isinstance(previous, list) and isinstance(value, list):
            merged[key] = list(previous)
            merged[key].extend(v for v in value if v not in merged[key])
        elif isinstance(previous, dict) and isinstance(value, dict):
            merged[key] = merge_metadata(previous, value)
        else:
            if key in old and previous != value:
                values = history.setdefault(key, [])
                values.extend(v for v in (previous, value) if v not in values)
            merged[key] = value
    if history:
        merged["metadata_history"] = history
    if dates:
        merged["event_dates"] = sorted(
            {
                str(d)
                for d in (*old.get("event_dates", []), *new.get("event_dates", []), *dates)
                if d
            }
        )
    return merged


def reuse_source(db, request, metadata: dict, fact) -> str | None:
    """Reuse a live source and merge the new observation; no orphan parent."""
    candidates = db.execute(
        "SELECT m.* FROM memories m WHERE m.profile_id=? AND m.content=? "
        "AND m.scope=? AND m.role=? AND m.speaker=? "
        "AND EXISTS (SELECT 1 FROM atomic_facts f WHERE f.memory_id=m.memory_id "
        "AND f.lifecycle IN ('active','warm','cold') "
        "AND COALESCE(f.quarantined,0)=0 "
        "AND NOT EXISTS (SELECT 1 FROM fact_temporal_validity v "
        "WHERE v.fact_id=f.fact_id AND v.system_expired_at IS NOT NULL))",
        (request.profile_id, request.content, request.scope, request.role, request.speaker),
    )
    for row in candidates:
        stored = dict(row)
        # A source row whose metadata or audience is bytes, not text, cannot be
        # compared and must not be merged into (the merge would write over its
        # bytes): it is not a match, and the new observation gets its own row.
        if is_damaged(stored.get("metadata_json")) or is_damaged(stored.get("shared_with")):
            continue
        old = load_json_column(stored.get("metadata_json"), {}, column="memories.metadata_json")
        if _picture_identity(old) != _picture_identity(metadata):
            continue
        from superlocalmemory.core.project_identity import project_key

        if project_key(old.get("project")) != project_key(metadata.get("project")):
            continue
        if set(load_json_column(stored.get("shared_with"), [], column="memories.shared_with") or []) != set(
            request.shared_with or []
        ):
            continue
        merged = merge_metadata(old, metadata, (stored.get("session_date"), request.session_date))
        db.execute(
            "UPDATE memories SET metadata_json=?, session_date=? WHERE memory_id=?",
            (json.dumps(merged), fact.observation_date, stored["memory_id"]),
        )
        fact.memory_id = stored["memory_id"]
        return fact.memory_id
    return None


def _picture_identity(metadata: dict):
    """Pictures deduplicate by content hash, with legacy attachment IDs as fallback."""
    source = metadata.get("_slm_source") or {}
    if source.get("type") not in ("media", "document"):
        return None
    return (source.get("type"), source.get("content_sha256") or source.get("media_id"),
            source.get("document_id"), source.get("page"))


def matching_fact(db, fact):
    """Derived assertions only collapse within the same source observation.

    Runs inside every save's write transaction, so it must cost the same on any
    store size. ``ORDER BY created_at`` let SQLite pick ``idx_facts_recent``
    (profile_id, created_at, ...) to avoid a sort, and walk every fact of the
    profile comparing content; ``+created_at`` keeps the order but leaves the
    choice to ``idx_facts_source_identity``. The retired check is a primary-key
    probe per candidate instead of a list of every retired fact. Measured on a
    copy of a 27,689-fact store (14 Oct 2026): 1.3-1.7 s idle -> under 1 ms,
    and 5-30 s under load, which held the write lock long enough to fail mesh
    heartbeats with "database is locked".
    """
    rows = db.execute(
        "SELECT f.fact_id,f.shared_with FROM atomic_facts f WHERE f.profile_id=? "
        "AND f.content=? AND f.scope=? AND f.memory_id=? "
        "AND f.lifecycle IN ('active','warm','cold') AND COALESCE(f.quarantined,0)=0 "
        "AND NOT EXISTS (SELECT 1 FROM fact_temporal_validity v "
        "WHERE v.fact_id=f.fact_id AND v.system_expired_at IS NOT NULL) "
        "ORDER BY +f.created_at",
        (fact.profile_id, fact.content, fact.scope or "personal", fact.memory_id),
    )
    audience = set(fact.shared_with or [])
    return [row for row in rows
            if set(load_json_column(row["shared_with"], [], column="atomic_facts.shared_with",
                                    row=row["fact_id"]) or []) == audience][
        :1
    ]


def ensure_indexes(conn) -> None:
    """Additive indexes, created after schema bootstrap on new and existing stores."""
    for table, index, columns in (
        (
            "atomic_facts",
            "idx_facts_source_identity",
            ("profile_id", "content", "scope", "memory_id"),
        ),
        ("memories", "idx_memories_source_identity", ("profile_id", "content", "scope")),
    ):
        have = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if set(columns) <= have:
            conn.execute(f"CREATE INDEX IF NOT EXISTS {index} ON {table}({','.join(columns)})")


def store_source_fact(db, record, fact) -> str:
    """Store a caller-built source fact in its caller's transaction."""
    reused = reuse_source(db, record, record.metadata, fact)
    if reused:
        record.memory_id = reused
    else:
        db.store_memory(record)
    fact.memory_id = record.memory_id
    return db.store_fact(fact)
