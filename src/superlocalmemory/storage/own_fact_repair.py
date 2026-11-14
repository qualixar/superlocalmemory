# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Give back the searchable fact enrichment wrongly took from a memory.

Before 4.1.22, enrichment deleted a memory's own first-queryable fact when it
judged it a near-duplicate of ANOTHER memory's fact (fixed going forward by
``core/receipt_guard.py``). Those memories are still stored but have no fact,
so recall can never find them. This finds them and re-promotes one queryable
fact from the memory's own text, through the same builder the save path uses.

A memory qualifies only with POSITIVE PROOF that enrichment removed its fact,
never because it merely has no fact (a person may have deleted it): its save's
ingestion record names the fact and the consolidation log records that fact as
a near-duplicate, or the save finished with facts that all exist and all
belong to other memories (folded into them). Without proof it is counted as
``unproven`` and untouched. With proof it is still held back — counted, never
repaired — when any of these hold:

* ``erased``     a tombstone, erasure receipt or erase obligation names the
                 memory, its fact or its profile, or the fact is being erased
                 now (``erasure_fence``): an erased memory never comes back;
* ``erasure_after_save``  a person/entity erasure ran in its profile after the
                 save (the text may mention who was erased; review by hand);
* ``changed``    the save record no longer matches the memory (scrubbed, edited,
                 other profile, unfinished), or the text is empty;
* ``refused``    the save path's own admission or ingest gate would refuse it;
* ``withheld``   its own or a sibling's fact (the one it was folded into) is
                 quarantined: withheld text is never published through it.
Siblings count for ``erased`` too: an erased sibling means its near-identical
text was erased. An unrelated fact erasure elsewhere in the profile does not
hold a memory back: the repair re-creates a fact only from this memory's own
stored text, which list/fetch already show. Every check runs again inside the
write (``promote``).

Reads ids, states and counts; the text is only handed to the fact builder.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

HELD = ("unproven", "erased", "withheld", "erasure_after_save", "changed", "refused",
        "duplicate")


@dataclass(frozen=True)
class Candidate:
    memory_id: str
    profile_id: str
    operation_id: str
    removed_fact_ids: tuple[str, ...]


def _has(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (table,)).fetchone() is not None


def _ids(raw: Any) -> list[str]:
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return []
    return [str(v) for v in value] if isinstance(value, list) else []


def _factless(conn: sqlite3.Connection) -> list[tuple]:
    if not all(_has(conn, t) for t in ("memories", "atomic_facts", "ingestion_operations",
                                       "consolidation_log")):
        return []
    return conn.execute(
        "SELECT m.memory_id, m.profile_id, m.content, m.metadata_json, m.created_at "
        "FROM memories m WHERE NOT EXISTS "
        "(SELECT 1 FROM atomic_facts f WHERE f.memory_id = m.memory_id) "
        "ORDER BY m.created_at, m.memory_id").fetchall()


def _erased(conn: sqlite3.Connection, profile_id: str, memory_id: str | None,
            fact_ids: list[str]) -> bool:
    from superlocalmemory.storage.erasure_fence import is_erasing

    subjects = [s for s in (memory_id, *fact_ids) if s]
    if not subjects:
        return False
    marks = ",".join("?" * len(subjects))
    if any(is_erasing(profile_id, f) for f in fact_ids):
        return True
    if _has(conn, "projection_tombstones") and conn.execute(
            f"SELECT 1 FROM projection_tombstones WHERE memory_id IN ({marks}) "  # noqa: S608
            f"OR fact_id IN ({marks})", (*subjects, *subjects)).fetchone():
        return True
    if _has(conn, "erasure_receipts") and conn.execute(
            f"SELECT 1 FROM erasure_receipts WHERE subject_id IN ({marks}) OR "  # noqa: S608
            "(subject_type = 'profile' AND subject_id = ?)", (*subjects, profile_id)).fetchone():
        return True
    return _has(conn, "projection_obligations") and conn.execute(
        f"SELECT 1 FROM projection_obligations WHERE kind = 'erase' AND subject_id IN ({marks})",  # noqa: S608
        tuple(subjects)).fetchone() is not None


def _erasure_after(conn: sqlite3.Connection, profile_id: str, created_at: str) -> bool:
    if not _has(conn, "erasure_receipts"):
        return False
    from datetime import datetime

    try:
        saved = datetime.fromisoformat(str(created_at).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return True  # cannot order it against the erasure: hold it
    return conn.execute(
        "SELECT 1 FROM erasure_receipts WHERE profile_id = ? AND subject_type = 'entity' "
        "AND requested_at >= ?",
        (profile_id, saved - 86400)).fetchone() is not None


def _logged_near_duplicate(conn: sqlite3.Connection, profile_id: str,
                           fact_ids: list[str]) -> bool:
    """Proof 1: the consolidation log records its fact as a near-duplicate."""
    return bool(fact_ids) and conn.execute(
        f"SELECT 1 FROM consolidation_log WHERE profile_id = ? AND action_type = 'noop' "  # noqa: S608
        f"AND new_fact_id IN ({','.join('?' * len(fact_ids))})",
        (profile_id, *fact_ids)).fetchone() is not None


def _folded_into_others(conn: sqlite3.Connection, memory_id: str, final_ids: list[str]) -> bool:
    """Proof 2: its save finished with facts that ALL exist and ALL belong to other
    memories (folded into them). A person's delete leaves its own final fact
    missing instead, so it never matches."""
    if not final_ids:
        return False
    owners = [conn.execute("SELECT memory_id FROM atomic_facts WHERE fact_id = ?",
                           (f,)).fetchone() for f in final_ids]
    return all(o is not None and o[0] != memory_id for o in owners)


def _siblings(conn: sqlite3.Connection, profile_id: str, memory_id: str,
              fact_ids: list[str], final_ids: list[str]) -> list[str]:
    """The other memories' facts this one was folded into (noop targets + final facts)."""
    targets: list[str] = []
    if fact_ids:
        targets = [str(r[0]) for r in conn.execute(
            f"SELECT existing_fact_id FROM consolidation_log WHERE profile_id = ? "  # noqa: S608
            f"AND action_type = 'noop' AND new_fact_id IN ({','.join('?' * len(fact_ids))})",
            (profile_id, *fact_ids)) if r[0]]
    for fid in final_ids:
        owner = conn.execute("SELECT memory_id FROM atomic_facts WHERE fact_id = ?",
                             (fid,)).fetchone()
        if owner is None or owner[0] != memory_id:
            targets.append(fid)
    return sorted(set(targets))


def _withheld(conn: sqlite3.Connection, memory_id: str, siblings: list[str]) -> bool:
    """Its own or a sibling's fact is quarantined (withheld): never publish that text."""
    marks = ",".join("?" * len(siblings)) or "NULL"
    return conn.execute(
        f"SELECT 1 FROM atomic_facts WHERE quarantined = 1 AND "  # noqa: S608
        f"(memory_id = ? OR fact_id IN ({marks})) LIMIT 1",
        (memory_id, *siblings)).fetchone() is not None


def _judge(conn: sqlite3.Connection, row: Any) -> tuple[str | None, Candidate | None]:
    """Why ``row`` (a factless memory) is held back, or the candidate to repair."""
    from superlocalmemory.core.engine_ingestion import content_passes_admission

    memory_id, profile_id, content, metadata_json, created_at = row
    op_id = str(_meta(metadata_json).get("ingestion_operation_id") or "")
    op = conn.execute(
        "SELECT profile_id, state, raw_content, queryable_fact_ids_json, final_fact_ids_json "
        "FROM ingestion_operations WHERE operation_id = ?", (op_id,)).fetchone() if op_id else None
    fact_ids = _ids(op[3]) if op else []
    final_ids = _ids(op[4]) if op else []
    if not (_logged_near_duplicate(conn, profile_id, fact_ids)
            or _folded_into_others(conn, memory_id, final_ids)):
        return "unproven", None  # no proof enrichment removed it: never touched
    siblings = _siblings(conn, profile_id, memory_id, fact_ids, final_ids)
    if (_erased(conn, profile_id, memory_id, fact_ids)
            or _erased(conn, profile_id, None, siblings)):
        return "erased", None
    if _withheld(conn, memory_id, siblings):
        return "withheld", None
    if _erasure_after(conn, profile_id, created_at):
        return "erasure_after_save", None
    if (op[0] != profile_id or op[1] != "complete" or not str(content or "").strip()
            or op[2] != content or any(conn.execute(
                "SELECT 1 FROM atomic_facts WHERE fact_id = ?", (f,)).fetchone() is None
                for f in siblings if f not in final_ids)):
        return "changed", None  # edited, scrubbed, unfinished, moved, or its sibling deleted
    if not content_passes_admission(content):
        return "refused", None
    return None, Candidate(memory_id, profile_id, op_id, tuple(fact_ids))


def classify(conn: sqlite3.Connection) -> tuple[list[Candidate], dict[str, int]]:
    """Memories to repair, and how many were held back for each reason."""
    found: list[Candidate] = []
    held = dict.fromkeys(HELD, 0)
    for row in _factless(conn):
        reason, candidate = _judge(conn, row)
        if candidate is not None and _words_already_a_fact(conn, candidate):
            # Storing it would fold onto that fact (promote creates nothing), and its
            # words are already found through the other memory: not repair work.
            held["duplicate"] += 1
        elif candidate is not None:
            found.append(candidate)
        else:
            held[reason] += 1
    return found, held


def _words_already_a_fact(conn: sqlite3.Connection, candidate: Candidate) -> bool:
    from superlocalmemory.core.ingest_gate import apply_ingest_gate

    row = conn.execute("SELECT content FROM memories WHERE memory_id = ?",
                       (candidate.memory_id,)).fetchone()
    if row is None:
        return False
    gate = apply_ingest_gate(row[0])
    if gate.rejected:
        return False
    return conn.execute(
        "SELECT 1 FROM atomic_facts WHERE profile_id = ? AND content = ? AND memory_id != ? "
        "LIMIT 1", (candidate.profile_id, gate.fact_content, candidate.memory_id)
    ).fetchone() is not None


def _meta(raw: Any) -> dict:
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def census(conn: sqlite3.Connection) -> dict[str, int]:
    """Counts for ``slm db integrity`` and the repair plan."""
    found, held = classify(conn)
    return {"to_repair": len(found), **{f"held_{k}": v for k, v in held.items()}}


class _NothingToCreate(Exception):
    """Identical words already belong to another memory: roll back, create nothing."""


def promote(db: Any, conn: sqlite3.Connection | None, candidate: Candidate) -> str | None:
    """Re-promote one queryable fact for ``candidate``; its id, or None if held back.

    Everything is decided again INSIDE one ``BEGIN IMMEDIATE`` write transaction
    on the connection that writes: the memory is still factless, still proven,
    nothing about it or its siblings was erased or withheld since it was listed.
    The immediate lock makes the database the arbiter between processes, so two
    repairs can never both create the fact. ``conn`` is accepted for callers of
    the earlier signature and not used. The fact carries the memory's scope,
    sharing, session, date and kind exactly as its save did.
    """
    del conn
    try:
        with db.raw_connection() as tx:   # the process write lock + its own connection
            tx.execute("BEGIN IMMEDIATE")
            return _promote_locked(db, tx, candidate)
    except _NothingToCreate:
        return None


def _promote_locked(db: Any, tx: sqlite3.Connection, candidate: Candidate) -> str | None:
    from superlocalmemory.core.ingest_gate import apply_ingest_gate
    from superlocalmemory.core.kind_assignment import assign_kinds, store_fact_keeping_kind
    from superlocalmemory.core.queryable_fact import queryable_fact

    current = tx.execute(
        "SELECT m.memory_id, m.profile_id, m.content, m.metadata_json, m.created_at FROM "
        "memories m WHERE m.memory_id = ? AND m.profile_id = ? AND NOT EXISTS "
        "(SELECT 1 FROM atomic_facts f WHERE f.memory_id = m.memory_id)",
        (candidate.memory_id, candidate.profile_id)).fetchone()
    if current is None:
        return None  # it has a fact again, or is gone
    reason, fresh = _judge(tx, tuple(current))
    if reason is not None or fresh is None or fresh.operation_id != candidate.operation_id:
        return None
    row = tx.execute(
        "SELECT m.content, m.scope, m.shared_with, m.session_id, m.session_date, m.created_at, "
        "o.raw_metadata_json, o.source_type, o.trusted_actor_id FROM memories m JOIN "
        "ingestion_operations o ON o.operation_id = ? WHERE m.memory_id = ?",
        (candidate.operation_id, candidate.memory_id)).fetchone()
    content, scope, shared, session_id, session_date, created_at, raw_meta, source, actor = row
    gate = apply_ingest_gate(content, source_type=source)
    if gate.rejected:
        return None
    metadata = _meta(raw_meta)
    fact = queryable_fact(
        gate.fact_content, profile_id=candidate.profile_id, scope=scope or "personal",
        shared_with=_ids(shared) if shared else None, session_id=session_id or "",
        observation_date=(session_date or str(created_at))[:10], created_at=str(created_at))
    fact = assign_kinds([fact], metadata=metadata, source_type=source, classifier=None, db=db)[0]
    fact.memory_id = candidate.memory_id
    stored = store_fact_keeping_kind(db, fact, metadata=metadata,
                                     request=SimpleNamespace(source_type=source,
                                                             trusted_actor_id=actor))
    mine = tx.execute("SELECT COUNT(*) FROM atomic_facts WHERE memory_id = ?",
                      (candidate.memory_id,)).fetchone()[0]
    owner = tx.execute("SELECT memory_id FROM atomic_facts WHERE fact_id = ?",
                       (stored,)).fetchone()
    if mine != 1 or owner is None or owner[0] != candidate.memory_id:
        # Identical words already stored for another memory folded onto that
        # fact (and may have touched it): undo all of it, create nothing.
        raise _NothingToCreate
    _queue_enrichment(db, candidate, stored, content=content, metadata=metadata, source=source,
                      actor=actor, scope=scope or "personal", shared=_ids(shared) if shared else [],
                      session_id=session_id or "", session_date=session_date or "")
    return stored


def _queue_enrichment(db: Any, candidate: Candidate, fact_id: str, *, content: str,
                      metadata: dict, source: str, actor: str, scope: str, shared: list[str],
                      session_id: str, session_date: str) -> None:
    """Queue the fact for enrichment exactly as a save queues its receipt.

    A first-queryable fact is found by words only until the materializer gives
    it its meaning vector, keyword tokens and graph links. One ingestion
    operation per memory (stable key), in the queryable state, holding this
    fact: the materializer promotes it in place, as for any save.
    """
    from superlocalmemory.core.ingestion_command import (
        IngestionOperationRepository,
        IngestionRequest,
        IngestionState,
    )

    repository = IngestionOperationRepository(db)
    operation, _ = repository.create_with_status(IngestionRequest(
        content=content, profile_id=candidate.profile_id, source_type=source,
        idempotency_key=f"own-fact-repair:{candidate.memory_id}", metadata=metadata,
        scope=scope, shared_with=tuple(shared), trusted_actor_id=actor,
        session_id=session_id, session_date=session_date))
    if operation.state is IngestionState.RAW:
        repository.transition(operation.operation_id, expected=IngestionState.RAW,
                              target=IngestionState.QUERYABLE, queryable_fact_ids=(fact_id,))


__all__ = ["Candidate", "HELD", "census", "classify", "promote"]
