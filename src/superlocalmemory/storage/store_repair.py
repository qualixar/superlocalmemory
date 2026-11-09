# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""One-time repair of two kinds of damage left by builds before 4.1.19.

1. Memories whose text was blanked when they were saved. Earlier builds
   replaced anything that looked like a secret with ``[REDACTED:...]`` -
   including ordinary long paths and links. The original text was kept in the
   memory's ingestion record, and its background enrichment never finished
   because the stored copy no longer matched it. A memory is restored only when
   the stored text is exactly the original with some spans replaced by those
   markers; anything else is left alone and counted.
2. Facts marked "replaced" by an automatic check that was wrong. Builds
   before 4.1.19 marked a fact as replaced when an embedding-geometry check, or
   a local model asked about it, judged a newer fact to contradict it; on a
   real store 0 of 22 sampled marks were genuine, and every marked fact was
   pushed down in every recall. Those marks are undone and the old reason is
   kept in the record. A user's own correction is never touched.
3. Entities that a question created. Asking about an unknown name used to
   create an empty entity for it. One is removed only when nothing in the store
   refers to it.

``plan_repair`` reads and changes nothing. ``apply_repair`` runs in one
transaction, and every write is guarded by the value it expects to replace, so
a row changed in the meantime is skipped, never overwritten.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from superlocalmemory.core.ingest_gate import apply_ingest_gate
from superlocalmemory.core.security_primitives import detect_secrets

_MARKER = re.compile(r"\[REDACTED:[A-Z_]+:[^\]]*\]")
_MISMATCH = "queryable ingestion memory content mismatch"
#: Reasons written by the automatic checks, never by a person.
_MACHINE_REASONS = ("LLM-verified contradiction", "Sheaf coboundary")
_REVERTED = "reverted in 4.1.19 (automatic check, not a correction): "

#: (table, column, is_json_list). Any mention keeps an entity.
_ENTITY_REFERENCES: tuple[tuple[str, str, bool], ...] = (
    ("entity_aliases", "entity_id", False),
    ("entity_profiles", "entity_id", False),
    ("temporal_events", "entity_id", False),
    ("entity_communities", "entity_id", False),
    ("fact_entity_associations", "entity_id", False),
    ("consolidated_summaries", "entity_id", False),
    ("memory_scenes", "entity_ids_json", True),
    ("community_summaries", "entity_ids_json", True),
    ("atomic_facts", "canonical_entities_json", True),
)


@dataclass(frozen=True, slots=True)
class RestoreItem:
    operation_id: str
    memory_id: str
    fact_ids: tuple[str, ...]
    blanked: str
    original: str


@dataclass(frozen=True, slots=True)
class RepairPlan:
    restorable: tuple[RestoreItem, ...]
    not_restorable: tuple[str, ...]      # operation ids left alone
    empty_entities: tuple[str, ...]      # entity ids nothing refers to
    wrong_replacements: tuple[str, ...] = ()  # fact ids wrongly marked replaced

    def summary(self) -> dict[str, int]:
        return {
            "memories_to_restore": len(self.restorable),
            "memories_left_alone": len(self.not_restorable),
            "empty_entities_to_remove": len(self.empty_entities),
            "wrong_replaced_marks_to_undo": len(self.wrong_replacements),
        }


@dataclass(frozen=True, slots=True)
class RepairResult:
    memories_restored: int
    memories_skipped: int
    entities_removed: int
    replaced_marks_undone: int = 0


def is_blanked_copy(stored: str, original: str) -> bool:
    """True when ``stored`` is ``original`` with some spans replaced by markers.

    Each marker stands for at least one character of the original. The text
    between markers must appear in the original in the same order, so the
    check walks the pieces with ``str.find`` taking the earliest match of each:
    the earliest placement leaves the most room for the rest, so it finds a
    split whenever one exists. Linear in the text, where one regex of N lazy
    gaps backtracked polynomially.
    """
    if not isinstance(stored, str) or not isinstance(original, str):
        return False
    if stored == original or not _MARKER.search(stored):
        return False
    parts = _MARKER.split(stored)
    head, tail = parts[0], parts[-1]
    if not original.startswith(head) or not original.endswith(tail):
        return False
    pos, end = len(head), len(original) - len(tail)
    for piece in parts[1:-1]:
        found = original.find(piece, pos + 1, end)
        if found < 0:
            return False
        pos = found + len(piece)
    return end - pos >= 1


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(r[1]) for r in conn.execute(f"PRAGMA table_info('{table}')")}


def _plan_restores(conn: sqlite3.Connection) -> tuple[list[RestoreItem], list[str]]:
    restorable: list[RestoreItem] = []
    left: list[str] = []
    rows = conn.execute(
        "SELECT operation_id, raw_content, queryable_fact_ids_json "
        "FROM ingestion_operations WHERE state='failed' AND last_error=?",
        (_MISMATCH,),
    ).fetchall()
    for operation_id, original, ids_json in rows:
        try:
            fact_ids = tuple(str(x) for x in json.loads(ids_json or "[]"))
        except (TypeError, ValueError):
            fact_ids = ()
        if not fact_ids:
            left.append(operation_id)
            continue
        found = conn.execute(
            "SELECT m.memory_id, m.content FROM atomic_facts f "
            "JOIN memories m ON m.memory_id = f.memory_id WHERE f.fact_id=?",
            (fact_ids[0],),
        ).fetchone()
        # A span the old redactor blanked may be a real credential: a person
        # decides, never this repair (the same rule save_repair applies).
        if (found is None or not is_blanked_copy(found[1], original)
                or apply_ingest_gate(original).rejected or detect_secrets(original)):
            left.append(operation_id)
            continue
        restorable.append(RestoreItem(operation_id, str(found[0]), fact_ids,
                                      str(found[1]), str(original)))
    return restorable, left


def _plan_empty_entities(conn: sqlite3.Connection) -> list[str]:
    candidates = [str(r[0]) for r in conn.execute(
        "SELECT entity_id FROM canonical_entities WHERE COALESCE(fact_count, 0) = 0")]
    if not candidates:
        return []
    referenced: set[str] = set()
    for table, column, is_json in _ENTITY_REFERENCES:
        if column not in _table_columns(conn, table):
            continue
        if not is_json:
            where = f"{column} IS NOT NULL"
            if table == "entity_aliases" and "source" in _table_columns(conn, table):
                # Every entity is created with its own name as an alias; that
                # self-alias is not something else referring to it.
                where += " AND COALESCE(source, '') != 'canonical'"
            referenced.update(str(r[0]) for r in conn.execute(
                f"SELECT DISTINCT {column} FROM {table} WHERE {where}"))
            continue
        for (blob,) in conn.execute(
                f"SELECT {column} FROM {table} WHERE {column} IS NOT NULL AND {column} != ''"):
            referenced.update(eid for eid in candidates if eid in str(blob))
    return [eid for eid in candidates if eid not in referenced]


def _plan_wrong_replacements(conn: sqlite3.Connection) -> list[str]:
    if "invalidation_reason" not in _table_columns(conn, "fact_temporal_validity"):
        return []
    clauses = " OR ".join("invalidation_reason LIKE ?" for _ in _MACHINE_REASONS)
    return [str(r[0]) for r in conn.execute(
        "SELECT fact_id FROM fact_temporal_validity WHERE system_expired_at IS NOT NULL "
        f"AND ({clauses}) ORDER BY fact_id",
        tuple(f"{r}%" for r in _MACHINE_REASONS))]


def plan_repair(conn: sqlite3.Connection) -> RepairPlan:
    """What a repair would change. Reads only."""
    restorable, left = _plan_restores(conn)
    return RepairPlan(tuple(restorable), tuple(left), tuple(_plan_empty_entities(conn)),
                      tuple(_plan_wrong_replacements(conn)))


def _rows_of(conn: sqlite3.Connection) -> Callable[..., list]:
    def run(sql: str, params: tuple = ()) -> list:
        return conn.execute(sql, params).fetchall()
    return run


def restore_blanked_text(
    execute: Callable[..., Sequence], memory_id: str, blanked: str, original: str,
    fact_ids: Sequence[str] = (), *, accept_restored_facts: bool = False,
) -> bool:
    """Put ``original`` back over its blanked copy: the memory and its facts, or nothing.

    The guarded statements of the store repair, shared with ``save_repair``.
    ``execute(sql, params)`` returns the rows of a statement inside a write
    transaction the caller holds. Every row is verified before any is written:
    the memory must hold ``blanked`` and every fact the gate's form of it. A
    memory over the ingest gate's 24k-char cap has a CLAMPED fact (head plus
    tail), never the memory's text verbatim, so both sides pass through
    :func:`apply_ingest_gate`; short text passes through unchanged. Each write
    is guarded by the value it replaces (``UPDATE ... WHERE content=<expected>``).

    False, with nothing written, when a row is not what was planned or when the
    gate rejects ``original`` (restoring it would only fail again) or when it
    holds a credential shape (the redactor may have blanked a real secret; a
    person decides, and nothing here logs it).
    ``accept_restored_facts`` lets a fact that already holds the restored form
    stand, which a save repair needs and the one-time store repair never did.
    """
    gate = apply_ingest_gate(original)
    if gate.rejected or detect_secrets(original):
        return False
    expected_fact = apply_ingest_gate(blanked).fact_content
    restored_fact = gate.fact_content
    current_facts = {
        str(row[0]): row[1] for row in execute(
            "SELECT fact_id, content FROM atomic_facts WHERE fact_id IN ({})"
            .format(",".join("?" for _ in fact_ids)), tuple(fact_ids),
        )
    } if fact_ids else {}
    allowed = (expected_fact, restored_fact) if accept_restored_facts else (expected_fact,)
    facts_match = all(current_facts.get(fid) in allowed for fid in fact_ids)
    mem_row = execute("SELECT content FROM memories WHERE memory_id=?", (memory_id,))
    if not (mem_row and mem_row[0][0] == blanked and facts_match):
        return False
    execute("UPDATE memories SET content=? WHERE memory_id=? AND content=?",
            (original, memory_id, blanked))
    for fact_id in fact_ids:
        execute("UPDATE atomic_facts SET content=? WHERE fact_id=? AND content=?",
                (restored_fact, fact_id, expected_fact))
    return True


def apply_repair(conn: sqlite3.Connection, plan: RepairPlan) -> RepairResult:
    """Apply ``plan`` in one transaction. Rows changed since planning are skipped."""
    restored = skipped = removed = undone = 0
    conn.execute("BEGIN IMMEDIATE")
    try:
        run = _rows_of(conn)
        for item in plan.restorable:
            # One set of guarded statements, shared with save_repair: it verifies
            # the memory and every fact before writing any, so a restore is
            # counted only for what is fully applied, and an original the
            # ingest gate rejects is skipped (it would only fail again).
            if not restore_blanked_text(run, item.memory_id, item.blanked, item.original,
                                        item.fact_ids):
                skipped += 1
                continue
            conn.execute(
                "UPDATE ingestion_operations SET state='queryable', attempt_count=0, "
                "next_retry_at=0, last_error='', lease_owner='', lease_expires_at=0, "
                "updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') "
                "WHERE operation_id=? AND state='failed' AND last_error=?",
                (item.operation_id, _MISMATCH))
            restored += 1
        has_alias_source = "source" in _table_columns(conn, "entity_aliases")
        for entity_id in plan.empty_entities:
            if has_alias_source:
                conn.execute("DELETE FROM entity_aliases WHERE entity_id=? "
                             "AND source='canonical'", (entity_id,))
            cur = conn.execute(
                "DELETE FROM canonical_entities WHERE entity_id=? "
                "AND COALESCE(fact_count, 0) = 0", (entity_id,))
            removed += cur.rowcount
        clauses = " OR ".join("invalidation_reason LIKE ?" for _ in _MACHINE_REASONS)
        for fact_id in plan.wrong_replacements:
            cur = conn.execute(
                "UPDATE fact_temporal_validity SET valid_until=NULL, system_expired_at=NULL, "
                "invalidation_reason=? || invalidation_reason WHERE fact_id=? "
                "AND system_expired_at IS NOT NULL "
                f"AND ({clauses})",
                (_REVERTED, fact_id, *(f"{r}%" for r in _MACHINE_REASONS)))
            undone += cur.rowcount
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return RepairResult(restored, skipped, removed, undone)


__all__ = ["RepairPlan", "RepairResult", "RestoreItem", "apply_repair",
           "is_blanked_copy", "plan_repair", "restore_blanked_text"]
