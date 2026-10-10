# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The read-only picture ``slm db repair`` acts on: counts only, never text.

``plan`` is what the repair prints before it changes anything and again after
it ran: per class, how many rows it would remove, keep, scrub or settle.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from superlocalmemory.storage import integrity_census as census
from superlocalmemory.storage import integrity_obligations as obligations
from superlocalmemory.storage import own_fact_repair


def _has(conn: sqlite3.Connection, table: str) -> bool:
    return census._has(conn, table)


def erased_facts(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """``(profile_id, fact_id)`` of every erased fact: tombstoned and gone."""
    if not _has(conn, "projection_tombstones"):
        return []
    return [(str(r[0]), str(r[1])) for r in conn.execute(
        "SELECT t.profile_id, t.fact_id FROM projection_tombstones AS t WHERE NOT EXISTS "
        "(SELECT 1 FROM atomic_facts AS f WHERE f.fact_id = t.fact_id) ORDER BY t.created_at")]


def _ids(raw: Any) -> list[str]:
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return []
    return [str(v) for v in value] if isinstance(value, list) else []


def _live(conn: sqlite3.Connection, ids: list[str]) -> bool:
    return any(conn.execute("SELECT 1 FROM atomic_facts WHERE fact_id = ?", (i,)).fetchone()
               for i in ids)


def erased_text_leftovers(conn: sqlite3.Connection) -> dict[str, int]:
    """Copies of erased memories' words outside the projections (counts)."""
    out = {"erased_facts": 0, "journal_text": 0, "event_previews": 0,
           "entity_summaries": 0, "derived_summaries": 0, "archive_copies": 0}
    erased = erased_facts(conn)
    out["erased_facts"] = len(erased)
    journal_ops: set[str] = set()
    for profile_id, fact_id in erased:
        like = f"%{fact_id}%"
        if _has(conn, "ingestion_operations"):
            for op_id, q, f in conn.execute(
                    "SELECT operation_id, queryable_fact_ids_json, final_fact_ids_json FROM "
                    "ingestion_operations WHERE profile_id = ? AND raw_content != '' AND "
                    "(queryable_fact_ids_json LIKE ? OR final_fact_ids_json LIKE ?)",
                    (profile_id, like, like)):
                if not _live(conn, _ids(q) + _ids(f)):
                    journal_ops.add(str(op_id))
        if _has(conn, "memory_events"):
            out["event_previews"] += int(conn.execute(
                "SELECT COUNT(*) FROM memory_events WHERE payload LIKE ? AND payload LIKE "
                "'%content_preview%'", (like,)).fetchone()[0])
        if _has(conn, "entity_profiles"):
            out["entity_summaries"] += int(conn.execute(
                "SELECT COUNT(*) FROM entity_profiles WHERE profile_id = ? AND fact_ids_json "
                "LIKE ?", (profile_id, like)).fetchone()[0])
        for table, column in (("core_memory_blocks", "source_fact_ids"),
                              ("community_summaries", "fact_ids_json"),
                              ("consolidated_summaries", "source_fact_ids")):
            if _has(conn, table):
                out["derived_summaries"] += int(conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE profile_id = ? AND {column} LIKE ?",  # noqa: S608
                    (profile_id, like)).fetchone()[0])
        if _has(conn, "memory_archive"):
            out["archive_copies"] += int(conn.execute(
                "SELECT COUNT(*) FROM memory_archive WHERE fact_id = ?", (fact_id,)).fetchone()[0])
    out["journal_text"] = len(journal_ops)
    return out


def erased_text_targets(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    return erased_facts(conn)


def keyword_index_state(conn: sqlite3.Connection) -> dict[str, Any]:
    from superlocalmemory.storage import fts_residue

    state: dict[str, Any] = {}
    damaged: list[str] = []
    for table in fts_residue.FTS_TABLES:
        if _has(conn, table):
            state[table] = "secure_delete_on" if _secure_delete_on(conn, table) else "secure_delete_off"
            if fts_residue.keyword_index_damaged(conn, table):
                damaged.append(table)
    state["damaged"] = damaged
    purged = _has(conn, "integrity_repair_receipts") and bool(conn.execute(
        "SELECT 1 FROM integrity_repair_receipts WHERE action = 'purge_keyword_index' "
        "LIMIT 1").fetchone())
    state["purged_by_repair"] = bool(purged)
    return state


def _secure_delete_on(conn: sqlite3.Connection, table: str) -> bool:
    from superlocalmemory.storage.fts_residue import secure_delete_on

    try:
        return secure_delete_on(conn, table)
    except sqlite3.Error:
        return False


def unreachable_vectors(db_path: Any) -> int | None:
    """None when the vector extension cannot be loaded here (not checked)."""
    from superlocalmemory.storage.vector_residue import unreferenced_rowids, vec_connection

    with vec_connection(db_path) as conn:
        return None if conn is None else len(unreferenced_rowids(conn))


def plan(conn: sqlite3.Connection, *, foreign_keys: bool = True,
         lance: Any = None) -> dict[str, Any]:
    """``lance`` is the running vector projection when the caller has one (the
    repair inside SLM); otherwise it is found on disk and read, never written."""
    from superlocalmemory.storage import vector_parity

    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    return {
        "foreign_key_findings": census.foreign_key_findings(conn) if foreign_keys else None,
        "orphans": census.orphan_census(conn),
        "parentless_facts": census.quarantined_parentless_facts(conn),
        "unreachable_vectors": unreachable_vectors(db_path),
        "erased_text": erased_text_leftovers(conn),
        "keyword_index": keyword_index_state(conn),
        "failed_obligations": obligations.census(conn),
        "memories_without_own_fact": own_fact_repair.census(conn),
        "vector_parity": vector_parity.census(conn, db_path, lance),
    }


__all__ = ["erased_facts", "erased_text_leftovers", "erased_text_targets",
           "keyword_index_state", "plan", "unreachable_vectors"]
