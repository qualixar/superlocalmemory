# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory — per-profile isolation (I-4)

"""M021 — per-profile ingestion_log dedup (memory.db, deferred).

``ingestion_log`` is the compatibility ledger for external-adapter ingests. Its
dedup UNIQUE constraint was global — ``UNIQUE(source_type, dedup_key)`` — so a
second profile ingesting the same source key could not record its own ledger
row (INSERT OR IGNORE silently no-op'd), losing that profile's fact_ids in the
ledger. (The authoritative dedup already moved to the M018 ingestion_operations
table, which is profile-scoped, so the memory itself is not starved — this
migration fixes the ledger so per-profile bookkeeping is complete.)

SQLite cannot alter a UNIQUE constraint in place, so the table is rebuilt:
rename → create with profile_id + UNIQUE(profile_id, source_type, dedup_key) →
copy rows backfilling profile_id='default' → drop old. Idempotent and tolerant
of a missing table (fresh installs create it correctly via schema_v343).

Deferred because ingestion_log is created at engine init (apply_v343_schema),
after ``apply_all`` runs — same reason as M016.

Author: Varun Pratap Bhardwaj / Qualixar
"""

from __future__ import annotations

import sqlite3

NAME = "M021_ingestion_log_profile"
DB_TARGET = "memory"

# Documentation + drift hash. apply() below is the authoritative executor.
DDL = (
    "ALTER TABLE ingestion_log ADD COLUMN profile_id TEXT NOT NULL DEFAULT 'default';"
    "-- rebuild for UNIQUE(profile_id, source_type, dedup_key)"
)

_NEW_TABLE = """
CREATE TABLE ingestion_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id TEXT NOT NULL DEFAULT 'default',
    source_type TEXT NOT NULL,
    dedup_key TEXT NOT NULL,
    fact_ids TEXT DEFAULT '[]',
    metadata TEXT DEFAULT '{}',
    status TEXT DEFAULT 'ingested',
    ingested_at TEXT NOT NULL,
    UNIQUE(profile_id, source_type, dedup_key)
)
"""


def _cols(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def _old_row_ids(conn: sqlite3.Connection) -> set:
    """Row ids in the leftover table. DB errors propagate — callers only
    invoke this where the table provably exists, and an unreadable
    leftover must fail loudly, never read as an empty (dropped) set."""
    return {
        r[0]
        for r in conn.execute("SELECT id FROM _ingestion_log_old").fetchall()
    }


def _new_row_ids(conn: sqlite3.Connection) -> set:
    """Row ids in the canonical table. DB errors propagate (see above)."""
    return {
        r[0]
        for r in conn.execute("SELECT id FROM ingestion_log").fetchall()
    }


def _copy_from_old(conn: sqlite3.Connection) -> None:
    """Copy legacy rows under the 'default' profile from ``_ingestion_log_old``.

    Column set matches the pre-migration schema (id, source_type,
    dedup_key, fact_ids, metadata, status, ingested_at).
    """
    old_cols = _cols(conn, "_ingestion_log_old")
    has_meta = "metadata" in old_cols
    if has_meta:
        conn.execute(
            "INSERT INTO ingestion_log "
            "(id, profile_id, source_type, dedup_key, fact_ids, metadata, "
            " status, ingested_at) "
            "SELECT id, 'default', source_type, dedup_key, fact_ids, metadata, "
            " status, ingested_at FROM _ingestion_log_old"
        )
    else:
        conn.execute(
            "INSERT INTO ingestion_log "
            "(id, profile_id, source_type, dedup_key, fact_ids, status, ingested_at) "
            "SELECT id, 'default', source_type, dedup_key, fact_ids, "
            " status, ingested_at FROM _ingestion_log_old"
        )


def _rebuild_from_old(conn: sqlite3.Connection) -> None:
    """Complete an interrupted rebuild: the only copy lives in ``_old``."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(_NEW_TABLE)
        _copy_from_old(conn)
        conn.execute("DROP TABLE _ingestion_log_old")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ingestion_dedup "
            "ON ingestion_log(profile_id, source_type, dedup_key)"
        )
        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:  # pragma: no cover — best-effort
            pass
        raise


def apply(conn: sqlite3.Connection) -> None:
    """Rebuild ingestion_log with a profile-scoped dedup constraint.

    4.1.14 audit: the whole rebuild is ONE transaction (a crash between
    RENAME and DROP previously left the only copy in ``_old`` with no
    path back), and a leftover ``_old`` table resumes instead of
    restarting — restarting would RENAME a table that no longer exists.
    No-op on a fresh install (table already has profile_id) or when the
    table doesn't exist yet. Existing rows backfill to the 'default'
    profile.
    """
    table_exists = _table_exists(conn, "ingestion_log")
    old_exists = _table_exists(conn, "_ingestion_log_old")
    if not table_exists:
        if old_exists:
            _rebuild_from_old(conn)
        return
    if "profile_id" in _cols(conn, "ingestion_log"):
        if old_exists:
            # A leftover alongside a canonical table: resume by copying any
            # ids the canonical table is missing (the mid-rebuild crash
            # shape: RENAME+CREATE done, copy interrupted), then drop the
            # leftover only when every one of its rows exists canonically.
            # Row-identity subset, never a COUNT heuristic — a larger
            # disjoint set must fail loudly for manual review instead of
            # silently dropping user data.
            missing = _old_row_ids(conn) - _new_row_ids(conn)
            if missing:
                _old_has_meta = "metadata" in _cols(conn, "_ingestion_log_old")
                _meta_select = (
                    "metadata, " if _old_has_meta else "'{}', "
                )
                conn.execute("BEGIN IMMEDIATE")
                try:
                    for row_id in sorted(missing):
                        conn.execute(
                            "INSERT INTO ingestion_log "
                            "(id, profile_id, source_type, dedup_key, "
                            " fact_ids, metadata, status, ingested_at) "
                            "SELECT id, 'default', source_type, dedup_key, "
                            f" fact_ids, {_meta_select}status, ingested_at "
                            "FROM _ingestion_log_old WHERE id = ?",
                            (row_id,),
                        )
                    conn.execute("COMMIT")
                except Exception:
                    try:
                        conn.execute("ROLLBACK")
                    except sqlite3.Error:  # pragma: no cover — best-effort
                        pass
                    raise
            if _old_row_ids(conn) - _new_row_ids(conn):
                raise sqlite3.OperationalError(
                    "M021 leftover _ingestion_log_old holds rows missing "
                    "from ingestion_log; refusing automatic cleanup"
                )
            conn.execute("DROP TABLE _ingestion_log_old")
        return
    if old_exists:
        # Table present in old shape AND a leftover: external interference
        # (no path in this module produces that combination). Fail loudly.
        raise sqlite3.OperationalError(
            "M021 found ingestion_log without profile_id alongside a "
            "leftover _ingestion_log_old; refusing automatic rebuild"
        )

    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("ALTER TABLE ingestion_log RENAME TO _ingestion_log_old")
        conn.execute(_NEW_TABLE)
        _copy_from_old(conn)
        conn.execute("DROP TABLE _ingestion_log_old")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ingestion_dedup "
            "ON ingestion_log(profile_id, source_type, dedup_key)"
        )
        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:  # pragma: no cover — best-effort
            pass
        raise


def verify(conn: sqlite3.Connection) -> bool:
    """Applied once ingestion_log carries profile_id (or is absent on fresh DB).

    4.1.14 audit: ANY leftover ``_old`` table means unfinished business —
    whether the canonical table is missing (interrupted rebuild) or present
    (interrupted copy or ancient duplicate). Both verify False so the
    runner resumes instead of recording success over stranded data.
    """
    if _table_exists(conn, "_ingestion_log_old"):
        return False
    if not _table_exists(conn, "ingestion_log"):
        return True  # nothing to migrate; fresh install creates it correctly
    return "profile_id" in _cols(conn, "ingestion_log")


def repair(conn: sqlite3.Connection) -> None:
    """Re-run the idempotent apply as end-state repair (4.1.14 #133)."""
    apply(conn)
