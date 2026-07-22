# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
# Part of SuperLocalMemory V3
"""M021 — per-profile ingestion_log dedup (I-4).

The ledger's dedup was UNIQUE(source_type, dedup_key) — global — so a second
profile ingesting the same source key could not record its own ledger row.
M021 rebuilds the table with UNIQUE(profile_id, source_type, dedup_key),
backfilling existing rows to 'default'.
"""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.storage.migrations import M021_ingestion_log_profile as M


def _legacy_db(path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        "CREATE TABLE ingestion_log ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT, source_type TEXT NOT NULL,"
        " dedup_key TEXT NOT NULL, fact_ids TEXT DEFAULT '[]',"
        " metadata TEXT DEFAULT '{}', status TEXT DEFAULT 'ingested',"
        " ingested_at TEXT NOT NULL, UNIQUE(source_type, dedup_key));"
        "INSERT INTO ingestion_log (source_type, dedup_key, fact_ids, ingested_at) "
        "VALUES ('gmail','msgX','[\"f1\"]','2026-01-01');"
    )
    conn.commit()
    return conn


def test_migration_adds_profile_id_and_backfills_default(tmp_path):
    conn = _legacy_db(tmp_path / "m.db")
    assert M.verify(conn) is False
    M.apply(conn)
    conn.commit()
    cols = {r[1] for r in conn.execute("PRAGMA table_info(ingestion_log)")}
    assert "profile_id" in cols
    row = conn.execute(
        "SELECT profile_id, source_type, dedup_key FROM ingestion_log").fetchone()
    assert row == ("default", "gmail", "msgX")
    assert M.verify(conn) is True


def test_two_profiles_can_ingest_same_dedup_key(tmp_path):
    conn = _legacy_db(tmp_path / "m.db")
    M.apply(conn)
    conn.commit()
    # profile 'work' ingests the SAME (source_type, dedup_key) that 'default' has.
    conn.execute(
        "INSERT OR IGNORE INTO ingestion_log "
        "(profile_id, source_type, dedup_key, fact_ids, status, ingested_at) "
        "VALUES ('work','gmail','msgX','[\"f2\"]','ingested','2026-01-02')"
    )
    conn.commit()
    n = conn.execute(
        "SELECT COUNT(*) FROM ingestion_log "
        "WHERE source_type='gmail' AND dedup_key='msgX'").fetchone()[0]
    assert n == 2, "both profiles must record their own ledger row (no starvation)"


def test_same_profile_dedup_still_enforced(tmp_path):
    conn = _legacy_db(tmp_path / "m.db")
    M.apply(conn)
    conn.commit()
    # A duplicate within the SAME profile is still ignored.
    conn.execute(
        "INSERT OR IGNORE INTO ingestion_log "
        "(profile_id, source_type, dedup_key, fact_ids, status, ingested_at) "
        "VALUES ('default','gmail','msgX','[\"dup\"]','ingested','2026-01-03')"
    )
    conn.commit()
    n = conn.execute(
        "SELECT COUNT(*) FROM ingestion_log "
        "WHERE profile_id='default' AND source_type='gmail' AND dedup_key='msgX'"
    ).fetchone()[0]
    assert n == 1, "intra-profile dedup must still hold"


def test_idempotent_and_missing_table_safe(tmp_path):
    # Missing table → verify True, apply no-op.
    conn = sqlite3.connect(str(tmp_path / "empty.db"))
    assert M.verify(conn) is True
    M.apply(conn)  # must not raise
    # Re-apply on an already-migrated table is a no-op.
    conn2 = _legacy_db(tmp_path / "m.db")
    M.apply(conn2); conn2.commit()
    M.apply(conn2); conn2.commit()  # second apply is a no-op
    assert M.verify(conn2) is True
