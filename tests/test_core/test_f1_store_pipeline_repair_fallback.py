# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""F1 regression: entity fact_count must be incremented even when the M028
historical-backfill repair-state row is absent (partial migration failure).

Scenario reproduced: M028 DDL created the tables successfully but the
executescript INSERT for 'historical-backfill' was rolled back (e.g. by a
SIGKILL mid-migration).  Every _record_fact_entity_association call then
silently skips the fact_count increment because the JOIN on
fact_entity_association_repair_state returns zero rows.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.core.store_pipeline import _record_fact_entity_association
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.migrations import M028_fact_entity_associations as M028


def _seed_base_data(tmp_path: Path, db_path_name: str = "memory.db") -> sqlite3.Connection:
    """Seed all required base rows with FK constraints OFF (fixture-only helper).

    Returns a committed raw sqlite3 connection so callers can inspect state.
    Using PRAGMA foreign_keys=OFF in the fixture keeps setup independent of FK
    ordering quirks in Python 3.14's implicit transaction model.
    """
    conn = sqlite3.connect(str(tmp_path / db_path_name))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.executescript("""
        INSERT OR IGNORE INTO profiles(profile_id) VALUES ('work');
        INSERT OR IGNORE INTO memories(memory_id, profile_id, content)
            VALUES ('mem-1','work','test content');
        INSERT OR IGNORE INTO canonical_entities
            (entity_id, profile_id, canonical_name, fact_count)
            VALUES ('alice','work','Alice',0);
        INSERT OR IGNORE INTO atomic_facts
            (fact_id, memory_id, profile_id, content)
            VALUES ('fact-1','mem-1','work','Alice leads reliability');
    """)
    conn.close()


def _build_db(tmp_path: Path) -> DatabaseManager:
    """Return a DatabaseManager with M028 applied but the repair row deleted."""
    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(schema)
    _seed_base_data(tmp_path)

    with db.raw_connection() as conn:
        M028.apply(conn)
        # Simulate partial M028 failure: DDL ran but repair row was rolled back
        conn.execute(
            "DELETE FROM fact_entity_association_repair_state "
            "WHERE repair_key='historical-backfill'"
        )
    return db


def test_f1_fact_count_incremented_when_repair_row_absent(tmp_path: Path) -> None:
    """fact_count must reach 1 even when the M028 repair-state row is missing."""
    db = _build_db(tmp_path)

    # Confirm precondition: no repair row exists
    rows = db.execute(
        "SELECT 1 FROM fact_entity_association_repair_state "
        "WHERE repair_key='historical-backfill'",
        (),
    )
    assert not rows, "Precondition failed: repair row should be absent"

    _record_fact_entity_association(
        db,
        operation_id="op-abc123",
        profile_id="work",
        fact_id="fact-1",
        entity_id="alice",
    )

    result = db.execute(
        "SELECT fact_count FROM canonical_entities "
        "WHERE entity_id='alice' AND profile_id='work'",
        (),
    )
    assert result, "canonical_entities row must exist"
    fact_count = int(dict(result[0])["fact_count"])
    assert fact_count == 1, (
        f"F1: entity fact_count must be 1 after increment (got {fact_count!r}). "
        "repair-row-absent fallback path is not implemented."
    )


def test_f1_fallback_is_idempotent_on_retry(tmp_path: Path) -> None:
    """Calling twice for the same (fact_id, entity_id) must not double-count."""
    db = _build_db(tmp_path)

    _record_fact_entity_association(
        db, operation_id="op-abc123",
        profile_id="work", fact_id="fact-1", entity_id="alice",
    )
    _record_fact_entity_association(
        db, operation_id="op-abc123",
        profile_id="work", fact_id="fact-1", entity_id="alice",
    )

    result = db.execute(
        "SELECT fact_count FROM canonical_entities "
        "WHERE entity_id='alice' AND profile_id='work'",
        (),
    )
    fact_count = int(dict(result[0])["fact_count"])
    assert fact_count == 1, (
        f"F1: idempotency broken — fact_count must be 1 after two calls (got {fact_count!r})"
    )


def _seed_rows(tmp_path: Path, db_path_name: str, script: str) -> None:
    """Run an INSERT script against the DB with FK constraints OFF."""
    conn = sqlite3.connect(str(tmp_path / db_path_name))
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.executescript(script)
    conn.close()


def test_f1_normal_path_unaffected_when_repair_row_present(tmp_path: Path) -> None:
    """Normal path (repair row exists) must still work correctly after fix.

    Profile + entity are committed before M028 runs.  The fact is inserted
    after M028 so its rowid > target_fact_rowid, making count_applied=1.
    """
    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(schema)

    # Step 1: seed profile + entity before M028 captures the rowid boundary
    _seed_rows(tmp_path, "memory.db", """
        INSERT OR IGNORE INTO profiles(profile_id) VALUES ('work');
        INSERT OR IGNORE INTO canonical_entities
            (entity_id, profile_id, canonical_name, fact_count)
            VALUES ('bob','work','Bob',0);
    """)

    # Step 2: apply M028 — target_fact_rowid = 0 (no facts yet)
    with db.raw_connection() as conn:
        M028.apply(conn)
        # Repair row IS present (normal upgrade scenario)
        assert conn.execute(
            "SELECT 1 FROM fact_entity_association_repair_state "
            "WHERE repair_key='historical-backfill'"
        ).fetchone() is not None

    # Step 3: insert memory + fact AFTER M028 (rowid > target=0 → count_applied=1)
    _seed_rows(tmp_path, "memory.db", """
        INSERT OR IGNORE INTO memories(memory_id, profile_id, content)
            VALUES ('mem-1','work','x');
        INSERT OR IGNORE INTO atomic_facts
            (fact_id, memory_id, profile_id, content)
            VALUES ('fact-2','mem-1','work','Bob is the manager');
    """)

    _record_fact_entity_association(
        db, operation_id="op-xyz", profile_id="work",
        fact_id="fact-2", entity_id="bob",
    )

    result = db.execute(
        "SELECT fact_count FROM canonical_entities "
        "WHERE entity_id='bob' AND profile_id='work'",
        (),
    )
    fact_count = int(dict(result[0])["fact_count"])
    # A brand-new fact (rowid > target_fact_rowid since target was set before fact-2 was inserted)
    # will have count_applied=1 and increment the count.
    assert fact_count == 1, (
        f"Normal path must still produce fact_count=1 (got {fact_count!r})"
    )
