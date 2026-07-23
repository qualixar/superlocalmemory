# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""M030 Entity Explorer index migration contracts."""

from __future__ import annotations

import sqlite3

from superlocalmemory.storage.migrations import (
    M030_entity_explorer_indexes as m030,
)


def _complete_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE canonical_entities (
            entity_id TEXT PRIMARY KEY,
            profile_id TEXT NOT NULL,
            canonical_name TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            fact_count INTEGER NOT NULL
        );
        CREATE TABLE entity_profiles (
            entity_id TEXT NOT NULL,
            profile_id TEXT NOT NULL,
            project_name TEXT NOT NULL,
            last_compiled_at TEXT
        );
        """
    )


def test_apply_is_idempotent_and_verified() -> None:
    conn = sqlite3.connect(":memory:")
    _complete_schema(conn)

    m030.apply(conn)
    m030.apply(conn)

    assert m030.verify(conn) is True


def test_apply_tolerates_absent_optional_tables() -> None:
    conn = sqlite3.connect(":memory:")

    m030.apply(conn)

    assert m030.verify(conn) is True


def test_repair_restores_completed_migration_end_state() -> None:
    conn = sqlite3.connect(":memory:")
    _complete_schema(conn)
    m030.apply(conn)
    conn.execute("DROP INDEX idx_entities_profile_fact_count_id")

    assert m030.verify(conn) is False
    m030.repair(conn)
    assert m030.verify(conn) is True
