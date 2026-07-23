# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""M030 — page-first Entity Explorer indexes for large installations."""

from __future__ import annotations

import sqlite3

NAME = "M030_entity_explorer_indexes"
DB_TARGET = "memory"

DDL = """
CREATE INDEX IF NOT EXISTS idx_entities_profile_fact_count_id
    ON canonical_entities(profile_id, fact_count DESC, entity_id ASC);
CREATE INDEX IF NOT EXISTS idx_entities_profile_type_fact_count_id
    ON canonical_entities(
        profile_id, entity_type COLLATE NOCASE, fact_count DESC, entity_id ASC
    );
CREATE INDEX IF NOT EXISTS idx_entity_profiles_profile_entity_rank
    ON entity_profiles(
        profile_id, entity_id, last_compiled_at DESC,
        project_name COLLATE NOCASE
    );
"""

_INDEXES: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    (
        "idx_entities_profile_fact_count_id",
        "canonical_entities",
        ("profile_id", "fact_count", "entity_id"),
        "CREATE INDEX IF NOT EXISTS idx_entities_profile_fact_count_id "
        "ON canonical_entities(profile_id, fact_count DESC, entity_id ASC)",
    ),
    (
        "idx_entities_profile_type_fact_count_id",
        "canonical_entities",
        ("profile_id", "entity_type", "fact_count", "entity_id"),
        "CREATE INDEX IF NOT EXISTS idx_entities_profile_type_fact_count_id "
        "ON canonical_entities("
        "profile_id, entity_type COLLATE NOCASE, fact_count DESC, entity_id ASC"
        ")",
    ),
    (
        "idx_entity_profiles_profile_entity_rank",
        "entity_profiles",
        (
            "profile_id",
            "entity_id",
            "last_compiled_at",
            "project_name",
        ),
        "CREATE INDEX IF NOT EXISTS idx_entity_profiles_profile_entity_rank "
        "ON entity_profiles("
        "profile_id, entity_id, last_compiled_at DESC, "
        "project_name COLLATE NOCASE"
        ")",
    ),
)


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }


def apply(conn: sqlite3.Connection) -> None:
    """Create every index supported by the installed optional schema."""
    for _, table, required_columns, sql in _INDEXES:
        if set(required_columns) <= _columns(conn, table):
            conn.execute(sql)


def verify(conn: sqlite3.Connection) -> bool:
    """Require indexes only when their complete backing schema exists."""
    names = {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
    }
    return all(
        name in names
        for name, table, required_columns, _ in _INDEXES
        if set(required_columns) <= _columns(conn, table)
    )


def repair(conn: sqlite3.Connection) -> None:
    """Safely restore a dropped index without replaying data migrations."""
    apply(conn)
