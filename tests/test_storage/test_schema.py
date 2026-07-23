# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.storage.schema — DDL, indexes, FTS5, FKs.

Covers:
  - create_all_tables creates all 17 regular + 1 FTS table
  - FTS5 virtual table exists and is searchable
  - Indexes exist on hot query paths
  - Foreign key CASCADE delete works
  - profile_id column exists on all data tables
  - Schema version seeded on first run
  - drop_all_tables cleans up
  - get_table_names / get_fts_table_names return correct values
"""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.storage.schema import (
    SCHEMA_VERSION,
    _TABLES,
    _FTS_TABLES,
    create_all_tables,
    drop_all_tables,
    get_fts_table_names,
    get_table_names,
)


@pytest.fixture()
def conn() -> sqlite3.Connection:
    """In-memory SQLite database with schema applied."""
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    create_all_tables(c)
    c.commit()
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------

class TestCreateAllTables:
    def test_all_regular_tables_exist(self, conn: sqlite3.Connection) -> None:
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        for table_name in _TABLES:
            assert table_name in tables, f"Table '{table_name}' missing"

    def test_fts5_virtual_table_exists(self, conn: sqlite3.Connection) -> None:
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        for fts_name in _FTS_TABLES:
            assert fts_name in tables, f"FTS table '{fts_name}' missing"

    def test_idempotent_creation(self, conn: sqlite3.Connection) -> None:
        """Calling create_all_tables twice must not error."""
        create_all_tables(conn)
        conn.commit()
        # If we get here without error, the IF NOT EXISTS works.

    def test_fts_triggers_are_idempotent(self) -> None:
        """FTS trigger DDL must tolerate concurrent/repeated schema init."""
        from superlocalmemory.storage import schema

        assert "CREATE TRIGGER IF NOT EXISTS atomic_facts_fts_insert" in (
            schema._SQL_ATOMIC_FACTS_FTS
        )
        assert "CREATE TRIGGER IF NOT EXISTS atomic_facts_fts_delete" in (
            schema._SQL_ATOMIC_FACTS_FTS
        )
        assert "CREATE TRIGGER IF NOT EXISTS atomic_facts_fts_update" in (
            schema._SQL_ATOMIC_FACTS_FTS
        )

    def test_schema_version_seeded(self, conn: sqlite3.Connection) -> None:
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        assert row is not None
        assert row["version"] == SCHEMA_VERSION

    def test_default_profile_exists(self, conn: sqlite3.Connection) -> None:
        row = conn.execute(
            "SELECT * FROM profiles WHERE profile_id = 'default'"
        ).fetchone()
        assert row is not None
        assert row["name"] == "Default Profile"


# ---------------------------------------------------------------------------
# Profile ID on all data tables
# ---------------------------------------------------------------------------

class TestProfileIdColumn:
    """Every data table (except schema_version) must have profile_id."""

    _TABLES_WITHOUT_PROFILE_ID = {
        "schema_version",
        "entity_aliases",
        "config",
        # One durable cursor scans the shared atomic_facts rowid space while
        # every materialized association remains profile-scoped.
        "fact_entity_association_repair_state",
    }

    def test_profile_id_on_data_tables(self, conn: sqlite3.Connection) -> None:
        for table_name in _TABLES:
            if table_name in self._TABLES_WITHOUT_PROFILE_ID:
                continue
            cols = {
                row["name"]
                for row in conn.execute(
                    f"PRAGMA table_info({table_name})"
                ).fetchall()
            }
            assert "profile_id" in cols, (
                f"Table '{table_name}' missing profile_id column"
            )


# ---------------------------------------------------------------------------
# Indexes
# ---------------------------------------------------------------------------

class TestIndexes:
    def test_key_indexes_exist(self, conn: sqlite3.Connection) -> None:
        index_names = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        expected_indexes = [
            "idx_memories_profile",
            "idx_memories_session",
            "idx_facts_profile",
            "idx_facts_type",
            "idx_facts_lifecycle",
            "idx_entities_profile",
            "idx_entities_name_lower",
            "idx_edges_profile",
            "idx_edges_source",
            "idx_edges_target",
            "idx_tevents_profile",
            "idx_tevents_entity",
            "idx_trust_profile",
            "idx_audit_profile",
        ]
        for idx in expected_indexes:
            assert idx in index_names, f"Index '{idx}' missing"


# ---------------------------------------------------------------------------
# Foreign key CASCADE delete
# ---------------------------------------------------------------------------

class TestForeignKeyCascade:
    def test_delete_profile_cascades_to_memories(
        self, conn: sqlite3.Connection
    ) -> None:
        # Insert a non-default profile
        conn.execute(
            "INSERT INTO profiles (profile_id, name) VALUES ('p1', 'Test')"
        )
        conn.execute(
            "INSERT INTO memories "
            "(memory_id, profile_id, content) "
            "VALUES ('m1', 'p1', 'hello')"
        )
        conn.commit()

        # Delete the profile
        conn.execute("DELETE FROM profiles WHERE profile_id = 'p1'")
        conn.commit()

        # Memory should be gone
        row = conn.execute(
            "SELECT * FROM memories WHERE memory_id = 'm1'"
        ).fetchone()
        assert row is None, "CASCADE delete did not remove child memory"

    def test_delete_profile_cascades_to_facts(
        self, conn: sqlite3.Connection
    ) -> None:
        conn.execute(
            "INSERT INTO profiles (profile_id, name) VALUES ('p2', 'Test2')"
        )
        conn.execute(
            "INSERT INTO memories "
            "(memory_id, profile_id, content) "
            "VALUES ('m2', 'p2', 'test')"
        )
        conn.execute(
            "INSERT INTO atomic_facts "
            "(fact_id, memory_id, profile_id, content) "
            "VALUES ('f1', 'm2', 'p2', 'some fact')"
        )
        conn.commit()

        conn.execute("DELETE FROM profiles WHERE profile_id = 'p2'")
        conn.commit()

        row = conn.execute(
            "SELECT * FROM atomic_facts WHERE fact_id = 'f1'"
        ).fetchone()
        assert row is None

    def test_delete_memory_cascades_to_facts(
        self, conn: sqlite3.Connection
    ) -> None:
        conn.execute(
            "INSERT INTO memories "
            "(memory_id, profile_id, content) "
            "VALUES ('m3', 'default', 'test')"
        )
        conn.execute(
            "INSERT INTO atomic_facts "
            "(fact_id, memory_id, profile_id, content) "
            "VALUES ('f2', 'm3', 'default', 'fact text')"
        )
        conn.commit()

        conn.execute("DELETE FROM memories WHERE memory_id = 'm3'")
        conn.commit()

        row = conn.execute(
            "SELECT * FROM atomic_facts WHERE fact_id = 'f2'"
        ).fetchone()
        assert row is None


# ---------------------------------------------------------------------------
# FTS5 triggers (sync)
# ---------------------------------------------------------------------------

class TestFTS5Triggers:
    def test_insert_trigger_populates_fts(
        self, conn: sqlite3.Connection
    ) -> None:
        conn.execute(
            "INSERT INTO memories "
            "(memory_id, profile_id, content) "
            "VALUES ('m_fts', 'default', 'fts test')"
        )
        conn.execute(
            "INSERT INTO atomic_facts "
            "(fact_id, memory_id, profile_id, content) "
            "VALUES ('f_fts', 'm_fts', 'default', 'Alice loves hiking mountains')"
        )
        conn.commit()

        # Search FTS
        rows = conn.execute(
            "SELECT fact_id FROM atomic_facts_fts WHERE atomic_facts_fts MATCH 'hiking'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["fact_id"] == "f_fts"

    def test_delete_trigger_removes_from_fts(
        self, conn: sqlite3.Connection
    ) -> None:
        conn.execute(
            "INSERT INTO memories "
            "(memory_id, profile_id, content) "
            "VALUES ('m_del', 'default', 'del test')"
        )
        conn.execute(
            "INSERT INTO atomic_facts "
            "(fact_id, memory_id, profile_id, content) "
            "VALUES ('f_del', 'm_del', 'default', 'Bob hates swimming')"
        )
        conn.commit()

        conn.execute("DELETE FROM atomic_facts WHERE fact_id = 'f_del'")
        conn.commit()

        rows = conn.execute(
            "SELECT fact_id FROM atomic_facts_fts WHERE atomic_facts_fts MATCH 'swimming'"
        ).fetchall()
        assert len(rows) == 0


# ---------------------------------------------------------------------------
# Drop + utility functions
# ---------------------------------------------------------------------------

class TestDropAllTables:
    def test_drop_removes_all_tables(self, conn: sqlite3.Connection) -> None:
        drop_all_tables(conn)
        conn.commit()
        remaining = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        assert len(remaining) == 0


class TestUtilityFunctions:
    def test_get_table_names(self) -> None:
        names = get_table_names()
        assert isinstance(names, tuple)
        assert "profiles" in names
        assert "atomic_facts" in names
        assert len(names) == len(_TABLES)

    def test_get_fts_table_names(self) -> None:
        names = get_fts_table_names()
        assert isinstance(names, tuple)
        assert "atomic_facts_fts" in names
