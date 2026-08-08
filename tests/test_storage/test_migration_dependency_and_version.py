# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""Migration runner: downgrade guard on the deferred pass + dependency gating.

The deferred pass runs after engine init and writes DDL, so it must refuse a
database stamped by a newer build exactly as the up-front pass does. And a
migration whose declared dependency did not complete must be held back rather
than run against a base schema it assumes is present.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import superlocalmemory.storage.migration_runner as mr
from superlocalmemory.storage._migration_internals import Migration
from superlocalmemory.storage._schema_version import (
    SUPPORTED_SCHEMA_VERSION,
    SchemaVersionError,
    ensure_schema_version_table,
    read_schema_version,
    write_schema_version,
)


def test_v4_ports_mainline_migrations_under_unique_serials() -> None:
    """Main's M033/M034 invariants must not collide with V4's M033-M037."""
    eager_names = [migration.name for migration in mr.MIGRATIONS]
    deferred_names = [migration.name for migration in mr.DEFERRED_MIGRATIONS]
    all_names = eager_names + deferred_names

    assert "M038_learning_feedback_channel" in eager_names
    assert "M039_scene_fact_members" in deferred_names
    assert len(all_names) == len(set(all_names))
    targets = {
        migration.name: migration.db_target
        for migration in (*mr.MIGRATIONS, *mr.DEFERRED_MIGRATIONS)
    }
    assert targets["M033_projection_transactions"] == "memory"
    assert targets["M038_learning_feedback_channel"] == "learning"
    assert SUPPORTED_SCHEMA_VERSION == 39


def test_schema_39_is_stamped_only_after_m039_completes(tmp_path: Path) -> None:
    """A version-39 marker must prove the normalized scene schema exists."""
    from superlocalmemory.storage import schema

    learning_db = tmp_path / "learning.db"
    memory_db = tmp_path / "memory.db"
    with sqlite3.connect(memory_db) as conn:
        schema.create_all_tables(conn)

    eager = mr.apply_all(learning_db, memory_db)
    assert eager["failed"] == []
    assert read_schema_version(learning_db) < SUPPORTED_SCHEMA_VERSION
    assert read_schema_version(memory_db) < SUPPORTED_SCHEMA_VERSION

    deferred = mr.apply_deferred(learning_db, memory_db)
    assert deferred["failed"] == []
    assert read_schema_version(learning_db) == SUPPORTED_SCHEMA_VERSION
    assert read_schema_version(memory_db) == SUPPORTED_SCHEMA_VERSION
    with sqlite3.connect(memory_db) as conn:
        assert conn.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type='table' AND name='scene_fact_members'"
        ).fetchone() == (1,)


def test_apply_deferred_refuses_newer_schema(tmp_path: Path) -> None:
    learning_db = tmp_path / "learning.db"
    memory_db = tmp_path / "memory.db"
    sqlite3.connect(learning_db).close()
    with sqlite3.connect(memory_db) as conn:
        ensure_schema_version_table(conn)
        write_schema_version(conn, SUPPORTED_SCHEMA_VERSION + 5)

    with pytest.raises(SchemaVersionError):
        mr.apply_deferred(learning_db, memory_db)


def test_dependent_migration_skipped_when_dependency_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    learning_db = tmp_path / "learning.db"
    memory_db = tmp_path / "memory.db"
    sqlite3.connect(learning_db).close()
    sqlite3.connect(memory_db).close()

    dep = Migration(
        name="ZZZ_dep_fails", db_target="memory", ddl="THIS IS NOT VALID SQL;"
    )
    child = Migration(
        name="ZZZ_child",
        db_target="memory",
        ddl="CREATE TABLE zzz_child (id INTEGER);",
        dependencies=("ZZZ_dep_fails",),
    )
    monkeypatch.setattr(mr, "MIGRATIONS", [dep, child])

    stats = mr.apply_all(learning_db, memory_db)

    assert "ZZZ_dep_fails" in stats["failed"]
    assert "ZZZ_child" in stats["skipped"]
    assert "dependency not satisfied" in stats["details"]["ZZZ_child"]
    # The dependent's DDL must not have run.
    with sqlite3.connect(memory_db) as conn:
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert "zzz_child" not in tables


def test_version_singleton_coexists_with_legacy_schema_version(tmp_path: Path) -> None:
    """Regression: every real (3.x) database already carries the legacy multi-row
    ``schema_version`` history table (columns version/applied_at/description).  The
    version-ceiling singleton must create/read/write its own marker without
    colliding with that legacy table — previously it failed with
    "table schema_version has no column named id" on every migrated DB.
    """
    memory_db = tmp_path / "memory.db"
    with sqlite3.connect(memory_db) as conn:
        # Legacy history table exactly as schema.py / schema_v34x create it.
        conn.execute(
            "CREATE TABLE schema_version ("
            " version TEXT NOT NULL,"
            " applied_at TEXT NOT NULL DEFAULT (datetime('now')),"
            " description TEXT NOT NULL DEFAULT '')"
        )
        conn.execute(
            "INSERT INTO schema_version (version, description) VALUES ('3.4.11', 'legacy')"
        )
        conn.commit()

    with sqlite3.connect(memory_db) as conn:
        ensure_schema_version_table(conn)
        write_schema_version(conn, SUPPORTED_SCHEMA_VERSION)
        conn.commit()

    # The singleton guard now reads back the stamped ceiling (was 0 before the fix).
    assert read_schema_version(memory_db) == SUPPORTED_SCHEMA_VERSION

    # The legacy history table is untouched and still queryable.
    with sqlite3.connect(memory_db) as conn:
        legacy = {r[0] for r in conn.execute("SELECT version FROM schema_version")}
    assert "3.4.11" in legacy
