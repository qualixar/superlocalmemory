# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""O(1) idempotency contract for ingestion fact/entity observations."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from superlocalmemory.storage.migrations import (
    M028_fact_entity_associations as migration,
)


def _base_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE profiles (
            profile_id TEXT PRIMARY KEY
        );
        CREATE TABLE atomic_facts (
            fact_id TEXT PRIMARY KEY,
            profile_id TEXT NOT NULL,
            canonical_entities_json TEXT NOT NULL DEFAULT '[]'
        );
        CREATE TABLE canonical_entities (
            entity_id TEXT PRIMARY KEY,
            profile_id TEXT NOT NULL,
            fact_count INTEGER NOT NULL DEFAULT 0
        );
        """
    )


def test_m028_schema_is_complete_while_backfill_truth_remains_pending(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "memory.db"
    conn = sqlite3.connect(db_path)
    _base_tables(conn)
    conn.execute("INSERT INTO profiles VALUES ('work')")
    conn.execute(
        "INSERT INTO canonical_entities VALUES ('alice', 'work', 7)"
    )
    conn.execute(
        "INSERT INTO atomic_facts VALUES ('fact-1', 'work', '[\"alice\"]')"
    )

    conn.commit()
    migration.apply(conn)
    assert migration.verify(conn) is True
    conn.close()

    assert migration.get_repair_status(db_path)["state"] == "pending"


def test_m028_backfill_is_chunked_restartable_and_preserves_counts(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "memory.db"
    conn = sqlite3.connect(db_path)
    _base_tables(conn)
    conn.execute("INSERT INTO profiles VALUES ('work')")
    conn.execute("INSERT INTO canonical_entities VALUES ('alice', 'work', 7)")
    conn.executemany(
        "INSERT INTO atomic_facts VALUES (?, 'work', '[\"alice\"]')",
        [("fact-1",), ("fact-2",), ("fact-3",)],
    )
    conn.commit()
    migration.apply(conn)
    conn.close()

    first = migration.repair_fact_entity_associations(
        db_path, batch_size=1, max_batches=1,
    )
    second = migration.repair_fact_entity_associations(
        db_path, batch_size=1, max_batches=1,
    )
    third = migration.repair_fact_entity_associations(
        db_path, batch_size=1, max_batches=1,
    )
    complete = migration.repair_fact_entity_associations(
        db_path, batch_size=1, max_batches=1,
    )

    assert first == {"scanned": 1, "inserted": 1, "complete": False}
    assert second == {"scanned": 1, "inserted": 1, "complete": False}
    assert third == {"scanned": 1, "inserted": 1, "complete": False}
    assert complete == {"scanned": 0, "inserted": 0, "complete": True}
    status = migration.get_repair_status(db_path)
    assert status["state"] == "complete"
    assert status["last_fact_rowid"] == 3
    assert status["scanned"] == 3
    assert status["inserted"] == 3
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM fact_entity_associations"
        ).fetchone()[0] == 3
        assert conn.execute(
            "SELECT fact_count FROM canonical_entities WHERE entity_id='alice'"
        ).fetchone()[0] == 7
    finally:
        conn.close()


def test_m028_lookup_plan_uses_composite_primary_key() -> None:
    conn = sqlite3.connect(":memory:")
    _base_tables(conn)
    migration.apply(conn)

    plan = " ".join(
        str(column)
        for row in conn.execute(
            "EXPLAIN QUERY PLAN SELECT 1 FROM fact_entity_associations "
            "WHERE profile_id=? AND fact_id=? AND entity_id=?",
            ("work", "fact-1", "alice"),
        )
        for column in row
    )

    assert "USING" in plan
    assert "INDEX" in plan
    assert "SCAN fact_entity_associations" not in plan


def test_m028_repair_restores_dropped_index_without_recapturing_boundary() -> None:
    """Repair schema drift without broadening the captured historical scan."""
    conn = sqlite3.connect(":memory:")
    _base_tables(conn)
    conn.execute("INSERT INTO profiles VALUES ('work')")
    conn.execute(
        "INSERT INTO atomic_facts VALUES ('fact-1', 'work', '[\"alice\"]')"
    )
    migration.apply(conn)
    captured_boundary = conn.execute(
        "SELECT target_fact_rowid FROM fact_entity_association_repair_state "
        "WHERE repair_key='historical-backfill'"
    ).fetchone()[0]

    conn.execute(
        "DROP INDEX idx_fact_entity_associations_entity"
    )
    migration.repair(conn)

    repaired_boundary = conn.execute(
        "SELECT target_fact_rowid FROM fact_entity_association_repair_state "
        "WHERE repair_key='historical-backfill'"
    ).fetchone()[0]
    assert migration.verify(conn) is True
    assert repaired_boundary == captured_boundary


def test_m028_is_registered_in_forward_migration_runner(tmp_path: Path) -> None:
    from superlocalmemory.storage import migration_runner, schema
    from superlocalmemory.storage.database import DatabaseManager

    registered = {item.name for item in migration_runner.DEFERRED_MIGRATIONS}
    assert migration.NAME in registered

    learning_db = tmp_path / "learning.db"
    memory_db = tmp_path / "memory.db"
    pre_engine = migration_runner.apply_all(learning_db, memory_db)
    assert migration.NAME not in pre_engine["failed"]
    assert migration.NAME not in pre_engine["applied"]

    DatabaseManager(memory_db).initialize(schema)
    post_engine = migration_runner.apply_deferred(learning_db, memory_db)
    assert migration.NAME in post_engine["applied"]


def test_post_readiness_scheduler_publishes_durable_completion(
    tmp_path: Path,
) -> None:
    from superlocalmemory.server.unified_daemon import (
        _schedule_fact_entity_association_repair,
    )

    db_path = tmp_path / "memory.db"
    conn = sqlite3.connect(db_path)
    _base_tables(conn)
    migration.apply(conn)
    conn.close()
    application = SimpleNamespace(state=SimpleNamespace())

    async def run() -> None:
        task = _schedule_fact_entity_association_repair(
            application, db_path, batch_size=1, tick_seconds=0,
        )
        await task

    asyncio.run(run())

    status = application.state.fact_entity_association_repair_status
    assert status["state"] == "complete"
    assert status["source"] == "startup_background_repair"
    assert migration.get_repair_status(db_path)["state"] == "complete"


def test_post_readiness_scheduler_retries_transient_sqlite_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from superlocalmemory.server.unified_daemon import (
        _schedule_fact_entity_association_repair,
    )

    db_path = tmp_path / "memory.db"
    conn = sqlite3.connect(db_path)
    _base_tables(conn)
    migration.apply(conn)
    conn.close()
    application = SimpleNamespace(state=SimpleNamespace())
    actual_repair = migration.repair_fact_entity_associations
    attempts = 0

    def fail_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise sqlite3.OperationalError("database is temporarily locked")
        return actual_repair(*args, **kwargs)

    monkeypatch.setattr(
        migration, "repair_fact_entity_associations", fail_once,
    )

    async def run() -> None:
        task = _schedule_fact_entity_association_repair(
            application, db_path, batch_size=1, tick_seconds=0,
        )
        await task

    asyncio.run(run())

    assert attempts == 2
    status = application.state.fact_entity_association_repair_status
    assert status["state"] == "complete"
    assert status["last_error"] == ""
    assert migration.get_repair_status(db_path)["state"] == "complete"
