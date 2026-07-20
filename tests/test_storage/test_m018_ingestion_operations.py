# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Contract tests for the additive M018 ingestion-operation schema."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from superlocalmemory.storage.migrations import M018_ingestion_operations as migration


def _columns(conn: sqlite3.Connection) -> set[str]:
    return {
        row[1]
        for row in conn.execute("PRAGMA table_info(ingestion_operations)").fetchall()
    }


def test_m018_creates_durable_ingestion_operation_contract() -> None:
    conn = sqlite3.connect(":memory:")
    migration.apply(conn)

    assert {
        "operation_id",
        "profile_id",
        "source_type",
        "idempotency_key",
        "source_hash",
        "raw_content",
        "raw_metadata_json",
        "scope",
        "shared_with_json",
        "trusted_actor_id",
        "session_id",
        "session_date",
        "speaker",
        "role",
        "lease_owner",
        "lease_expires_at",
        "next_retry_at",
        "state",
        "queryable_fact_ids_json",
        "final_fact_ids_json",
        "derivation_version",
        "derivation_state_json",
        "attempt_count",
        "last_error",
        "created_at",
        "updated_at",
    } <= _columns(conn)
    assert migration.verify(conn) is True


def test_m018_is_idempotent_and_enforces_profile_scoped_idempotency() -> None:
    conn = sqlite3.connect(":memory:")
    migration.apply(conn)
    migration.apply(conn)

    values = (
        "op-1",
        "profile-a",
        "mcp",
        "same-key",
        "hash-a",
        "evidence",
    )
    conn.execute(
        "INSERT INTO ingestion_operations "
        "(operation_id, profile_id, source_type, idempotency_key, source_hash, raw_content) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        values,
    )
    with __import__("pytest").raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO ingestion_operations "
            "(operation_id, profile_id, source_type, idempotency_key, source_hash, raw_content) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("op-2", "profile-a", "mcp", "same-key", "hash-a", "evidence"),
        )

    conn.execute(
        "INSERT INTO ingestion_operations "
        "(operation_id, profile_id, source_type, idempotency_key, source_hash, raw_content) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("op-3", "profile-b", "mcp", "same-key", "hash-a", "evidence"),
    )


def test_m018_rejects_unknown_materialization_state() -> None:
    conn = sqlite3.connect(":memory:")
    migration.apply(conn)

    with __import__("pytest").raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO ingestion_operations "
            "(operation_id, profile_id, source_type, idempotency_key, source_hash, raw_content, state) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("op-1", "default", "cli", "k", "h", "evidence", "done-ish"),
        )


def test_m018_is_registered_in_forward_migration_runner(tmp_path: Path) -> None:
    from superlocalmemory.storage import migration_runner

    registered = {item.name for item in migration_runner.MIGRATIONS}
    assert migration.NAME in registered

    learning_db = tmp_path / "learning.db"
    memory_db = tmp_path / "memory.db"
    result = migration_runner.apply_all(learning_db, memory_db)

    assert migration.NAME in result["applied"]
    conn = sqlite3.connect(memory_db)
    try:
        assert migration.verify(conn) is True
    finally:
        conn.close()

    second = migration_runner.apply_all(learning_db, memory_db)
    assert migration.NAME in second["skipped"]
