# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Typed-command contracts for the canonical writer."""

from __future__ import annotations

import json
import sqlite3

import pytest


def _install_write_commits(db_path) -> None:
    from superlocalmemory.storage.migrations import M032_write_coordinator_admission

    conn = sqlite3.connect(db_path)
    try:
        M032_write_coordinator_admission.apply(conn)
        conn.commit()
    finally:
        conn.close()


def _admission_payload(name: str) -> dict[str, str]:
    return {
        "journal_id": f"journal:{name}",
        "request_hash": f"hash:{name}",
        "profile_id": "default",
        "idempotency_key": f"idempotency:{name}",
    }


def test_typed_command_commits_handler_mutation_and_immutable_receipt(tmp_path) -> None:
    """A handler runs in one BEGIN IMMEDIATE transaction with its receipt."""
    from superlocalmemory.storage.write_coordinator import (
        CommandKind,
        WriteCommand,
        WriteCoordinator,
        WriteResult,
    )

    db_path = tmp_path / "memory.db"
    _install_write_commits(db_path)
    coordinator = WriteCoordinator(db_path, owner_id="typed-command")
    assert coordinator.claim_ownership()
    try:

        def admit(conn, _capability, command):
            conn.execute("INSERT INTO admissions(value) VALUES (?)", (command.payload["value"],))
            return WriteResult.from_receipt(
                command,
                {"operation_id": "operation:first", "stored": command.payload["value"]},
            )

        coordinator.register_handler(CommandKind.ADMISSION, admit)
        coordinator.execute("CREATE TABLE admissions (value TEXT NOT NULL)")
        command = WriteCommand.create(
            CommandKind.ADMISSION,
            {**_admission_payload("first"), "value": "first"},
        )
        result = coordinator.submit(command, timeout=0.5)

        assert result.command_id == command.command_id
        assert result.kind is CommandKind.ADMISSION
        assert result.receipt == {
            "commit_sequence": 1,
            "operation_id": "operation:first",
            "stored": "first",
        }
        with pytest.raises(TypeError):
            result.receipt["stored"] = "mutated"  # type: ignore[index]

        row = coordinator.execute(
            "SELECT command_kind, receipt_json FROM write_commits WHERE command_id = ?",
            (command.command_id,),
        )[0]
        assert tuple(row) == (
            "admission",
            '{"commit_sequence":1,"operation_id":"operation:first","stored":"first"}',
        )
        assert coordinator.execute("SELECT value FROM admissions")[0][0] == "first"
    finally:
        coordinator.release_ownership()


def test_typed_command_replays_prior_receipt_without_reexecuting_handler(tmp_path) -> None:
    """Command ids are durable idempotency keys, not a caller-side convention."""
    from superlocalmemory.storage.write_coordinator import (
        CommandKind,
        WriteCommand,
        WriteCoordinator,
        WriteResult,
    )

    db_path = tmp_path / "memory.db"
    _install_write_commits(db_path)
    coordinator = WriteCoordinator(db_path)
    assert coordinator.claim_ownership()
    calls = 0
    try:

        def handler(_conn, _capability, command):
            nonlocal calls
            calls += 1
            return WriteResult.from_receipt(
                command,
                {"call": calls, "operation_id": "operation:replay"},
            )

        coordinator.register_handler(CommandKind.ADMISSION, handler)
        command = WriteCommand.create(
            CommandKind.ADMISSION,
            {**_admission_payload("replay"), "operation": "remember"},
        )
        expected = {
            "call": 1,
            "commit_sequence": 1,
            "operation_id": "operation:replay",
        }
        assert coordinator.submit(command, timeout=0.5).receipt == expected
        assert coordinator.submit(command, timeout=0.5).receipt == expected
        assert calls == 1
    finally:
        coordinator.release_ownership()


def test_typed_command_rolls_back_handler_mutation_when_receipt_cannot_commit(tmp_path) -> None:
    """There is no acknowledged admission without its durable receipt."""
    from superlocalmemory.storage.write_coordinator import (
        CommandKind,
        WriteCommand,
        WriteCoordinator,
        WriteCoordinatorError,
        WriteResult,
    )

    db_path = tmp_path / "memory.db"
    _install_write_commits(db_path)
    coordinator = WriteCoordinator(db_path)
    assert coordinator.claim_ownership()
    try:

        def malformed_receipt(conn, _capability, command):
            conn.execute("INSERT INTO admissions(value) VALUES ('orphan')")
            return WriteResult.from_receipt(
                command,
                {"bad": {1, 2}, "operation_id": "operation:malformed"},
            )

        coordinator.register_handler(CommandKind.ADMISSION, malformed_receipt)
        coordinator.execute("CREATE TABLE admissions (value TEXT NOT NULL)")
        with pytest.raises(WriteCoordinatorError, match="failed"):
            coordinator.submit(
                WriteCommand.create(
                    CommandKind.ADMISSION,
                    _admission_payload("malformed"),
                ),
                timeout=0.5,
            )
        assert coordinator.execute("SELECT COUNT(*) FROM admissions")[0][0] == 0
    finally:
        coordinator.release_ownership()


def test_typed_command_rejects_memory_text_in_immutable_receipt(tmp_path) -> None:
    """A future handler cannot retain deleted memory text in the audit ledger."""
    from superlocalmemory.storage.write_coordinator import (
        CommandKind,
        WriteCommand,
        WriteCoordinator,
        WriteCoordinatorError,
        WriteResult,
    )

    db_path = tmp_path / "memory.db"
    _install_write_commits(db_path)
    coordinator = WriteCoordinator(db_path)
    assert coordinator.claim_ownership()
    try:

        def leaking_receipt(conn, _capability, command):
            conn.execute("INSERT INTO admissions(value) VALUES ('must-roll-back')")
            return WriteResult.from_receipt(
                command,
                {
                    "operation_id": "operation:leaking-receipt",
                    "content_preview": "SECRET-RECEIPT-WITNESS",
                },
            )

        coordinator.register_handler(CommandKind.ADMISSION, leaking_receipt)
        coordinator.execute("CREATE TABLE admissions (value TEXT NOT NULL)")
        with pytest.raises(WriteCoordinatorError, match="metadata only"):
            coordinator.submit(
                WriteCommand.create(
                    CommandKind.ADMISSION,
                    _admission_payload("leaking-receipt"),
                ),
                timeout=0.5,
            )
        assert coordinator.execute("SELECT COUNT(*) FROM admissions")[0][0] == 0
        assert coordinator.execute("SELECT COUNT(*) FROM write_commits")[0][0] == 0
    finally:
        coordinator.release_ownership()


def test_receipt_ledger_rejects_update_and_delete_after_commit(tmp_path) -> None:
    """The durable idempotency ledger cannot be rewritten after acknowledgement."""
    from superlocalmemory.storage.write_coordinator import (
        CommandKind,
        WriteCommand,
        WriteCoordinator,
        WriteCoordinatorError,
        WriteResult,
    )

    db_path = tmp_path / "memory.db"
    _install_write_commits(db_path)
    coordinator = WriteCoordinator(db_path)
    assert coordinator.claim_ownership()
    try:
        coordinator.register_handler(
            CommandKind.ADMISSION,
            lambda _conn, _capability, command: WriteResult.from_receipt(
                command,
                {"accepted": True, "operation_id": "operation:immutable"},
            ),
        )
        result = coordinator.submit(
            WriteCommand.create(
                CommandKind.ADMISSION,
                _admission_payload("immutable"),
            ),
            timeout=0.5,
        )
        with pytest.raises(WriteCoordinatorError) as update_error:
            coordinator.execute(
                "UPDATE write_commits SET receipt_json = '{}' WHERE command_id = ?",
                (result.command_id,),
            )
        assert update_error.value.__cause__ is not None
        assert "immutable" in str(update_error.value.__cause__)
        with pytest.raises(WriteCoordinatorError) as delete_error:
            coordinator.execute(
                "DELETE FROM write_commits WHERE command_id = ?",
                (result.command_id,),
            )
        assert delete_error.value.__cause__ is not None
        assert "immutable" in str(delete_error.value.__cause__)
    finally:
        coordinator.release_ownership()


def test_m032_is_applied_by_the_startup_migration_runner(tmp_path) -> None:
    """An upgrade gets the receipt ledger before the daemon can accept writes."""
    from superlocalmemory.storage.migration_runner import apply_all
    from superlocalmemory.storage.migrations import M032_write_coordinator_admission

    memory_db = tmp_path / "memory.db"
    result = apply_all(tmp_path / "learning.db", memory_db)
    assert M032_write_coordinator_admission.NAME in result["applied"]
    conn = sqlite3.connect(memory_db)
    try:
        assert M032_write_coordinator_admission.verify(conn)
    finally:
        conn.close()


def test_pre_m032_profile_data_survives_migrate_start_and_restart(tmp_path) -> None:
    """Representative existing facts/FTS/operations survive the 3.8.6 upgrade."""
    from superlocalmemory.core.engine_ingestion import build_immediate_admission_handler
    from superlocalmemory.core.ingestion_command import (
        IngestionOperationRepository,
        IngestionRequest,
        IngestionState,
    )
    from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage.migration_runner import apply_all
    from superlocalmemory.storage.migrations import M018_ingestion_operations
    from superlocalmemory.storage.models import AtomicFact, MemoryRecord

    memory_db = tmp_path / "memory.db"
    learning_db = tmp_path / "learning.db"
    db = DatabaseManager(memory_db)
    db.initialize(schema)
    with db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='write_commits'"
        ).fetchone() is None

    expected: dict[str, tuple[str, str, str]] = {}
    repository = IngestionOperationRepository(db)
    for profile_id, token in (
        ("mode-a", "prem032alpha"),
        ("mode-b", "prem032bravo"),
        ("mode-c", "prem032charlie"),
    ):
        db.execute(
            "INSERT INTO profiles(profile_id, name, description) VALUES (?, ?, ?)",
            (profile_id, profile_id, f"legacy {profile_id} profile"),
        )
        memory_id = f"legacy-memory-{profile_id}"
        fact_id = f"legacy-fact-{profile_id}"
        content = f"{token} existing profile evidence must survive migration."
        db.store_memory(MemoryRecord(
            memory_id=memory_id,
            profile_id=profile_id,
            content=content,
            metadata={"legacy": True, "profile": profile_id},
        ))
        db.store_fact(AtomicFact(
            fact_id=fact_id,
            memory_id=memory_id,
            profile_id=profile_id,
            content=content,
            entities=[f"Entity-{profile_id}"],
        ))
        operation = repository.create(IngestionRequest(
            content=content,
            profile_id=profile_id,
            source_type="legacy-mcp",
            idempotency_key=f"legacy-operation:{profile_id}",
            metadata={"legacy": True},
        ))
        repository.transition(
            operation.operation_id,
            expected=IngestionState.RAW,
            target=IngestionState.QUERYABLE,
            queryable_fact_ids=(fact_id,),
        )
        expected[profile_id] = (memory_id, fact_id, operation.operation_id)

    before_counts = tuple(db.execute(
        "SELECT "
        "(SELECT COUNT(*) FROM profiles), "
        "(SELECT COUNT(*) FROM memories), "
        "(SELECT COUNT(*) FROM atomic_facts), "
        "(SELECT COUNT(*) FROM ingestion_operations)"
    )[0])
    before_rows = [
        tuple(row)
        for row in db.execute(
            "SELECT fact_id, memory_id, profile_id, content, entities_json "
            "FROM atomic_facts ORDER BY profile_id, fact_id"
        )
    ]

    first_migration = apply_all(learning_db, memory_db)
    assert first_migration["failed"] == []
    assert apply_all(learning_db, memory_db)["failed"] == []

    upgraded = DatabaseManager(memory_db)
    runtime = CanonicalRememberRuntime(
        db=upgraded,
        profile_id="mode-a",
        writer=build_immediate_admission_handler(upgraded, profile_id="mode-a"),
        journal_path=tmp_path / "admission_journal.db",
        owner_id="pre-m032-first-start",
    )
    runtime.start()
    runtime.stop()
    replacement_db = DatabaseManager(memory_db)
    replacement = CanonicalRememberRuntime(
        db=replacement_db,
        profile_id="mode-a",
        writer=build_immediate_admission_handler(
            replacement_db,
            profile_id="mode-a",
        ),
        journal_path=tmp_path / "admission_journal.db",
        owner_id="pre-m032-restart",
    )
    replacement.start()
    replacement.stop()

    verified = DatabaseManager(memory_db)
    after_counts = tuple(verified.execute(
        "SELECT "
        "(SELECT COUNT(*) FROM profiles), "
        "(SELECT COUNT(*) FROM memories), "
        "(SELECT COUNT(*) FROM atomic_facts), "
        "(SELECT COUNT(*) FROM ingestion_operations)"
    )[0])
    assert after_counts == before_counts
    assert [
        tuple(row)
        for row in verified.execute(
            "SELECT fact_id, memory_id, profile_id, content, entities_json "
            "FROM atomic_facts ORDER BY profile_id, fact_id"
        )
    ] == before_rows
    for profile_id, token in (
        ("mode-a", "prem032alpha"),
        ("mode-b", "prem032bravo"),
        ("mode-c", "prem032charlie"),
    ):
        memory_id, fact_id, operation_id = expected[profile_id]
        assert [fact.fact_id for fact in verified.search_facts_fts(token, profile_id)] == [
            fact_id
        ]
        memory_rows = verified.execute(
            "SELECT content, metadata_json FROM memories "
            "WHERE memory_id=? AND profile_id=?",
            (memory_id, profile_id),
        )
        assert len(memory_rows) == 1
        assert (
            memory_rows[0]["content"]
            == f"{token} existing profile evidence must survive migration."
        )
        assert json.loads(memory_rows[0]["metadata_json"]) == {
            "legacy": True,
            "profile": profile_id,
        }
        operation = IngestionOperationRepository(verified).get(operation_id)
        assert operation.state is IngestionState.QUERYABLE
        assert operation.queryable_fact_ids == (fact_id,)
        assert operation.metadata == {"legacy": True}
    with verified.raw_connection() as conn:
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"


def test_write_commit_ledger_allows_same_fact_updates_with_distinct_keys(tmp_path) -> None:
    """Mutation operation labels identify a fact, not one globally unique command."""
    from superlocalmemory.storage.write_coordinator import (
        CommandKind,
        WriteCommand,
        WriteCoordinator,
        WriteResult,
    )

    db_path = tmp_path / "memory.db"
    _install_write_commits(db_path)
    coordinator = WriteCoordinator(db_path)
    assert coordinator.claim_ownership()
    try:
        def update(_conn, _capability, command):
            return WriteResult.from_receipt(
                command,
                {"operation_id": "update:fact-1", "fact_id": "fact-1"},
            )

        coordinator.register_handler(CommandKind.UPDATE_FACT, update)
        first = WriteCommand(
            command_id="update-command-1",
            kind=CommandKind.UPDATE_FACT,
            payload={
                "journal_id": "update-journal-1",
                "request_hash": "update-hash-1",
                "profile_id": "default",
                "idempotency_key": "update-key-1",
            },
        )
        second = WriteCommand(
            command_id="update-command-2",
            kind=CommandKind.UPDATE_FACT,
            payload={
                "journal_id": "update-journal-2",
                "request_hash": "update-hash-2",
                "profile_id": "default",
                "idempotency_key": "update-key-2",
            },
        )
        cross_profile = WriteCommand(
            command_id="update-command-work",
            kind=CommandKind.UPDATE_FACT,
            payload={
                "journal_id": "update-journal-work",
                "request_hash": "update-hash-work",
                "profile_id": "work",
                "idempotency_key": "update-key-1",
            },
        )

        assert coordinator.submit(first, timeout=0.5).receipt["commit_sequence"] == 1
        assert coordinator.submit(second, timeout=0.5).receipt["commit_sequence"] == 2
        assert coordinator.submit(cross_profile, timeout=0.5).receipt["commit_sequence"] == 3
        assert coordinator.execute(
            "SELECT COUNT(*) FROM write_commits WHERE operation_id = ?",
            ("update:fact-1",),
        )[0][0] == 3
    finally:
        coordinator.release_ownership()


def test_m032_repairs_a_legacy_completed_ledger_schema(tmp_path) -> None:
    """A dev DB that recorded the provisional M032 safely upgrades in place."""
    from superlocalmemory.storage.migration_runner import apply_all
    from superlocalmemory.storage.migrations import (
        M003_migration_log,
        M032_write_coordinator_admission,
    )

    legacy_ddl = """
CREATE TABLE write_commits (
    commit_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    command_id TEXT NOT NULL UNIQUE,
    journal_id TEXT NOT NULL UNIQUE,
    command_kind TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    operation_id TEXT NOT NULL UNIQUE,
    receipt_json TEXT NOT NULL,
    committed_at REAL NOT NULL
);
CREATE INDEX idx_write_commits_committed_at ON write_commits (committed_at);
CREATE TRIGGER trg_write_commits_immutable_update BEFORE UPDATE ON write_commits
BEGIN SELECT RAISE(ABORT, 'write_commits receipts are immutable'); END;
CREATE TRIGGER trg_write_commits_immutable_delete BEFORE DELETE ON write_commits
BEGIN SELECT RAISE(ABORT, 'write_commits receipts are immutable'); END;
"""
    memory_db = tmp_path / "memory.db"
    conn = sqlite3.connect(memory_db)
    try:
        conn.executescript(M003_migration_log.DDL)
        conn.executescript(legacy_ddl)
        conn.execute(
            "INSERT INTO write_commits("
            "command_id, journal_id, command_kind, request_hash, profile_id, "
            "idempotency_key, operation_id, receipt_json, committed_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "legacy-command",
                "legacy-journal",
                "admission",
                "legacy-hash",
                "default",
                "legacy-key",
                "legacy-operation",
                "{}",
                0.0,
            ),
        )
        conn.execute(
            "INSERT INTO migration_log(name, applied_at, ddl_sha256, rows_affected, status) "
            "VALUES (?, '2026-07-27T00:00:00+00:00', ?, 0, 'complete')",
            (
                M032_write_coordinator_admission.NAME,
                "e45df41becba3d0c3342eca5ec3bd83aa899eef76943c819d2da73b4ca1625a7",
            ),
        )
        conn.commit()
    finally:
        conn.close()

    result = apply_all(tmp_path / "learning.db", memory_db)

    assert M032_write_coordinator_admission.NAME in result["applied"]
    conn = sqlite3.connect(memory_db)
    try:
        assert M032_write_coordinator_admission.verify(conn)
        row = conn.execute("SELECT command_id FROM write_commits").fetchone()
        assert row is not None
        assert row[0] == "legacy-command"
    finally:
        conn.close()
