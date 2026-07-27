# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""DatabaseManager contract for a coordinator-owned SQLite connection."""

from __future__ import annotations

import sqlite3
import threading

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


def test_bound_database_transaction_never_commits_or_closes_worker_connection(tmp_path) -> None:
    """The coordinator owns BEGIN/COMMIT; DatabaseManager only reuses its conn."""
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage.write_coordinator import (
        CommandKind,
        WriteCommand,
        WriteCoordinator,
        WriteResult,
    )

    db_path = tmp_path / "memory.db"
    _install_write_commits(db_path)
    db = DatabaseManager(db_path)
    coordinator = WriteCoordinator(db_path)
    assert coordinator.claim_ownership()
    try:

        def handler(conn, capability, command):
            with db._bind_coordinator_connection(conn, capability):
                with db.transaction():
                    db.execute("INSERT INTO binding_probe(value) VALUES (?)", ("inside",))
                assert conn.in_transaction is True
                assert db.execute("SELECT value FROM binding_probe")[0][0] == "inside"
            return WriteResult.from_receipt(
                command,
                {"operation_id": "operation:binding", "transaction_open": True},
            )

        coordinator.register_handler(CommandKind.ADMISSION, handler)
        coordinator.execute("CREATE TABLE binding_probe (value TEXT NOT NULL)")
        result = coordinator.submit(
            WriteCommand.create(
                CommandKind.ADMISSION,
                _admission_payload("binding"),
            ),
            timeout=0.5,
        )
        assert result.receipt == {
            "commit_sequence": 1,
            "operation_id": "operation:binding",
            "transaction_open": True,
        }
        assert coordinator.execute("SELECT value FROM binding_probe")[0][0] == "inside"
    finally:
        coordinator.release_ownership()


def test_bound_database_rejects_path_mismatch_nested_binding_and_untrusted_capability(
    tmp_path,
) -> None:
    """A DatabaseManager cannot steal a connection from another DB or worker."""
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage.write_coordinator import (
        CommandKind,
        WriteCommand,
        WriteCoordinator,
        WriteCoordinatorError,
        WriteResult,
    )

    db_path = tmp_path / "memory.db"
    other_path = tmp_path / "other.db"
    _install_write_commits(db_path)
    db = DatabaseManager(db_path)
    wrong_db = DatabaseManager(other_path)
    coordinator = WriteCoordinator(db_path)
    assert coordinator.claim_ownership()
    try:

        def handler(conn, capability, command):
            with pytest.raises(WriteCoordinatorError, match="untrusted"):
                with db._bind_coordinator_connection(conn, object()):
                    pass
            with pytest.raises(WriteCoordinatorError, match="different database"):
                with wrong_db._bind_coordinator_connection(conn, capability):
                    pass
            with db._bind_coordinator_connection(conn, capability):
                with pytest.raises(WriteCoordinatorError, match="already bound"):
                    with db._bind_coordinator_connection(conn, capability):
                        pass
            thread_errors: list[BaseException] = []

            def bind_from_wrong_thread() -> None:
                try:
                    with db._bind_coordinator_connection(conn, capability):
                        pass
                except BaseException as exc:  # thread assertion propagated below
                    thread_errors.append(exc)

            thread = threading.Thread(target=bind_from_wrong_thread)
            thread.start()
            thread.join(timeout=1)
            assert not thread.is_alive()
            assert len(thread_errors) == 1
            assert "worker thread" in str(thread_errors[0])
            return WriteResult.from_receipt(
                command,
                {"operation_id": "operation:validation", "validated": True},
            )

        coordinator.register_handler(CommandKind.ADMISSION, handler)
        result = coordinator.submit(
            WriteCommand.create(
                CommandKind.ADMISSION,
                _admission_payload("validation"),
            ),
            timeout=0.5,
        )
        assert result.receipt == {
            "commit_sequence": 1,
            "operation_id": "operation:validation",
            "validated": True,
        }
    finally:
        coordinator.release_ownership()
