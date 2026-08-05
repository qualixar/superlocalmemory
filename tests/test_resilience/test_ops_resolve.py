# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""TDD RED: ops_remediation.resolve_operation tests.

Tests:
  - cancel: DLQ entry removed from list
  - resolve result shape
  - invalid action raises ValueError
"""

from __future__ import annotations

import sqlite3
import time

import pytest


def _apply_migrations(conn: sqlite3.Connection) -> None:
    from superlocalmemory.storage.migrations import (
        M031_dead_letter_operations,
        M033_projection_transactions,
        M034_obligation_integrity,
    )
    M031_dead_letter_operations.apply(conn)
    M033_projection_transactions.apply(conn)
    M034_obligation_integrity.apply(conn)
    conn.commit()


def _seed_dead_letter(conn: sqlite3.Connection, profile_id: str = "default",
                       op_id: str = "dlq-resolve-1") -> None:
    conn.execute(
        "INSERT INTO dead_letter_operations "
        "(original_op_id, operation_type, content, error, attempt_count, profile_id, dead_lettered_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (op_id, "M018", "test content", "embedding failed", 10, profile_id, time.time()),
    )
    conn.commit()


def _seed_exhausted_obligation(conn: sqlite3.Connection, profile_id: str = "default",
                                op_id: str = "exh-resolve-1") -> None:
    now = time.time()
    conn.execute(
        "INSERT INTO projection_obligations "
        "(operation_id, profile_id, owner, kind, subject_id, revision, "
        " state, attempts, context_digest, verify_attempts, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (op_id, profile_id, "graph_owner", "apply", "sub-1", 1,
         "failed", 10, "digest-abc", 0, now, now),
    )
    conn.commit()


class TestResolveOperationImport:
    def test_resolve_operation_importable(self):
        from superlocalmemory.core.ops_remediation import resolve_operation
        assert callable(resolve_operation)


class TestResolveCancel:
    def test_cancel_removes_dlq_entry_from_list(self, tmp_path):
        """After cancel, the DLQ entry no longer appears in list_failed_operations."""
        from superlocalmemory.core.ops_remediation import (
            list_failed_operations,
            resolve_operation,
        )
        db_path = tmp_path / "memory.db"
        conn = sqlite3.connect(db_path)
        _apply_migrations(conn)
        _seed_dead_letter(conn, op_id="dlq-cancel-1")
        conn.close()

        # Confirm it's listed
        before = list_failed_operations(db_path)
        assert len(before["dead_letter"]) == 1

        result = resolve_operation(db_path, None, "dlq-cancel-1", "cancel")
        assert result["success"] is True
        assert result["action"] == "cancel"

        # Now it should not appear
        after = list_failed_operations(db_path)
        assert len(after["dead_letter"]) == 0

    def test_cancel_exhausted_obligation_removes_from_list(self, tmp_path):
        """cancel on exhausted obligation marks it so it no longer surfaces."""
        from superlocalmemory.core.ops_remediation import (
            list_failed_operations,
            resolve_operation,
        )
        db_path = tmp_path / "memory.db"
        conn = sqlite3.connect(db_path)
        _apply_migrations(conn)
        _seed_exhausted_obligation(conn, op_id="exh-cancel-1")
        conn.close()

        before = list_failed_operations(db_path)
        assert len(before["exhausted_obligations"]) == 1

        result = resolve_operation(db_path, None, "exh-cancel-1", "cancel")
        assert result["success"] is True

        after = list_failed_operations(db_path)
        assert len(after["exhausted_obligations"]) == 0

    def test_cancel_result_has_required_fields(self, tmp_path):
        from superlocalmemory.core.ops_remediation import resolve_operation
        db_path = tmp_path / "memory.db"
        conn = sqlite3.connect(db_path)
        _apply_migrations(conn)
        _seed_dead_letter(conn, op_id="dlq-shape-1")
        conn.close()

        result = resolve_operation(db_path, None, "dlq-shape-1", "cancel")
        assert "success" in result
        assert "action" in result
        assert "operation_id" in result
        assert result["operation_id"] == "dlq-shape-1"


class TestResolveInvalidAction:
    def test_invalid_action_raises_value_error(self, tmp_path):
        from superlocalmemory.core.ops_remediation import resolve_operation
        db_path = tmp_path / "memory.db"
        conn = sqlite3.connect(db_path)
        _apply_migrations(conn)
        conn.close()

        with pytest.raises(ValueError, match="invalid action"):
            resolve_operation(db_path, None, "some-op", "invalid_action")

    def test_valid_actions_accepted(self, tmp_path):
        """retry, force_reconcile, cancel are valid (even if op doesn't exist)."""
        from superlocalmemory.core.ops_remediation import resolve_operation
        db_path = tmp_path / "memory.db"
        conn = sqlite3.connect(db_path)
        _apply_migrations(conn)
        conn.close()

        # These shouldn't raise ValueError (unknown op is ok, returns not_found)
        for action in ("retry", "force_reconcile", "cancel"):
            result = resolve_operation(db_path, None, "nonexistent-op", action)
            # success may be False (op not found) but no ValueError
            assert "success" in result


class TestResolveOperationNotFound:
    def test_unknown_operation_id_returns_not_found(self, tmp_path):
        from superlocalmemory.core.ops_remediation import resolve_operation
        db_path = tmp_path / "memory.db"
        conn = sqlite3.connect(db_path)
        _apply_migrations(conn)
        conn.close()

        result = resolve_operation(db_path, None, "unknown-op-xyz", "cancel")
        assert result["success"] is False
        assert "not_found" in result.get("reason", "").lower()


class TestResolveRetryEngineWiring:
    """Regression: with a REAL engine and a found dead-letter, retry must build the
    ingestion command via build_engine_ingestion_command(engine) — not
    IngestionCommand(engine), which is missing the write_queryable/materialize
    collaborators and raised TypeError on the live daemon.
    """

    def test_retry_with_real_engine_has_no_constructor_error(
        self, engine_with_mock_deps
    ):
        from superlocalmemory.core.ops_remediation import resolve_operation

        engine = engine_with_mock_deps
        db_path = engine._db.db_path
        # Seed a dead-letter row so _action_retry proceeds past the DLQ
        # existence check and reaches command construction.
        conn = sqlite3.connect(str(db_path))
        try:
            _seed_dead_letter(conn, op_id="dlq-retry-wiring")
        finally:
            conn.close()

        result = resolve_operation(db_path, engine, "dlq-retry-wiring", "retry")

        # The old constructor bug surfaced as this exact TypeError text; it must
        # be gone regardless of the op's eventual retry outcome.
        blob = f"{result.get('reason', '')} {result.get('message', '')}"
        assert "keyword-only" not in blob
        assert "write_queryable" not in blob
        assert "materialize" not in blob
        assert "action" in result and result["action"] == "retry"
