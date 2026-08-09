# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""TDD RED: ops_remediation.list_failed_operations tests.

Validates:
  - Dead-letter rows, DEGRADED manifests, exhausted obligations all appear
  - profile_id filter works
  - Empty result when nothing is broken
"""

from __future__ import annotations

import sqlite3
import time
from types import SimpleNamespace

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


def _seed_dead_letter(conn: sqlite3.Connection, profile_id: str, op_id: str = "dlq-op-1") -> None:
    conn.execute(
        "INSERT INTO dead_letter_operations "
        "(original_op_id, operation_type, content, error, attempt_count, profile_id, dead_lettered_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (op_id, "M018", "test content", "embedding failed", 10, profile_id, time.time()),
    )
    conn.commit()


def _seed_degraded_manifest(conn: sqlite3.Connection, profile_id: str, op_id: str = "deg-op-1") -> None:
    conn.execute(
        "INSERT INTO completion_manifests "
        "(operation_id, profile_id, state, all_met, obligation_count, "
        " owner_evidence_json, manifest_hash, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (op_id, profile_id, "DEGRADED", 0, 2, '{}', 'hash123', time.time(), time.time()),
    )
    conn.commit()


def _seed_exhausted_obligation(
    conn: sqlite3.Connection,
    profile_id: str,
    op_id: str = "exh-op-1",
    attempts: int = 10,
) -> None:
    now = time.time()
    conn.execute(
        "INSERT INTO projection_obligations "
        "(operation_id, profile_id, owner, kind, subject_id, revision, "
        " state, attempts, context_digest, verify_attempts, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (op_id, profile_id, "graph_owner", "apply", "sub-1", 1,
         "failed", attempts, "digest-abc", 0, now, now),
    )
    conn.commit()


class TestListFailedOperationsImport:
    def test_module_importable(self):
        from superlocalmemory.core import ops_remediation  # noqa: F401
        assert hasattr(ops_remediation, "list_failed_operations")

    def test_resolve_operation_importable(self):
        from superlocalmemory.core.ops_remediation import resolve_operation  # noqa: F401


class TestListFailedOperationsBasic:
    def test_empty_db_returns_empty_lists(self, tmp_path):
        from superlocalmemory.core.ops_remediation import list_failed_operations
        db_path = tmp_path / "memory.db"
        conn = sqlite3.connect(db_path)
        _apply_migrations(conn)
        conn.close()

        result = list_failed_operations(db_path)
        assert result["dead_letter"] == []
        assert result["degraded_manifests"] == []
        assert result["exhausted_obligations"] == []
        assert result["total"] == 0

    def test_dead_letter_row_surfaced(self, tmp_path):
        from superlocalmemory.core.ops_remediation import list_failed_operations
        db_path = tmp_path / "memory.db"
        conn = sqlite3.connect(db_path)
        _apply_migrations(conn)
        _seed_dead_letter(conn, "default", "dlq-op-1")
        conn.close()

        result = list_failed_operations(db_path)
        assert len(result["dead_letter"]) == 1
        assert result["dead_letter"][0]["operation_id"] == "dlq-op-1"
        assert result["total"] == 1

    def test_degraded_manifest_surfaced(self, tmp_path):
        from superlocalmemory.core.ops_remediation import list_failed_operations
        db_path = tmp_path / "memory.db"
        conn = sqlite3.connect(db_path)
        _apply_migrations(conn)
        _seed_degraded_manifest(conn, "default", "deg-op-1")
        conn.close()

        result = list_failed_operations(db_path)
        assert len(result["degraded_manifests"]) == 1
        assert result["degraded_manifests"][0]["operation_id"] == "deg-op-1"

    def test_exhausted_obligation_surfaced(self, tmp_path):
        from superlocalmemory.core.ops_remediation import list_failed_operations
        db_path = tmp_path / "memory.db"
        conn = sqlite3.connect(db_path)
        _apply_migrations(conn)
        _seed_exhausted_obligation(conn, "default", "exh-op-1", attempts=10)
        conn.close()

        result = list_failed_operations(db_path)
        assert len(result["exhausted_obligations"]) == 1
        assert result["exhausted_obligations"][0]["operation_id"] == "exh-op-1"

    def test_non_exhausted_obligation_not_surfaced(self, tmp_path):
        """Obligations with attempts < MAX should not appear."""
        from superlocalmemory.core.ops_remediation import list_failed_operations
        db_path = tmp_path / "memory.db"
        conn = sqlite3.connect(db_path)
        _apply_migrations(conn)
        _seed_exhausted_obligation(conn, "default", "few-op-1", attempts=3)
        conn.close()

        result = list_failed_operations(db_path)
        assert result["exhausted_obligations"] == []

    def test_all_three_categories_returned(self, tmp_path):
        from superlocalmemory.core.ops_remediation import list_failed_operations
        db_path = tmp_path / "memory.db"
        conn = sqlite3.connect(db_path)
        _apply_migrations(conn)
        _seed_dead_letter(conn, "default", "dlq-op-x")
        _seed_degraded_manifest(conn, "default", "deg-op-x")
        _seed_exhausted_obligation(conn, "default", "exh-op-x", attempts=10)
        conn.close()

        result = list_failed_operations(db_path)
        assert len(result["dead_letter"]) == 1
        assert len(result["degraded_manifests"]) == 1
        assert len(result["exhausted_obligations"]) == 1
        assert result["total"] == 3


class TestOpsStatusFailureCounts:
    def test_status_helper_reports_persisted_failure_surfaces(self, tmp_path):
        """The daemon status route must not silently report zero for a bad DB."""
        from superlocalmemory.server.unified_daemon import _ops_failure_counts

        db_path = tmp_path / "memory.db"
        conn = sqlite3.connect(db_path)
        _apply_migrations(conn)
        _seed_dead_letter(conn, "default", "dlq-status")
        _seed_degraded_manifest(conn, "default", "deg-status")
        _seed_exhausted_obligation(conn, "default", "exh-status", attempts=10)
        conn.close()

        application = SimpleNamespace(
            state=SimpleNamespace(
                config=SimpleNamespace(db_path=db_path),
                canonical_remember_runtime=None,
                write_coordinator=None,
            )
        )
        counts = _ops_failure_counts(engine=None, application=application)

        assert counts["dead_letter_count"] == 1
        assert counts["degraded_operations"] == 1
        assert counts["exhausted_obligations"] == 1


class TestListFailedOperationsProfileFilter:
    def test_profile_filter_excludes_other_profiles(self, tmp_path):
        from superlocalmemory.core.ops_remediation import list_failed_operations
        db_path = tmp_path / "memory.db"
        conn = sqlite3.connect(db_path)
        _apply_migrations(conn)
        _seed_dead_letter(conn, "profile-A", "dlq-A")
        _seed_dead_letter(conn, "profile-B", "dlq-B")
        conn.close()

        result = list_failed_operations(db_path, profile_id="profile-A")
        assert len(result["dead_letter"]) == 1
        assert result["dead_letter"][0]["operation_id"] == "dlq-A"

    def test_no_profile_filter_returns_all(self, tmp_path):
        from superlocalmemory.core.ops_remediation import list_failed_operations
        db_path = tmp_path / "memory.db"
        conn = sqlite3.connect(db_path)
        _apply_migrations(conn)
        _seed_dead_letter(conn, "profile-A", "dlq-A")
        _seed_dead_letter(conn, "profile-B", "dlq-B")
        conn.close()

        result = list_failed_operations(db_path)
        assert len(result["dead_letter"]) == 2


class TestListFailedOperationsResultShape:
    def test_dead_letter_entry_has_required_fields(self, tmp_path):
        from superlocalmemory.core.ops_remediation import list_failed_operations
        db_path = tmp_path / "memory.db"
        conn = sqlite3.connect(db_path)
        _apply_migrations(conn)
        _seed_dead_letter(conn, "default")
        conn.close()

        result = list_failed_operations(db_path)
        entry = result["dead_letter"][0]
        assert "operation_id" in entry
        assert "error" in entry
        assert "attempts" in entry
        assert "profile_id" in entry
        assert "category" in entry
        assert entry["category"] == "dead_letter"

    def test_degraded_manifest_entry_has_required_fields(self, tmp_path):
        from superlocalmemory.core.ops_remediation import list_failed_operations
        db_path = tmp_path / "memory.db"
        conn = sqlite3.connect(db_path)
        _apply_migrations(conn)
        _seed_degraded_manifest(conn, "default")
        conn.close()

        result = list_failed_operations(db_path)
        entry = result["degraded_manifests"][0]
        assert "operation_id" in entry
        assert "state" in entry
        assert "profile_id" in entry
        assert "category" in entry
        assert entry["category"] == "degraded_manifest"

    def test_exhausted_obligation_entry_has_required_fields(self, tmp_path):
        from superlocalmemory.core.ops_remediation import list_failed_operations
        db_path = tmp_path / "memory.db"
        conn = sqlite3.connect(db_path)
        _apply_migrations(conn)
        _seed_exhausted_obligation(conn, "default")
        conn.close()

        result = list_failed_operations(db_path)
        entry = result["exhausted_obligations"][0]
        assert "operation_id" in entry
        assert "attempts" in entry
        assert "profile_id" in entry
        assert "category" in entry
        assert entry["category"] == "exhausted_obligation"


class TestOpsOperationKinds:
    def test_ops_inspect_in_operation_kind_enum(self):
        from superlocalmemory.core.operation_request import OperationKind
        assert hasattr(OperationKind, "OPS_INSPECT")

    def test_ops_resolve_in_operation_kind_enum(self):
        from superlocalmemory.core.operation_request import OperationKind
        assert hasattr(OperationKind, "OPS_RESOLVE")

    def test_ops_inspect_policy_registered(self):
        from superlocalmemory.core.operation_policy_registry import _DEFAULT_REGISTRY
        from superlocalmemory.core.operation_request import OperationKind
        policy = _DEFAULT_REGISTRY.get_policy(OperationKind.OPS_INSPECT)
        assert policy is not None

    def test_ops_resolve_policy_registered(self):
        from superlocalmemory.core.operation_policy_registry import _DEFAULT_REGISTRY
        from superlocalmemory.core.operation_request import OperationKind
        policy = _DEFAULT_REGISTRY.get_policy(OperationKind.OPS_RESOLVE)
        assert policy is not None

    def test_ops_inspect_requires_owner_or_admin_role(self):
        from superlocalmemory.core.actor_context import ActorContext, ActorRole, Transport
        from superlocalmemory.core.operation_policy_registry import _DEFAULT_REGISTRY
        from superlocalmemory.core.operation_request import OperationKind

        member_actor = ActorContext(
            principal_id="member-user",
            roles=frozenset({ActorRole.MEMBER}),
            transport=Transport.HTTP,
            client_host="127.0.0.1",
        )
        decision = _DEFAULT_REGISTRY.evaluate(
            OperationKind.OPS_INSPECT, member_actor, mode="company"
        )
        assert not decision.allowed, "MEMBER should NOT be allowed OPS_INSPECT"

    def test_ops_inspect_allows_owner(self):
        from superlocalmemory.core.actor_context import ActorContext, ActorRole, Transport
        from superlocalmemory.core.operation_policy_registry import _DEFAULT_REGISTRY
        from superlocalmemory.core.operation_request import OperationKind

        owner_actor = ActorContext(
            principal_id="owner",
            roles=frozenset({ActorRole.OWNER}),
            transport=Transport.HTTP,
            client_host="127.0.0.1",
        )
        decision = _DEFAULT_REGISTRY.evaluate(
            OperationKind.OPS_INSPECT, owner_actor, mode="company"
        )
        assert decision.allowed

    def test_ops_resolve_denied_to_member(self):
        from superlocalmemory.core.actor_context import ActorContext, ActorRole, Transport
        from superlocalmemory.core.operation_policy_registry import _DEFAULT_REGISTRY
        from superlocalmemory.core.operation_request import OperationKind

        member_actor = ActorContext(
            principal_id="member-user",
            roles=frozenset({ActorRole.MEMBER}),
            transport=Transport.HTTP,
            client_host="127.0.0.1",
        )
        decision = _DEFAULT_REGISTRY.evaluate(
            OperationKind.OPS_RESOLVE, member_actor, mode="company"
        )
        assert not decision.allowed
