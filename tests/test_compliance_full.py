# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for V3 Compliance System — Task 6 of V3 build.

Covers: ABAC engine, Audit chain, Retention engine, Retention scheduler.
"""

from __future__ import annotations

import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

from superlocalmemory.compliance.abac import ABACEngine, AccessDenied
from superlocalmemory.compliance.audit import AuditChain
from superlocalmemory.compliance.retention import RetentionEngine
from superlocalmemory.compliance.scheduler import RetentionScheduler


# ==================================================================
# ABAC Engine Tests
# ==================================================================


class TestABACDefaultBehavior:
    """Test ABAC default-allow semantics."""

    def test_default_allows_all(self):
        abac = ABACEngine()
        assert abac.check("agent-1", "profile-1", "read") is True
        assert abac.check("agent-1", "profile-1", "write") is True

    def test_default_allows_all_actions(self):
        abac = ABACEngine()
        for action in ("read", "write", "delete", "admin"):
            assert abac.check("any-agent", "any-profile", action) is True


class TestABACDenyPolicies:
    """Test ABAC deny policy matching."""

    def test_deny_policy_blocks_action(self):
        abac = ABACEngine()
        abac.add_policy("profile-secret", "agent-x", "write", deny=True)
        assert abac.check("agent-x", "profile-secret", "write") is False
        assert abac.check("agent-x", "profile-secret", "read") is True

    def test_deny_raises_access_denied(self):
        abac = ABACEngine()
        abac.add_policy("p1", "bad-agent", "delete", deny=True)
        with pytest.raises(AccessDenied):
            abac.check_or_raise("bad-agent", "p1", "delete")

    def test_other_agent_unaffected(self):
        abac = ABACEngine()
        abac.add_policy("p1", "bad-agent", "write", deny=True)
        assert abac.check("good-agent", "p1", "write") is True

    def test_other_profile_unaffected(self):
        abac = ABACEngine()
        abac.add_policy("secret-profile", "agent-1", "read", deny=True)
        assert abac.check("agent-1", "other-profile", "read") is True

    def test_multiple_deny_policies(self):
        abac = ABACEngine()
        abac.add_policy("p1", "a1", "write", deny=True)
        abac.add_policy("p1", "a1", "delete", deny=True)
        assert abac.check("a1", "p1", "write") is False
        assert abac.check("a1", "p1", "delete") is False
        assert abac.check("a1", "p1", "read") is True


class TestABACPolicyManagement:
    """Test ABAC policy add/remove/list operations."""

    def test_remove_policy_restores_access(self):
        abac = ABACEngine()
        abac.add_policy("p1", "agent-1", "write", deny=True)
        assert abac.check("agent-1", "p1", "write") is False
        abac.remove_policy("p1", "agent-1", "write")
        assert abac.check("agent-1", "p1", "write") is True

    def test_list_all_policies(self):
        abac = ABACEngine()
        abac.add_policy("p1", "a1", "write")
        abac.add_policy("p2", "a2", "read")
        all_policies = abac.list_policies()
        assert len(all_policies) == 2

    def test_list_policies_filtered_by_profile(self):
        abac = ABACEngine()
        abac.add_policy("p1", "a1", "write")
        abac.add_policy("p2", "a2", "read")
        p1_policies = abac.list_policies(profile_id="p1")
        assert len(p1_policies) == 1
        assert p1_policies[0]["profile_id"] == "p1"

    def test_list_policies_empty(self):
        abac = ABACEngine()
        assert abac.list_policies() == []

    def test_check_or_raise_passes_when_allowed(self):
        abac = ABACEngine()
        # Should not raise — default allow
        abac.check_or_raise("agent-1", "p1", "read")


# ==================================================================
# Audit Chain Tests
# ==================================================================


class TestAuditChainLogging:
    """Test audit chain log and query operations."""

    def test_log_returns_hash(self):
        with tempfile.TemporaryDirectory() as td:
            audit = AuditChain(db_path=Path(td) / "audit.db")
            h = audit.log("store", agent_id="a1", profile_id="p1",
                          content_hash="abc123")
            assert isinstance(h, str)
            assert len(h) == 64  # SHA-256 hex

    def test_log_and_query(self):
        with tempfile.TemporaryDirectory() as td:
            audit = AuditChain(db_path=Path(td) / "audit.db")
            audit.log("store", agent_id="a1", profile_id="p1",
                      content_hash="abc123")
            audit.log("recall", agent_id="a1", profile_id="p1")
            events = audit.query(profile_id="p1")
            assert len(events) >= 2
            ops = {e["operation"] for e in events}
            assert "store" in ops
            assert "recall" in ops

    def test_query_filter_by_operation(self):
        with tempfile.TemporaryDirectory() as td:
            audit = AuditChain(db_path=Path(td) / "audit.db")
            audit.log("store", agent_id="a1", profile_id="p1")
            audit.log("recall", agent_id="a1", profile_id="p1")
            audit.log("delete", agent_id="a1", profile_id="p1")
            events = audit.query(operation="store")
            assert len(events) == 1
            assert events[0]["operation"] == "store"

    def test_query_filter_by_agent(self):
        with tempfile.TemporaryDirectory() as td:
            audit = AuditChain(db_path=Path(td) / "audit.db")
            audit.log("store", agent_id="a1", profile_id="p1")
            audit.log("store", agent_id="a2", profile_id="p1")
            events = audit.query(agent_id="a1")
            assert len(events) == 1

    def test_query_limit(self):
        with tempfile.TemporaryDirectory() as td:
            audit = AuditChain(db_path=Path(td) / "audit.db")
            for i in range(10):
                audit.log("store", agent_id="a1", profile_id="p1")
            events = audit.query(limit=3)
            assert len(events) == 3

    def test_log_with_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            audit = AuditChain(db_path=Path(td) / "audit.db")
            audit.log("store", agent_id="a1", profile_id="p1",
                      metadata={"source": "test", "count": 5})
            events = audit.query()
            assert len(events) == 1


class TestAuditChainIntegrity:
    """Test audit chain tamper detection."""

    def test_integrity_valid_chain(self):
        with tempfile.TemporaryDirectory() as td:
            audit = AuditChain(db_path=Path(td) / "audit.db")
            audit.log("store", agent_id="a1", profile_id="p1",
                      content_hash="abc")
            audit.log("recall", agent_id="a2", profile_id="p1")
            audit.log("delete", agent_id="a1", profile_id="p1",
                      content_hash="def")
            assert audit.verify_integrity() is True

    def test_integrity_empty_chain(self):
        with tempfile.TemporaryDirectory() as td:
            audit = AuditChain(db_path=Path(td) / "audit.db")
            assert audit.verify_integrity() is True

    def test_detects_content_hash_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "audit.db"
            audit = AuditChain(db_path=db_path)
            audit.log("store", agent_id="a1", profile_id="p1",
                      content_hash="abc")
            audit.log("recall", agent_id="a1", profile_id="p1")
            # Tamper directly
            conn = sqlite3.connect(str(db_path))
            conn.execute(
                "UPDATE audit_chain SET content_hash='tampered' WHERE id=1"
            )
            conn.commit()
            conn.close()
            # Re-open and verify
            audit2 = AuditChain(db_path=db_path)
            assert audit2.verify_integrity() is False

    def test_detects_operation_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "audit.db"
            audit = AuditChain(db_path=db_path)
            audit.log("store", agent_id="a1", profile_id="p1")
            audit.log("recall", agent_id="a1", profile_id="p1")
            # Tamper operation field
            conn = sqlite3.connect(str(db_path))
            conn.execute(
                "UPDATE audit_chain SET operation='hacked' WHERE id=1"
            )
            conn.commit()
            conn.close()
            audit2 = AuditChain(db_path=db_path)
            assert audit2.verify_integrity() is False


class TestAuditChainStats:
    """Test audit chain statistics."""

    def test_stats_counts_by_operation(self):
        with tempfile.TemporaryDirectory() as td:
            audit = AuditChain(db_path=Path(td) / "audit.db")
            audit.log("store", agent_id="a1", profile_id="p1")
            audit.log("store", agent_id="a1", profile_id="p1")
            audit.log("recall", agent_id="a1", profile_id="p1")
            stats = audit.get_stats()
            assert stats.get("store", 0) >= 2
            assert stats.get("recall", 0) >= 1
            assert stats.get("total", 0) >= 3

    def test_stats_empty_chain(self):
        with tempfile.TemporaryDirectory() as td:
            audit = AuditChain(db_path=Path(td) / "audit.db")
            stats = audit.get_stats()
            assert stats.get("total", 0) == 0


class TestAuditChainInMemory:
    """Test audit chain with in-memory database."""

    def test_in_memory_works(self):
        audit = AuditChain()
        audit.log("store", agent_id="a1", profile_id="p1")
        events = audit.query()
        assert len(events) == 1
        assert audit.verify_integrity() is True


# ==================================================================
# Retention Engine Tests
# ==================================================================


class TestRetentionRules:
    """Test retention rule CRUD operations."""

    def _make_db(self) -> sqlite3.Connection:
        """Create an in-memory DB with retention_rules table."""
        conn = sqlite3.connect(":memory:")
        return conn

    def test_add_and_get_rules(self):
        db = self._make_db()
        engine = RetentionEngine(db)
        engine.add_rule("p1", "GDPR-30d", 30, "GDPR 30-day retention")
        rules = engine.get_rules("p1")
        assert len(rules) == 1
        assert rules[0]["rule_name"] == "GDPR-30d"
        assert rules[0]["days"] == 30

    def test_remove_rule(self):
        db = self._make_db()
        engine = RetentionEngine(db)
        engine.add_rule("p1", "GDPR-30d", 30)
        engine.remove_rule("p1", "GDPR-30d")
        rules = engine.get_rules("p1")
        assert len(rules) == 0

    def test_multiple_rules_per_profile(self):
        db = self._make_db()
        engine = RetentionEngine(db)
        engine.add_rule("p1", "GDPR-30d", 30)
        engine.add_rule("p1", "HIPAA-7y", 2555)
        rules = engine.get_rules("p1")
        assert len(rules) == 2

    def test_rules_scoped_to_profile(self):
        db = self._make_db()
        engine = RetentionEngine(db)
        engine.add_rule("p1", "GDPR-30d", 30)
        engine.add_rule("p2", "Custom-90d", 90)
        assert len(engine.get_rules("p1")) == 1
        assert len(engine.get_rules("p2")) == 1
        assert len(engine.get_rules("p3")) == 0

    def test_upsert_same_rule_name(self):
        db = self._make_db()
        engine = RetentionEngine(db)
        engine.add_rule("p1", "GDPR", 30)
        engine.add_rule("p1", "GDPR", 60)  # UPDATE
        rules = engine.get_rules("p1")
        assert len(rules) == 1
        assert rules[0]["days"] == 60


class TestRetentionExpiration:
    """Test expired fact detection."""

    def test_no_rules_returns_empty(self):
        db = sqlite3.connect(":memory:")
        engine = RetentionEngine(db)
        assert engine.get_expired_facts("p1") == []

    def test_no_facts_table_returns_empty(self):
        db = sqlite3.connect(":memory:")
        engine = RetentionEngine(db)
        engine.add_rule("p1", "GDPR-30d", 30)
        assert engine.get_expired_facts("p1") == []

    def test_expired_facts_detected(self):
        db = sqlite3.connect(":memory:")
        engine = RetentionEngine(db)
        # Real atomic_facts is keyed by fact_id (TEXT) + has a lifecycle column.
        db.execute(
            "CREATE TABLE atomic_facts ("
            "  fact_id TEXT PRIMARY KEY, profile_id TEXT, "
            "  lifecycle TEXT DEFAULT 'active', created_at TEXT)"
        )
        db.execute(
            "INSERT INTO atomic_facts (fact_id, profile_id, created_at) "
            "VALUES ('f_old', 'p1', '2020-01-01T00:00:00+00:00')"
        )
        db.commit()
        engine.add_rule("p1", "GDPR-30d", 30)
        expired = engine.get_expired_facts("p1")
        assert expired == ["f_old"]

    def test_fresh_facts_not_expired(self):
        from datetime import datetime, timezone
        db = sqlite3.connect(":memory:")
        engine = RetentionEngine(db)
        db.execute(
            "CREATE TABLE atomic_facts ("
            "  fact_id TEXT PRIMARY KEY, profile_id TEXT, "
            "  lifecycle TEXT DEFAULT 'active', created_at TEXT)"
        )
        now = datetime.now(timezone.utc).isoformat()
        db.execute(
            "INSERT INTO atomic_facts (fact_id, profile_id, created_at) "
            "VALUES ('f_new', 'p1', ?)", (now,)
        )
        db.commit()
        engine.add_rule("p1", "GDPR-30d", 30)
        expired = engine.get_expired_facts("p1")
        assert len(expired) == 0


class TestRetentionEnforcement:
    """Test retention enforcement (deletion)."""

    def test_enforce_archives_expired_soft_state(self):
        """Enforcement is GDPR-safe soft-state (archive), NOT a raw delete.

        A rule created via add_rule defaults to action='archive', so an expired
        fact moves to lifecycle='archived' and is retained (never deleted). This
        replaces the old behavior that hard-deleted rows (and crashed on the
        real fact_id-keyed schema).
        """
        db = sqlite3.connect(":memory:")
        engine = RetentionEngine(db)
        db.execute(
            "CREATE TABLE atomic_facts ("
            "  fact_id TEXT PRIMARY KEY, profile_id TEXT, "
            "  lifecycle TEXT DEFAULT 'active', created_at TEXT)"
        )
        db.execute(
            "INSERT INTO atomic_facts (fact_id, profile_id, created_at) "
            "VALUES ('f_old', 'p1', '2020-01-01T00:00:00+00:00')"
        )
        db.commit()
        engine.add_rule("p1", "GDPR-30d", 30)
        result = engine.enforce("p1")
        assert result["archived"] == 1
        assert result["tombstoned"] == 0
        # The fact is RETAINED (soft-state), moved to the archived zone.
        row = db.execute(
            "SELECT lifecycle FROM atomic_facts WHERE fact_id='f_old'"
        ).fetchone()
        assert row[0] == "archived"

    def test_enforce_tombstone_action(self):
        """A tombstone rule lands expired facts in the 'archived' lifecycle zone
        (the only valid terminal state) and flags them purgeable via
        archive_status — the real atomic_facts CHECK forbids 'tombstoned'."""
        db = sqlite3.connect(":memory:")
        engine = RetentionEngine(db)
        db.execute(
            "CREATE TABLE atomic_facts ("
            "  fact_id TEXT PRIMARY KEY, profile_id TEXT, "
            "  lifecycle TEXT DEFAULT 'active', archive_status TEXT, created_at TEXT)"
        )
        db.execute(
            "INSERT INTO atomic_facts (fact_id, profile_id, created_at) "
            "VALUES ('f_old', 'p1', '2020-01-01T00:00:00+00:00')"
        )
        db.commit()
        engine.create_rule(name="tomb", framework="gdpr", retention_days=30,
                           action="tombstone", applies_to=None, profile_id="p1")
        result = engine.enforce("p1")
        assert result["tombstoned"] == 1
        assert result["deleted_count"] == 1  # legacy alias == tombstoned
        row = db.execute(
            "SELECT lifecycle, archive_status FROM atomic_facts WHERE fact_id='f_old'"
        ).fetchone()
        assert row[0] == "archived"
        assert row[1] == "tombstoned"

    def test_enforce_no_rules_no_deletions(self):
        db = sqlite3.connect(":memory:")
        engine = RetentionEngine(db)
        result = engine.enforce("p1")
        assert result["deleted_count"] == 0
        assert result["archived"] == 0


# ==================================================================
# Retention Scheduler Tests
# ==================================================================


class TestRetentionScheduler:
    """Test background retention scheduler."""

    def test_start_and_stop(self):
        db = sqlite3.connect(":memory:")
        engine = RetentionEngine(db)
        scheduler = RetentionScheduler(engine, interval_seconds=3600)
        assert scheduler.is_running is False
        scheduler.start()
        assert scheduler.is_running is True
        scheduler.stop()
        assert scheduler.is_running is False

    def test_double_start_is_idempotent(self):
        db = sqlite3.connect(":memory:")
        engine = RetentionEngine(db)
        scheduler = RetentionScheduler(engine, interval_seconds=3600)
        scheduler.start()
        scheduler.start()  # Should not fail
        assert scheduler.is_running is True
        scheduler.stop()

    def test_stop_when_not_running(self):
        db = sqlite3.connect(":memory:")
        engine = RetentionEngine(db)
        scheduler = RetentionScheduler(engine, interval_seconds=3600)
        scheduler.stop()  # Should not fail
        assert scheduler.is_running is False

    def test_run_once(self):
        db = sqlite3.connect(":memory:")
        engine = RetentionEngine(db)
        scheduler = RetentionScheduler(engine, interval_seconds=3600)
        result = scheduler.run_once()
        assert "profiles_processed" in result
        assert "results" in result
        assert result["profiles_processed"] == 0
