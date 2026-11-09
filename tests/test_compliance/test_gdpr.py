# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.compliance.gdpr — GDPR Compliance.

Covers:
  - export_profile_data: exports all tables for a profile
  - forget_profile: raises ValueError for "default" profile
  - forget_profile: deletes all profile data across tables
  - forget_entity: deletes entity and related facts
  - forget_entity: handles non-existent entity
  - forget_entity: creates proper audit entries
  - get_audit_trail: returns compliance audit entries
  - _audit: internal audit logging
  - Profile scoping: operations are profile-isolated

Previously known bugs (NOW FIXED):
  BUG-1 (FIXED): forget_profile no longer includes entity_aliases
         in its table scan loop (entity_aliases has no profile_id column).
  BUG-2 (FIXED): _audit() now accepts an explicit profile_id parameter;
         forget_entity passes the real profile_id instead of entity name.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.compliance.gdpr import GDPRCompliance
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import (
    AtomicFact,
    CanonicalEntity,
    MemoryRecord,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    db_path = tmp_path / "gdpr_test.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize(real_schema)
    return mgr


@pytest.fixture()
def gdpr(db: DatabaseManager) -> GDPRCompliance:
    return GDPRCompliance(db)


@pytest.fixture()
def seeded_db(db: DatabaseManager) -> DatabaseManager:
    """DB with a non-default profile, memories, facts, and entities."""
    db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) "
        "VALUES ('user_x', 'User X')"
    )
    db.store_memory(MemoryRecord(
        memory_id="mx1", profile_id="user_x", content="Hello from X"
    ))
    db.store_memory(MemoryRecord(
        memory_id="mx2", profile_id="user_x", content="Second message"
    ))
    db.store_fact(AtomicFact(
        fact_id="fx1", memory_id="mx1", profile_id="user_x",
        content="X likes coffee", canonical_entities=["ex1"],
    ))
    db.store_fact(AtomicFact(
        fact_id="fx2", memory_id="mx1", profile_id="user_x",
        content="X works at Acme",
    ))
    db.store_entity(CanonicalEntity(
        entity_id="ex1", profile_id="user_x",
        canonical_name="UserX", entity_type="person",
    ))
    return db


# ---------------------------------------------------------------------------
# export_profile_data
# ---------------------------------------------------------------------------

class TestExportProfileData:
    def test_exports_all_data(
        self, gdpr: GDPRCompliance, seeded_db: DatabaseManager
    ) -> None:
        data = gdpr.export_profile_data("user_x")
        assert data["profile_id"] == "user_x"
        assert "exported_at" in data
        assert len(data["memories"]) == 2
        assert len(data["facts"]) == 2
        assert len(data["entities"]) == 1

    def test_total_items_calculated(
        self, gdpr: GDPRCompliance, seeded_db: DatabaseManager
    ) -> None:
        data = gdpr.export_profile_data("user_x")
        # 2 memories + 2 facts + 1 entity + 0 edges + 0 trust + 0 feedback
        assert data["total_items"] >= 5

    def test_export_creates_audit_entry(
        self, gdpr: GDPRCompliance, seeded_db: DatabaseManager
    ) -> None:
        gdpr.export_profile_data("user_x")
        trail = gdpr.get_audit_trail("user_x")
        assert len(trail) >= 1
        actions = [e["action"] for e in trail]
        assert "export" in actions

    def test_export_empty_profile(self, gdpr: GDPRCompliance) -> None:
        data = gdpr.export_profile_data("default")
        # Export creates 1 compliance_audit + 1 profile_record for "default"
        assert data["total_items"] <= 2

    def test_export_includes_all_keys(
        self, gdpr: GDPRCompliance, seeded_db: DatabaseManager
    ) -> None:
        data = gdpr.export_profile_data("user_x")
        expected_keys = {
            "profile_id", "exported_at", "memories", "facts",
            "entities", "edges", "trust_scores", "feedback",
            "entity_profiles", "scenes", "temporal_events",
            "consolidation_log", "behavioral_patterns",
            "action_outcomes", "compliance_audit",
            "provenance", "entity_aliases", "profile_record",
            "total_items",
        }
        assert expected_keys.issubset(set(data.keys()))


# ---------------------------------------------------------------------------
# forget_profile
# ---------------------------------------------------------------------------

class TestForgetProfile:
    def test_deletes_all_profile_data(
        self, gdpr: GDPRCompliance, seeded_db: DatabaseManager
    ) -> None:
        """Deletes all data for a profile (BUG-1 fixed: entity_aliases removed from table scan)."""
        counts = gdpr.forget_profile("user_x")
        assert counts["memories"] == 2
        assert counts["atomic_facts"] == 2
        assert counts["profiles"] == 1

    def test_raises_for_default_profile(self, gdpr: GDPRCompliance) -> None:
        with pytest.raises(ValueError, match="Cannot delete the default profile"):
            gdpr.forget_profile("default")

    def test_audit_deleted_by_table_scan_before_crash(
        self, gdpr: GDPRCompliance, seeded_db: DatabaseManager
    ) -> None:
        """The _audit call (line 99) inserts audit with profile_id='user_x'.
        Then the table scan loop deletes compliance_audit rows FIRST (it's
        first in the list), so the audit entry is gone before the loop
        crashes on entity_aliases. Net result: no audit trail survives."""
        try:
            gdpr.forget_profile("user_x")
        except sqlite3.OperationalError:
            pass
        # Audit was created then deleted (compliance_audit is first in loop)
        trail = gdpr.get_audit_trail("user_x")
        assert len(trail) == 0  # Audit entry was deleted before crash


# ---------------------------------------------------------------------------
# forget_entity
# ---------------------------------------------------------------------------

class TestForgetEntity:
    def test_deletes_entity_and_related_facts(
        self, gdpr: GDPRCompliance, seeded_db: DatabaseManager
    ) -> None:
        """Deletes entity + related facts (BUG-2 fixed: _audit uses profile_id param)."""
        counts = gdpr.forget_entity("UserX", "user_x")
        assert counts["entity"] == 1
        assert counts["facts"] >= 1

    def test_nonexistent_entity_returns_not_found(
        self, gdpr: GDPRCompliance, seeded_db: DatabaseManager
    ) -> None:
        """Returns {found: False} for nonexistent entity (BUG-2 fixed)."""
        result = gdpr.forget_entity("NonExistent", "user_x")
        assert result["found"] is False
        assert result["deleted"] == 0

    def test_only_entity_related_facts_deleted(
        self, gdpr: GDPRCompliance, seeded_db: DatabaseManager
    ) -> None:
        """fx2 survives since it doesn't reference ex1 (BUG-2 fixed)."""
        gdpr.forget_entity("UserX", "user_x")
        remaining = seeded_db.get_all_facts("user_x")
        assert len(remaining) == 1
        assert remaining[0].fact_id == "fx2"

    def test_forget_entity_creates_audit_entry(
        self, gdpr: GDPRCompliance, seeded_db: DatabaseManager
    ) -> None:
        """Verify forget_entity creates a proper audit entry (BUG-2 fixed)."""
        gdpr.forget_entity("UserX", "user_x")
        trail = gdpr.get_audit_trail("user_x")
        delete_entries = [e for e in trail if e["action"] == "delete"]
        assert len(delete_entries) >= 1
        assert delete_entries[0]["target_type"] == "entity"
        # The audit row names the erasure, never the erased name.
        from superlocalmemory.compliance.erasure_outcome import subject_ref

        assert delete_entries[0]["target_id"] == subject_ref("entity", "UserX")
        assert "UserX" not in str(delete_entries[0])


# ---------------------------------------------------------------------------
# get_audit_trail
# ---------------------------------------------------------------------------

class TestGetAuditTrail:
    def test_returns_audit_entries(
        self, gdpr: GDPRCompliance, seeded_db: DatabaseManager
    ) -> None:
        gdpr.export_profile_data("user_x")
        trail = gdpr.get_audit_trail("user_x")
        assert len(trail) >= 1
        assert "action" in trail[0]
        assert "timestamp" in trail[0]

    def test_limit_respected(
        self, gdpr: GDPRCompliance, seeded_db: DatabaseManager
    ) -> None:
        for _ in range(5):
            gdpr.export_profile_data("user_x")
        trail = gdpr.get_audit_trail("user_x", limit=2)
        assert len(trail) == 2

    def test_empty_for_unknown_profile(self, gdpr: GDPRCompliance) -> None:
        trail = gdpr.get_audit_trail("ghost_profile")
        assert trail == []

    def test_audit_entry_has_expected_fields(
        self, gdpr: GDPRCompliance, seeded_db: DatabaseManager
    ) -> None:
        gdpr.export_profile_data("user_x")
        trail = gdpr.get_audit_trail("user_x")
        entry = trail[0]
        assert "audit_id" in entry
        assert "profile_id" in entry
        assert "action" in entry
        assert "target_type" in entry
        assert "target_id" in entry
        assert "details" in entry
        assert "timestamp" in entry


# ---------------------------------------------------------------------------
# _audit internal method (tested indirectly)
# ---------------------------------------------------------------------------

class TestAuditInternal:
    def test_export_logs_audit_with_correct_action(
        self, gdpr: GDPRCompliance, seeded_db: DatabaseManager
    ) -> None:
        gdpr.export_profile_data("user_x")
        trail = gdpr.get_audit_trail("user_x")
        export_entries = [e for e in trail if e["action"] == "export"]
        assert len(export_entries) == 1
        assert export_entries[0]["target_type"] == "profile"

    def test_audit_uses_target_id_as_profile_id(
        self, gdpr: GDPRCompliance, seeded_db: DatabaseManager
    ) -> None:
        """Document that _audit stores target_id as profile_id (line 197).
        For export/forget_profile this works because target_id IS the profile_id.
        For forget_entity this is a bug (entity name is not a profile_id)."""
        gdpr.export_profile_data("user_x")
        trail = gdpr.get_audit_trail("user_x")
        assert len(trail) >= 1
        # The profile_id in the audit row equals the target_id passed to _audit
        assert trail[0]["profile_id"] == "user_x"
