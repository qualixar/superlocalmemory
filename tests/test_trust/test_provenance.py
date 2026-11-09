# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.trust.provenance — Provenance Tracking.

Covers:
  - record: creates provenance entry in DB
  - get_provenance: retrieves by fact_id + profile_id
  - get_provenance: returns None for non-existent fact
  - get_facts_by_source: filters by source_type
  - get_provenance_for_profile: all records for a profile with limit
  - Profile isolation: provenance scoped to profile_id
  - Multiple provenance records for different facts
  - ProvenanceRecord field correctness
"""

from __future__ import annotations

from pathlib import Path

import pytest

from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import (
    AtomicFact,
    MemoryRecord,
    ProvenanceRecord,
)
from superlocalmemory.trust.provenance import ProvenanceTracker

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    db_path = tmp_path / "prov_test.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize(real_schema)
    return mgr


@pytest.fixture()
def tracker(db: DatabaseManager) -> ProvenanceTracker:
    return ProvenanceTracker(db)


@pytest.fixture()
def seeded_db(db: DatabaseManager) -> DatabaseManager:
    """DB with a parent memory and a fact so provenance FK is satisfied."""
    db.store_memory(MemoryRecord(memory_id="m1", content="parent"))
    db.store_fact(AtomicFact(fact_id="f1", memory_id="m1", content="fact one"))
    db.store_fact(AtomicFact(fact_id="f2", memory_id="m1", content="fact two"))
    db.store_fact(AtomicFact(fact_id="f3", memory_id="m1", content="fact three"))
    return db


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------

class TestRecord:
    def test_creates_provenance_record(
        self, tracker: ProvenanceTracker, seeded_db: DatabaseManager
    ) -> None:
        rec = tracker.record("f1", "default", "conversation", "sess1", "agent_v3")
        assert isinstance(rec, ProvenanceRecord)
        assert rec.fact_id == "f1"
        assert rec.source_type == "conversation"
        assert rec.source_id == "sess1"
        assert rec.created_by == "agent_v3"
        assert rec.profile_id == "default"
        assert rec.provenance_id  # non-empty

    def test_record_persists_to_db(
        self, tracker: ProvenanceTracker, seeded_db: DatabaseManager
    ) -> None:
        tracker.record("f1", "default", "import", "batch_42")
        rows = seeded_db.execute(
            "SELECT * FROM provenance WHERE fact_id = ?", ("f1",)
        )
        assert len(rows) == 1
        assert dict(rows[0])["source_type"] == "import"

    def test_record_default_source_id_and_created_by(
        self, tracker: ProvenanceTracker, seeded_db: DatabaseManager
    ) -> None:
        rec = tracker.record("f2", "default", "consolidation")
        assert rec.source_id == ""
        assert rec.created_by == ""


# ---------------------------------------------------------------------------
# get_provenance
# ---------------------------------------------------------------------------

class TestGetProvenance:
    def test_retrieves_existing_record(
        self, tracker: ProvenanceTracker, seeded_db: DatabaseManager
    ) -> None:
        tracker.record("f1", "default", "conversation", "s1", "user")
        result = tracker.get_provenance("f1", "default")
        assert result is not None
        assert result.fact_id == "f1"
        assert result.source_type == "conversation"

    def test_returns_none_for_nonexistent(
        self, tracker: ProvenanceTracker, seeded_db: DatabaseManager
    ) -> None:
        assert tracker.get_provenance("no_such_fact", "default") is None

    def test_returns_none_for_wrong_profile(
        self, tracker: ProvenanceTracker, seeded_db: DatabaseManager
    ) -> None:
        tracker.record("f1", "default", "conversation")
        assert tracker.get_provenance("f1", "other_profile") is None


# ---------------------------------------------------------------------------
# get_facts_by_source
# ---------------------------------------------------------------------------

class TestGetFactsBySource:
    def test_filters_by_source_type(
        self, tracker: ProvenanceTracker, seeded_db: DatabaseManager
    ) -> None:
        tracker.record("f1", "default", "conversation")
        tracker.record("f2", "default", "import")
        tracker.record("f3", "default", "conversation")

        conv_records = tracker.get_facts_by_source("conversation", "default")
        assert len(conv_records) == 2
        assert all(r.source_type == "conversation" for r in conv_records)

    def test_empty_for_unknown_source_type(
        self, tracker: ProvenanceTracker, seeded_db: DatabaseManager
    ) -> None:
        result = tracker.get_facts_by_source("migration", "default")
        assert result == []

    def test_limit_respected(
        self, tracker: ProvenanceTracker, seeded_db: DatabaseManager
    ) -> None:
        tracker.record("f1", "default", "conversation")
        tracker.record("f2", "default", "conversation")
        tracker.record("f3", "default", "conversation")

        result = tracker.get_facts_by_source("conversation", "default", limit=2)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# get_provenance_for_profile
# ---------------------------------------------------------------------------

class TestGetProvenanceForProfile:
    def test_returns_all_records(
        self, tracker: ProvenanceTracker, seeded_db: DatabaseManager
    ) -> None:
        tracker.record("f1", "default", "conversation")
        tracker.record("f2", "default", "import")
        records = tracker.get_provenance_for_profile("default")
        assert len(records) == 2

    def test_limit_respected(
        self, tracker: ProvenanceTracker, seeded_db: DatabaseManager
    ) -> None:
        tracker.record("f1", "default", "conversation")
        tracker.record("f2", "default", "import")
        tracker.record("f3", "default", "consolidation")

        records = tracker.get_provenance_for_profile("default", limit=1)
        assert len(records) == 1

    def test_empty_for_unknown_profile(
        self, tracker: ProvenanceTracker, seeded_db: DatabaseManager
    ) -> None:
        records = tracker.get_provenance_for_profile("ghost_profile")
        assert records == []


# ---------------------------------------------------------------------------
# Profile isolation
# ---------------------------------------------------------------------------

class TestProfileIsolation:
    def test_records_separated_by_profile(
        self, tracker: ProvenanceTracker, db: DatabaseManager
    ) -> None:
        db.execute(
            "INSERT OR IGNORE INTO profiles (profile_id, name) "
            "VALUES ('work', 'Work')"
        )
        # Seed facts in both profiles
        db.store_memory(MemoryRecord(memory_id="md", content="d", profile_id="default"))
        db.store_memory(MemoryRecord(memory_id="mw", content="w", profile_id="work"))
        db.store_fact(AtomicFact(
            fact_id="fd", memory_id="md", profile_id="default", content="df"
        ))
        db.store_fact(AtomicFact(
            fact_id="fw", memory_id="mw", profile_id="work", content="wf"
        ))

        tracker.record("fd", "default", "conversation")
        tracker.record("fw", "work", "import")

        default_recs = tracker.get_provenance_for_profile("default")
        work_recs = tracker.get_provenance_for_profile("work")

        assert len(default_recs) == 1
        assert default_recs[0].fact_id == "fd"
        assert len(work_recs) == 1
        assert work_recs[0].fact_id == "fw"


# ---------------------------------------------------------------------------
# _row_to_record (static method)
# ---------------------------------------------------------------------------

class TestRowToRecord:
    def test_static_method_parses_row(
        self, tracker: ProvenanceTracker, seeded_db: DatabaseManager
    ) -> None:
        tracker.record("f1", "default", "conversation", "s1", "agent")
        rows = seeded_db.execute(
            "SELECT * FROM provenance WHERE fact_id = ?", ("f1",)
        )
        rec = ProvenanceTracker._row_to_record(rows[0])
        assert isinstance(rec, ProvenanceRecord)
        assert rec.fact_id == "f1"
        assert rec.source_type == "conversation"
