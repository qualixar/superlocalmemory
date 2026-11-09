# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.encoding.observation_builder.

Covers:
  - ObservationBuilder.update_profile() — create and update
  - ObservationBuilder.get_profile()
  - ObservationBuilder.build_all_profiles()
  - Summary building from facts
  - Idempotent fact addition
"""

from __future__ import annotations

from pathlib import Path

import pytest

from superlocalmemory.encoding.observation_builder import ObservationBuilder
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import (
    AtomicFact,
    CanonicalEntity,
    EntityProfile,
    MemoryRecord,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    db_path = tmp_path / "test.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize(real_schema)
    return mgr


@pytest.fixture()
def builder(db: DatabaseManager) -> ObservationBuilder:
    return ObservationBuilder(db=db)


def _setup_entity_and_fact(
    db: DatabaseManager,
    entity_id: str = "ent_alice",
    entity_name: str = "Alice",
    fact_id: str = "f1",
    content: str = "Alice works at Google",
    profile_id: str = "default",
) -> AtomicFact:
    """Create entity + memory + fact in DB, return the fact."""
    db.store_entity(CanonicalEntity(
        entity_id=entity_id, profile_id=profile_id,
        canonical_name=entity_name, entity_type="person",
    ))
    mem_id = f"m_{fact_id}"
    db.store_memory(MemoryRecord(memory_id=mem_id, content="parent"))
    fact = AtomicFact(
        fact_id=fact_id, memory_id=mem_id, profile_id=profile_id,
        content=content, canonical_entities=[entity_id],
    )
    db.store_fact(fact)
    return fact


# ---------------------------------------------------------------------------
# update_profile — create
# ---------------------------------------------------------------------------

class TestUpdateProfileCreate:
    def test_creates_new_profile(
        self, builder: ObservationBuilder, db: DatabaseManager,
    ) -> None:
        fact = _setup_entity_and_fact(db)
        profile = builder.update_profile("ent_alice", fact, "default")
        assert isinstance(profile, EntityProfile)
        assert profile.entity_id == "ent_alice"
        assert "f1" in profile.fact_ids
        assert "Alice works at Google" in profile.knowledge_summary

    def test_persists_to_db(
        self, builder: ObservationBuilder, db: DatabaseManager,
    ) -> None:
        fact = _setup_entity_and_fact(db)
        builder.update_profile("ent_alice", fact, "default")
        # Verify via DB query
        rows = db.execute(
            "SELECT * FROM entity_profiles WHERE entity_id = 'ent_alice'"
        )
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# update_profile — update existing
# ---------------------------------------------------------------------------

class TestUpdateProfileExisting:
    def test_appends_new_fact(
        self, builder: ObservationBuilder, db: DatabaseManager,
    ) -> None:
        fact1 = _setup_entity_and_fact(db, fact_id="f1", content="Alice works at Google")
        builder.update_profile("ent_alice", fact1, "default")

        # Add second fact
        db.store_memory(MemoryRecord(memory_id="m_f2", content="parent2"))
        fact2 = AtomicFact(
            fact_id="f2", memory_id="m_f2", profile_id="default",
            content="Alice likes hiking",
            canonical_entities=["ent_alice"],
        )
        db.store_fact(fact2)

        profile = builder.update_profile("ent_alice", fact2, "default")
        assert "f1" in profile.fact_ids
        assert "f2" in profile.fact_ids

    def test_idempotent_addition(
        self, builder: ObservationBuilder, db: DatabaseManager,
    ) -> None:
        fact1 = _setup_entity_and_fact(db, fact_id="f1", content="Alice at Google")
        builder.update_profile("ent_alice", fact1, "default")
        profile = builder.update_profile("ent_alice", fact1, "default")
        # f1 should appear only once
        assert profile.fact_ids.count("f1") == 1

    def test_summary_includes_all_facts(
        self, builder: ObservationBuilder, db: DatabaseManager,
    ) -> None:
        fact1 = _setup_entity_and_fact(db, fact_id="f1", content="Alice works at Google")
        builder.update_profile("ent_alice", fact1, "default")

        db.store_memory(MemoryRecord(memory_id="m_f2", content="parent2"))
        fact2 = AtomicFact(
            fact_id="f2", memory_id="m_f2", profile_id="default",
            content="Alice lives in San Francisco",
            canonical_entities=["ent_alice"],
        )
        db.store_fact(fact2)

        profile = builder.update_profile("ent_alice", fact2, "default")
        assert "Google" in profile.knowledge_summary
        assert "San Francisco" in profile.knowledge_summary


# ---------------------------------------------------------------------------
# get_profile
# ---------------------------------------------------------------------------

class TestGetProfile:
    def test_returns_existing(
        self, builder: ObservationBuilder, db: DatabaseManager,
    ) -> None:
        fact = _setup_entity_and_fact(db)
        builder.update_profile("ent_alice", fact, "default")
        profile = builder.get_profile("ent_alice", "default")
        assert profile is not None
        assert profile.entity_id == "ent_alice"

    def test_returns_none_for_missing(self, builder: ObservationBuilder) -> None:
        profile = builder.get_profile("nonexistent", "default")
        assert profile is None


# ---------------------------------------------------------------------------
# build_all_profiles
# ---------------------------------------------------------------------------

class TestBuildAllProfiles:
    def test_builds_all(
        self, builder: ObservationBuilder, db: DatabaseManager,
    ) -> None:
        _setup_entity_and_fact(
            db, entity_id="ent_alice", entity_name="Alice",
            fact_id="f1", content="Alice at Google",
        )
        _setup_entity_and_fact(
            db, entity_id="ent_bob", entity_name="Bob",
            fact_id="f2", content="Bob at Apple",
        )

        profiles = builder.build_all_profiles("default")
        assert len(profiles) == 2
        entity_ids = {p.entity_id for p in profiles}
        assert "ent_alice" in entity_ids
        assert "ent_bob" in entity_ids

    def test_empty_database(self, builder: ObservationBuilder) -> None:
        profiles = builder.build_all_profiles("default")
        assert profiles == []

    def test_entity_without_facts(
        self, builder: ObservationBuilder, db: DatabaseManager,
    ) -> None:
        db.store_entity(CanonicalEntity(
            entity_id="ent_lonely", profile_id="default",
            canonical_name="Lonely", entity_type="person",
        ))
        profiles = builder.build_all_profiles("default")
        # Entity without facts should not produce a profile
        assert len(profiles) == 0


# ---------------------------------------------------------------------------
# Summary building limits
# ---------------------------------------------------------------------------

class TestSummaryLimits:
    def test_limits_to_last_20_facts(
        self, builder: ObservationBuilder, db: DatabaseManager,
    ) -> None:
        db.store_entity(CanonicalEntity(
            entity_id="ent_many", profile_id="default",
            canonical_name="ManyFacts", entity_type="concept",
        ))
        for i in range(25):
            mem_id = f"m_many_{i}"
            fact_id = f"f_many_{i}"
            db.store_memory(MemoryRecord(memory_id=mem_id, content="parent"))
            db.store_fact(AtomicFact(
                fact_id=fact_id, memory_id=mem_id, profile_id="default",
                content=f"Fact number {i} about ManyFacts",
                canonical_entities=["ent_many"],
            ))

        # Build profile with all 25 facts in fact_ids
        fact_ids = [f"f_many_{i}" for i in range(25)]
        summary = builder._build_summary("ent_many", fact_ids, "default")
        # Summary should include text from last 20 only
        # Fact 0 through 4 are earliest, so fact_ids[-20:] = facts 5-24
        parts = summary.split(" | ")
        assert len(parts) <= 20
