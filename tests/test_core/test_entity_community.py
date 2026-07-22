# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for core.entity_community (Wave Q backbone).

Entity co-occurrence graph + community detection (Louvain) — the single
principled clustering backbone shared by Q2 (community summaries) and Q3
(progressive-abstraction tiers).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from superlocalmemory.core.entity_community import EntityCommunityBuilder
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, MemoryRecord


@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    mgr = DatabaseManager(tmp_path / "test.db")
    mgr.initialize(real_schema)
    return mgr


def _fact(fid: str, entities: list[str]) -> AtomicFact:
    return AtomicFact(
        fact_id=fid, memory_id="m0", profile_id="default",
        content=f"fact {fid}", canonical_entities=list(entities),
    )


def _seed_two_clusters(db: DatabaseManager) -> None:
    db.store_memory(MemoryRecord(memory_id="m0", content="parent"))
    # Cluster A (work): e1,e2,e3 co-occur.
    db.store_fact(_fact("f1", ["e1", "e2", "e3"]))
    db.store_fact(_fact("f2", ["e1", "e2"]))
    db.store_fact(_fact("f3", ["e2", "e3"]))
    # Cluster B (travel): e4,e5,e6 co-occur.
    db.store_fact(_fact("f4", ["e4", "e5"]))
    db.store_fact(_fact("f5", ["e4", "e5", "e6"]))
    # Singleton entity — no co-occurrence, must be filtered out.
    db.store_fact(_fact("f6", ["e7"]))


class TestEntityCommunityDetection:
    def test_two_clusters_detected(self, db: DatabaseManager) -> None:
        _seed_two_clusters(db)
        result = EntityCommunityBuilder(db).compute_and_store("default")
        assert result["community_count"] == 2

    def test_co_occurring_entities_share_community(self, db: DatabaseManager) -> None:
        _seed_two_clusters(db)
        builder = EntityCommunityBuilder(db)
        builder.compute_and_store("default")
        comms = builder.get_communities("default")
        # Find the community each entity landed in.
        loc = {e: cid for cid, ents in comms.items() for e in ents}
        assert loc["e1"] == loc["e2"] == loc["e3"]
        assert loc["e4"] == loc["e5"]
        assert loc["e1"] != loc["e4"]  # distinct clusters

    def test_singleton_filtered(self, db: DatabaseManager) -> None:
        _seed_two_clusters(db)
        builder = EntityCommunityBuilder(db)
        builder.compute_and_store("default")
        comms = builder.get_communities("default")
        all_entities = {e for ents in comms.values() for e in ents}
        assert "e7" not in all_entities  # min-size filter drops it

    def test_get_community_for_entity(self, db: DatabaseManager) -> None:
        _seed_two_clusters(db)
        builder = EntityCommunityBuilder(db)
        builder.compute_and_store("default")
        assert builder.get_community_for_entity("e1", "default") is not None
        assert builder.get_community_for_entity("e7", "default") is None
        assert builder.get_community_for_entity("nope", "default") is None

    def test_recompute_replaces(self, db: DatabaseManager) -> None:
        _seed_two_clusters(db)
        builder = EntityCommunityBuilder(db)
        builder.compute_and_store("default")
        first = builder.get_communities("default")
        # Recompute must not accumulate stale rows.
        builder.compute_and_store("default")
        second = builder.get_communities("default")
        n1 = sum(len(v) for v in first.values())
        n2 = sum(len(v) for v in second.values())
        assert n1 == n2

    def test_empty_profile_safe(self, db: DatabaseManager) -> None:
        result = EntityCommunityBuilder(db).compute_and_store("empty")
        assert result["community_count"] == 0
        assert EntityCommunityBuilder(db).get_communities("empty") == {}

    def test_profile_isolation(self, db: DatabaseManager) -> None:
        _seed_two_clusters(db)
        db.execute(
            "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
            ("other", "Other"),
        )
        db.store_memory(MemoryRecord(memory_id="mx", content="other", profile_id="other"))
        other = AtomicFact(
            fact_id="fx", memory_id="mx", profile_id="other",
            content="x", canonical_entities=["z1", "z2"],
        )
        db.store_fact(other)
        builder = EntityCommunityBuilder(db)
        builder.compute_and_store("default")
        # 'default' communities must never contain 'other' profile entities.
        comms = builder.get_communities("default")
        all_entities = {e for ents in comms.values() for e in ents}
        assert "z1" not in all_entities
