# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for core.progressive_abstraction (Wave Q3).

Top-tier persona roll-up over the entity-community backbone + summaries, plus
get_sources drill-down (persona -> communities -> source atoms). Recall-gated
and size-bounded (no hot-path pollution).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from superlocalmemory.core.community_summary import CommunitySummaryBuilder
from superlocalmemory.core.entity_community import EntityCommunityBuilder
from superlocalmemory.core.progressive_abstraction import ProgressiveAbstraction
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import (
    AtomicFact,
    CanonicalEntity,
    MemoryRecord,
)

_PERSONA_MAX_CHARS = 2048


@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    mgr = DatabaseManager(tmp_path / "test.db")
    mgr.initialize(real_schema)
    return mgr


def _fact(fid: str, content: str, entities: list[str]) -> AtomicFact:
    return AtomicFact(
        fact_id=fid, memory_id="m0", profile_id="default",
        content=content, canonical_entities=list(entities),
    )


def _seed(db: DatabaseManager) -> None:
    db.store_memory(MemoryRecord(memory_id="m0", content="parent"))
    for eid, name in [("e1", "Accenture"), ("e2", "Varun"), ("e3", "manager"),
                      ("e4", "Paris"), ("e5", "vacation")]:
        db.store_entity(CanonicalEntity(
            entity_id=eid, profile_id="default",
            canonical_name=name, entity_type="thing",
        ))
    db.store_fact(_fact("f1", "Varun works at Accenture as an architect", ["e1", "e2", "e3"]))
    db.store_fact(_fact("f2", "Varun met his manager at Accenture", ["e1", "e2", "e3"]))
    db.store_fact(_fact("f3", "Accenture promoted Varun", ["e1", "e2"]))
    db.store_fact(_fact("f4", "Varun booked a vacation to Paris", ["e4", "e5"]))
    db.store_fact(_fact("f5", "The Paris vacation was relaxing", ["e4", "e5"]))
    EntityCommunityBuilder(db).compute_and_store("default")
    CommunitySummaryBuilder(db).compute_and_store("default")


class TestPersonaRollup:
    def test_persona_built(self, db: DatabaseManager) -> None:
        _seed(db)
        result = ProgressiveAbstraction(db).compute_and_store("default")
        assert result["built"] is True
        persona = ProgressiveAbstraction(db).get_persona("default")
        assert persona is not None
        assert persona["summary"]
        assert persona["community_ids"]  # references its member communities

    def test_persona_size_bounded(self, db: DatabaseManager) -> None:
        _seed(db)
        ProgressiveAbstraction(db).compute_and_store("default")
        persona = ProgressiveAbstraction(db).get_persona("default")
        assert len(persona["summary"]) <= _PERSONA_MAX_CHARS

    def test_mode_bc_uses_summarizer(self, db: DatabaseManager) -> None:
        _seed(db)

        class _Summ:
            def summarize_cluster(self, members: list[dict]) -> str:
                return "PERSONA: senior architect who travels."

        ProgressiveAbstraction(db, summarizer=_Summ()).compute_and_store("default")
        persona = ProgressiveAbstraction(db).get_persona("default")
        assert persona["summary"] == "PERSONA: senior architect who travels."

    def test_summarizer_failure_fails_open(self, db: DatabaseManager) -> None:
        _seed(db)

        class _Boom:
            def summarize_cluster(self, members: list[dict]) -> str:
                raise RuntimeError("down")

        result = ProgressiveAbstraction(db, summarizer=_Boom()).compute_and_store("default")
        assert result["built"] is True
        assert ProgressiveAbstraction(db).get_persona("default")["summary"]

    def test_recompute_replaces(self, db: DatabaseManager) -> None:
        _seed(db)
        pa = ProgressiveAbstraction(db)
        pa.compute_and_store("default")
        pa.compute_and_store("default")
        rows = db.execute(
            "SELECT COUNT(*) AS n FROM persona_summary WHERE profile_id='default'"
        )
        assert dict(rows[0])["n"] == 1  # one persona row per profile

    def test_empty_profile_safe(self, db: DatabaseManager) -> None:
        result = ProgressiveAbstraction(db).compute_and_store("empty")
        assert result["built"] is False
        assert ProgressiveAbstraction(db).get_persona("empty") is None


class TestDrillDown:
    def test_persona_sources_return_atoms(self, db: DatabaseManager) -> None:
        _seed(db)
        pa = ProgressiveAbstraction(db)
        pa.compute_and_store("default")
        sources = pa.get_sources("default", "persona")
        assert sources["node_type"] == "persona"
        assert sources["communities"]                 # child communities
        # Drill-down reaches the source atoms (market bar: non-empty provenance).
        assert set(sources["fact_ids"]) >= {"f1", "f2"}

    def test_community_sources_return_its_atoms(self, db: DatabaseManager) -> None:
        _seed(db)
        pa = ProgressiveAbstraction(db)
        pa.compute_and_store("default")
        comms = EntityCommunityBuilder(db).get_communities("default")
        # Pick the community that owns f1 (the work cluster).
        summaries = CommunitySummaryBuilder(db).get_summaries("default")
        import json
        work_cid = next(
            s["community_id"] for s in summaries
            if "f1" in json.loads(s["fact_ids_json"])
        )
        sources = pa.get_sources("default", work_cid)
        assert sources["node_type"] == "community"
        assert "f1" in sources["fact_ids"]

    def test_unknown_node_returns_empty(self, db: DatabaseManager) -> None:
        _seed(db)
        pa = ProgressiveAbstraction(db)
        pa.compute_and_store("default")
        sources = pa.get_sources("default", 9999)
        assert sources["fact_ids"] == []
