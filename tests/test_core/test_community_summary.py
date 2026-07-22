# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for core.community_summary (Wave Q2).

One synthesized report per entity community. Reuses the entity-community
backbone + core.Summarizer (Mode A heuristic / Mode B/C LLM, fail-open).
Superseded facts are excluded from the summary (market CRIT-3).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from superlocalmemory.core.community_summary import CommunitySummaryBuilder
from superlocalmemory.core.entity_community import EntityCommunityBuilder
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import (
    AtomicFact,
    CanonicalEntity,
    MemoryRecord,
)


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
    # Cluster A (work): e1,e2,e3
    db.store_fact(_fact("f1", "Varun works at Accenture as an architect", ["e1", "e2", "e3"]))
    db.store_fact(_fact("f2", "Varun met his manager at Accenture", ["e1", "e2", "e3"]))
    db.store_fact(_fact("f3", "Accenture promoted Varun", ["e1", "e2"]))
    # Cluster B (travel): e4,e5
    db.store_fact(_fact("f4", "Varun booked a vacation to Paris", ["e4", "e5"]))
    db.store_fact(_fact("f5", "The Paris vacation was relaxing", ["e4", "e5"]))
    EntityCommunityBuilder(db).compute_and_store("default")


class TestCommunitySummaryGeneration:
    def test_summaries_created_per_community(self, db: DatabaseManager) -> None:
        _seed(db)
        result = CommunitySummaryBuilder(db).compute_and_store("default")
        assert result["summaries_written"] == 2
        summaries = CommunitySummaryBuilder(db).get_summaries("default")
        assert len(summaries) == 2
        for s in summaries:
            assert s["summary"]            # non-empty (Mode A heuristic)
            assert s["keywords"]           # non-empty keyword-dense signal
            assert s["fact_count"] >= 2
            assert json.loads(s["fact_ids_json"])   # drill-down handle present

    def test_superseded_facts_excluded(self, db: DatabaseManager) -> None:
        _seed(db)
        # Supersede f3 (a work-cluster fact).
        db.store_temporal_validity("f3", "default")
        db.invalidate_fact_temporal("f3", invalidated_by="f2", invalidation_reason="test")
        CommunitySummaryBuilder(db).compute_and_store("default")
        summaries = CommunitySummaryBuilder(db).get_summaries("default")
        all_fids = {
            fid for s in summaries for fid in json.loads(s["fact_ids_json"])
        }
        assert "f3" not in all_fids  # superseded -> excluded from summary
        assert "f1" in all_fids

    def test_mode_bc_uses_summarizer(self, db: DatabaseManager) -> None:
        _seed(db)

        class _Summ:
            def summarize_cluster(self, members: list[dict]) -> str:
                return "LLM COMMUNITY REPORT"

        CommunitySummaryBuilder(db, summarizer=_Summ()).compute_and_store("default")
        summaries = CommunitySummaryBuilder(db).get_summaries("default")
        assert all(s["summary"] == "LLM COMMUNITY REPORT" for s in summaries)

    def test_summarizer_failure_fails_open(self, db: DatabaseManager) -> None:
        _seed(db)

        class _Boom:
            def summarize_cluster(self, members: list[dict]) -> str:
                raise RuntimeError("llm down")

        # Must not raise; must still write keyword-dense Mode-A summaries.
        result = CommunitySummaryBuilder(db, summarizer=_Boom()).compute_and_store("default")
        assert result["summaries_written"] == 2
        summaries = CommunitySummaryBuilder(db).get_summaries("default")
        assert all(s["summary"] for s in summaries)

    def test_recompute_replaces(self, db: DatabaseManager) -> None:
        _seed(db)
        builder = CommunitySummaryBuilder(db)
        builder.compute_and_store("default")
        builder.compute_and_store("default")
        assert len(builder.get_summaries("default")) == 2  # no duplicates

    def test_empty_profile_safe(self, db: DatabaseManager) -> None:
        result = CommunitySummaryBuilder(db).compute_and_store("empty")
        assert result["summaries_written"] == 0
        assert CommunitySummaryBuilder(db).get_summaries("empty") == []

    def test_get_summary_by_community(self, db: DatabaseManager) -> None:
        _seed(db)
        builder = CommunitySummaryBuilder(db)
        builder.compute_and_store("default")
        comms = EntityCommunityBuilder(db).get_communities("default")
        cid = next(iter(comms))
        s = builder.get_summary("default", cid)
        assert s is not None
        assert s["community_id"] == cid
