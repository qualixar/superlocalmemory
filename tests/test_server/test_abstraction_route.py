# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for the progressive-abstraction read API (Wave Q3).

Handlers are exercised directly with a connection to a seeded temp DB (the
builder read-methods they delegate to are covered at the core level).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import superlocalmemory.server.routes.abstraction as ab
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


def _fact(fid: str, content: str, entities: list[str]) -> AtomicFact:
    return AtomicFact(
        fact_id=fid, memory_id="m0", profile_id="default",
        content=content, canonical_entities=list(entities),
    )


@pytest.fixture()
def seeded_db_path(tmp_path: Path, monkeypatch) -> Path:
    dbp = tmp_path / "memory.db"
    mgr = DatabaseManager(dbp)
    mgr.initialize(real_schema)
    mgr.store_memory(MemoryRecord(memory_id="m0", content="parent"))
    for eid, name in [("e1", "Accenture"), ("e2", "Varun"), ("e3", "manager"),
                      ("e4", "Paris"), ("e5", "vacation")]:
        mgr.store_entity(CanonicalEntity(
            entity_id=eid, profile_id="default",
            canonical_name=name, entity_type="thing",
        ))
    mgr.store_fact(_fact("f1", "Varun works at Accenture", ["e1", "e2", "e3"]))
    mgr.store_fact(_fact("f2", "Varun met his manager at Accenture", ["e1", "e2", "e3"]))
    mgr.store_fact(_fact("f3", "Accenture promoted Varun", ["e1", "e2"]))
    mgr.store_fact(_fact("f4", "Varun booked a vacation to Paris", ["e4", "e5"]))
    mgr.store_fact(_fact("f5", "The Paris vacation was relaxing", ["e4", "e5"]))
    EntityCommunityBuilder(mgr).compute_and_store("default")
    CommunitySummaryBuilder(mgr).compute_and_store("default")
    ProgressiveAbstraction(mgr).compute_and_store("default")

    def _fake_conn() -> sqlite3.Connection:
        c = sqlite3.connect(str(dbp))
        c.row_factory = sqlite3.Row
        return c

    monkeypatch.setattr(ab, "_conn", _fake_conn)
    return dbp


def _body(resp) -> dict:
    return json.loads(resp.body)


class TestAbstractionRoutes:
    def test_persona_endpoint(self, seeded_db_path) -> None:
        body = _body(ab.get_persona(profile="default"))
        assert body["profile"] == "default"
        assert body["persona"] is not None
        assert body["persona"]["summary"]

    def test_communities_endpoint(self, seeded_db_path) -> None:
        body = _body(ab.get_communities(profile="default"))
        assert len(body["communities"]) == 2
        assert all(c["summary"] for c in body["communities"])

    def test_sources_persona_drilldown(self, seeded_db_path) -> None:
        body = _body(ab.get_sources(profile="default", node="persona"))
        s = body["sources"]
        assert s["node_type"] == "persona"
        assert set(s["fact_ids"]) >= {"f1", "f2"}

    def test_sources_community_drilldown(self, seeded_db_path) -> None:
        # Resolve the work community id from the communities endpoint.
        comms = _body(ab.get_communities(profile="default"))["communities"]
        work_cid = next(
            c["community_id"] for c in comms
            if "f1" in json.loads(c["fact_ids_json"])
        )
        body = _body(ab.get_sources(profile="default", node=str(work_cid)))
        assert body["sources"]["node_type"] == "community"
        assert "f1" in body["sources"]["fact_ids"]

    def test_missing_db_is_soft(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(ab, "_conn", lambda: None)
        body = _body(ab.get_persona(profile="default"))
        assert body["persona"] is None
