# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.storage.database — DatabaseManager CRUD.

Covers:
  - store_memory + retrieve via execute
  - store_fact + get_all_facts + get_facts_by_type + get_facts_by_entity
  - store_entity + get_entity_by_name (case-insensitive)
  - store_edge + get_edges_for_node
  - store_temporal_event + get_temporal_events
  - BM25 token persistence (store + retrieve)
  - FTS5 search (store facts, search by text)
  - Transaction commit + rollback
  - Profile isolation (two profiles, data separated)
  - update_fact + delete_fact
  - get_fact_count

Uses the real schema module (storage.schema) since database.py was aligned
to schema.py table names in S17.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import (
    AtomicFact,
    CanonicalEntity,
    EdgeType,
    EntityAlias,
    FactType,
    GraphEdge,
    MemoryRecord,
    TemporalEvent,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    """DatabaseManager wired to a temp directory with schema applied."""
    db_path = tmp_path / "test.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize(real_schema)
    return mgr


@pytest.fixture()
def db_with_profile(db: DatabaseManager) -> DatabaseManager:
    """DB with an extra profile 'work' pre-created."""
    db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('work', 'Work')"
    )
    return db


# ---------------------------------------------------------------------------
# Memory CRUD
# ---------------------------------------------------------------------------

class TestStoreMemory:
    def test_store_and_retrieve(self, db: DatabaseManager) -> None:
        record = MemoryRecord(
            memory_id="m1",
            profile_id="default",
            content="Hello world",
            session_id="s1",
            speaker="user",
        )
        result_id = db.store_memory(record)
        assert result_id == "m1"

        rows = db.execute(
            "SELECT * FROM memories WHERE memory_id = ?", ("m1",)
        )
        assert len(rows) == 1
        assert dict(rows[0])["content"] == "Hello world"

    def test_store_memory_upsert(self, db: DatabaseManager) -> None:
        r1 = MemoryRecord(memory_id="m_dup", content="v1")
        r2 = MemoryRecord(memory_id="m_dup", content="v2")
        db.store_memory(r1)
        db.store_memory(r2)
        rows = db.execute(
            "SELECT content FROM memories WHERE memory_id = ?", ("m_dup",)
        )
        assert dict(rows[0])["content"] == "v2"


# ---------------------------------------------------------------------------
# Fact CRUD
# ---------------------------------------------------------------------------

class TestStoreFact:
    def _store_parent_memory(self, db: DatabaseManager, memory_id: str = "m0") -> None:
        db.store_memory(MemoryRecord(memory_id=memory_id, content="parent"))

    def test_store_and_get_all_facts(self, db: DatabaseManager) -> None:
        self._store_parent_memory(db)
        f = AtomicFact(
            fact_id="f1", memory_id="m0",
            content="Alice is an engineer",
            fact_type=FactType.SEMANTIC,
        )
        result_id = db.store_fact(f)
        assert result_id == "f1"

        facts = db.get_all_facts("default")
        assert len(facts) == 1
        assert facts[0].content == "Alice is an engineer"
        assert facts[0].fact_type == FactType.SEMANTIC

    def test_get_facts_by_type(self, db: DatabaseManager) -> None:
        self._store_parent_memory(db)
        db.store_fact(AtomicFact(
            fact_id="f_sem", memory_id="m0", content="cats are mammals",
            fact_type=FactType.SEMANTIC,
        ))
        db.store_fact(AtomicFact(
            fact_id="f_epi", memory_id="m0", content="Alice met Bob",
            fact_type=FactType.EPISODIC,
        ))

        semantic = db.get_facts_by_type(FactType.SEMANTIC, "default")
        assert len(semantic) == 1
        assert semantic[0].fact_id == "f_sem"

        episodic = db.get_facts_by_type(FactType.EPISODIC, "default")
        assert len(episodic) == 1
        assert episodic[0].fact_id == "f_epi"

    def test_get_facts_by_entity(self, db: DatabaseManager) -> None:
        self._store_parent_memory(db)
        db.store_fact(AtomicFact(
            fact_id="fe1", memory_id="m0",
            content="Alice works at Acme",
            canonical_entities=["ent_alice"],
        ))
        db.store_fact(AtomicFact(
            fact_id="fe2", memory_id="m0",
            content="Bob works at Beta",
            canonical_entities=["ent_bob"],
        ))

        alice_facts = db.get_facts_by_entity("ent_alice", "default")
        assert len(alice_facts) == 1
        assert alice_facts[0].fact_id == "fe1"

    def test_update_fact(self, db: DatabaseManager) -> None:
        self._store_parent_memory(db)
        db.store_fact(AtomicFact(
            fact_id="f_upd", memory_id="m0", content="old content",
        ))
        db.update_fact("f_upd", {"content": "new content", "confidence": 0.9})

        facts = db.get_all_facts("default")
        match = [f for f in facts if f.fact_id == "f_upd"]
        assert len(match) == 1
        assert match[0].content == "new content"
        assert match[0].confidence == 0.9

    def test_update_fact_empty_raises(self, db: DatabaseManager) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            db.update_fact("f_any", {})

    def test_delete_fact(self, db: DatabaseManager) -> None:
        self._store_parent_memory(db)
        db.store_fact(AtomicFact(
            fact_id="f_del", memory_id="m0", content="to delete",
        ))
        db.delete_fact("f_del")
        assert db.get_fact_count("default") == 0

    def test_get_fact_count(self, db: DatabaseManager) -> None:
        self._store_parent_memory(db)
        assert db.get_fact_count("default") == 0
        db.store_fact(AtomicFact(fact_id="fc1", memory_id="m0", content="a"))
        db.store_fact(AtomicFact(fact_id="fc2", memory_id="m0", content="b"))
        assert db.get_fact_count("default") == 2


# ---------------------------------------------------------------------------
# Entity CRUD
# ---------------------------------------------------------------------------

class TestStoreEntity:
    def test_store_and_get_by_name(self, db: DatabaseManager) -> None:
        entity = CanonicalEntity(
            entity_id="e1", canonical_name="Alice",
            entity_type="person",
        )
        db.store_entity(entity)
        found = db.get_entity_by_name("Alice", "default")
        assert found is not None
        assert found.entity_id == "e1"
        assert found.entity_type == "person"

    def test_get_entity_case_insensitive(self, db: DatabaseManager) -> None:
        db.store_entity(CanonicalEntity(
            entity_id="e2", canonical_name="Bob",
        ))
        assert db.get_entity_by_name("bob", "default") is not None
        assert db.get_entity_by_name("BOB", "default") is not None

    def test_get_entity_not_found(self, db: DatabaseManager) -> None:
        assert db.get_entity_by_name("nonexistent", "default") is None


# ---------------------------------------------------------------------------
# Alias CRUD
# ---------------------------------------------------------------------------

class TestStoreAlias:
    def test_store_and_get_aliases(self, db: DatabaseManager) -> None:
        db.store_entity(CanonicalEntity(entity_id="ea", canonical_name="Alice"))
        db.store_alias(EntityAlias(
            alias_id="a1", entity_id="ea", alias="Ali", source="test",
        ))
        db.store_alias(EntityAlias(
            alias_id="a2", entity_id="ea", alias="A", source="test",
        ))
        aliases = db.get_aliases_for_entity("ea")
        assert len(aliases) == 2
        alias_names = {a.alias for a in aliases}
        assert alias_names == {"Ali", "A"}


# ---------------------------------------------------------------------------
# Edge CRUD
# ---------------------------------------------------------------------------

class TestStoreEdge:
    def test_store_and_get_edges(self, db: DatabaseManager) -> None:
        edge = GraphEdge(
            edge_id="edge1", source_id="f1", target_id="f2",
            edge_type=EdgeType.SEMANTIC, weight=0.9,
        )
        db.store_edge(edge)
        edges = db.get_edges_for_node("f1", "default")
        assert len(edges) == 1
        assert edges[0].edge_type == EdgeType.SEMANTIC
        assert edges[0].weight == 0.9

    def test_get_edges_for_node_as_target(self, db: DatabaseManager) -> None:
        db.store_edge(GraphEdge(
            edge_id="e_t", source_id="x", target_id="y",
        ))
        edges = db.get_edges_for_node("y", "default")
        assert len(edges) == 1


# ---------------------------------------------------------------------------
# Temporal events
# ---------------------------------------------------------------------------

class TestTemporalEvents:
    def test_store_and_get(self, db: DatabaseManager) -> None:
        db.store_entity(CanonicalEntity(entity_id="te_e", canonical_name="Eve"))
        db.store_memory(MemoryRecord(memory_id="te_m", content="event"))
        db.store_fact(AtomicFact(fact_id="te_f", memory_id="te_m", content="event fact"))

        event = TemporalEvent(
            event_id="te1", entity_id="te_e", fact_id="te_f",
            observation_date="2026-03-11",
            referenced_date="2026-03-10",
            description="Eve started a new job",
        )
        db.store_temporal_event(event)

        events = db.get_temporal_events("te_e", "default")
        assert len(events) == 1
        assert events[0].description == "Eve started a new job"
        assert events[0].observation_date == "2026-03-11"


# ---------------------------------------------------------------------------
# BM25 token persistence
# ---------------------------------------------------------------------------

class TestBM25Tokens:
    def test_store_and_retrieve(self, db: DatabaseManager) -> None:
        db.store_bm25_tokens("f_bm", "default", ["alice", "works", "acme"])
        index = db.get_all_bm25_tokens("default")
        assert "f_bm" in index
        assert index["f_bm"] == ["alice", "works", "acme"]

    def test_upsert_replaces(self, db: DatabaseManager) -> None:
        db.store_bm25_tokens("f_bm2", "default", ["old"])
        db.store_bm25_tokens("f_bm2", "default", ["new", "tokens"])
        index = db.get_all_bm25_tokens("default")
        assert index["f_bm2"] == ["new", "tokens"]

    def test_empty_for_unknown_profile(self, db: DatabaseManager) -> None:
        index = db.get_all_bm25_tokens("nonexistent")
        assert index == {}


# ---------------------------------------------------------------------------
# FTS5 search
# ---------------------------------------------------------------------------

class TestFTS5Search:
    def test_search_by_text(self, db: DatabaseManager) -> None:
        db.store_memory(MemoryRecord(memory_id="m_fts", content="parent"))
        db.store_fact(AtomicFact(
            fact_id="fts_1", memory_id="m_fts",
            content="Alice loves hiking in the mountains",
        ))
        db.store_fact(AtomicFact(
            fact_id="fts_2", memory_id="m_fts",
            content="Bob enjoys swimming in the ocean",
        ))

        results = db.search_facts_fts("hiking", "default")
        assert len(results) == 1
        assert results[0].fact_id == "fts_1"

    def test_fts_no_results(self, db: DatabaseManager) -> None:
        results = db.search_facts_fts("xyzzyx_no_match", "default")
        assert results == []


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------

class TestTransaction:
    def test_commit_on_success(self, db: DatabaseManager) -> None:
        with db.transaction():
            db.execute(
                "INSERT INTO memories (memory_id, profile_id, content) "
                "VALUES ('txn_ok', 'default', 'committed')"
            )
        rows = db.execute(
            "SELECT * FROM memories WHERE memory_id = 'txn_ok'"
        )
        assert len(rows) == 1

    def test_rollback_on_error(self, db: DatabaseManager) -> None:
        try:
            with db.transaction():
                db.execute(
                    "INSERT INTO memories (memory_id, profile_id, content) "
                    "VALUES ('txn_fail', 'default', 'should rollback')"
                )
                raise RuntimeError("force rollback")
        except RuntimeError:
            pass

        rows = db.execute(
            "SELECT * FROM memories WHERE memory_id = 'txn_fail'"
        )
        assert len(rows) == 0, "Transaction should have rolled back"

    def test_other_thread_never_reuses_uncommitted_transaction_connection(
        self, db: DatabaseManager,
    ) -> None:
        writer_ready = threading.Event()
        release_writer = threading.Event()
        writer_errors: list[Exception] = []

        def writer() -> None:
            try:
                with db.transaction():
                    db.execute(
                        "INSERT INTO memories (memory_id, profile_id, content) "
                        "VALUES ('thread_txn', 'default', 'uncommitted')"
                    )
                    writer_ready.set()
                    assert release_writer.wait(timeout=5)
            except Exception as exc:  # pragma: no cover - asserted below
                writer_errors.append(exc)

        thread = threading.Thread(target=writer)
        thread.start()
        assert writer_ready.wait(timeout=5)
        try:
            rows = db.execute(
                "SELECT * FROM memories WHERE memory_id='thread_txn'"
            )
            assert rows == []
        finally:
            release_writer.set()
            thread.join(timeout=5)

        assert not thread.is_alive()
        assert writer_errors == []


# ---------------------------------------------------------------------------
# Profile isolation
# ---------------------------------------------------------------------------

class TestProfileIsolation:
    def test_two_profiles_data_separated(
        self, db_with_profile: DatabaseManager
    ) -> None:
        db = db_with_profile

        # Store memory + fact in default
        db.store_memory(MemoryRecord(
            memory_id="md", profile_id="default", content="default data",
        ))
        db.store_fact(AtomicFact(
            fact_id="fd", memory_id="md", profile_id="default",
            content="default fact",
        ))

        # Store memory + fact in work
        db.store_memory(MemoryRecord(
            memory_id="mw", profile_id="work", content="work data",
        ))
        db.store_fact(AtomicFact(
            fact_id="fw", memory_id="mw", profile_id="work",
            content="work fact",
        ))

        # Verify isolation
        default_facts = db.get_all_facts("default")
        work_facts = db.get_all_facts("work")

        assert len(default_facts) == 1
        assert default_facts[0].fact_id == "fd"

        assert len(work_facts) == 1
        assert work_facts[0].fact_id == "fw"

    def test_entity_isolated_by_profile(
        self, db_with_profile: DatabaseManager
    ) -> None:
        db = db_with_profile
        db.store_entity(CanonicalEntity(
            entity_id="e_def", profile_id="default", canonical_name="Alice",
        ))
        db.store_entity(CanonicalEntity(
            entity_id="e_wrk", profile_id="work", canonical_name="Alice",
        ))

        default_alice = db.get_entity_by_name("Alice", "default")
        work_alice = db.get_entity_by_name("Alice", "work")

        assert default_alice is not None
        assert work_alice is not None
        assert default_alice.entity_id != work_alice.entity_id


# ---------------------------------------------------------------------------
# Config store
# ---------------------------------------------------------------------------

class TestConfigStore:
    def test_set_and_get_config(self, db: DatabaseManager) -> None:
        db.set_config("mode", "a")
        assert db.get_config("mode") == "a"

    def test_get_config_returns_none_for_missing(self, db: DatabaseManager) -> None:
        assert db.get_config("nonexistent") is None

    def test_set_config_upsert(self, db: DatabaseManager) -> None:
        db.set_config("k", "v1")
        db.set_config("k", "v2")
        assert db.get_config("k") == "v2"


# ---------------------------------------------------------------------------
# Context manager protocol
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_works_as_context_manager(self, tmp_path: Path) -> None:
        db_path = tmp_path / "ctx.db"
        with DatabaseManager(db_path) as mgr:
            mgr.initialize(real_schema)
            mgr.store_memory(MemoryRecord(memory_id="ctx1", content="ctx test"))
            rows = mgr.execute("SELECT * FROM memories WHERE memory_id = 'ctx1'")
            assert len(rows) == 1

    def test_list_tables(self, db: DatabaseManager) -> None:
        tables = db.list_tables()
        assert "memories" in tables
        assert "atomic_facts" in tables
        assert "canonical_entities" in tables


# ---------------------------------------------------------------------------
# WAL PRAGMA ordering
# ---------------------------------------------------------------------------

class TestEnableWal:
    def test_enable_wal_sets_busy_timeout(self, tmp_path: Path) -> None:
        """_enable_wal() sets busy_timeout so subsequent connections inherit WAL mode safely."""
        db_path = tmp_path / "test.db"
        db = DatabaseManager(db_path)
        db.initialize(__import__("superlocalmemory.storage.schema", fromlist=["schema"]))
        # Verify busy_timeout is correctly configured on DatabaseManager-managed connections
        timeout = db.execute("PRAGMA busy_timeout")[0][0]
        assert timeout == 10000, f"Expected busy_timeout=10000, got {timeout}"
        # Verify WAL mode is active
        journal = db.execute("PRAGMA journal_mode")[0][0]
        assert journal.lower() == "wal", f"Expected wal, got {journal}"


# ---------------------------------------------------------------------------
# v3.6.4: Idempotent fact writes (content-level dedup)
# ---------------------------------------------------------------------------
#
# Invariant: storing the same fact content twice for the same profile is a
# no-op on row count — it reinforces the existing fact instead of creating a
# duplicate. This prevents the duplicate explosion that poisons importance
# ranking and core-memory promotion (root cause of recall noise).

class TestStoreFactIdempotent:
    def _parent(self, db: DatabaseManager, mid: str = "m0") -> None:
        db.store_memory(MemoryRecord(memory_id=mid, content="parent"))

    def test_duplicate_content_creates_one_fact(self, db: DatabaseManager) -> None:
        self._parent(db)
        c = "Varun ships SLM v3.6.4"
        id1 = db.store_fact(AtomicFact(fact_id="f1", memory_id="m0", content=c,
                                       fact_type=FactType.SEMANTIC))
        id2 = db.store_fact(AtomicFact(fact_id="f2", memory_id="m0", content=c,
                                       fact_type=FactType.SEMANTIC))
        facts = db.get_all_facts("default")
        assert len(facts) == 1, "duplicate content must not create a second fact"
        # Second write returns the canonical (first) fact_id, not a new one.
        assert id1 == "f1"
        assert id2 == "f1"

    def test_duplicate_reinforces_evidence_count(self, db: DatabaseManager) -> None:
        self._parent(db)
        c = "Idempotent writes are a memory-system invariant"
        for fid in ("a", "b", "c3"):
            db.store_fact(AtomicFact(fact_id=fid, memory_id="m0", content=c,
                                     fact_type=FactType.SEMANTIC, evidence_count=1))
        facts = db.get_all_facts("default")
        assert len(facts) == 1
        # Three writes of the same content → evidence reinforced (>= 3).
        assert facts[0].evidence_count >= 3

    def test_distinct_content_creates_separate_facts(self, db: DatabaseManager) -> None:
        self._parent(db)
        db.store_fact(AtomicFact(fact_id="x", memory_id="m0", content="fact A",
                                 fact_type=FactType.SEMANTIC))
        db.store_fact(AtomicFact(fact_id="y", memory_id="m0", content="fact B",
                                 fact_type=FactType.SEMANTIC))
        assert len(db.get_all_facts("default")) == 2

    def test_dedup_is_profile_scoped(self, db_with_profile: DatabaseManager) -> None:
        self._parent(db_with_profile)
        c = "shared content across profiles"
        db_with_profile.store_fact(AtomicFact(fact_id="p1", memory_id="m0",
                                              profile_id="default", content=c,
                                              fact_type=FactType.SEMANTIC))
        db_with_profile.store_fact(AtomicFact(fact_id="p2", memory_id="m0",
                                              profile_id="work", content=c,
                                              fact_type=FactType.SEMANTIC))
        # Same content under two profiles → two facts (profile isolation holds).
        assert len(db_with_profile.get_all_facts("default")) == 1
        assert len(db_with_profile.get_all_facts("work")) == 1

    def test_dedup_mutates_fact_id_to_canonical(self, db: DatabaseManager) -> None:
        # The passed fact's id is rewritten to the canonical id so downstream
        # embedding/graph writes (keyed on fact.fact_id) target the real fact.
        self._parent(db)
        c = "canonical id rewrite check"
        db.store_fact(AtomicFact(fact_id="orig", memory_id="m0", content=c,
                                 fact_type=FactType.SEMANTIC))
        dup = AtomicFact(fact_id="dup", memory_id="m0", content=c,
                         fact_type=FactType.SEMANTIC)
        db.store_fact(dup)
        assert dup.fact_id == "orig"

    def test_dedup_matches_warm_and_cold_facts(self, db: DatabaseManager) -> None:
        # P0-2: a fact that ages to warm/cold must still dedup — otherwise the
        # 9944 warm + 3672 cold facts re-open the duplication window.
        self._parent(db)
        for zone in ("warm", "cold"):
            c = f"{zone} lifecycle dedup check"
            id1 = db.store_fact(AtomicFact(fact_id=f"{zone}1", memory_id="m0",
                                           content=c, fact_type=FactType.SEMANTIC))
            db.execute("UPDATE atomic_facts SET lifecycle = ? WHERE fact_id = ?",
                       (zone, id1))
            dup = AtomicFact(fact_id=f"{zone}2", memory_id="m0", content=c,
                             fact_type=FactType.SEMANTIC)
            id2 = db.store_fact(dup)
            assert id2 == id1, f"{zone} fact was not deduped"
            assert dup.fact_id == id1
            rows = db.execute("SELECT COUNT(*) AS c FROM atomic_facts WHERE content = ?", (c,))
            assert dict(rows[0])["c"] == 1, f"duplicate row created for {zone} fact"

    def test_gc_removes_orphaned_embedding_metadata(self, db: DatabaseManager) -> None:
        # P1-3: an orphan arises when a fact is deleted via an FK-OFF connection.
        import sqlite3
        self._parent(db)
        db.store_fact(AtomicFact(fact_id="real1", memory_id="m0", content="kept",
                                 fact_type=FactType.SEMANTIC))
        db.execute(
            "INSERT INTO embedding_metadata (vec_rowid, fact_id, profile_id) "
            "VALUES (1, 'real1', 'default')"
        )
        # Insert an orphan exactly how production does — a connection with FK OFF.
        raw = sqlite3.connect(str(db.db_path))
        raw.execute(
            "INSERT INTO embedding_metadata (vec_rowid, fact_id, profile_id) "
            "VALUES (2, 'ghost', 'default')"
        )
        raw.commit()
        raw.close()

        removed = db.gc_orphaned_embedding_metadata()
        assert removed == 1
        rows = db.execute("SELECT fact_id FROM embedding_metadata ORDER BY fact_id")
        assert [dict(r)["fact_id"] for r in rows] == ["real1"]
        # Idempotent: second sweep removes nothing.
        assert db.gc_orphaned_embedding_metadata() == 0

    def test_delete_fact_removes_its_embedding_metadata(self, db: DatabaseManager) -> None:
        self._parent(db)
        db.store_fact(AtomicFact(fact_id="d1", memory_id="m0", content="bye",
                                 fact_type=FactType.SEMANTIC))
        db.execute(
            "INSERT INTO embedding_metadata (vec_rowid, fact_id, profile_id) "
            "VALUES (9, 'd1', 'default')"
        )
        db.delete_fact("d1")
        rows = db.execute("SELECT COUNT(*) AS c FROM embedding_metadata WHERE fact_id='d1'")
        assert dict(rows[0])["c"] == 0

    def test_get_all_facts_respects_sql_limit(self, db: DatabaseManager) -> None:
        # memory-bounding-02: limit pushed into SQL, newest-first.
        self._parent(db)
        for i in range(5):
            db.store_fact(AtomicFact(
                fact_id=f"lf{i}", memory_id="m0", content=f"limited fact number {i}",
                fact_type=FactType.SEMANTIC, created_at=f"2026-01-0{i + 1}T00:00:00",
            ))
        limited = db.get_all_facts("default", limit=2)
        assert len(limited) == 2
        assert limited[0].fact_id == "lf4", "must return newest first under LIMIT"
        # Default (no limit) returns all.
        assert len(db.get_all_facts("default")) == 5

    def test_dedup_excludes_archived(self, db: DatabaseManager) -> None:
        # Archived == soft-deleted / forgotten. Re-storing the same content must
        # create a FRESH active fact (re-learning), NOT resurrect the archived one.
        self._parent(db)
        c = "archived must not dedup-match"
        id1 = db.store_fact(AtomicFact(fact_id="arch1", memory_id="m0", content=c,
                                       fact_type=FactType.SEMANTIC))
        db.execute("UPDATE atomic_facts SET lifecycle = 'archived' WHERE fact_id = ?", (id1,))
        id2 = db.store_fact(AtomicFact(fact_id="arch2", memory_id="m0", content=c,
                                       fact_type=FactType.SEMANTIC))
        assert id2 == "arch2", "archived fact was wrongly resurrected by dedup"
        rows = db.execute(
            "SELECT COUNT(*) AS c FROM atomic_facts WHERE content = ? AND lifecycle != 'archived'",
            (c,),
        )
        assert dict(rows[0])["c"] == 1


# ---------------------------------------------------------------------------
# _jl sentinel — explicit default=None must round-trip as None, not []
# ---------------------------------------------------------------------------

class TestJlSentinel:
    """Regression: _jl(raw, None) must return None for NULL/empty raw.

    The pre-fix implementation collapsed "no default supplied" and "explicit
    None default" into the same branch (`default if default is not None else
    []`), so optional-list/optional-vector columns like fisher_variance,
    embedding, langevin_position, shared_with loaded as [] instead of None,
    defeating downstream `is None` guards.
    """

    def test_no_default_returns_empty_list_for_null(self) -> None:
        from superlocalmemory.storage.database import _jl
        assert _jl(None) == []
        assert _jl("") == []

    def test_explicit_none_default_returns_none_for_null(self) -> None:
        from superlocalmemory.storage.database import _jl
        assert _jl(None, None) is None
        assert _jl("", None) is None

    def test_explicit_other_default_preserved(self) -> None:
        from superlocalmemory.storage.database import _jl
        assert _jl(None, 0) == 0
        assert _jl(None, {"k": "v"}) == {"k": "v"}

    def test_non_null_raw_always_parsed(self) -> None:
        from superlocalmemory.storage.database import _jl
        assert _jl("[1, 2, 3]") == [1, 2, 3]
        assert _jl("[1, 2, 3]", None) == [1, 2, 3]
        assert _jl('{"a": 1}', None) == {"a": 1}


# ---------------------------------------------------------------------------
# Env-tunable SQLite endurance knobs (#53) — bad/absent env must fall back to
# the documented defaults so unset config is byte-identical to the old code.
# ---------------------------------------------------------------------------

class TestDbEnvTuning:
    def test_env_int_absent_returns_default(self, monkeypatch) -> None:
        from superlocalmemory.storage.database import _env_int
        monkeypatch.delenv("SLM_DB_BUSY_TIMEOUT_MS", raising=False)
        assert _env_int("SLM_DB_BUSY_TIMEOUT_MS", 10_000) == 10_000

    def test_env_int_valid_override(self, monkeypatch) -> None:
        from superlocalmemory.storage.database import _env_int
        monkeypatch.setenv("SLM_DB_BUSY_TIMEOUT_MS", "30000")
        assert _env_int("SLM_DB_BUSY_TIMEOUT_MS", 10_000) == 30_000

    def test_env_int_garbage_falls_back(self, monkeypatch) -> None:
        from superlocalmemory.storage.database import _env_int
        monkeypatch.setenv("SLM_DB_MAX_RETRIES", "not-a-number")
        assert _env_int("SLM_DB_MAX_RETRIES", 5) == 5

    def test_env_int_nonpositive_falls_back(self, monkeypatch) -> None:
        from superlocalmemory.storage.database import _env_int
        monkeypatch.setenv("SLM_DB_MAX_RETRIES", "0")
        assert _env_int("SLM_DB_MAX_RETRIES", 5) == 5

    def test_env_float_valid_and_garbage(self, monkeypatch) -> None:
        from superlocalmemory.storage.database import _env_float
        monkeypatch.setenv("SLM_DB_RETRY_BASE_DELAY", "0.5")
        assert _env_float("SLM_DB_RETRY_BASE_DELAY", 0.1) == 0.5
        monkeypatch.setenv("SLM_DB_RETRY_BASE_DELAY", "nope")
        assert _env_float("SLM_DB_RETRY_BASE_DELAY", 0.1) == 0.1
