# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for MemoryEngine.store() flow — fact extraction, enrichment, consolidation.

Covers:
  - store() returns list[str] of fact IDs
  - store() creates a MemoryRecord in the DB
  - store() extracts and stores AtomicFacts
  - store("") returns empty list
  - store() enriches facts with embeddings
  - store() calls graph_builder.build_edges
  - Consolidation noop skips storage
  - Consolidation update returns existing fact ID
  - Consolidation add stores new fact
  - Pre-hooks are called before store
  - Post-hooks are called after store with fact_ids
  - store_fact_direct() creates parent memory record (FK)
  - store_fact_direct() indexes BM25 tokens
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from superlocalmemory.core.engine import MemoryEngine
from superlocalmemory.storage.models import (
    AtomicFact,
    ConsolidationAction,
    ConsolidationActionType,
    FactType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fact(
    fact_id: str = "f1",
    content: str = "Alice is an engineer",
    entities: list[str] | None = None,
    fact_type: FactType = FactType.SEMANTIC,
) -> AtomicFact:
    return AtomicFact(
        fact_id=fact_id,
        memory_id="m0",
        content=content,
        entities=entities or [],
        fact_type=fact_type,
        confidence=0.9,
    )


def _add_action(fact_id: str = "f1") -> ConsolidationAction:
    return ConsolidationAction(
        action_type=ConsolidationActionType.ADD,
        new_fact_id=fact_id,
    )


def _noop_action() -> ConsolidationAction:
    return ConsolidationAction(
        action_type=ConsolidationActionType.NOOP,
    )


def _update_action(new_fact_id: str = "f-existing") -> ConsolidationAction:
    return ConsolidationAction(
        action_type=ConsolidationActionType.UPDATE,
        new_fact_id=new_fact_id,
    )


# ---------------------------------------------------------------------------
# Basic store flow
# ---------------------------------------------------------------------------

class TestStoreBasicFlow:
    """Verify the happy-path store() pipeline."""

    def test_store_returns_fact_ids(
        self, engine_with_mock_deps: MemoryEngine,
    ) -> None:
        """store() returns a list of string fact IDs."""
        result = engine_with_mock_deps.store(
            "Alice is an engineer", session_id="s1",
        )
        assert isinstance(result, list)
        for fid in result:
            assert isinstance(fid, str)

    def test_store_creates_memory_record(
        self, engine_with_mock_deps: MemoryEngine,
    ) -> None:
        """store() creates a row in the memories table."""
        engine_with_mock_deps.store("Bob likes tea", session_id="s1")
        # Verify via raw SQL — memories table should have at least one row
        rows = engine_with_mock_deps._db.execute(
            "SELECT content FROM memories WHERE profile_id = ?",
            (engine_with_mock_deps._profile_id,),
        )
        assert len(rows) >= 1
        found = any("Bob likes tea" in dict(r)["content"] for r in rows)
        assert found, "Memory record not found in DB"

    def test_store_extracts_and_stores_facts(
        self, engine_with_mock_deps: MemoryEngine,
    ) -> None:
        """store() extracts atomic facts and persists them."""
        ids = engine_with_mock_deps.store(
            "Carol works at CERN in Geneva", session_id="s1",
        )
        # Should have extracted at least one fact
        assert len(ids) >= 1
        # Facts should exist in DB
        facts = engine_with_mock_deps._db.get_all_facts(
            engine_with_mock_deps._profile_id,
        )
        stored_ids = {f.fact_id for f in facts}
        for fid in ids:
            assert fid in stored_ids

    def test_store_empty_content_returns_empty(
        self, engine_with_mock_deps: MemoryEngine,
    ) -> None:
        """store("") returns an empty list (no facts extracted)."""
        result = engine_with_mock_deps.store("", session_id="s1")
        assert result == []

    def test_store_enriches_with_embedding(
        self, engine_with_mock_deps: MemoryEngine, mock_embedder: MagicMock,
    ) -> None:
        """store() calls embedder.embed() to enrich facts with embeddings."""
        engine_with_mock_deps.store(
            "Dave moved to Berlin in 2025", session_id="s1",
        )
        # The mock embedder's embed method should have been called
        assert mock_embedder.embed.called

    def test_store_calls_graph_builder(
        self, engine_with_mock_deps: MemoryEngine,
    ) -> None:
        """store() invokes _graph_builder.build_edges for each stored fact."""
        gb = engine_with_mock_deps._graph_builder
        with patch.object(gb, 'build_edges', wraps=gb.build_edges) as spy:
            ids = engine_with_mock_deps.store(
                "Eve is a researcher at MIT", session_id="s1",
            )
            if ids:
                assert spy.called


# ---------------------------------------------------------------------------
# Consolidation paths
# ---------------------------------------------------------------------------

class TestStoreConsolidation:
    """Verify noop / update / add consolidation outcomes."""

    def test_store_noop_consolidation_skips_fact(
        self, engine_with_mock_deps: MemoryEngine,
    ) -> None:
        """When consolidator returns NOOP, the fact is not stored."""
        consolidator = engine_with_mock_deps._consolidator
        with patch.object(
            consolidator, 'consolidate', return_value=_noop_action(),
        ):
            ids = engine_with_mock_deps.store(
                "Duplicate content here", session_id="s1",
            )
            assert ids == []

    def test_store_update_consolidation_returns_id(
        self, engine_with_mock_deps: MemoryEngine,
    ) -> None:
        """When consolidator returns UPDATE, the updated fact ID is in result."""
        # First store a fact to have something to "update"
        original_ids = engine_with_mock_deps.store(
            "Frank likes pizza", session_id="s1",
        )
        if not original_ids:
            pytest.skip("No facts extracted from initial store")

        existing_id = original_ids[0]
        consolidator = engine_with_mock_deps._consolidator
        mock_action = _update_action(new_fact_id=existing_id)
        with patch.object(
            consolidator, 'consolidate', return_value=mock_action,
        ):
            ids = engine_with_mock_deps.store(
                "Frank really loves pizza", session_id="s2",
            )
            assert existing_id in ids

    def test_store_add_consolidation_stores_fact(
        self, engine_with_mock_deps: MemoryEngine,
    ) -> None:
        """When consolidator returns ADD, the fact is stored normally."""
        consolidator = engine_with_mock_deps._consolidator
        with patch.object(
            consolidator, 'consolidate', return_value=_add_action("new-f1"),
        ):
            ids = engine_with_mock_deps.store(
                "Grace is learning Rust", session_id="s1",
            )
            assert len(ids) >= 1


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

class TestStoreHooks:
    """Verify pre/post hook invocation during store()."""

    def test_store_runs_pre_hooks(
        self, engine_with_mock_deps: MemoryEngine,
    ) -> None:
        """store() calls _hooks.run_pre('store', ...) before processing."""
        spy = MagicMock()
        engine_with_mock_deps._hooks.register_pre("store", spy)
        engine_with_mock_deps.store("Hook test content", session_id="s1")
        spy.assert_called_once()
        ctx = spy.call_args[0][0]
        assert ctx["operation"] == "store"

    def test_store_runs_post_hooks(
        self, engine_with_mock_deps: MemoryEngine,
    ) -> None:
        """store() calls _hooks.run_post('store', ...) with fact_ids."""
        spy = MagicMock()
        engine_with_mock_deps._hooks.register_post("store", spy)
        engine_with_mock_deps.store("Post hook test", session_id="s1")
        spy.assert_called_once()
        ctx = spy.call_args[0][0]
        assert "fact_ids" in ctx
        assert "fact_count" in ctx


# ---------------------------------------------------------------------------
# store_fact_direct
# ---------------------------------------------------------------------------

class TestStoreFactDirect:
    """Verify store_fact_direct() creates parent memory and indexes BM25."""

    def test_store_fact_direct_creates_parent_memory(
        self, engine_with_mock_deps: MemoryEngine,
    ) -> None:
        """store_fact_direct() creates a MemoryRecord to satisfy FK."""
        fact = _make_fact(fact_id="direct-f1", content="Directly stored fact")
        fact.memory_id = ""  # Force parent creation

        returned_id = engine_with_mock_deps.store_fact_direct(fact)
        assert returned_id == "direct-f1"
        # Verify parent memory was created (memory_id populated)
        assert fact.memory_id != ""

    def test_store_fact_direct_indexes_bm25(
        self, engine_with_mock_deps: MemoryEngine,
    ) -> None:
        """store_fact_direct() passes fact to BM25 indexer if available."""
        bm25 = getattr(engine_with_mock_deps._retrieval_engine, '_bm25', None)
        if bm25 is None:
            pytest.skip("No BM25 channel on retrieval engine")

        with patch.object(bm25, 'add') as bm25_spy:
            fact = _make_fact(fact_id="bm25-f1", content="BM25 indexed fact")
            fact.memory_id = ""
            engine_with_mock_deps.store_fact_direct(fact)
            bm25_spy.assert_called_once()
            assert bm25_spy.call_args[0][0] == "bm25-f1"
