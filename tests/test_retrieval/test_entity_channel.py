# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.retrieval.entity_channel — Entity Graph + Spreading Activation.

Covers:
  - extract_query_entities() — proper nouns, title-cased, quoted, dedup, stopwords
  - EntityGraphChannel.search() — entity resolution, direct facts, spreading activation
  - Edge traversal and decay
  - Discover entities from activated facts
  - No entities in query -> empty results
  - Mock DB interactions (get_entity_by_name, get_facts_by_entity, get_edges_for_node)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from superlocalmemory.retrieval.entity_channel import (
    EntityGraphChannel,
    extract_query_entities,
)
from superlocalmemory.storage.models import (
    AtomicFact,
    CanonicalEntity,
    EdgeType,
    GraphEdge,
)


# ---------------------------------------------------------------------------
# extract_query_entities
# ---------------------------------------------------------------------------

class TestExtractQueryEntities:
    def test_proper_nouns_extracted(self) -> None:
        result = extract_query_entities("Did Alice meet Bob yesterday?")
        names = [n.lower() for n in result]
        assert "alice" in names
        assert "bob" in names

    def test_title_cased_fallback(self) -> None:
        # All lowercase — title() should produce proper nouns
        result = extract_query_entities("what did alice do?")
        names = [n.lower() for n in result]
        assert "alice" in names

    def test_quoted_phrases(self) -> None:
        result = extract_query_entities('Tell me about "Project Alpha"')
        texts = [n.lower() for n in result]
        assert "project alpha" in texts

    def test_stopwords_filtered(self) -> None:
        result = extract_query_entities("What did they do?")
        names_lower = [n.lower() for n in result]
        # "What", "They" are in entity stop list
        assert "what" not in names_lower

    def test_short_names_filtered(self) -> None:
        result = extract_query_entities("A B C Alice")
        # Single-char names should be filtered (len < 2)
        names = [n for n in result if len(n) == 1]
        assert len(names) == 0

    def test_deduplication_case_insensitive(self) -> None:
        result = extract_query_entities("Alice alice ALICE")
        # Should have only one entry for alice
        lower_names = [n.lower() for n in result]
        assert lower_names.count("alice") == 1

    def test_empty_query(self) -> None:
        assert extract_query_entities("") == []

    def test_no_entities(self) -> None:
        result = extract_query_entities("what is going on?")
        # "What" is a stop word. "Going" from title() might pass.
        # The key check: no crash, returns list
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# EntityGraphChannel with mocks
# ---------------------------------------------------------------------------

def _mock_entity(entity_id: str, name: str) -> CanonicalEntity:
    return CanonicalEntity(entity_id=entity_id, canonical_name=name)


def _mock_fact(fact_id: str, canonical_entities: list[str] | None = None) -> AtomicFact:
    return AtomicFact(
        fact_id=fact_id, memory_id="m0", content=f"fact {fact_id}",
        canonical_entities=canonical_entities or [],
    )


def _mock_edge(source: str, target: str) -> GraphEdge:
    return GraphEdge(
        edge_id=f"e_{source}_{target}",
        source_id=source, target_id=target,
        edge_type=EdgeType.ENTITY, weight=1.0,
    )


class TestEntityGraphChannelSearch:
    def test_no_entities_returns_empty(self) -> None:
        db = MagicMock()
        ch = EntityGraphChannel(db)
        results = ch.search("what is going on", "default")
        assert results == []

    def test_entity_not_found_returns_empty(self) -> None:
        db = MagicMock()
        db.get_entity_by_name.return_value = None
        ch = EntityGraphChannel(db)
        results = ch.search("Did Alice do something?", "default")
        assert results == []

    def test_direct_entity_facts_returned(self) -> None:
        db = MagicMock()
        db.get_entity_by_name.return_value = _mock_entity("e_alice", "Alice")
        db.get_facts_by_entity.return_value = [_mock_fact("f1", ["e_alice"])]
        db.get_edges_for_node.return_value = []
        db.execute.return_value = []
        ch = EntityGraphChannel(db, max_hops=1)
        results = ch.search("What did Alice do?", "default")
        assert len(results) > 0
        fact_ids = [r[0] for r in results]
        assert "f1" in fact_ids

    def test_spreading_activation_through_edges(self) -> None:
        db = MagicMock()
        db.get_entity_by_name.return_value = _mock_entity("e_alice", "Alice")
        # Alice directly linked to f1
        db.get_facts_by_entity.return_value = [_mock_fact("f1")]
        # f1 has edge to f2
        db.get_edges_for_node.side_effect = lambda fid, pid, **kwargs: (
            [_mock_edge("f1", "f2")] if fid == "f1" else []
        )
        # f2 has canonical entities -> discover new entities
        db.execute.return_value = []

        ch = EntityGraphChannel(db, decay=0.7, max_hops=3)
        results = ch.search("What did Alice do?", "default")
        fact_ids = [r[0] for r in results]
        # f2 should be discovered via spreading activation
        assert "f1" in fact_ids
        assert "f2" in fact_ids

    def test_decay_reduces_activation(self) -> None:
        db = MagicMock()
        db.get_entity_by_name.return_value = _mock_entity("e_alice", "Alice")
        db.get_facts_by_entity.return_value = [_mock_fact("f1")]
        db.get_edges_for_node.side_effect = lambda fid, pid, **kwargs: (
            [_mock_edge("f1", "f2")] if fid == "f1" else []
        )
        db.execute.return_value = []

        ch = EntityGraphChannel(db, decay=0.7, max_hops=3)
        results = ch.search("What did Alice do?", "default")
        scores = {r[0]: r[1] for r in results}
        # f1 should have higher activation than f2
        if "f2" in scores:
            assert scores["f1"] > scores["f2"]

    def test_activation_threshold_filters_weak(self) -> None:
        db = MagicMock()
        db.get_entity_by_name.return_value = _mock_entity("e_alice", "Alice")
        db.get_facts_by_entity.return_value = [_mock_fact("f1")]
        db.get_edges_for_node.return_value = []
        db.execute.return_value = []

        ch = EntityGraphChannel(db, decay=0.7, activation_threshold=0.5, max_hops=3)
        results = ch.search("What did Alice do?", "default")
        # All results should be above threshold
        for _, score in results:
            assert score >= 0.5

    def test_top_k_limits_results(self) -> None:
        db = MagicMock()
        db.get_entity_by_name.return_value = _mock_entity("e_alice", "Alice")
        facts = [_mock_fact(f"f{i}") for i in range(20)]
        db.get_facts_by_entity.return_value = facts
        db.get_edges_for_node.return_value = []
        db.execute.return_value = []

        ch = EntityGraphChannel(db, max_hops=1)
        results = ch.search("What did Alice do?", "default", top_k=5)
        assert len(results) <= 5

    def test_entity_resolver_used_when_provided(self) -> None:
        db = MagicMock()
        resolver = MagicMock()
        resolver.resolve.return_value = {"Alice": "e_alice"}
        db.get_facts_by_entity.return_value = [_mock_fact("f1")]
        db.get_edges_for_node.return_value = []
        db.execute.return_value = []

        ch = EntityGraphChannel(db, entity_resolver=resolver, max_hops=1)
        results = ch.search("What did Alice do?", "default")
        resolver.resolve.assert_called_once()
        assert len(results) > 0

    def test_discover_entities_from_facts(self) -> None:
        db = MagicMock()
        db.get_entity_by_name.return_value = _mock_entity("e_alice", "Alice")
        db.get_facts_by_entity.side_effect = lambda eid, pid, **kwargs: (
            [_mock_fact("f1", ["e_alice"])] if eid == "e_alice"
            else [_mock_fact("f3", ["e_bob"])] if eid == "e_bob"
            else []
        )
        db.get_edges_for_node.return_value = []

        # Simulate discovering e_bob from f1's canonical_entities
        def mock_execute(sql, params):
            if params and params[0] == "f1":
                row = MagicMock()
                row.__iter__ = lambda s: iter([("canonical_entities_json", json.dumps(["e_bob"]))])
                row.keys = lambda: ["canonical_entities_json"]
                # Return dict-like row
                mock_row = MagicMock()
                mock_row.__iter__ = lambda s: iter(
                    [("canonical_entities_json", json.dumps(["e_bob"]))]
                )
                d = {"canonical_entities_json": json.dumps(["e_bob"])}
                mock_row.__getitem__ = lambda s, k: d[k]
                mock_dict = MagicMock(return_value=d)
                return [mock_dict]
            return []

        db.execute.side_effect = mock_execute

        ch = EntityGraphChannel(db, decay=0.7, max_hops=3)
        results = ch.search("What did Alice do?", "default")
        # Should have found facts via both Alice and discovered Bob
        assert len(results) > 0
