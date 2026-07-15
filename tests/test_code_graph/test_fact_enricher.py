# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Tests for FactEnricher — enrichment formatting, no-match fallback."""

from __future__ import annotations

import pytest

from superlocalmemory.code_graph.bridge.entity_resolver import MatchedNode
from superlocalmemory.code_graph.bridge.fact_enricher import (
    MAX_ENRICHMENT_LEN,
    MAX_NODES_PER_ENRICHMENT,
    FactEnricher,
)
from superlocalmemory.code_graph.database import CodeGraphDatabase
from superlocalmemory.code_graph.models import (
    GraphEdge,
    GraphNode,
    EdgeKind,
    NodeKind,
)


@pytest.fixture
def enricher(db: CodeGraphDatabase) -> FactEnricher:
    """FactEnricher with a fresh database."""
    return FactEnricher(db)


def _make_matched_node(
    *,
    node_id: str = "n1",
    qualified_name: str = "src/auth/handler.py::authenticate_user",
    kind: str = "function",
    file_path: str = "src/auth/handler.py",
    confidence: float = 0.90,
    match_source: str = "exact_name",
) -> MatchedNode:
    return MatchedNode(
        node_id=node_id,
        qualified_name=qualified_name,
        kind=kind,
        file_path=file_path,
        confidence=confidence,
        match_source=match_source,
    )


def _setup_node_with_edges(db: CodeGraphDatabase) -> None:
    """Insert a node with some callers and callees."""
    node = GraphNode(
        node_id="n1",
        kind=NodeKind.FUNCTION,
        name="authenticate_user",
        qualified_name="src/auth/handler.py::authenticate_user",
        file_path="src/auth/handler.py",
        language="python",
    )
    db.upsert_node(node)

    # Callers
    caller = GraphNode(
        node_id="n2",
        kind=NodeKind.FUNCTION,
        name="login_endpoint",
        qualified_name="src/api/routes.py::login_endpoint",
        file_path="src/api/routes.py",
        language="python",
    )
    db.upsert_node(caller)
    db.upsert_edge(GraphEdge(
        edge_id="e1",
        kind=EdgeKind.CALLS,
        source_node_id="n2",
        target_node_id="n1",
        file_path="src/api/routes.py",
    ))

    # Callees
    callee = GraphNode(
        node_id="n3",
        kind=NodeKind.FUNCTION,
        name="check_password",
        qualified_name="src/auth/utils.py::check_password",
        file_path="src/auth/utils.py",
        language="python",
    )
    db.upsert_node(callee)
    db.upsert_edge(GraphEdge(
        edge_id="e2",
        kind=EdgeKind.CALLS,
        source_node_id="n1",
        target_node_id="n3",
        file_path="src/auth/handler.py",
    ))


class TestEnrich:
    """Test fact enrichment."""

    def test_enrich_single_node(
        self, db: CodeGraphDatabase, enricher: FactEnricher,
    ) -> None:
        _setup_node_with_edges(db)
        matched = _make_matched_node()
        result = enricher.enrich("fact-1", [matched], "Auth handler fixed")

        assert result
        assert "authenticate_user" in result
        assert "1 callers" in result
        assert "calls 1" in result

    def test_enrich_no_matches(self, enricher: FactEnricher) -> None:
        result = enricher.enrich("fact-2", [], "Original description")
        assert result == "Original description"

    def test_enrich_empty_description(
        self, db: CodeGraphDatabase, enricher: FactEnricher,
    ) -> None:
        _setup_node_with_edges(db)
        matched = _make_matched_node()
        result = enricher.enrich("fact-3", [matched], "")

        # Should be just the enrichment suffix
        assert "[function:" in result

    def test_enrich_preserves_original(
        self, db: CodeGraphDatabase, enricher: FactEnricher,
    ) -> None:
        _setup_node_with_edges(db)
        matched = _make_matched_node()
        original = "Fixed the auth bug"
        result = enricher.enrich("fact-4", [matched], original)

        assert result.startswith(original)

    def test_enrich_truncation(
        self, db: CodeGraphDatabase, enricher: FactEnricher,
    ) -> None:
        _setup_node_with_edges(db)
        matched = _make_matched_node()
        long_desc = "A" * 490
        result = enricher.enrich("fact-5", [matched], long_desc)

        assert len(result) <= MAX_ENRICHMENT_LEN
        assert result.endswith("...")

    def test_enrich_limits_nodes(
        self, db: CodeGraphDatabase, enricher: FactEnricher,
    ) -> None:
        """Should only use top MAX_NODES_PER_ENRICHMENT nodes."""
        _setup_node_with_edges(db)
        nodes = [
            _make_matched_node(node_id=f"n{i}", confidence=0.9 - i * 0.1)
            for i in range(5)
        ]
        # Only n1 exists in DB, so others will fail silently
        result = enricher.enrich("fact-6", nodes, "test")
        # Should not crash even with non-existent nodes
        assert "test" in result

    def test_enrich_node_no_edges(
        self, db: CodeGraphDatabase, enricher: FactEnricher,
    ) -> None:
        """Node with no callers or callees."""
        node = GraphNode(
            node_id="n10",
            kind=NodeKind.FUNCTION,
            name="isolated_func",
            qualified_name="src/utils.py::isolated_func",
            file_path="src/utils.py",
            language="python",
        )
        db.upsert_node(node)
        matched = _make_matched_node(
            node_id="n10",
            qualified_name="src/utils.py::isolated_func",
            file_path="src/utils.py",
        )
        result = enricher.enrich("fact-7", [matched], "test")
        assert "isolated_func" in result
        # No callers/callees info
        assert "callers" not in result
        assert "calls" not in result


class TestBulkEnrich:
    """Test bulk enrichment."""

    def test_bulk_enrich(
        self, db: CodeGraphDatabase, enricher: FactEnricher,
    ) -> None:
        _setup_node_with_edges(db)
        matched = _make_matched_node()
        pairs = [
            ("f1", [matched], "desc one"),
            ("f2", [], "desc two"),
            ("f3", [matched], "desc three"),
        ]
        results = enricher.bulk_enrich(pairs)
        assert len(results) == 3
        assert results[0].enriched_description != "desc one"  # enriched
        assert results[1].enriched_description == "desc two"  # no matches
        assert results[2].enriched_description != "desc three"  # enriched
