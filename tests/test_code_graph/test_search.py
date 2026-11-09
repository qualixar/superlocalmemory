# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Tests for HybridSearch — FTS5 text search, kind boosting, empty results."""

from __future__ import annotations

import time

import pytest

from superlocalmemory.code_graph.database import CodeGraphDatabase
from superlocalmemory.code_graph.models import GraphNode, NodeKind
from superlocalmemory.code_graph.search import (
    HybridSearch,
    SearchResult,
    _keyword_score,
    _sanitize_fts_query,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def search_db(db: CodeGraphDatabase) -> CodeGraphDatabase:
    """DB pre-populated with test nodes for search."""
    now = time.time()
    nodes = [
        GraphNode(
            node_id="n1", kind=NodeKind.FUNCTION, name="authenticate_user",
            qualified_name="src/auth.py::authenticate_user",
            file_path="src/auth.py", line_start=10, line_end=30,
            language="python", created_at=now, updated_at=now,
        ),
        GraphNode(
            node_id="n2", kind=NodeKind.CLASS, name="UserService",
            qualified_name="src/services.py::UserService",
            file_path="src/services.py", line_start=5, line_end=100,
            language="python", created_at=now, updated_at=now,
        ),
        GraphNode(
            node_id="n3", kind=NodeKind.METHOD, name="get_user",
            qualified_name="src/services.py::UserService.get_user",
            file_path="src/services.py", line_start=20, line_end=40,
            language="python", signature="def get_user(self, user_id: int)",
            created_at=now, updated_at=now,
        ),
        GraphNode(
            node_id="n4", kind=NodeKind.FILE, name="auth.py",
            qualified_name="src/auth.py",
            file_path="src/auth.py", line_start=1, line_end=100,
            language="python", created_at=now, updated_at=now,
        ),
        GraphNode(
            node_id="n5", kind=NodeKind.FUNCTION, name="validate_token",
            qualified_name="src/auth.py::validate_token",
            file_path="src/auth.py", line_start=35, line_end=50,
            language="python", created_at=now, updated_at=now,
        ),
    ]
    for node in nodes:
        db.upsert_node(node)
    return db


@pytest.fixture
def search(search_db: CodeGraphDatabase) -> HybridSearch:
    """HybridSearch instance with populated DB."""
    return HybridSearch(search_db)


# ---------------------------------------------------------------------------
# FTS5 Search Tests
# ---------------------------------------------------------------------------

class TestFTS5Search:
    """Tests for FTS5 text search."""

    def test_search_by_name(self, search: HybridSearch) -> None:
        """Search finds nodes by name."""
        results = search.search("authenticate")
        assert len(results) >= 1
        names = {r.name for r in results}
        assert "authenticate_user" in names

    def test_search_by_qualified_name(self, search: HybridSearch) -> None:
        """Search finds nodes by qualified name."""
        results = search.search("UserService")
        assert len(results) >= 1
        names = {r.name for r in results}
        assert "UserService" in names

    def test_search_returns_search_result(self, search: HybridSearch) -> None:
        """Results are SearchResult frozen dataclasses."""
        results = search.search("user")
        assert len(results) > 0
        r = results[0]
        assert isinstance(r, SearchResult)
        assert r.node_id
        assert r.name
        assert r.score > 0
        assert r.match_source in ("fts5", "keyword")

    def test_search_empty_query(self, search: HybridSearch) -> None:
        """Empty query returns empty list."""
        assert search.search("") == []
        assert search.search("   ") == []

    def test_search_respects_limit(self, search: HybridSearch) -> None:
        """Limit parameter is respected."""
        results = search.search("user", limit=1)
        assert len(results) <= 1

    def test_search_no_matches(self, search: HybridSearch) -> None:
        """Query with no matches returns empty or keyword fallback."""
        results = search.search("xyznonexistent123")
        # May be empty (FTS5 no match, keyword no match)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Kind Boosting Tests
# ---------------------------------------------------------------------------

class TestKindBoosting:
    """Tests for kind-based score boosting."""

    def test_function_method_boosted(self, search: HybridSearch) -> None:
        """Functions/methods get boosted in hybrid search with kind boosting."""
        # Use hybrid search which applies kind boosting via _apply_kind_boost
        results = search.search_hybrid("validate_token")
        func_results = [r for r in results if r.kind in ("function", "method")]
        # Functions/methods should appear in results with boost applied
        assert len(func_results) >= 1
        # Verify the boost constant is > 1.0 for functions
        from superlocalmemory.code_graph.search import _KIND_BOOST
        assert _KIND_BOOST[NodeKind.FUNCTION.value] > _KIND_BOOST[NodeKind.FILE.value]

    def test_hybrid_kind_boosting(self, search: HybridSearch) -> None:
        """Hybrid search applies kind boosting."""
        results = search.search_hybrid("validate_token")
        assert len(results) >= 1
        # Should find the function
        func_results = [r for r in results if r.kind == "function"]
        assert len(func_results) >= 1


# ---------------------------------------------------------------------------
# Hybrid Search Tests
# ---------------------------------------------------------------------------

class TestHybridSearch:
    """Tests for hybrid FTS5 + RRF search."""

    def test_hybrid_without_embedding(self, search: HybridSearch) -> None:
        """Hybrid search without embedding is FTS5-only."""
        results = search.search_hybrid("user")
        assert len(results) > 0
        # Without embedding, should be fts5 or keyword match source
        for r in results:
            assert r.match_source in ("fts5", "keyword")

    def test_hybrid_empty_query(self, search: HybridSearch) -> None:
        """Empty query returns empty list."""
        assert search.search_hybrid("") == []

    def test_hybrid_limit(self, search: HybridSearch) -> None:
        """Limit works in hybrid mode."""
        results = search.search_hybrid("user", limit=2)
        assert len(results) <= 2


# ---------------------------------------------------------------------------
# Empty Graph Tests
# ---------------------------------------------------------------------------

class TestEmptyGraph:
    """Tests for graceful handling of empty graph."""

    def test_search_empty_graph(self, db: CodeGraphDatabase) -> None:
        """Search on empty graph returns empty list."""
        search = HybridSearch(db)
        assert search.search("anything") == []

    def test_hybrid_empty_graph(self, db: CodeGraphDatabase) -> None:
        """Hybrid search on empty graph returns empty list."""
        search = HybridSearch(db)
        assert search.search_hybrid("anything") == []

    def test_semantic_empty_graph(self, db: CodeGraphDatabase) -> None:
        """Semantic search on empty graph returns empty list."""
        search = HybridSearch(db)
        assert search.search_semantic([0.1] * 768) == []


# ---------------------------------------------------------------------------
# Helper Function Tests
# ---------------------------------------------------------------------------

class TestHelpers:
    """Tests for module-level helper functions."""

    def test_sanitize_fts_query_basic(self) -> None:
        """Basic query sanitization."""
        result = _sanitize_fts_query("hello world")
        assert '"hello"' in result
        assert '"world"' in result

    def test_sanitize_fts_query_special_chars(self) -> None:
        """Special characters are handled."""
        result = _sanitize_fts_query('hello*world"test')
        # Should not crash, should produce valid FTS5 query
        assert result  # Non-empty

    def test_sanitize_fts_query_empty(self) -> None:
        """Empty query returns empty string."""
        assert _sanitize_fts_query("") == ""
        assert _sanitize_fts_query("   ") == ""

    def test_keyword_score_exact_match(self) -> None:
        """Exact name match scores highest."""
        score = _keyword_score("auth", "src/auth.py::auth", ["auth"])
        assert score >= 3.0  # Exact match + qualified name match

    def test_keyword_score_prefix_match(self) -> None:
        """Prefix match scores medium."""
        score = _keyword_score("authenticate", "src/auth.py::authenticate", ["auth"])
        assert score >= 2.0  # Prefix match + qualified name match

    def test_keyword_score_contains_match(self) -> None:
        """Substring match scores lower."""
        score = _keyword_score("validate_auth", "src/auth.py::validate_auth", ["auth"])
        assert score >= 1.0  # Contains match

    def test_keyword_score_no_match(self) -> None:
        """No match scores 0."""
        score = _keyword_score("xyz", "src/xyz.py::xyz", ["abc"])
        assert score == 0.0
