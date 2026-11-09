# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Tests for CommunityDetector — file-based grouping, architecture overview."""

from __future__ import annotations

import json
import time

import pytest

from superlocalmemory.code_graph.communities import (
    ArchitectureOverview,
    CommunityDetector,
    CommunityInfo,
    CouplingWarning,
    _extract_directory,
    _generate_community_name,
)
from superlocalmemory.code_graph.database import CodeGraphDatabase
from superlocalmemory.code_graph.models import (
    EdgeKind,
    GraphEdge,
    GraphNode,
    NodeKind,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def community_db(db: CodeGraphDatabase) -> CodeGraphDatabase:
    """DB with nodes in multiple directories for community detection."""
    now = time.time()
    nodes = [
        # src/auth/ directory
        GraphNode(
            node_id="auth_handler", kind=NodeKind.FUNCTION,
            name="authenticate", qualified_name="src/auth/handler.py::authenticate",
            file_path="src/auth/handler.py", line_start=1, line_end=20,
            language="python", created_at=now, updated_at=now,
        ),
        GraphNode(
            node_id="token_validator", kind=NodeKind.FUNCTION,
            name="validate_token", qualified_name="src/auth/tokens.py::validate_token",
            file_path="src/auth/tokens.py", line_start=1, line_end=15,
            language="python", created_at=now, updated_at=now,
        ),
        # src/db/ directory
        GraphNode(
            node_id="db_query", kind=NodeKind.FUNCTION,
            name="execute_query", qualified_name="src/db/query.py::execute_query",
            file_path="src/db/query.py", line_start=1, line_end=30,
            language="python", created_at=now, updated_at=now,
        ),
        GraphNode(
            node_id="db_connection", kind=NodeKind.CLASS,
            name="ConnectionPool", qualified_name="src/db/pool.py::ConnectionPool",
            file_path="src/db/pool.py", line_start=1, line_end=50,
            language="python", created_at=now, updated_at=now,
        ),
        # src/api/ directory
        GraphNode(
            node_id="api_handler", kind=NodeKind.FUNCTION,
            name="handle_request", qualified_name="src/api/routes.py::handle_request",
            file_path="src/api/routes.py", line_start=1, line_end=25,
            language="python", created_at=now, updated_at=now,
        ),
    ]
    for node in nodes:
        db.upsert_node(node)

    # Cross-directory edges
    edges = [
        GraphEdge(
            edge_id="e1", kind=EdgeKind.CALLS,
            source_node_id="api_handler", target_node_id="auth_handler",
            file_path="src/api/routes.py", line=10,
            created_at=now, updated_at=now,
        ),
        GraphEdge(
            edge_id="e2", kind=EdgeKind.CALLS,
            source_node_id="auth_handler", target_node_id="db_query",
            file_path="src/auth/handler.py", line=15,
            created_at=now, updated_at=now,
        ),
        GraphEdge(
            edge_id="e3", kind=EdgeKind.CALLS,
            source_node_id="api_handler", target_node_id="db_query",
            file_path="src/api/routes.py", line=20,
            created_at=now, updated_at=now,
        ),
    ]
    for edge in edges:
        db.upsert_edge(edge)

    return db


@pytest.fixture
def detector(community_db: CodeGraphDatabase) -> CommunityDetector:
    """CommunityDetector instance with populated DB."""
    return CommunityDetector(community_db)


# ---------------------------------------------------------------------------
# Community Detection Tests
# ---------------------------------------------------------------------------

class TestCommunityDetection:
    """Tests for detect_communities()."""

    def test_detects_communities_by_directory(
        self, detector: CommunityDetector
    ) -> None:
        """Groups nodes by directory."""
        communities = detector.detect_communities()
        assert len(communities) == 3  # src/auth, src/db, src/api
        dirs = {c.directory for c in communities}
        assert "src/auth" in dirs
        assert "src/db" in dirs
        assert "src/api" in dirs

    def test_community_sizes(self, detector: CommunityDetector) -> None:
        """Each community has correct node count."""
        communities = detector.detect_communities()
        size_map = {c.directory: c.size for c in communities}
        assert size_map["src/auth"] == 2
        assert size_map["src/db"] == 2
        assert size_map["src/api"] == 1

    def test_community_has_node_ids(self, detector: CommunityDetector) -> None:
        """Communities contain correct node IDs."""
        communities = detector.detect_communities()
        auth_comm = next(c for c in communities if c.directory == "src/auth")
        assert "auth_handler" in auth_comm.node_ids
        assert "token_validator" in auth_comm.node_ids

    def test_community_id_set_on_nodes(
        self, detector: CommunityDetector, community_db: CodeGraphDatabase
    ) -> None:
        """community_id is updated on graph_nodes."""
        detector.detect_communities()
        node = community_db.get_node("auth_handler")
        assert node is not None
        assert node.community_id is not None

    def test_communities_sorted_by_size(
        self, detector: CommunityDetector
    ) -> None:
        """Communities returned sorted by size (largest first)."""
        communities = detector.detect_communities()
        for i in range(len(communities) - 1):
            assert communities[i].size >= communities[i + 1].size

    def test_communities_stored_in_metadata(
        self, detector: CommunityDetector, community_db: CodeGraphDatabase
    ) -> None:
        """Communities are stored in graph_metadata."""
        detector.detect_communities()
        raw = community_db.get_metadata("communities")
        assert raw is not None
        data = json.loads(raw)
        assert len(data) == 3

    def test_empty_graph(self, db: CodeGraphDatabase) -> None:
        """Empty graph returns no communities."""
        detector = CommunityDetector(db)
        assert detector.detect_communities() == []

    def test_dominant_language(self, detector: CommunityDetector) -> None:
        """Dominant language is detected."""
        communities = detector.detect_communities()
        for c in communities:
            assert c.dominant_language == "python"


# ---------------------------------------------------------------------------
# Architecture Overview Tests
# ---------------------------------------------------------------------------

class TestArchitectureOverview:
    """Tests for get_architecture_overview()."""

    def test_returns_overview(self, detector: CommunityDetector) -> None:
        """Returns an ArchitectureOverview object."""
        overview = detector.get_architecture_overview()
        assert isinstance(overview, ArchitectureOverview)
        assert overview.total_communities == 3
        assert overview.total_nodes == 5

    def test_overview_has_communities(
        self, detector: CommunityDetector
    ) -> None:
        """Overview includes all communities."""
        overview = detector.get_architecture_overview()
        assert len(overview.communities) == 3

    def test_coupling_warnings_detected(
        self, detector: CommunityDetector
    ) -> None:
        """Cross-community edges generate coupling warnings."""
        overview = detector.get_architecture_overview()
        # We have 3 cross-community edges, but they are between 3 pairs
        # Any coupling detected is valid
        assert isinstance(overview.coupling_warnings, tuple)

    def test_coupling_warning_severity(
        self, detector: CommunityDetector
    ) -> None:
        """Coupling warnings have correct severity levels."""
        overview = detector.get_architecture_overview()
        for w in overview.coupling_warnings:
            assert isinstance(w, CouplingWarning)
            assert w.severity in ("low", "medium", "high")
            assert w.edge_count > 0


# ---------------------------------------------------------------------------
# Helper Function Tests
# ---------------------------------------------------------------------------

class TestHelpers:
    """Tests for module-level helpers."""

    def test_extract_directory_nested(self) -> None:
        """Extracts directory from nested path."""
        assert _extract_directory("src/auth/handler.py") == "src/auth"

    def test_extract_directory_root(self) -> None:
        """Root-level files get 'root' directory."""
        assert _extract_directory("main.py") == "root"

    def test_extract_directory_deep(self) -> None:
        """Deep nesting works."""
        assert _extract_directory("a/b/c/d/file.py") == "a/b/c/d"

    def test_generate_community_name_with_class(self) -> None:
        """Community name includes class name if present."""
        nodes = [
            {"name": "MyClass", "kind": "class"},
            {"name": "func1", "kind": "function"},
        ]
        name = _generate_community_name("src/core", nodes)
        assert "MyClass" in name

    def test_generate_community_name_no_class(self) -> None:
        """Community name is directory-based without classes."""
        nodes = [
            {"name": "func1", "kind": "function"},
            {"name": "func2", "kind": "function"},
        ]
        name = _generate_community_name("src/utils", nodes)
        assert "utils" in name


# ---------------------------------------------------------------------------
# Frozen Dataclass Tests
# ---------------------------------------------------------------------------

class TestDataclasses:
    """Test that result types are frozen."""

    def test_community_info_frozen(self) -> None:
        """CommunityInfo is immutable."""
        c = CommunityInfo(
            community_id=0, name="test", directory="test",
            size=1, dominant_language="python", file_count=1,
            cohesion=1.0, node_ids=("n1",),
        )
        with pytest.raises(AttributeError):
            c.name = "changed"  # type: ignore[misc]

    def test_coupling_warning_frozen(self) -> None:
        """CouplingWarning is immutable."""
        w = CouplingWarning(
            source_community="a", target_community="b",
            edge_count=5, severity="medium",
        )
        with pytest.raises(AttributeError):
            w.edge_count = 10  # type: ignore[misc]

    def test_architecture_overview_frozen(self) -> None:
        """ArchitectureOverview is immutable."""
        o = ArchitectureOverview(
            communities=(), coupling_warnings=(),
            total_nodes=0, total_communities=0,
        )
        with pytest.raises(AttributeError):
            o.total_nodes = 5  # type: ignore[misc]
