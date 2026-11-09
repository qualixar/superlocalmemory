# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Tests for GraphStore — thin persistence layer."""

from __future__ import annotations

import pytest

from superlocalmemory.code_graph.database import CodeGraphDatabase
from superlocalmemory.code_graph.graph_store import GraphStore
from superlocalmemory.code_graph.models import (
    EdgeKind,
    FileRecord,
    GraphEdge,
    GraphNode,
    NodeKind,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(db: CodeGraphDatabase) -> GraphStore:
    """GraphStore backed by a fresh test DB."""
    return GraphStore(db)


def _make_node(
    node_id: str = "n1",
    name: str = "foo",
    qname: str = "mod.foo",
    file_path: str = "src/mod.py",
    kind: NodeKind = NodeKind.FUNCTION,
) -> GraphNode:
    return GraphNode(
        node_id=node_id,
        kind=kind,
        name=name,
        qualified_name=qname,
        file_path=file_path,
        line_start=1,
        line_end=10,
        language="python",
    )


def _make_edge(
    edge_id: str = "e1",
    kind: EdgeKind = EdgeKind.CALLS,
    source_node_id: str = "n1",
    target_node_id: str = "n2",
    file_path: str = "src/mod.py",
) -> GraphEdge:
    return GraphEdge(
        edge_id=edge_id,
        kind=kind,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        file_path=file_path,
        line=5,
        confidence=1.0,
    )


def _make_file_record(
    file_path: str = "src/mod.py",
    content_hash: str = "abc123",
    node_count: int = 2,
    edge_count: int = 1,
) -> FileRecord:
    return FileRecord(
        file_path=file_path,
        content_hash=content_hash,
        mtime=1000.0,
        language="python",
        node_count=node_count,
        edge_count=edge_count,
    )


# ---------------------------------------------------------------------------
# Tests: store_file_nodes_edges
# ---------------------------------------------------------------------------

class TestStoreFileNodesEdges:
    """Tests for atomic file replacement."""

    def test_store_and_retrieve_nodes(self, store: GraphStore) -> None:
        n1 = _make_node("n1", "foo", "mod.foo")
        n2 = _make_node("n2", "bar", "mod.bar")
        edge = _make_edge("e1", EdgeKind.CALLS, "n1", "n2")
        fr = _make_file_record()

        store.store_file_nodes_edges("src/mod.py", [n1, n2], [edge], fr)

        nodes = store.get_nodes_by_file("src/mod.py")
        assert len(nodes) == 2
        assert {n.node_id for n in nodes} == {"n1", "n2"}

    def test_store_replaces_old_data(self, store: GraphStore) -> None:
        # First store
        n1 = _make_node("n1", "foo", "mod.foo")
        fr1 = _make_file_record(node_count=1, edge_count=0)
        store.store_file_nodes_edges("src/mod.py", [n1], [], fr1)

        # Second store with different node
        n2 = _make_node("n2", "bar", "mod.bar")
        fr2 = _make_file_record(content_hash="def456", node_count=1)
        store.store_file_nodes_edges("src/mod.py", [n2], [], fr2)

        nodes = store.get_nodes_by_file("src/mod.py")
        assert len(nodes) == 1
        assert nodes[0].node_id == "n2"

    def test_store_updates_file_record(self, store: GraphStore) -> None:
        n1 = _make_node()
        fr = _make_file_record(content_hash="hash1")
        store.store_file_nodes_edges("src/mod.py", [n1], [], fr)

        rec = store.get_file_record("src/mod.py")
        assert rec is not None
        assert rec.content_hash == "hash1"

    def test_store_bumps_version(self, store: GraphStore) -> None:
        v_before = store.version
        n1 = _make_node()
        fr = _make_file_record()
        store.store_file_nodes_edges("src/mod.py", [n1], [], fr)
        assert store.version > v_before

    def test_store_empty_lists(self, store: GraphStore) -> None:
        fr = _make_file_record(
            file_path="src/empty.py", node_count=0, edge_count=0,
        )
        store.store_file_nodes_edges("src/empty.py", [], [], fr)

        rec = store.get_file_record("src/empty.py")
        assert rec is not None
        assert rec.node_count == 0


# ---------------------------------------------------------------------------
# Tests: remove_file
# ---------------------------------------------------------------------------

class TestRemoveFile:
    """Tests for file removal."""

    def test_remove_deletes_nodes_and_record(self, store: GraphStore) -> None:
        n1 = _make_node()
        fr = _make_file_record()
        store.store_file_nodes_edges("src/mod.py", [n1], [], fr)

        store.remove_file("src/mod.py")

        assert store.get_nodes_by_file("src/mod.py") == []
        assert store.get_file_record("src/mod.py") is None

    def test_remove_nonexistent_is_noop(self, store: GraphStore) -> None:
        # Should not raise
        store.remove_file("nonexistent.py")

    def test_remove_bumps_version(self, store: GraphStore) -> None:
        n1 = _make_node()
        fr = _make_file_record()
        store.store_file_nodes_edges("src/mod.py", [n1], [], fr)

        v_before = store.version
        store.remove_file("src/mod.py")
        assert store.version > v_before


# ---------------------------------------------------------------------------
# Tests: get_all_nodes_and_edges
# ---------------------------------------------------------------------------

class TestGetAllNodesAndEdges:
    """Tests for bulk loading."""

    def test_empty_graph(self, store: GraphStore) -> None:
        nodes, edges = store.get_all_nodes_and_edges()
        assert nodes == []
        assert edges == []

    def test_returns_all_data(self, store: GraphStore) -> None:
        n1 = _make_node("n1", "foo", "mod.foo")
        n2 = _make_node("n2", "bar", "mod.bar")
        edge = _make_edge("e1", EdgeKind.CALLS, "n1", "n2")
        fr = _make_file_record()
        store.store_file_nodes_edges("src/mod.py", [n1, n2], [edge], fr)

        nodes, edges = store.get_all_nodes_and_edges()
        assert len(nodes) == 2
        assert len(edges) == 1

    def test_multiple_files(self, store: GraphStore) -> None:
        n1 = _make_node("n1", "foo", "a.foo", "src/a.py")
        n2 = _make_node("n2", "bar", "b.bar", "src/b.py")
        fr_a = _make_file_record("src/a.py", "h1", 1, 0)
        fr_b = _make_file_record("src/b.py", "h2", 1, 0)

        store.store_file_nodes_edges("src/a.py", [n1], [], fr_a)
        store.store_file_nodes_edges("src/b.py", [n2], [], fr_b)

        nodes, edges = store.get_all_nodes_and_edges()
        assert len(nodes) == 2


# ---------------------------------------------------------------------------
# Tests: find_dependents
# ---------------------------------------------------------------------------

class TestFindDependents:
    """Tests for dependent file tracing."""

    def test_finds_importing_files(self, store: GraphStore) -> None:
        # a.py has node n_a, b.py has node n_b
        # b.py imports from a.py (edge from n_b -> n_a, file_path=b.py)
        n_a = _make_node("n_a", "func_a", "a.func_a", "src/a.py")
        n_b = _make_node("n_b", "func_b", "b.func_b", "src/b.py")
        edge = _make_edge(
            "e1", EdgeKind.IMPORTS, "n_b", "n_a", "src/b.py"
        )
        fr_a = _make_file_record("src/a.py", "h1", 1, 0)
        fr_b = _make_file_record("src/b.py", "h2", 1, 1)

        store.store_file_nodes_edges("src/a.py", [n_a], [], fr_a)
        store.store_file_nodes_edges("src/b.py", [n_b], [edge], fr_b)

        deps = store.find_dependents("src/a.py")
        assert "src/b.py" in deps

    def test_no_self_reference(self, store: GraphStore) -> None:
        n1 = _make_node("n1", "foo", "a.foo", "src/a.py")
        n2 = _make_node("n2", "bar", "a.bar", "src/a.py")
        edge = _make_edge("e1", EdgeKind.CALLS, "n1", "n2", "src/a.py")
        fr = _make_file_record("src/a.py", "h1", 2, 1)

        store.store_file_nodes_edges("src/a.py", [n1, n2], [edge], fr)

        deps = store.find_dependents("src/a.py")
        assert "src/a.py" not in deps

    def test_empty_when_no_dependents(self, store: GraphStore) -> None:
        n1 = _make_node("n1", "foo", "a.foo", "src/a.py")
        fr = _make_file_record("src/a.py")
        store.store_file_nodes_edges("src/a.py", [n1], [], fr)

        deps = store.find_dependents("src/a.py")
        assert deps == set()


# ---------------------------------------------------------------------------
# Tests: get_stats
# ---------------------------------------------------------------------------

class TestGetStats:
    """Tests for stats delegation."""

    def test_empty_stats(self, store: GraphStore) -> None:
        stats = store.get_stats()
        assert stats["nodes"] == 0
        assert stats["edges"] == 0
        assert stats["files"] == 0

    def test_stats_after_store(self, store: GraphStore) -> None:
        n1 = _make_node("n1", "foo", "mod.foo")
        n2 = _make_node("n2", "bar", "mod.bar")
        edge = _make_edge("e1", EdgeKind.CALLS, "n1", "n2")
        fr = _make_file_record()
        store.store_file_nodes_edges("src/mod.py", [n1, n2], [edge], fr)

        stats = store.get_stats()
        assert stats["nodes"] == 2
        assert stats["edges"] == 1
        assert stats["files"] == 1
