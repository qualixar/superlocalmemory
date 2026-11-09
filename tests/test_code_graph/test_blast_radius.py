# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Tests for BlastRadius — bidirectional BFS impact analysis."""

from __future__ import annotations

import pytest

try:
    import rustworkx  # noqa: F401
    HAS_RUSTWORKX = True
except ImportError:
    HAS_RUSTWORKX = False

pytestmark = pytest.mark.skipif(
    not HAS_RUSTWORKX, reason="rustworkx not installed"
)

from superlocalmemory.code_graph.blast_radius import (
    BlastRadius,
    BlastRadiusResult,
)
from superlocalmemory.code_graph.database import CodeGraphDatabase
from superlocalmemory.code_graph.graph_engine import GraphEngine
from superlocalmemory.code_graph.graph_store import GraphStore
from superlocalmemory.code_graph.models import (
    EdgeKind,
    FileRecord,
    GraphEdge,
    GraphNode,
    NodeKind,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node(
    nid: str, name: str, qname: str, fp: str = "src/mod.py",
) -> GraphNode:
    return GraphNode(
        node_id=nid, kind=NodeKind.FUNCTION, name=name,
        qualified_name=qname, file_path=fp,
        line_start=1, line_end=10, language="python",
    )


def _edge(
    eid: str, src: str, tgt: str, fp: str = "src/mod.py",
    kind: EdgeKind = EdgeKind.CALLS,
) -> GraphEdge:
    return GraphEdge(
        edge_id=eid, kind=kind, source_node_id=src, target_node_id=tgt,
        file_path=fp, line=5, confidence=1.0,
    )


def _fr(fp: str = "src/mod.py") -> FileRecord:
    return FileRecord(
        file_path=fp, content_hash="h", mtime=1.0, language="python",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(db: CodeGraphDatabase) -> GraphStore:
    return GraphStore(db)


@pytest.fixture
def engine(store: GraphStore) -> GraphEngine:
    return GraphEngine(store)


@pytest.fixture
def br(engine: GraphEngine) -> BlastRadius:
    return BlastRadius(engine)


# ---------------------------------------------------------------------------
# Tests: empty / no seeds
# ---------------------------------------------------------------------------

class TestBlastRadiusEmpty:
    def test_empty_graph(self, br: BlastRadius) -> None:
        result = br.compute(changed_files=["nonexistent.py"])
        assert result.changed_nodes == frozenset()
        assert result.impacted_nodes == frozenset()

    def test_no_seeds(self, br: BlastRadius) -> None:
        result = br.compute()
        assert result == BlastRadiusResult()


# ---------------------------------------------------------------------------
# Tests: linear chain A -> B -> C -> D
# ---------------------------------------------------------------------------

class TestLinearChain:
    @pytest.fixture(autouse=True)
    def _setup(self, store: GraphStore) -> None:
        # Use a single file_path for all nodes so atomic replacement works.
        # The graph is: A --CALLS--> B --CALLS--> C --CALLS--> D
        fp = "src/mod.py"
        nodes = [
            _node("a", "a", "mod.a", fp),
            _node("b", "b", "mod.b", fp),
            _node("c", "c", "mod.c", fp),
            _node("d", "d", "mod.d", fp),
        ]
        edges = [
            _edge("e1", "a", "b", fp),
            _edge("e2", "b", "c", fp),
            _edge("e3", "c", "d", fp),
        ]
        store.store_file_nodes_edges(fp, nodes, edges, _fr(fp))

    def test_forward_depth_1(self, br: BlastRadius) -> None:
        result = br.compute(
            seed_node_ids=["b"], max_depth=1, direction="forward",
        )
        assert "b" in result.changed_nodes
        assert "c" in result.impacted_nodes
        assert "d" not in result.impacted_nodes

    def test_forward_depth_2(self, br: BlastRadius) -> None:
        result = br.compute(
            seed_node_ids=["b"], max_depth=2, direction="forward",
        )
        assert "c" in result.impacted_nodes
        assert "d" in result.impacted_nodes

    def test_reverse_depth_1(self, br: BlastRadius) -> None:
        result = br.compute(
            seed_node_ids=["c"], max_depth=1, direction="reverse",
        )
        assert "b" in result.impacted_nodes
        assert "a" not in result.impacted_nodes

    def test_both_directions(self, br: BlastRadius) -> None:
        result = br.compute(
            seed_node_ids=["b"], max_depth=1, direction="both",
        )
        # Forward: b->c. Reverse: a->b so a discovered
        assert "a" in result.impacted_nodes
        assert "c" in result.impacted_nodes

    def test_impacted_files(self, br: BlastRadius) -> None:
        result = br.compute(
            seed_node_ids=["b"], max_depth=1, direction="forward",
        )
        assert "src/mod.py" in result.impacted_files

    def test_depth_reached(self, br: BlastRadius) -> None:
        result = br.compute(
            seed_node_ids=["a"], max_depth=3, direction="forward",
        )
        assert result.depth_reached <= 3

    def test_by_changed_files(self, br: BlastRadius) -> None:
        result = br.compute(
            changed_files=["src/mod.py"], max_depth=1, direction="forward",
        )
        # All 4 nodes are in src/mod.py, so all become seeds
        assert len(result.changed_nodes) == 4


# ---------------------------------------------------------------------------
# Tests: truncation
# ---------------------------------------------------------------------------

class TestTruncation:
    def test_max_nodes_truncation(
        self, store: GraphStore, br: BlastRadius
    ) -> None:
        # Build a wide graph: hub -> n1, n2, ..., n20
        hub = _node("hub", "hub", "mod.hub")
        targets = [
            _node(f"t{i}", f"t{i}", f"mod.t{i}") for i in range(20)
        ]
        edges = [
            _edge(f"e{i}", "hub", f"t{i}") for i in range(20)
        ]
        store.store_file_nodes_edges(
            "src/mod.py", [hub] + targets, edges, _fr()
        )

        result = br.compute(
            seed_node_ids=["hub"], max_depth=1, max_nodes=5,
            direction="forward",
        )
        assert result.truncated is True
        # Total visited (seed + impacted) should be <= max_nodes
        total = len(result.changed_nodes) + len(result.impacted_nodes)
        assert total <= 5


# ---------------------------------------------------------------------------
# Tests: edge kind filtering
# ---------------------------------------------------------------------------

class TestEdgeKindFilter:
    def test_filter_by_kind(
        self, store: GraphStore, br: BlastRadius
    ) -> None:
        a = _node("a", "a", "mod.a")
        b = _node("b", "b", "mod.b")
        c = _node("c", "c", "mod.c")
        e1 = _edge("e1", "a", "b", kind=EdgeKind.CALLS)
        e2 = _edge("e2", "a", "c", kind=EdgeKind.IMPORTS)
        store.store_file_nodes_edges(
            "src/mod.py", [a, b, c], [e1, e2], _fr()
        )

        # Only follow CALLS
        result = br.compute(
            seed_node_ids=["a"], max_depth=1, direction="forward",
            edge_kinds={EdgeKind.CALLS.value},
        )
        assert "b" in result.impacted_nodes
        assert "c" not in result.impacted_nodes
