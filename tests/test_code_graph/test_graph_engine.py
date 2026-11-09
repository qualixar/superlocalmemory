# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Tests for GraphEngine — rustworkx in-memory graph."""

# ruff: noqa: E402 (the imports below deliberately follow the rustworkx availability guard and skip marker)

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

from superlocalmemory.code_graph.database import CodeGraphDatabase
from superlocalmemory.code_graph.graph_engine import (
    GraphEngine,
    NodeNotFoundError,
)
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
    return GraphStore(db)


@pytest.fixture
def engine(store: GraphStore) -> GraphEngine:
    return GraphEngine(store)


def _node(
    nid: str, name: str, qname: str, fp: str = "src/mod.py",
    is_test: bool = False, kind: NodeKind = NodeKind.FUNCTION,
) -> GraphNode:
    return GraphNode(
        node_id=nid, kind=kind, name=name, qualified_name=qname,
        file_path=fp, line_start=1, line_end=10, language="python",
        is_test=is_test,
    )


def _edge(
    eid: str, kind: EdgeKind, src: str, tgt: str, fp: str = "src/mod.py",
) -> GraphEdge:
    return GraphEdge(
        edge_id=eid, kind=kind, source_node_id=src, target_node_id=tgt,
        file_path=fp, line=5, confidence=1.0,
    )


def _fr(fp: str = "src/mod.py") -> FileRecord:
    return FileRecord(
        file_path=fp, content_hash="h", mtime=1.0, language="python",
    )


def _populate_simple_graph(store: GraphStore) -> tuple[str, str, str]:
    """Create A --CALLS--> B --CALLS--> C  and return (a_id, b_id, c_id)."""
    a = _node("a", "func_a", "mod.func_a")
    b = _node("b", "func_b", "mod.func_b")
    c = _node("c", "func_c", "mod.func_c")
    e_ab = _edge("e_ab", EdgeKind.CALLS, "a", "b")
    e_bc = _edge("e_bc", EdgeKind.CALLS, "b", "c")
    fr = _fr()
    store.store_file_nodes_edges("src/mod.py", [a, b, c], [e_ab, e_bc], fr)
    return "a", "b", "c"


# ---------------------------------------------------------------------------
# Tests: build_graph
# ---------------------------------------------------------------------------

class TestBuildGraph:
    def test_empty_graph(self, engine: GraphEngine) -> None:
        g = engine.build_graph()
        assert g.num_nodes() == 0
        assert g.num_edges() == 0

    def test_nodes_and_edges_loaded(
        self, store: GraphStore, engine: GraphEngine
    ) -> None:
        _populate_simple_graph(store)
        g = engine.build_graph()
        assert g.num_nodes() == 3
        assert g.num_edges() == 2

    def test_cache_hit(
        self, store: GraphStore, engine: GraphEngine
    ) -> None:
        _populate_simple_graph(store)
        g1 = engine.build_graph()
        g2 = engine.build_graph()
        assert g1 is g2  # Same object = cache hit

    def test_cache_invalidation_on_write(
        self, store: GraphStore, engine: GraphEngine
    ) -> None:
        _populate_simple_graph(store)
        g1 = engine.build_graph()

        # Write new data → version changes
        n_d = _node("d", "func_d", "mod.func_d")
        fr2 = FileRecord(
            file_path="src/other.py", content_hash="h2",
            mtime=2.0, language="python",
        )
        store.store_file_nodes_edges("src/other.py", [n_d], [], fr2)

        g2 = engine.build_graph()
        assert g2 is not g1  # Rebuilt
        assert g2.num_nodes() == 4

    def test_graph_index_types(
        self, store: GraphStore, engine: GraphEngine
    ) -> None:
        _populate_simple_graph(store)
        engine.build_graph()
        idx = engine.index

        # id_to_rx: str -> int
        for k, v in idx.id_to_rx.items():
            assert isinstance(k, str)
            assert isinstance(v, int)

        # rx_to_id: int -> str
        for k, v in idx.rx_to_id.items():
            assert isinstance(k, int)
            assert isinstance(v, str)


# ---------------------------------------------------------------------------
# Tests: get_callers / get_callees
# ---------------------------------------------------------------------------

class TestCallersCallees:
    def test_get_callers(
        self, store: GraphStore, engine: GraphEngine
    ) -> None:
        a_id, b_id, c_id = _populate_simple_graph(store)
        callers = engine.get_callers(b_id)
        assert len(callers) == 1
        assert callers[0]["node"]["node_id"] == a_id

    def test_get_callees(
        self, store: GraphStore, engine: GraphEngine
    ) -> None:
        a_id, b_id, c_id = _populate_simple_graph(store)
        callees = engine.get_callees(b_id)
        assert len(callees) == 1
        assert callees[0]["node"]["node_id"] == c_id

    def test_no_callers_for_root(
        self, store: GraphStore, engine: GraphEngine
    ) -> None:
        a_id, _, _ = _populate_simple_graph(store)
        callers = engine.get_callers(a_id)
        assert callers == []

    def test_no_callees_for_leaf(
        self, store: GraphStore, engine: GraphEngine
    ) -> None:
        _, _, c_id = _populate_simple_graph(store)
        callees = engine.get_callees(c_id)
        assert callees == []

    def test_edge_kind_filter(
        self, store: GraphStore, engine: GraphEngine
    ) -> None:
        a_id, b_id, _ = _populate_simple_graph(store)
        # Filter for IMPORTS only → no results
        callers = engine.get_callers(b_id, edge_kinds={EdgeKind.IMPORTS.value})
        assert callers == []

    def test_node_not_found(self, engine: GraphEngine) -> None:
        with pytest.raises(NodeNotFoundError):
            engine.get_callers("nonexistent")


# ---------------------------------------------------------------------------
# Tests: get_tests_for
# ---------------------------------------------------------------------------

class TestGetTestsFor:
    def test_tested_by_edge(
        self, store: GraphStore, engine: GraphEngine
    ) -> None:
        func = _node("f1", "my_func", "mod.my_func")
        test = _node("t1", "test_my_func", "test.test_my_func",
                      "tests/test.py", is_test=True)
        edge = _edge("e1", EdgeKind.TESTED_BY, "f1", "t1")
        store.store_file_nodes_edges("src/mod.py", [func], [], _fr())
        store.store_file_nodes_edges(
            "tests/test.py", [test], [edge], _fr("tests/test.py")
        )

        tests = engine.get_tests_for("f1")
        assert len(tests) == 1
        assert tests[0]["node_id"] == "t1"

    def test_calls_from_test_node(
        self, store: GraphStore, engine: GraphEngine
    ) -> None:
        func = _node("f1", "my_func", "mod.my_func")
        test = _node("t1", "test_my_func", "test.test_my_func",
                      "tests/test.py", is_test=True)
        # CALLS edge from test → func (not TESTED_BY, but test node)
        edge = _edge("e1", EdgeKind.CALLS, "t1", "f1", "tests/test.py")
        store.store_file_nodes_edges("src/mod.py", [func], [], _fr())
        store.store_file_nodes_edges(
            "tests/test.py", [test], [edge], _fr("tests/test.py")
        )

        tests = engine.get_tests_for("f1")
        assert len(tests) == 1
        assert tests[0]["node_id"] == "t1"

    def test_deduplicates(
        self, store: GraphStore, engine: GraphEngine
    ) -> None:
        func = _node("f1", "my_func", "mod.my_func")
        test = _node("t1", "test_it", "test.test_it",
                      "tests/test.py", is_test=True)
        # Both TESTED_BY and CALLS edge
        e1 = _edge("e1", EdgeKind.TESTED_BY, "f1", "t1")
        e2 = _edge("e2", EdgeKind.CALLS, "t1", "f1", "tests/test.py")
        store.store_file_nodes_edges("src/mod.py", [func], [], _fr())
        store.store_file_nodes_edges(
            "tests/test.py", [test], [e1, e2], _fr("tests/test.py")
        )

        tests = engine.get_tests_for("f1")
        assert len(tests) == 1  # Deduplicated

    def test_no_tests(
        self, store: GraphStore, engine: GraphEngine
    ) -> None:
        func = _node("f1", "my_func", "mod.my_func")
        store.store_file_nodes_edges("src/mod.py", [func], [], _fr())

        tests = engine.get_tests_for("f1")
        assert tests == []


# ---------------------------------------------------------------------------
# Tests: get_connected_component
# ---------------------------------------------------------------------------

class TestConnectedComponent:
    def test_single_component(
        self, store: GraphStore, engine: GraphEngine
    ) -> None:
        a_id, b_id, c_id = _populate_simple_graph(store)
        comp = engine.get_connected_component(b_id)
        assert set(comp) == {a_id, b_id, c_id}

    def test_isolated_node(
        self, store: GraphStore, engine: GraphEngine
    ) -> None:
        _populate_simple_graph(store)
        isolated = _node("iso", "isolated", "mod.isolated", "src/other.py")
        store.store_file_nodes_edges(
            "src/other.py", [isolated], [], _fr("src/other.py")
        )

        comp = engine.get_connected_component("iso")
        assert comp == ["iso"]


# ---------------------------------------------------------------------------
# Tests: invalidate
# ---------------------------------------------------------------------------

class TestInvalidate:
    def test_invalidate_forces_rebuild(
        self, store: GraphStore, engine: GraphEngine
    ) -> None:
        _populate_simple_graph(store)
        g1 = engine.build_graph()
        engine.invalidate()
        g2 = engine.build_graph()
        assert g2 is not g1
