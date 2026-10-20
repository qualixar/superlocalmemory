# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Issue #296: a promoted Cozo store held 4.3 M edges while canonical SQLite held 2.9 M.

After any sequence of writes, the incrementally drained Cozo graph must hold exactly the
logical edges a fresh ``bulk_import_from_sqlite`` would build (``storage.logical_edges``): no
edge to a withheld fact, none a delete or prune removed, and none missing.
"""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.core.projection_drain import ProjectionDrain
from superlocalmemory.graph.cozo_backend import _COZO_AVAILABLE, CozoDBGraphBackend
from superlocalmemory.storage import projection_outbox
from superlocalmemory.storage.logical_edges import iter_logical_edges
from superlocalmemory.storage.models import AtomicFact, EdgeType, GraphEdge, MemoryRecord

pytestmark = pytest.mark.skipif(not _COZO_AVAILABLE, reason="CozoDB not installed")

_PARENT = "m-drift"


@pytest.fixture
def setup(engine_with_mock_deps, tmp_path):
    db = engine_with_mock_deps._db
    db.store_memory(MemoryRecord(memory_id=_PARENT, profile_id="default", content="parent"))
    db.execute("DELETE FROM projection_outbox")
    graph = CozoDBGraphBackend(str(tmp_path / "cozo"))
    yield db, graph, ProjectionDrain(db, lambda: graph, lambda: None)
    graph.close()


def _fact(db, text: str) -> str:
    fact = AtomicFact(memory_id=_PARENT, profile_id="default", content=text, embedding=[0.1] * 8)
    db.store_fact(fact)
    return fact.fact_id


def _edge(db, a: str, b: str, kind=EdgeType.SEMANTIC) -> None:
    db.store_edge(GraphEdge(profile_id="default", source_id=a, target_id=b, edge_type=kind,
                            weight=0.9))


def _canonical(db) -> set[tuple[str, str, str]]:
    conn = sqlite3.connect(db.db_path)
    try:
        return {(r[0], r[1], r[2]) for r in iter_logical_edges(conn, "default")}
    finally:
        conn.close()


def _projected(graph) -> set[tuple[str, str, str]]:
    rows = graph._db.run("?[a, b, t] := *edge{from_id: a, to_id: b, edge_type: t}").values.tolist()
    return {tuple(r) for r in rows}


def _drain(db, drain) -> None:
    while projection_outbox.depth(db):
        drain.drain_once()


def test_a_neighbour_reprojected_after_a_fact_is_withheld_does_not_bring_its_edge_back(setup):
    db, graph, drain = setup
    visible, withheld = _fact(db, "stays visible"), _fact(db, "about to be withheld")
    _edge(db, visible, withheld)
    _drain(db, drain)
    assert _projected(graph) == _canonical(db) != set()

    db.execute("UPDATE atomic_facts SET quarantined = 1 WHERE fact_id = ?", (withheld,))
    projection_outbox.enqueue(db, withheld, "default")
    _drain(db, drain)
    projection_outbox.enqueue(db, visible, "default")  # any later reason to re-project it
    _drain(db, drain)

    assert _projected(graph) == _canonical(db) == set()


def test_a_fact_shown_again_gets_its_edges_back(setup):
    db, graph, drain = setup
    a, b = _fact(db, "a"), _fact(db, "b")
    _edge(db, a, b)
    db.execute("UPDATE atomic_facts SET quarantined = 1 WHERE fact_id = ?", (b,))
    _drain(db, drain)
    db.execute("UPDATE atomic_facts SET quarantined = 0 WHERE fact_id = ?", (b,))
    projection_outbox.enqueue(db, b, "default")
    _drain(db, drain)
    assert _projected(graph) == _canonical(db) != set()


def test_deletes_and_prunes_leave_the_projection_equal_to_the_store(setup):
    from superlocalmemory.core.graph_pruner import prune_graph

    db, graph, drain = setup
    facts = [_fact(db, f"fact {i}") for i in range(6)]
    for i in range(5):
        _edge(db, facts[i], facts[i + 1])
    _edge(db, facts[0], facts[0])  # a self loop the pruner removes
    _drain(db, drain)
    db.delete_fact(facts[3])
    prune_graph(db, "default")
    _drain(db, drain)
    assert _projected(graph) == _canonical(db)
