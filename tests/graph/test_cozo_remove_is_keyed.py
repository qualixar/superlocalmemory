# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Issue #296: removing a fact's edges must not scan the whole edge relation.

``edge`` is keyed by ``from_id`` first, so outgoing edges are a key seek. Incoming edges are found
through ``edge_rev`` (keyed by ``to_id``), which must mirror ``edge`` exactly, including on stores
that were created before it existed.
"""

from __future__ import annotations

import time

import pytest

from superlocalmemory.graph.cozo_backend import _COZO_AVAILABLE, CozoDBGraphBackend

pytestmark = pytest.mark.skipif(
    not _COZO_AVAILABLE,
    reason="CozoDB optional dependency not installed (pip install superlocalmemory[cozo])",
)


def _edges(be: CozoDBGraphBackend) -> set[tuple[str, str, str]]:
    rows = be._db.run("?[a, b, t] := *edge{from_id: a, to_id: b, edge_type: t}").values.tolist()
    return {tuple(r) for r in rows}


def _mirror(be: CozoDBGraphBackend) -> set[tuple[str, str, str]]:
    rows = be._db.run("?[a, b, t] := *edge_rev{to_id: b, from_id: a, edge_type: t}").values.tolist()
    return {tuple(r) for r in rows}


@pytest.fixture
def backend(tmp_path):
    be = CozoDBGraphBackend(str(tmp_path / "cozo"))
    yield be
    be.close()


def _graph(be: CozoDBGraphBackend) -> None:
    be.add_edge("a", "b", "rel", 1.0)
    be.add_edge("c", "a", "rel", 1.0)
    be.add_edge("a", "a", "self", 1.0)
    be.add_edge("x", "y", "rel", 1.0)


def test_add_edge_keeps_the_reverse_relation_in_step(backend):
    _graph(backend)
    assert _mirror(backend) == _edges(backend) and len(_edges(backend)) == 4


def test_remove_fact_drops_outgoing_and_incoming_edges_and_their_mirror(backend):
    _graph(backend)
    backend.remove_fact("a")
    assert _edges(backend) == {("x", "y", "rel")}
    assert _mirror(backend) == _edges(backend)


def test_remove_entity_drops_outgoing_and_incoming_edges_and_their_mirror(backend):
    _graph(backend)
    backend.remove_entity("a")
    assert _edges(backend) == {("x", "y", "rel")}
    assert _mirror(backend) == _edges(backend)


def test_remove_profile_keeps_the_mirror(backend):
    _graph(backend)
    backend.add_edge("p", "q", "rel", 1.0, profile_id="other")
    backend.remove_profile("default")
    assert _edges(backend) == {("p", "q", "rel")}
    assert _mirror(backend) == _edges(backend)


def test_a_store_from_before_the_reverse_relation_is_backfilled_on_open(tmp_path):
    path = str(tmp_path / "cozo")
    be = CozoDBGraphBackend(path)
    _graph(be)
    be._db.run("::remove edge_rev")  # what a 4.2.10 store looks like
    be._db.run("::remove graph_meta")
    be.close()
    be = CozoDBGraphBackend(path)
    try:
        assert _mirror(be) == _edges(be)
        be.remove_fact("a")
        assert _edges(be) == {("x", "y", "rel")} and _mirror(be) == _edges(be)
    finally:
        be.close()


def _bulk(be: CozoDBGraphBackend, n: int) -> None:
    rows = [[f"f{i}", f"f{(i * 7919) % n}", "rel", 1.0, "{}", "default", ""] for i in range(n)]
    for s in range(0, n, 50_000):
        part = rows[s:s + 50_000]
        be._db.run("?[from_id, to_id, edge_type, weight, metadata, profile_id, created_at] <- $r"
                   " :put edge {from_id, to_id, edge_type => weight, metadata, profile_id, created_at}",
                   {"r": part})
        be._db.run("?[to_id, from_id, edge_type] <- $r :put edge_rev {to_id, from_id, edge_type}",
                   {"r": [[r[1], r[0], r[2]] for r in part]})


def _removal_ms(be: CozoDBGraphBackend, n: int) -> float:
    start = time.perf_counter()
    for k in range(20):
        be.remove_fact(f"f{(k * 104729) % n}")
    return (time.perf_counter() - start) / 20 * 1000


def test_removal_cost_does_not_grow_with_the_edge_count(tmp_path):
    small = CozoDBGraphBackend(str(tmp_path / "small"))
    large = CozoDBGraphBackend(str(tmp_path / "large"))
    try:
        _bulk(small, 10_000)
        _bulk(large, 200_000)  # 20x the edges
        small_ms, large_ms = _removal_ms(small, 10_000), _removal_ms(large, 200_000)
        # A scan grows ~20x here (measured 4x per 4x on 4.2.10); a key seek stays flat.
        assert large_ms < max(5 * small_ms, 5.0), (small_ms, large_ms)
    finally:
        small.close()
        large.close()
