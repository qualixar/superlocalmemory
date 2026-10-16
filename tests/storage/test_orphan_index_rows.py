# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Index rows of a memory that is no longer stored are removed; live ones never.

The 13 Oct 2026 state on the author's store: the fact and its embedding_metadata
row were gone, but its vector_row_map row and its vec0 vector remained. The
vector was referenced by the map row, so the unreferenced-vector sweep never
took it, and nothing else looked at map rows.
"""

from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("sqlite_vec")


@pytest.fixture
def store(tmp_path):
    from superlocalmemory.retrieval.vector_store import VectorStore, VectorStoreConfig

    path = tmp_path / "memory.db"
    vs = VectorStore(path, VectorStoreConfig(dimension=4))
    if not vs.available:
        pytest.skip("sqlite-vec cannot load on this Python")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY, profile_id TEXT)")
    for i in range(4):
        conn.execute("INSERT INTO atomic_facts VALUES (?, 'default')", (f"f{i}",))
    conn.commit()
    conn.close()
    for i in range(4):
        assert vs.upsert(f"f{i}", "default", [float(i + 1), 0.0, 1.0, 0.5])
    return vs, path


def _count(path, table: str) -> int:
    from superlocalmemory.storage.vector_residue import vec_connection

    with vec_connection(path) as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])  # noqa: S608


def _strand(path, fact_id: str, *, keep_metadata: bool = False) -> None:
    """The fact is deleted; its index rows are not (what a failed delete leaves)."""
    conn = sqlite3.connect(path)
    conn.execute("DELETE FROM atomic_facts WHERE fact_id = ?", (fact_id,))
    if not keep_metadata:
        conn.execute("DELETE FROM embedding_metadata WHERE fact_id = ?", (fact_id,))
    conn.commit()
    conn.close()


def test_stray_map_row_and_its_vector_are_removed(store):
    from superlocalmemory.storage.vector_residue import (
        sweep_unreferenced_vectors,
        unreferenced_rowids,
        vec_connection,
    )

    vs, path = store
    _strand(path, "f1")
    with vec_connection(path) as conn:
        assert unreferenced_rowids(conn) == []  # the old sweep cannot see it
    assert sweep_unreferenced_vectors(path) == 2  # its vector and its map row
    assert _count(path, "fact_embeddings") == 3
    assert _count(path, "vector_row_map") == 3
    assert vs.indexed_fact_ids("default") == {"f0", "f2", "f3"}
    assert sweep_unreferenced_vectors(path) == 0  # idempotent


def test_repair_reports_each_kind_and_leaves_live_memories(store):
    from superlocalmemory.storage.vector_residue import repair_orphan_index_rows

    vs, path = store
    _strand(path, "f0", keep_metadata=True)  # whole trio left behind
    _strand(path, "f2")                      # metadata already gone
    out = repair_orphan_index_rows(path)
    assert (out.vectors, out.metadata, out.row_map) == (2, 1, 2)
    assert vs.indexed_fact_ids("default") == {"f1", "f3"}
    assert repair_orphan_index_rows(path).total == 0


def test_a_vector_a_live_fact_still_points_at_is_kept(store):
    from superlocalmemory.storage.vector_residue import remove_orphan_index_rows, vec_connection

    _vs, path = store
    conn = sqlite3.connect(path)
    rowid = conn.execute("SELECT vec_rowid FROM vector_row_map WHERE fact_id='f1'").fetchone()[0]
    conn.execute("INSERT INTO vector_row_map (fact_id, profile_id, vec_rowid) "
                 "VALUES ('ghost', 'default', ?)", (rowid,))
    conn.commit()
    conn.close()
    with vec_connection(path) as conn:
        out = remove_orphan_index_rows(conn)
    assert (out.vectors, out.row_map) == (0, 1)  # the ghost's map row goes; f1's vector stays
    assert _count(path, "fact_embeddings") == 4


def test_without_the_vector_extension_map_rows_still_go(store):
    from superlocalmemory.storage.vector_residue import remove_orphan_index_rows

    _vs, path = store
    _strand(path, "f3")
    conn = sqlite3.connect(path, isolation_level=None)  # no extension loaded here
    try:
        out = remove_orphan_index_rows(conn)
    finally:
        conn.close()
    assert out.vectors == 0 and out.row_map == 1
    from superlocalmemory.storage.vector_residue import sweep_unreferenced_vectors
    assert sweep_unreferenced_vectors(path) == 1  # the routine sweep takes the vector now
    assert _count(path, "fact_embeddings") == 3


def test_a_store_without_facts_table_is_left_alone(tmp_path):
    from superlocalmemory.retrieval.vector_store import VectorStore, VectorStoreConfig
    from superlocalmemory.storage.vector_residue import repair_orphan_index_rows

    path = tmp_path / "memory.db"
    vs = VectorStore(path, VectorStoreConfig(dimension=4))
    if not vs.available:
        pytest.skip("sqlite-vec cannot load on this Python")
    vs.upsert("f0", "default", [1.0, 0.0, 1.0, 0.5])
    assert repair_orphan_index_rows(path).total == 0
    assert vs.indexed_fact_ids("default") == {"f0"}
