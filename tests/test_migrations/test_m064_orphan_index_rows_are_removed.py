# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M064: a store with a stray index row is clean after its first start on 4.2.8.

The first start is ``apply_all`` then ``apply_deferred`` (engine init sits
between them), with no command for the user to run. Temporary stores only.
"""

from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("sqlite_vec")

from superlocalmemory.storage import schema
from superlocalmemory.storage._migration_internals import _MODULES
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.migration_runner import DEFERRED_MIGRATIONS, apply_all, apply_deferred
from superlocalmemory.storage.migrations import M064_orphan_index_rows_are_removed as M064


def _store(tmp_path):
    """A 4.2.7 store holding three memories and one stray: its fact and metadata
    are gone, its map row and vector remain (the state of 13 Oct 2026)."""
    from superlocalmemory.retrieval.vector_store import VectorStore, VectorStoreConfig

    memory, learning = tmp_path / "memory.db", tmp_path / "learning.db"
    manager = DatabaseManager(memory)
    manager.initialize(schema)
    manager.close()
    vs = VectorStore(memory, VectorStoreConfig(dimension=4))
    if not vs.available:
        pytest.skip("sqlite-vec cannot load on this Python")
    conn = sqlite3.connect(memory)
    for i in range(4):
        conn.execute("INSERT INTO memories (memory_id, profile_id, content) "
                     "VALUES (?, 'default', 'x')", (f"m{i}",))
        conn.execute("INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content) "
                     "VALUES (?, ?, 'default', 'x')", (f"f{i}", f"m{i}"))
    conn.commit()
    conn.close()
    for i in range(4):
        assert vs.upsert(f"f{i}", "default", [float(i + 1), 0.0, 1.0, 0.5])
    conn = sqlite3.connect(memory)
    conn.execute("DELETE FROM atomic_facts WHERE fact_id = 'f2'")
    conn.execute("DELETE FROM embedding_metadata WHERE fact_id = 'f2'")
    conn.commit()
    conn.close()
    return memory, learning, vs


def _counts(memory):
    from superlocalmemory.storage.vector_residue import vec_connection

    with vec_connection(memory) as conn:
        return tuple(int(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])  # noqa: S608
                     for t in ("fact_embeddings", "vector_row_map", "embedding_metadata"))


def test_registered_deferred_with_a_floor_and_no_replay():
    assert _MODULES[M064.NAME] is M064
    assert any(m.name == M064.NAME for m in DEFERRED_MIGRATIONS)
    assert M064.DB_TARGET == "memory" and M064.DOWNGRADE_FLOOR == 51
    assert M064.REPAIR_NOT_APPLICABLE.strip()


def test_first_start_on_the_new_version_leaves_no_stray_row(tmp_path):
    memory, learning, vs = _store(tmp_path)
    assert _counts(memory) == (4, 4, 3)  # the stray is there

    apply_all(learning, memory)
    result = apply_deferred(learning, memory)

    assert M064.NAME in result["applied"], result
    assert _counts(memory) == (3, 3, 3)
    assert vs.indexed_fact_ids("default") == {"f0", "f1", "f3"}


def test_it_runs_once_and_a_later_stray_is_left_to_maintenance(tmp_path):
    memory, learning, vs = _store(tmp_path)
    apply_all(learning, memory)
    apply_deferred(learning, memory)
    again = apply_deferred(learning, memory)
    assert M064.NAME not in again["applied"] and M064.NAME in again["skipped"]

    from superlocalmemory.storage.vector_residue import sweep_unreferenced_vectors

    conn = sqlite3.connect(memory)
    conn.execute("DELETE FROM atomic_facts WHERE fact_id = 'f1'")
    conn.commit()
    conn.close()
    assert _counts(memory) == (3, 3, 3)
    assert sweep_unreferenced_vectors(memory) > 0  # the routine pass takes it, no marker needed
    assert _counts(memory) == (2, 2, 2)


def test_apply_is_idempotent_and_safe_before_the_facts_table_exists(tmp_path):
    conn = sqlite3.connect(tmp_path / "empty.db", isolation_level=None)
    M064.apply(conn)  # no atomic_facts: nothing to do, no error
    conn.close()

    memory, _learning, _vs = _store(tmp_path)
    conn = sqlite3.connect(memory, isolation_level=None)
    M064.apply(conn)
    M064.apply(conn)
    conn.close()
    assert _counts(memory) == (3, 3, 3)
