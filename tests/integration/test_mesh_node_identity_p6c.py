# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
#
# Tests for the shared per-node identity primitive (3c-0).
# xdist-UNSAFE by convention with the mesh suite — run serially.

from __future__ import annotations

import sqlite3

from superlocalmemory.mesh.node_identity import get_node_id


def test_stable_across_calls(tmp_path):
    """Same DB → same node_id on repeated calls (persisted)."""
    db = str(tmp_path / "mesh.db")
    a = get_node_id(db)
    b = get_node_id(db)
    assert a == b
    assert len(a) >= 8


def test_distinct_dbs_distinct_ids(tmp_path):
    """Two different DBs (two nodes) get distinct ids."""
    a = get_node_id(str(tmp_path / "a.db"))
    b = get_node_id(str(tmp_path / "b.db"))
    assert a != b


def test_survives_reopen(tmp_path):
    """Id persists across a fresh connection (simulated restart)."""
    db = str(tmp_path / "mesh.db")
    first = get_node_id(db)
    # Independent connection sees the same persisted row.
    conn = sqlite3.connect(db)
    try:
        row = conn.execute(
            "SELECT node_id FROM mesh_node_identity WHERE id = 1"
        ).fetchone()
    finally:
        conn.close()
    assert row is not None and row[0] == first
    assert get_node_id(db) == first


def test_fail_soft_on_bad_path(tmp_path):
    """An unwritable/invalid DB path returns a process-stable fallback, no raise."""
    bad = str(tmp_path / "no_such_dir" / "nested" / "mesh.db")
    v1 = get_node_id(bad)
    v2 = get_node_id(bad)
    assert v1 == v2  # process-stable fallback
    assert v1  # non-empty
