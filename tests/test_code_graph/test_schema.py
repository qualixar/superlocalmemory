# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Tests for CodeGraph schema creation."""

import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.storage.schema_code_graph import (
    create_all_tables,
    drop_all_tables,
)


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    """Fresh in-memory-like connection with schema."""
    db = tmp_path / "test_schema.db"
    c = sqlite3.connect(str(db))
    c.row_factory = sqlite3.Row
    create_all_tables(c)
    return c


class TestSchemaCreation:
    def test_tables_created(self, conn: sqlite3.Connection):
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        assert "graph_nodes" in tables
        assert "graph_edges" in tables
        assert "graph_files" in tables
        assert "graph_metadata" in tables
        assert "code_memory_links" in tables

    def test_fts5_table(self, conn: sqlite3.Connection):
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "graph_nodes_fts" in tables

    def test_idempotent(self, conn: sqlite3.Connection):
        # Second call should not raise
        create_all_tables(conn)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        assert len(tables) >= 5

    def test_indexes_created(self, conn: sqlite3.Connection):
        indexes = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
            ).fetchall()
        }
        assert "idx_gn_file_path" in indexes
        assert "idx_gn_kind" in indexes
        assert "idx_ge_source" in indexes
        assert "idx_ge_target" in indexes
        assert "idx_cml_node" in indexes
        assert "idx_cml_stale" in indexes
        assert "idx_gn_community" in indexes

    def test_triggers_created(self, conn: sqlite3.Connection):
        triggers = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            ).fetchall()
        }
        assert "trg_gn_fts_insert" in triggers
        assert "trg_gn_fts_delete" in triggers
        assert "trg_gn_fts_update" in triggers

    def test_foreign_keys_enforced(self, conn: sqlite3.Connection):
        conn.execute("PRAGMA foreign_keys=ON")
        # Insert a node first
        conn.execute(
            "INSERT INTO graph_nodes (node_id, kind, name, qualified_name, file_path, created_at, updated_at) "
            "VALUES ('n1', 'function', 'foo', 'test.py::foo', 'test.py', 1.0, 1.0)"
        )
        # Edge with valid FK should succeed
        conn.execute(
            "INSERT INTO graph_edges (edge_id, kind, source_node_id, target_node_id, file_path, created_at, "
            "updated_at) "
            "VALUES ('e1', 'calls', 'n1', 'n1', 'test.py', 1.0, 1.0)"
        )
        # Edge with invalid FK should fail
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO graph_edges (edge_id, kind, source_node_id, target_node_id, file_path, created_at, "
                "updated_at) "
                "VALUES ('e2', 'calls', 'bogus', 'n1', 'test.py', 1.0, 1.0)"
            )

    def test_cascade_delete(self, conn: sqlite3.Connection):
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(
            "INSERT INTO graph_nodes (node_id, kind, name, qualified_name, file_path, created_at, updated_at) "
            "VALUES ('n1', 'function', 'foo', 'test.py::foo', 'test.py', 1.0, 1.0)"
        )
        conn.execute(
            "INSERT INTO graph_edges (edge_id, kind, source_node_id, target_node_id, file_path, created_at, "
            "updated_at) "
            "VALUES ('e1', 'calls', 'n1', 'n1', 'test.py', 1.0, 1.0)"
        )
        conn.execute("DELETE FROM graph_nodes WHERE node_id = 'n1'")
        edges = conn.execute("SELECT * FROM graph_edges").fetchall()
        assert len(edges) == 0

    def test_check_constraint_kind(self, conn: sqlite3.Connection):
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO graph_nodes (node_id, kind, name, qualified_name, file_path, created_at, updated_at) "
                "VALUES ('n1', 'invalid_kind', 'foo', 'test.py::foo', 'test.py', 1.0, 1.0)"
            )

    def test_check_constraint_confidence(self, conn: sqlite3.Connection):
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(
            "INSERT INTO graph_nodes (node_id, kind, name, qualified_name, file_path, created_at, updated_at) "
            "VALUES ('n1', 'function', 'foo', 'test.py::foo', 'test.py', 1.0, 1.0)"
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO graph_edges (edge_id, kind, source_node_id, target_node_id, file_path, confidence, "
                "created_at, updated_at) "
                "VALUES ('e1', 'calls', 'n1', 'n1', 'test.py', 1.5, 1.0, 1.0)"
            )

    def test_fts5_auto_sync_on_insert(self, conn: sqlite3.Connection):
        conn.execute(
            "INSERT INTO graph_nodes (node_id, kind, name, qualified_name, file_path, signature, created_at, "
            "updated_at) "
            "VALUES ('n1', 'function', 'authenticate_user', 'auth.py::authenticate_user', 'auth.py', 'def "
            "authenticate_user(token)', 1.0, 1.0)"
        )
        conn.commit()
        results = conn.execute(
            "SELECT * FROM graph_nodes_fts WHERE graph_nodes_fts MATCH 'authenticate'"
        ).fetchall()
        assert len(results) == 1

    def test_drop_all(self, conn: sqlite3.Connection):
        drop_all_tables(conn)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        assert len(tables) == 0
