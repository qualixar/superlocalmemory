# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3.2 | Phase 6 API Tests

"""Shared fixtures for Phase 6 API endpoint tests.

Provides:
- seeded_db: tmp DB with V3 + V32 schema and seed data
- api_client: FastAPI TestClient wired to a fresh app
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")


def _setup_v32_tables(conn: sqlite3.Connection) -> None:
    """Create V32 tables (the ones Phase 6 endpoints query)."""
    from superlocalmemory.storage.schema_v32 import V32_DDL
    for ddl in V32_DDL:
        for stmt in ddl.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                try:
                    conn.execute(stmt)
                except Exception:
                    pass
    conn.commit()


def _seed_profile(conn: sqlite3.Connection, pid: str = "default") -> None:
    """Insert a profile row."""
    conn.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name, description) "
        "VALUES (?, ?, ?)",
        (pid, pid, f"Test profile: {pid}"),
    )
    conn.commit()


def _seed_facts(conn: sqlite3.Connection, pid: str = "default", count: int = 5) -> list[str]:
    """Insert seed atomic_facts (with parent memories) and return fact_ids."""
    fact_ids = []
    for i in range(count):
        fid = f"fact_{uuid.uuid4().hex[:8]}"
        mid = f"mem_{uuid.uuid4().hex[:8]}"
        content = f"Test fact number {i}: some content about topic {i}"
        # Insert parent memory row first (FK target)
        conn.execute(
            "INSERT INTO memories (memory_id, profile_id, content) "
            "VALUES (?, ?, ?)",
            (mid, pid, content),
        )
        conn.execute(
            "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, "
            "fact_type, confidence) "
            "VALUES (?, ?, ?, ?, 'semantic', 0.9)",
            (fid, mid, pid, content),
        )
        fact_ids.append(fid)
    conn.commit()
    return fact_ids


def _seed_edges(
    conn: sqlite3.Connection, fact_ids: list[str], pid: str = "default",
) -> list[str]:
    """Insert seed association_edges between consecutive facts."""
    edge_ids = []
    for i in range(len(fact_ids) - 1):
        eid = f"edge_{uuid.uuid4().hex[:8]}"
        assoc_type = "auto_link" if i % 2 == 0 else "hebbian"
        conn.execute(
            "INSERT INTO association_edges "
            "(edge_id, profile_id, source_fact_id, target_fact_id, "
            "association_type, weight, co_access_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (eid, pid, fact_ids[i], fact_ids[i + 1], assoc_type, 0.7 + i * 0.05, i),
        )
        edge_ids.append(eid)
    conn.commit()
    return edge_ids


def _seed_core_blocks(
    conn: sqlite3.Connection, pid: str = "default",
) -> list[str]:
    """Insert seed core_memory_blocks."""
    block_ids = []
    for btype in ("user_profile", "project_context"):
        bid = f"blk_{uuid.uuid4().hex[:8]}"
        content = f"Core memory block for {btype}: test content"
        conn.execute(
            "INSERT INTO core_memory_blocks "
            "(block_id, profile_id, block_type, content, char_count, "
            "version, compiled_by) "
            "VALUES (?, ?, ?, ?, ?, 1, 'rules')",
            (bid, pid, btype, content, len(content)),
        )
        block_ids.append(bid)
    conn.commit()
    return block_ids


@pytest.fixture
def seeded_db(tmp_path):
    """Create a temporary DB with all tables and seed data.

    Returns (db_path, fact_ids, edge_ids, block_ids).
    """
    from superlocalmemory.storage import schema

    db_path = tmp_path / "memory.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    schema.create_all_tables(conn)
    _setup_v32_tables(conn)
    _seed_profile(conn)
    fact_ids = _seed_facts(conn)
    edge_ids = _seed_edges(conn, fact_ids)
    block_ids = _seed_core_blocks(conn)
    conn.close()
    return db_path, fact_ids, edge_ids, block_ids


@pytest.fixture
def empty_db(tmp_path):
    """Create a temporary DB with all tables but no seed data.

    Returns db_path.
    """
    from superlocalmemory.storage import schema

    db_path = tmp_path / "memory.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    schema.create_all_tables(conn)
    _setup_v32_tables(conn)
    _seed_profile(conn)
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# LightGBM mock — prevents real C training in test_api/
# test_dashboard_phase_truth.py calls _retrain_ranker_impl which calls
# lgb.Dataset + lgb.train. With real C training + LanceDB background thread
# (started by other test_api tests via FastAPI/starlette) the process crashes
# with SIGSEGV on macOS ARM (multi-libomp OpenMP race).
# ---------------------------------------------------------------------------
from tests.fixtures.lgb_mock import MockBooster, MockDataset, mock_lgb_train  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def _mock_lgb_training_for_api_tests():
    """Patch lightgbm.Dataset and lightgbm.train for this test directory."""
    with (
        patch("lightgbm.Dataset", MockDataset),
        patch("lightgbm.train", mock_lgb_train),
    ):
        yield
