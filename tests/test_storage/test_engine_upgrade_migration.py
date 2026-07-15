# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Regression: MemoryEngine.initialize() must apply the deferred M016 scope
migration on an EXISTING pre-3.6.15 database — not only the daemon lifespan.

Pre-release gate (v3.6.15) caught this on a real 760MB production DB: a user
on `slm remember --sync`, the Python API, or a LangChain/CrewAI integration
(i.e. WITHOUT the daemon) hit `table memories has no column named scope` on the
first scoped write, because only the daemon ran apply_deferred. This guards the
fix in engine._init_db_layer().
"""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest

from superlocalmemory.storage import schema
from superlocalmemory.core.config import SLMConfig
from superlocalmemory.core.engine_capabilities import Capabilities
from superlocalmemory.storage.models import Mode


def _make_pre_scope_db(db_path) -> None:
    """Build a current-schema DB, then strip scope/shared_with from `memories`
    to simulate a 3.6.14 database that predates the scope columns."""
    conn = sqlite3.connect(str(db_path))
    schema.create_all_tables(conn)
    # Indexes must go before the columns they reference (SQLite DROP COLUMN).
    for idx in ("idx_memories_scope", "idx_memories_profile_scope"):
        conn.execute(f"DROP INDEX IF EXISTS {idx}")
    conn.execute("ALTER TABLE memories DROP COLUMN shared_with")
    conn.execute("ALTER TABLE memories DROP COLUMN scope")
    conn.execute(
        "INSERT INTO memories (memory_id, profile_id, content) "
        "VALUES ('m_legacy', 'default', 'a legacy 3.6.14 memory')"
    )
    conn.execute(
        "INSERT INTO ccq_consolidated_blocks "
        "(block_id, profile_id, content, cluster_id) "
        "VALUES ('ccq_legacy', 'default', 'a legacy consolidated memory', 'cluster_1')"
    )
    conn.commit()
    conn.close()


def test_engine_init_migrates_pre_scope_db(tmp_path, mock_embedder):
    db = tmp_path / "memory.db"
    _make_pre_scope_db(db)

    # Sanity: the simulated DB really lacks the scope column.
    conn = sqlite3.connect(str(db))
    pre = {r[1] for r in conn.execute("PRAGMA table_info(memories)")}
    conn.close()
    assert "scope" not in pre, "test setup failed — scope column should be absent"

    # Engine init must re-apply the deferred M016 migration.
    cfg = SLMConfig.for_mode(Mode.A, base_dir=tmp_path)
    cfg.retrieval.use_cross_encoder = False
    from superlocalmemory.core.engine import MemoryEngine

    eng = MemoryEngine(cfg)
    with patch(
        "superlocalmemory.core.engine_wiring.init_embedder",
        return_value=mock_embedder,
    ):
        eng.initialize()
        eng._embedder = mock_embedder

    conn = sqlite3.connect(str(db))
    post = {r[1] for r in conn.execute("PRAGMA table_info(memories)")}
    ccq_post = {
        r[1] for r in conn.execute("PRAGMA table_info(ccq_consolidated_blocks)")
    }
    ccq_indexes = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND tbl_name='ccq_consolidated_blocks'"
        )
    }
    n = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    ccq_row = conn.execute(
        "SELECT content, scope FROM ccq_consolidated_blocks "
        "WHERE block_id = 'ccq_legacy'"
    ).fetchone()
    conn.close()

    assert "scope" in post and "shared_with" in post, (
        "engine.initialize() did not apply the deferred M016 scope migration"
    )
    assert n == 1, "legacy row lost during migration"
    assert "scope" in ccq_post, (
        "engine.initialize() did not schedule the deferred M017 CCQ migration"
    )
    assert {
        "idx_ccq_consolidated_blocks_scope",
        "idx_ccq_consolidated_blocks_profile_scope",
    } <= ccq_indexes
    assert ccq_row == ("a legacy consolidated memory", "personal"), (
        "legacy CCQ row lost or changed during M017 migration"
    )

    # And a scoped write must now succeed instead of crashing.
    ids = eng.store_fast("a fresh post-upgrade fact", scope="global")
    assert ids
    row = eng._db.execute(
        "SELECT scope FROM atomic_facts WHERE fact_id = ?", (ids[0],)
    )
    assert row[0][0] == "global"
    eng.close()


def test_engine_init_fails_closed_when_m018_cannot_be_applied(tmp_path):
    """Startup must not advertise a writable engine without canonical ingest."""
    cfg = SLMConfig.for_mode(Mode.A, base_dir=tmp_path)
    from superlocalmemory.core.engine import MemoryEngine

    eng = MemoryEngine(cfg, capabilities=Capabilities.LIGHT)
    with (
        patch("superlocalmemory.storage.migration_runner.apply_all", return_value={}),
        patch("superlocalmemory.storage.migration_runner.apply_deferred", return_value={}),
        pytest.raises(RuntimeError, match="required ingestion migration failed"),
    ):
        eng.initialize()

    assert eng._initialized is False


def test_fresh_engine_bootstraps_learning_schema_before_forward_migrations(tmp_path):
    cfg = SLMConfig.for_mode(Mode.A, base_dir=tmp_path)
    from superlocalmemory.core.engine import MemoryEngine
    from superlocalmemory.storage.migration_runner import status

    eng = MemoryEngine(cfg, capabilities=Capabilities.LIGHT)
    eng.initialize()

    migration_status = status(tmp_path / "learning.db", tmp_path / "memory.db")
    assert migration_status["M001_add_signal_features_columns"] == "complete"
    assert migration_status["M002_model_state_history"] == "complete"
    assert migration_status["M009_model_lineage"] == "complete"
    eng.close()
