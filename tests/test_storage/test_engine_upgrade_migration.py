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

from superlocalmemory.storage import schema
from superlocalmemory.core.config import SLMConfig
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
    n = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    conn.close()

    assert "scope" in post and "shared_with" in post, (
        "engine.initialize() did not apply the deferred M016 scope migration"
    )
    assert n == 1, "legacy row lost during migration"

    # And a scoped write must now succeed instead of crashing.
    ids = eng.store_fast("a fresh post-upgrade fact", scope="global")
    assert ids
    row = eng._db.execute(
        "SELECT scope FROM atomic_facts WHERE fact_id = ?", (ids[0],)
    )
    assert row[0][0] == "global"
    eng.close()
