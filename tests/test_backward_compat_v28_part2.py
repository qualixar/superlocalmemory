# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""Comprehensive backward compatibility tests for v2.8 upgrade.

Validates that:
1. Fresh installs get v2.8 schema correctly
2. Upgrades from v2.7.6 preserve existing data
3. All v2.8 modules are importable and operational
4. Lifecycle, Behavioral, and Compliance engines work independently
5. No regressions in core memory operations after upgrade
6. Event-driven loose coupling holds (engine failures don't cascade)
7. Feature vector dimensions are correct (20 features)
8. MCP tool handlers are available
"""
import sqlite3
import tempfile
import os
import sys
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime, timedelta

# Ensure src/ is on sys.path for all imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ============================================================================
# Helpers
# ============================================================================

def _create_v27_db(db_path: str, num_memories: int = 2) -> None:
    """Create a v2.7.6-compatible DB with NO lifecycle columns.

    Includes FTS virtual table + triggers that v2.7 would have had,
    so that MemoryStoreV2._init_db migration doesn't corrupt FTS sync.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            summary TEXT,
            project_path TEXT,
            project_name TEXT,
            tags TEXT,
            category TEXT,
            parent_id INTEGER,
            tree_path TEXT,
            depth INTEGER DEFAULT 0,
            memory_type TEXT DEFAULT 'session',
            importance INTEGER DEFAULT 5,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_accessed TIMESTAMP,
            access_count INTEGER DEFAULT 0,
            content_hash TEXT UNIQUE,
            cluster_id INTEGER,
            profile TEXT DEFAULT 'default'
        )
    """)
    # FTS table + triggers (present in v2.7.6)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
        USING fts5(content, summary, tags, content='memories', content_rowid='id')
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, content, summary, tags)
            VALUES (new.id, new.content, new.summary, new.tags);
        END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, summary, tags)
            VALUES('delete', old.id, old.content, old.summary, old.tags);
        END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, summary, tags)
            VALUES('delete', old.id, old.content, old.summary, old.tags);
            INSERT INTO memories_fts(rowid, content, summary, tags)
            VALUES (new.id, new.content, new.summary, new.tags);
        END
    """)
    for i in range(num_memories):
        h = hashlib.sha256(f"v27_memory_{i}".encode()).hexdigest()[:32]
        conn.execute(
            "INSERT INTO memories (content, content_hash, profile) VALUES (?, ?, 'default')",
            (f"existing memory from v2.7 #{i}", h),
        )
    conn.commit()
    conn.close()


def _get_columns(db_path: str, table: str = "memories") -> set:
    """Return the set of column names in the given table."""
    conn = sqlite3.connect(db_path)
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    conn.close()
    return cols


# ============================================================================
# 1. FRESH INSTALL TESTS
# ============================================================================

class TestUpgradeFromV27:
    """Simulates upgrade from v2.7.6 (existing data, no lifecycle columns)."""

    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "memory.db")
        _create_v27_db(self.db_path, num_memories=2)

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # -- Schema migration --

    def test_schema_migrates_on_init(self):
        """MemoryStoreV2 init should add lifecycle columns to existing DB."""
        from memory_store_v2 import MemoryStoreV2
        MemoryStoreV2(self.db_path)
        cols = _get_columns(self.db_path)
        assert "lifecycle_state" in cols
        assert "lifecycle_updated_at" in cols
        assert "lifecycle_history" in cols
        assert "access_level" in cols

    def test_migration_is_idempotent(self):
        """Running migration twice should not error or duplicate columns."""
        from memory_store_v2 import MemoryStoreV2
        MemoryStoreV2(self.db_path)
        MemoryStoreV2(self.db_path)  # second init
        cols = _get_columns(self.db_path)
        # Count lifecycle_state appearances
        conn = sqlite3.connect(self.db_path)
        info = conn.execute("PRAGMA table_info(memories)").fetchall()
        conn.close()
        lifecycle_cols = [r for r in info if r[1] == "lifecycle_state"]
        assert len(lifecycle_cols) == 1

    def test_existing_data_preserved(self):
        from memory_store_v2 import MemoryStoreV2
        store = MemoryStoreV2(self.db_path)
        results = store.list_all(limit=10)
        assert len(results) == 2
        contents = {r["content"] for r in results}
        assert "existing memory from v2.7 #0" in contents
        assert "existing memory from v2.7 #1" in contents

    def test_existing_memory_ids_preserved(self):
        from memory_store_v2 import MemoryStoreV2
        store = MemoryStoreV2(self.db_path)
        results = store.list_all(limit=10)
        ids = {r["id"] for r in results}
        assert 1 in ids
        assert 2 in ids

    def test_existing_memories_default_active(self):
        from memory_store_v2 import MemoryStoreV2
        MemoryStoreV2(self.db_path)
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT lifecycle_state FROM memories").fetchall()
        conn.close()
        for row in rows:
            assert row[0] == "active"

    def test_existing_memories_default_access_level(self):
        from memory_store_v2 import MemoryStoreV2
        MemoryStoreV2(self.db_path)
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT access_level FROM memories").fetchall()
        conn.close()
        for row in rows:
            assert row[0] == "public"

    def test_search_still_works_after_upgrade(self):
        from memory_store_v2 import MemoryStoreV2
        store = MemoryStoreV2(self.db_path)
        results = store.search("existing memory", limit=5)
        assert len(results) >= 1

    def test_list_all_still_works_after_upgrade(self):
        from memory_store_v2 import MemoryStoreV2
        store = MemoryStoreV2(self.db_path)
        results = store.list_all(limit=50)
        assert len(results) == 2

    def test_new_features_work_on_upgraded_db(self):
        from memory_store_v2 import MemoryStoreV2
        MemoryStoreV2(self.db_path)
        from lifecycle.lifecycle_engine import LifecycleEngine
        engine = LifecycleEngine(self.db_path)
        dist = engine.get_state_distribution()
        assert dist["active"] == 2

    def test_create_new_memory_after_upgrade(self):
        from memory_store_v2 import MemoryStoreV2
        store = MemoryStoreV2(self.db_path)
        mem_id = store.add_memory(content="new v2.8 memory", tags=["new"])
        assert mem_id is not None
        results = store.list_all(limit=10)
        assert len(results) == 3

    def test_new_memory_has_lifecycle_state_after_upgrade(self):
        from memory_store_v2 import MemoryStoreV2
        store = MemoryStoreV2(self.db_path)
        mem_id = store.add_memory(content="new v2.8 lifecycle check", tags=["lc"])
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT lifecycle_state FROM memories WHERE id=?", (mem_id,)
        ).fetchone()
        conn.close()
        assert row[0] == "active"

    def test_lifecycle_transition_on_upgraded_memory(self):
        from memory_store_v2 import MemoryStoreV2
        MemoryStoreV2(self.db_path)
        from lifecycle.lifecycle_engine import LifecycleEngine
        engine = LifecycleEngine(self.db_path)
        # Transition existing memory (id=1) from active to warm
        result = engine.transition_memory(1, "warm", reason="aging")
        assert result["success"] is True
        assert engine.get_memory_state(1) == "warm"

    def test_lifecycle_history_recorded_on_upgrade(self):
        from memory_store_v2 import MemoryStoreV2
        MemoryStoreV2(self.db_path)
        from lifecycle.lifecycle_engine import LifecycleEngine
        engine = LifecycleEngine(self.db_path)
        engine.transition_memory(1, "warm", reason="test history")
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT lifecycle_history FROM memories WHERE id=1"
        ).fetchone()
        conn.close()
        history = json.loads(row[0])
        assert len(history) == 1
        assert history[0]["from"] == "active"
        assert history[0]["to"] == "warm"

    def test_get_stats_still_works_after_upgrade(self):
        from memory_store_v2 import MemoryStoreV2
        store = MemoryStoreV2(self.db_path)
        stats = store.get_stats()
        assert stats["total_memories"] >= 2

    def test_dedup_still_works_after_upgrade(self):
        from memory_store_v2 import MemoryStoreV2
        store = MemoryStoreV2(self.db_path)
        id1 = store.add_memory(content="unique content for dedup test", tags=["d"])
        id2 = store.add_memory(content="unique content for dedup test", tags=["d"])
        assert id1 == id2  # Deduplication returns existing ID


class TestUpgradeFromV27LargeDataset:
    """Upgrade from v2.7 with many memories to test migration performance."""

    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "memory.db")
        _create_v27_db(self.db_path, num_memories=50)

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_all_50_memories_get_lifecycle_state(self):
        from memory_store_v2 import MemoryStoreV2
        MemoryStoreV2(self.db_path)
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT lifecycle_state FROM memories").fetchall()
        conn.close()
        assert len(rows) == 50
        for row in rows:
            assert row[0] == "active"

    def test_state_distribution_correct_after_bulk_migration(self):
        from memory_store_v2 import MemoryStoreV2
        MemoryStoreV2(self.db_path)
        from lifecycle.lifecycle_engine import LifecycleEngine
        engine = LifecycleEngine(self.db_path)
        dist = engine.get_state_distribution()
        assert dist["active"] == 50
        assert dist["warm"] == 0

    def test_lifecycle_transition_bulk(self):
        from memory_store_v2 import MemoryStoreV2
        MemoryStoreV2(self.db_path)
        from lifecycle.lifecycle_engine import LifecycleEngine
        engine = LifecycleEngine(self.db_path)
        for i in range(1, 11):
            engine.transition_memory(i, "warm", reason="bulk test")
        dist = engine.get_state_distribution()
        assert dist["warm"] == 10
        assert dist["active"] == 40


# ============================================================================
# 3. MODULE AVAILABILITY TESTS
# ============================================================================

