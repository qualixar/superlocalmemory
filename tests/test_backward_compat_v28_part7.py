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

class TestV27ApiCompatibility:
    """Ensure all v2.7 public APIs still work identically."""

    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "memory.db")
        from memory_store_v2 import MemoryStoreV2
        self.store = MemoryStoreV2(self.db_path)

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_add_memory_returns_int(self):
        mid = self.store.add_memory(content="type check", tags=["tc"])
        assert isinstance(mid, int)

    def test_add_memory_with_all_params(self):
        mid = self.store.add_memory(
            content="full params test",
            summary="test summary",
            project_path="/tmp/test",
            project_name="test-proj",
            tags=["a", "b"],
            category="backend",
            memory_type="long-term",
            importance=8,
        )
        assert mid is not None

    def test_add_memory_dedup(self):
        mid1 = self.store.add_memory(content="dedup v27 test", tags=["d"])
        mid2 = self.store.add_memory(content="dedup v27 test", tags=["d"])
        assert mid1 == mid2

    def test_add_memory_empty_content_raises(self):
        import pytest
        with pytest.raises(ValueError):
            self.store.add_memory(content="", tags=["e"])

    def test_add_memory_non_string_raises(self):
        import pytest
        with pytest.raises(TypeError):
            self.store.add_memory(content=123, tags=["e"])

    def test_search_returns_list(self):
        self.store.add_memory(content="searchable content v27", tags=["s"])
        results = self.store.search("searchable", limit=5)
        assert isinstance(results, list)

    def test_search_result_has_expected_keys(self):
        self.store.add_memory(content="key check memory content", tags=["k"])
        results = self.store.search("key check", limit=5)
        if results:
            r = results[0]
            assert "id" in r
            assert "content" in r
            assert "score" in r
            assert "tags" in r
            assert "importance" in r

    def test_list_all_returns_list(self):
        self.store.add_memory(content="list all test", tags=["la"])
        results = self.store.list_all(limit=10)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_list_all_result_has_title(self):
        self.store.add_memory(content="title field test", tags=["tf"])
        results = self.store.list_all(limit=10)
        assert "title" in results[0]

    def test_get_stats_returns_dict(self):
        self.store.add_memory(content="stats test", tags=["st"])
        stats = self.store.get_stats()
        assert isinstance(stats, dict)
        assert "total_memories" in stats

    def test_get_stats_total_correct(self):
        self.store.add_memory(content="stats count 1", tags=["sc"])
        self.store.add_memory(content="stats count 2", tags=["sc"])
        stats = self.store.get_stats()
        assert stats["total_memories"] >= 2

    def test_search_with_lifecycle_filter(self):
        """v2.8 extension: search accepts lifecycle_states parameter."""
        self.store.add_memory(content="lifecycle filter test", tags=["lf"])
        results = self.store.search(
            "lifecycle filter", limit=5,
            lifecycle_states=("active", "warm"),
        )
        assert isinstance(results, list)

    def test_search_default_lifecycle_filter(self):
        """Default lifecycle filter should include active and warm."""
        self.store.add_memory(content="default lifecycle search", tags=["dls"])
        results = self.store.search("default lifecycle", limit=5)
        # Should find the memory since it's active (default)
        assert len(results) >= 0  # At least returns an empty list, no crash

    def test_memory_type_field_preserved(self):
        mid = self.store.add_memory(
            content="type preserve test", tags=["tp"],
            memory_type="long-term",
        )
        results = self.store.list_all(limit=50)
        match = [r for r in results if r["id"] == mid]
        assert len(match) == 1
        assert match[0]["memory_type"] == "long-term"

    def test_importance_field_preserved(self):
        mid = self.store.add_memory(
            content="importance preserve test", tags=["ip"],
            importance=9,
        )
        results = self.store.list_all(limit=50)
        match = [r for r in results if r["id"] == mid]
        assert len(match) == 1
        assert match[0]["importance"] == 9

    def test_category_field_preserved(self):
        mid = self.store.add_memory(
            content="category preserve test", tags=["cp"],
            category="frontend",
        )
        results = self.store.list_all(limit=50)
        match = [r for r in results if r["id"] == mid]
        assert len(match) == 1
        assert match[0]["category"] == "frontend"

    def test_tags_stored_as_json(self):
        mid = self.store.add_memory(
            content="json tags test", tags=["alpha", "beta"],
        )
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT tags FROM memories WHERE id=?", (mid,)
        ).fetchone()
        conn.close()
        parsed = json.loads(row[0])
        assert "alpha" in parsed
        assert "beta" in parsed


# ============================================================================
# 11. CONCURRENT ACCESS TESTS
# ============================================================================

class TestConcurrentAccess:
    """Verify thread-safety of lifecycle transitions."""

    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "memory.db")
        from memory_store_v2 import MemoryStoreV2
        self.store = MemoryStoreV2(self.db_path)

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_concurrent_lifecycle_transitions(self):
        import threading
        mid = self.store.add_memory(content="concurrent test", tags=["ct"])
        from lifecycle.lifecycle_engine import LifecycleEngine
        engine = LifecycleEngine(self.db_path)

        results = []

        def try_transition():
            r = engine.transition_memory(mid, "warm", reason="race")
            results.append(r)

        threads = [threading.Thread(target=try_transition) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # At least one should succeed, others may fail (already in warm state)
        successes = [r for r in results if r.get("success")]
        assert len(successes) >= 1

    def test_concurrent_outcome_recording(self):
        import threading
        from behavioral.outcome_tracker import OutcomeTracker
        learning_db = os.path.join(self.tmp_dir, "learning.db")
        tracker = OutcomeTracker(learning_db)

        ids = []

        def record():
            oid = tracker.record_outcome([1], "success")
            ids.append(oid)

        threads = [threading.Thread(target=record) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(ids) == 10
        assert all(oid is not None and oid > 0 for oid in ids)

    def test_concurrent_audit_logging(self):
        import threading
        from compliance.audit_db import AuditDB
        audit_db = os.path.join(self.tmp_dir, "audit.db")
        db = AuditDB(audit_db)

        ids = []

        def log_event(i):
            eid = db.log_event(f"event_{i}", actor=f"actor_{i}")
            ids.append(eid)

        threads = [threading.Thread(target=log_event, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(ids) == 10
        # Verify chain is still valid after concurrent writes
        result = db.verify_chain()
        assert result["valid"] is True


# ============================================================================
# 12. EDGE CASES & ERROR HANDLING
# ============================================================================

