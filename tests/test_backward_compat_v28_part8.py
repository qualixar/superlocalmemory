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

class TestEdgeCases:
    """Edge cases and error boundaries."""

    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "memory.db")

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_lifecycle_engine_get_state_missing_memory(self):
        from memory_store_v2 import MemoryStoreV2
        MemoryStoreV2(self.db_path)
        from lifecycle.lifecycle_engine import LifecycleEngine
        engine = LifecycleEngine(self.db_path)
        state = engine.get_memory_state(99999)
        assert state is None

    def test_transition_with_empty_reason(self):
        from memory_store_v2 import MemoryStoreV2
        store = MemoryStoreV2(self.db_path)
        mid = store.add_memory(content="empty reason test", tags=["er"])
        from lifecycle.lifecycle_engine import LifecycleEngine
        engine = LifecycleEngine(self.db_path)
        result = engine.transition_memory(mid, "warm", reason="")
        assert result["success"] is True

    def test_outcome_empty_memory_ids(self):
        from behavioral.outcome_tracker import OutcomeTracker
        learning_db = os.path.join(self.tmp_dir, "learning.db")
        tracker = OutcomeTracker(learning_db)
        oid = tracker.record_outcome([], "success")
        assert oid is not None  # Empty list is technically valid JSON

    def test_audit_event_no_resource_id(self):
        from compliance.audit_db import AuditDB
        audit_db = os.path.join(self.tmp_dir, "audit.db")
        db = AuditDB(audit_db)
        eid = db.log_event("system.startup", actor="system")
        assert eid > 0

    def test_audit_event_with_details_none(self):
        from compliance.audit_db import AuditDB
        audit_db = os.path.join(self.tmp_dir, "audit.db")
        db = AuditDB(audit_db)
        eid = db.log_event("test", actor="sys", details=None)
        assert eid > 0

    def test_lifecycle_state_distribution_empty_db(self):
        from memory_store_v2 import MemoryStoreV2
        MemoryStoreV2(self.db_path)
        from lifecycle.lifecycle_engine import LifecycleEngine
        engine = LifecycleEngine(self.db_path)
        dist = engine.get_state_distribution()
        assert all(v == 0 for v in dist.values())

    def test_outcome_tracker_get_outcomes_empty(self):
        from behavioral.outcome_tracker import OutcomeTracker
        learning_db = os.path.join(self.tmp_dir, "learning.db")
        tracker = OutcomeTracker(learning_db)
        outcomes = tracker.get_outcomes()
        assert outcomes == []

    def test_audit_query_events_empty(self):
        from compliance.audit_db import AuditDB
        audit_db = os.path.join(self.tmp_dir, "audit.db")
        db = AuditDB(audit_db)
        events = db.query_events()
        assert events == []

    def test_unicode_content_preserved(self):
        from memory_store_v2 import MemoryStoreV2
        store = MemoryStoreV2(self.db_path)
        content = "Unicode test: \u2603 \u2764 \u00e9\u00e8\u00ea \u4e16\u754c"
        mid = store.add_memory(content=content, tags=["unicode"])
        results = store.list_all(limit=10)
        assert results[0]["content"] == content

    def test_large_content_memory(self):
        from memory_store_v2 import MemoryStoreV2
        store = MemoryStoreV2(self.db_path)
        content = "x" * 100_000  # 100KB
        mid = store.add_memory(content=content, tags=["large"])
        assert mid is not None

    def test_max_tags(self):
        from memory_store_v2 import MemoryStoreV2
        store = MemoryStoreV2(self.db_path)
        tags = [f"tag{i}" for i in range(20)]
        mid = store.add_memory(content="max tags test", tags=tags)
        assert mid is not None

    def test_too_many_tags_raises(self):
        import pytest
        from memory_store_v2 import MemoryStoreV2
        store = MemoryStoreV2(self.db_path)
        tags = [f"tag{i}" for i in range(21)]
        with pytest.raises(ValueError):
            store.add_memory(content="too many tags", tags=tags)
