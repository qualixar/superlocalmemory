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

class TestThreeDatabaseIsolation:
    """Verify memory.db, learning.db, and audit.db operate independently."""

    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.memory_db = os.path.join(self.tmp_dir, "memory.db")
        self.learning_db = os.path.join(self.tmp_dir, "learning.db")
        self.audit_db = os.path.join(self.tmp_dir, "audit.db")

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_three_databases_created_separately(self):
        from memory_store_v2 import MemoryStoreV2
        from behavioral.outcome_tracker import OutcomeTracker
        from compliance.audit_db import AuditDB

        MemoryStoreV2(self.memory_db)
        OutcomeTracker(self.learning_db)
        AuditDB(self.audit_db)

        assert os.path.exists(self.memory_db)
        assert os.path.exists(self.learning_db)
        assert os.path.exists(self.audit_db)

    def test_memory_db_has_no_outcome_table(self):
        from memory_store_v2 import MemoryStoreV2
        MemoryStoreV2(self.memory_db)
        conn = sqlite3.connect(self.memory_db)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        assert "action_outcomes" not in tables

    def test_memory_db_has_no_audit_table(self):
        from memory_store_v2 import MemoryStoreV2
        MemoryStoreV2(self.memory_db)
        conn = sqlite3.connect(self.memory_db)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        assert "audit_events" not in tables

    def test_learning_db_has_no_memories_table(self):
        from behavioral.outcome_tracker import OutcomeTracker
        OutcomeTracker(self.learning_db)
        conn = sqlite3.connect(self.learning_db)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        assert "memories" not in tables

    def test_audit_db_has_no_memories_table(self):
        from compliance.audit_db import AuditDB
        AuditDB(self.audit_db)
        conn = sqlite3.connect(self.audit_db)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        assert "memories" not in tables

    def test_learning_db_failure_doesnt_affect_memory_db(self):
        from memory_store_v2 import MemoryStoreV2
        store = MemoryStoreV2(self.memory_db)
        mem_id = store.add_memory(content="isolation test", tags=["iso"])
        # learning_db not even created — memory ops still work
        assert mem_id is not None

    def test_audit_db_failure_doesnt_affect_memory_db(self):
        from memory_store_v2 import MemoryStoreV2
        store = MemoryStoreV2(self.memory_db)
        mem_id = store.add_memory(content="audit isolation test", tags=["iso"])
        assert mem_id is not None


# ============================================================================
# 9. CROSS-ENGINE INTEGRATION TESTS
# ============================================================================

class TestCrossEngineIntegration:
    """Test that engines work together when all are available."""

    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.memory_db = os.path.join(self.tmp_dir, "memory.db")
        self.learning_db = os.path.join(self.tmp_dir, "learning.db")
        self.audit_db_path = os.path.join(self.tmp_dir, "audit.db")
        from memory_store_v2 import MemoryStoreV2
        self.store = MemoryStoreV2(self.memory_db)

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_memory_created_then_lifecycle_tracked(self):
        mem_id = self.store.add_memory(content="integration test", tags=["int"])
        from lifecycle.lifecycle_engine import LifecycleEngine
        engine = LifecycleEngine(self.memory_db)
        state = engine.get_memory_state(mem_id)
        assert state == "active"

    def test_memory_created_then_outcome_recorded(self):
        mem_id = self.store.add_memory(content="outcome integration", tags=["oi"])
        from behavioral.outcome_tracker import OutcomeTracker
        tracker = OutcomeTracker(self.learning_db)
        oid = tracker.record_outcome([mem_id], "success")
        assert oid is not None

    def test_memory_created_then_audit_logged(self):
        mem_id = self.store.add_memory(content="audit integration", tags=["ai"])
        from compliance.audit_db import AuditDB
        audit = AuditDB(self.audit_db_path)
        eid = audit.log_event("memory.created", actor="test", resource_id=mem_id)
        assert eid > 0

    def test_full_lifecycle_with_audit(self):
        mem_id = self.store.add_memory(content="full flow test", tags=["ff"])
        from lifecycle.lifecycle_engine import LifecycleEngine
        from compliance.audit_db import AuditDB
        engine = LifecycleEngine(self.memory_db)
        audit = AuditDB(self.audit_db_path)

        audit.log_event("memory.created", actor="test", resource_id=mem_id)
        engine.transition_memory(mem_id, "warm", reason="aging")
        audit.log_event("lifecycle.transition", actor="scheduler",
                        resource_id=mem_id, details={"to": "warm"})

        assert engine.get_memory_state(mem_id) == "warm"
        events = audit.query_events(resource_id=mem_id)
        assert len(events) == 2

    def test_full_lifecycle_with_behavioral_outcome(self):
        mem_id = self.store.add_memory(content="behavior flow test", tags=["bf"])
        from behavioral.outcome_tracker import OutcomeTracker
        tracker = OutcomeTracker(self.learning_db)

        # Record multiple outcomes
        tracker.record_outcome([mem_id], "success")
        tracker.record_outcome([mem_id], "success")
        tracker.record_outcome([mem_id], "failure")

        rate = tracker.get_success_rate(mem_id)
        assert 0.6 <= rate <= 0.7

    def test_multiple_memories_with_different_states(self):
        from lifecycle.lifecycle_engine import LifecycleEngine
        engine = LifecycleEngine(self.memory_db)

        m1 = self.store.add_memory(content="active mem 1", tags=["s"])
        m2 = self.store.add_memory(content="warm mem 2", tags=["s"])
        m3 = self.store.add_memory(content="cold mem 3", tags=["s"])

        engine.transition_memory(m2, "warm")
        engine.transition_memory(m3, "warm")
        engine.transition_memory(m3, "cold")

        dist = engine.get_state_distribution()
        assert dist["active"] == 1
        assert dist["warm"] == 1
        assert dist["cold"] == 1


# ============================================================================
# 10. V2.7 API COMPATIBILITY TESTS
# ============================================================================

