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

class TestFreshInstall:
    """v2.8 on a fresh system (no existing data)."""

    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "memory.db")

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # -- Schema creation --

    def test_fresh_memory_store_creates_schema(self):
        from memory_store_v2 import MemoryStoreV2
        MemoryStoreV2(self.db_path)
        cols = _get_columns(self.db_path)
        assert "lifecycle_state" in cols
        assert "access_level" in cols

    def test_fresh_schema_has_lifecycle_updated_at(self):
        from memory_store_v2 import MemoryStoreV2
        MemoryStoreV2(self.db_path)
        cols = _get_columns(self.db_path)
        assert "lifecycle_updated_at" in cols

    def test_fresh_schema_has_lifecycle_history(self):
        from memory_store_v2 import MemoryStoreV2
        MemoryStoreV2(self.db_path)
        cols = _get_columns(self.db_path)
        assert "lifecycle_history" in cols

    def test_fresh_schema_has_v27_columns(self):
        """All original v2.7 columns must still be present."""
        from memory_store_v2 import MemoryStoreV2
        MemoryStoreV2(self.db_path)
        cols = _get_columns(self.db_path)
        expected = {
            "id", "content", "summary", "project_path", "project_name",
            "tags", "category", "parent_id", "tree_path", "depth",
            "memory_type", "importance", "created_at", "updated_at",
            "last_accessed", "access_count", "content_hash", "cluster_id",
        }
        assert expected.issubset(cols), f"Missing columns: {expected - cols}"

    def test_fresh_schema_fts_table_exists(self):
        from memory_store_v2 import MemoryStoreV2
        MemoryStoreV2(self.db_path)
        conn = sqlite3.connect(self.db_path)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        assert "memories_fts" in tables

    def test_fresh_schema_sessions_table_exists(self):
        from memory_store_v2 import MemoryStoreV2
        MemoryStoreV2(self.db_path)
        conn = sqlite3.connect(self.db_path)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        assert "sessions" in tables

    def test_fresh_schema_creator_metadata_exists(self):
        from memory_store_v2 import MemoryStoreV2
        MemoryStoreV2(self.db_path)
        conn = sqlite3.connect(self.db_path)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        assert "creator_metadata" in tables

    # -- Default values for new memories --

    def test_fresh_store_lifecycle_default_active(self):
        from memory_store_v2 import MemoryStoreV2
        store = MemoryStoreV2(self.db_path)
        mem_id = store.add_memory(content="fresh memory", tags=["test"])
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT lifecycle_state FROM memories WHERE id=?", (mem_id,)
        ).fetchone()
        conn.close()
        assert row[0] == "active"

    def test_fresh_store_access_level_default_public(self):
        from memory_store_v2 import MemoryStoreV2
        store = MemoryStoreV2(self.db_path)
        mem_id = store.add_memory(content="public memory", tags=["acl"])
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT access_level FROM memories WHERE id=?", (mem_id,)
        ).fetchone()
        conn.close()
        assert row[0] == "public"

    def test_fresh_store_lifecycle_history_default_empty(self):
        from memory_store_v2 import MemoryStoreV2
        store = MemoryStoreV2(self.db_path)
        mem_id = store.add_memory(content="history test", tags=["h"])
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT lifecycle_history FROM memories WHERE id=?", (mem_id,)
        ).fetchone()
        conn.close()
        history = json.loads(row[0]) if row[0] else []
        assert history == []

    # -- Lifecycle engine on fresh db --

    def test_lifecycle_engine_works_fresh(self):
        from memory_store_v2 import MemoryStoreV2
        store = MemoryStoreV2(self.db_path)
        mem_id = store.add_memory(content="lifecycle test", tags=["t"])
        from lifecycle.lifecycle_engine import LifecycleEngine
        engine = LifecycleEngine(self.db_path)
        state = engine.get_memory_state(mem_id)
        assert state == "active"

    def test_lifecycle_state_distribution_fresh(self):
        from memory_store_v2 import MemoryStoreV2
        store = MemoryStoreV2(self.db_path)
        store.add_memory(content="mem1 for dist", tags=["d"])
        store.add_memory(content="mem2 for dist", tags=["d"])
        from lifecycle.lifecycle_engine import LifecycleEngine
        engine = LifecycleEngine(self.db_path)
        dist = engine.get_state_distribution()
        assert dist["active"] == 2
        assert dist["warm"] == 0
        assert dist["cold"] == 0

    def test_lifecycle_transition_fresh(self):
        from memory_store_v2 import MemoryStoreV2
        store = MemoryStoreV2(self.db_path)
        mem_id = store.add_memory(content="transition test", tags=["tr"])
        from lifecycle.lifecycle_engine import LifecycleEngine
        engine = LifecycleEngine(self.db_path)
        result = engine.transition_memory(mem_id, "warm", reason="test")
        assert result["success"] is True
        assert engine.get_memory_state(mem_id) == "warm"

    def test_lifecycle_invalid_transition_fresh(self):
        from memory_store_v2 import MemoryStoreV2
        store = MemoryStoreV2(self.db_path)
        mem_id = store.add_memory(content="invalid transition test", tags=["inv"])
        from lifecycle.lifecycle_engine import LifecycleEngine
        engine = LifecycleEngine(self.db_path)
        # active -> cold is not valid (must go through warm first)
        result = engine.transition_memory(mem_id, "cold", reason="skip")
        assert result["success"] is False

    def test_lifecycle_reactivation_fresh(self):
        from memory_store_v2 import MemoryStoreV2
        store = MemoryStoreV2(self.db_path)
        mem_id = store.add_memory(content="reactivation test", tags=["re"])
        from lifecycle.lifecycle_engine import LifecycleEngine
        engine = LifecycleEngine(self.db_path)
        engine.transition_memory(mem_id, "warm", reason="cool down")
        result = engine.reactivate_memory(mem_id, trigger="explicit")
        assert result["success"] is True
        assert engine.get_memory_state(mem_id) == "active"

    # -- Behavioral engine on fresh db --

    def test_behavioral_engine_works_fresh(self):
        from behavioral.outcome_tracker import OutcomeTracker
        learning_db = os.path.join(self.tmp_dir, "learning.db")
        tracker = OutcomeTracker(learning_db)
        oid = tracker.record_outcome([1], "success")
        assert oid is not None
        assert oid > 0

    def test_behavioral_outcome_tracker_failure(self):
        from behavioral.outcome_tracker import OutcomeTracker
        learning_db = os.path.join(self.tmp_dir, "learning.db")
        tracker = OutcomeTracker(learning_db)
        oid = tracker.record_outcome([1], "failure")
        assert oid is not None
        assert oid > 0

    def test_behavioral_outcome_tracker_partial(self):
        from behavioral.outcome_tracker import OutcomeTracker
        learning_db = os.path.join(self.tmp_dir, "learning.db")
        tracker = OutcomeTracker(learning_db)
        oid = tracker.record_outcome([1], "partial")
        assert oid is not None
        assert oid > 0

    def test_behavioral_invalid_outcome_returns_none(self):
        from behavioral.outcome_tracker import OutcomeTracker
        learning_db = os.path.join(self.tmp_dir, "learning.db")
        tracker = OutcomeTracker(learning_db)
        oid = tracker.record_outcome([1], "invalid_outcome")
        assert oid is None

    def test_behavioral_get_outcomes(self):
        from behavioral.outcome_tracker import OutcomeTracker
        learning_db = os.path.join(self.tmp_dir, "learning.db")
        tracker = OutcomeTracker(learning_db)
        tracker.record_outcome([10], "success")
        tracker.record_outcome([10], "failure")
        outcomes = tracker.get_outcomes()
        assert len(outcomes) >= 2

    def test_behavioral_success_rate(self):
        from behavioral.outcome_tracker import OutcomeTracker
        learning_db = os.path.join(self.tmp_dir, "learning.db")
        tracker = OutcomeTracker(learning_db)
        tracker.record_outcome([42], "success")
        tracker.record_outcome([42], "success")
        tracker.record_outcome([42], "failure")
        rate = tracker.get_success_rate(42)
        assert 0.6 <= rate <= 0.7  # 2/3

    def test_behavioral_multiple_memory_ids(self):
        from behavioral.outcome_tracker import OutcomeTracker
        learning_db = os.path.join(self.tmp_dir, "learning.db")
        tracker = OutcomeTracker(learning_db)
        oid = tracker.record_outcome([1, 2, 3], "success")
        assert oid is not None

    # -- Compliance engine on fresh db --

    def test_compliance_engine_works_fresh(self):
        from compliance.audit_db import AuditDB
        audit_db = os.path.join(self.tmp_dir, "audit.db")
        db = AuditDB(audit_db)
        eid = db.log_event("test.event", actor="user", resource_id=1)
        assert eid > 0

    def test_compliance_audit_hash_chain_valid(self):
        from compliance.audit_db import AuditDB
        audit_db = os.path.join(self.tmp_dir, "audit.db")
        db = AuditDB(audit_db)
        db.log_event("first.event", actor="user1")
        db.log_event("second.event", actor="user2")
        result = db.verify_chain()
        assert result["valid"] is True
        assert result["entries_checked"] == 2

    def test_compliance_audit_empty_chain_valid(self):
        from compliance.audit_db import AuditDB
        audit_db = os.path.join(self.tmp_dir, "audit.db")
        db = AuditDB(audit_db)
        result = db.verify_chain()
        assert result["valid"] is True
        assert result["entries_checked"] == 0

    def test_compliance_audit_query_by_event_type(self):
        from compliance.audit_db import AuditDB
        audit_db = os.path.join(self.tmp_dir, "audit.db")
        db = AuditDB(audit_db)
        db.log_event("memory.created", actor="agent1", resource_id=1)
        db.log_event("memory.recalled", actor="agent2", resource_id=2)
        events = db.query_events(event_type="memory.created")
        assert len(events) == 1
        assert events[0]["event_type"] == "memory.created"

    def test_compliance_audit_query_by_actor(self):
        from compliance.audit_db import AuditDB
        audit_db = os.path.join(self.tmp_dir, "audit.db")
        db = AuditDB(audit_db)
        db.log_event("ev1", actor="alice", resource_id=1)
        db.log_event("ev2", actor="bob", resource_id=2)
        events = db.query_events(actor="alice")
        assert len(events) == 1

    def test_compliance_audit_details_stored(self):
        from compliance.audit_db import AuditDB
        audit_db = os.path.join(self.tmp_dir, "audit.db")
        db = AuditDB(audit_db)
        db.log_event("test.detail", actor="sys", details={"key": "val"})
        events = db.query_events(event_type="test.detail")
        assert len(events) == 1
        details = json.loads(events[0]["details"])
        assert details["key"] == "val"


# ============================================================================
# 2. UPGRADE FROM v2.7 TESTS
# ============================================================================

