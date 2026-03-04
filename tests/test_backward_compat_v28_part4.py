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

class TestLifecycleStateMachine:
    """Verify the lifecycle state machine rules."""

    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "memory.db")
        from memory_store_v2 import MemoryStoreV2
        self.store = MemoryStoreV2(self.db_path)
        from lifecycle.lifecycle_engine import LifecycleEngine
        self.engine = LifecycleEngine(self.db_path)

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _create_memory(self, content_suffix=""):
        return self.store.add_memory(
            content=f"state machine test {content_suffix} {datetime.now().isoformat()}",
            tags=["sm"],
        )

    # Valid transitions
    def test_active_to_warm(self):
        mid = self._create_memory("a2w")
        r = self.engine.transition_memory(mid, "warm")
        assert r["success"] is True

    def test_warm_to_cold(self):
        mid = self._create_memory("w2c")
        self.engine.transition_memory(mid, "warm")
        r = self.engine.transition_memory(mid, "cold")
        assert r["success"] is True

    def test_cold_to_archived(self):
        mid = self._create_memory("c2a")
        self.engine.transition_memory(mid, "warm")
        self.engine.transition_memory(mid, "cold")
        r = self.engine.transition_memory(mid, "archived")
        assert r["success"] is True

    def test_archived_to_tombstoned(self):
        mid = self._create_memory("a2t")
        self.engine.transition_memory(mid, "warm")
        self.engine.transition_memory(mid, "cold")
        self.engine.transition_memory(mid, "archived")
        r = self.engine.transition_memory(mid, "tombstoned")
        assert r["success"] is True

    def test_warm_to_active_reactivation(self):
        mid = self._create_memory("w2a")
        self.engine.transition_memory(mid, "warm")
        r = self.engine.transition_memory(mid, "active")
        assert r["success"] is True

    def test_cold_to_active_reactivation(self):
        mid = self._create_memory("c2a")
        self.engine.transition_memory(mid, "warm")
        self.engine.transition_memory(mid, "cold")
        r = self.engine.transition_memory(mid, "active")
        assert r["success"] is True

    def test_archived_to_active_reactivation(self):
        mid = self._create_memory("ar2a")
        self.engine.transition_memory(mid, "warm")
        self.engine.transition_memory(mid, "cold")
        self.engine.transition_memory(mid, "archived")
        r = self.engine.transition_memory(mid, "active")
        assert r["success"] is True

    # Invalid transitions
    def test_active_to_cold_invalid(self):
        mid = self._create_memory("a2c")
        r = self.engine.transition_memory(mid, "cold")
        assert r["success"] is False

    def test_active_to_archived_invalid(self):
        mid = self._create_memory("a2ar")
        r = self.engine.transition_memory(mid, "archived")
        assert r["success"] is False

    def test_active_to_tombstoned_invalid(self):
        mid = self._create_memory("a2t")
        r = self.engine.transition_memory(mid, "tombstoned")
        assert r["success"] is False

    def test_warm_to_archived_invalid(self):
        mid = self._create_memory("w2ar")
        self.engine.transition_memory(mid, "warm")
        r = self.engine.transition_memory(mid, "archived")
        assert r["success"] is False

    def test_warm_to_tombstoned_invalid(self):
        mid = self._create_memory("w2t")
        self.engine.transition_memory(mid, "warm")
        r = self.engine.transition_memory(mid, "tombstoned")
        assert r["success"] is False

    def test_cold_to_warm_invalid(self):
        mid = self._create_memory("c2w")
        self.engine.transition_memory(mid, "warm")
        self.engine.transition_memory(mid, "cold")
        r = self.engine.transition_memory(mid, "warm")
        assert r["success"] is False

    def test_cold_to_tombstoned_invalid(self):
        mid = self._create_memory("c2t")
        self.engine.transition_memory(mid, "warm")
        self.engine.transition_memory(mid, "cold")
        r = self.engine.transition_memory(mid, "tombstoned")
        assert r["success"] is False

    def test_tombstoned_is_terminal(self):
        mid = self._create_memory("tomb")
        self.engine.transition_memory(mid, "warm")
        self.engine.transition_memory(mid, "cold")
        self.engine.transition_memory(mid, "archived")
        self.engine.transition_memory(mid, "tombstoned")
        for target in ("active", "warm", "cold", "archived"):
            r = self.engine.transition_memory(mid, target)
            assert r["success"] is False, f"Tombstoned -> {target} should fail"

    def test_nonexistent_memory_transition(self):
        r = self.engine.transition_memory(99999, "warm")
        assert r["success"] is False

    # Transition history tracking
    def test_transition_history_grows(self):
        mid = self._create_memory("hist")
        self.engine.transition_memory(mid, "warm", reason="step1")
        self.engine.transition_memory(mid, "cold", reason="step2")
        self.engine.transition_memory(mid, "active", reason="reactivate")
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT lifecycle_history FROM memories WHERE id=?", (mid,)
        ).fetchone()
        conn.close()
        history = json.loads(row[0])
        assert len(history) == 3
        assert history[0]["to"] == "warm"
        assert history[1]["to"] == "cold"
        assert history[2]["to"] == "active"

    def test_transition_history_has_timestamps(self):
        mid = self._create_memory("ts")
        self.engine.transition_memory(mid, "warm", reason="ts test")
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT lifecycle_history FROM memories WHERE id=?", (mid,)
        ).fetchone()
        conn.close()
        history = json.loads(row[0])
        assert "timestamp" in history[0]

    def test_transition_history_has_reason(self):
        mid = self._create_memory("reason")
        self.engine.transition_memory(mid, "warm", reason="custom reason")
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT lifecycle_history FROM memories WHERE id=?", (mid,)
        ).fetchone()
        conn.close()
        history = json.loads(row[0])
        assert history[0]["reason"] == "custom reason"

    # State validation helpers
    def test_is_valid_transition_method(self):
        assert self.engine.is_valid_transition("active", "warm") is True
        assert self.engine.is_valid_transition("active", "cold") is False

    def test_states_constant(self):
        from lifecycle.lifecycle_engine import LifecycleEngine
        assert "active" in LifecycleEngine.STATES
        assert "warm" in LifecycleEngine.STATES
        assert "cold" in LifecycleEngine.STATES
        assert "archived" in LifecycleEngine.STATES
        assert "tombstoned" in LifecycleEngine.STATES
        assert len(LifecycleEngine.STATES) == 5

    def test_all_states_present_in_distribution(self):
        dist = self.engine.get_state_distribution()
        for state in ("active", "warm", "cold", "archived", "tombstoned"):
            assert state in dist


# ============================================================================
# 5. BEHAVIORAL LEARNING ENGINE TESTS
# ============================================================================

