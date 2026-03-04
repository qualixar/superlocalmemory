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

class TestModuleAvailability:
    """Verify all v2.8 modules are importable and report correct status."""

    def test_lifecycle_available(self):
        from lifecycle import get_status
        status = get_status()
        assert status["lifecycle_available"] is True

    def test_behavioral_available(self):
        from behavioral import get_status
        status = get_status()
        assert status["behavioral_available"] is True

    def test_compliance_available(self):
        from compliance import get_status
        status = get_status()
        assert status["compliance_available"] is True

    def test_lifecycle_engine_importable(self):
        from lifecycle.lifecycle_engine import LifecycleEngine
        assert LifecycleEngine is not None

    def test_lifecycle_evaluator_importable(self):
        from lifecycle.lifecycle_evaluator import LifecycleEvaluator
        assert LifecycleEvaluator is not None

    def test_lifecycle_retention_policy_importable(self):
        from lifecycle.retention_policy import RetentionPolicyManager
        assert RetentionPolicyManager is not None

    def test_lifecycle_bounded_growth_importable(self):
        from lifecycle.bounded_growth import BoundedGrowthEnforcer
        assert BoundedGrowthEnforcer is not None

    def test_behavioral_outcome_tracker_importable(self):
        from behavioral.outcome_tracker import OutcomeTracker
        assert OutcomeTracker is not None

    def test_behavioral_patterns_importable(self):
        from behavioral.behavioral_patterns import BehavioralPatternExtractor
        assert BehavioralPatternExtractor is not None

    def test_compliance_abac_importable(self):
        from compliance.abac_engine import ABACEngine
        assert ABACEngine is not None

    def test_compliance_audit_db_importable(self):
        from compliance.audit_db import AuditDB
        assert AuditDB is not None

    def test_feature_vector_is_20(self):
        from learning.feature_extractor import NUM_FEATURES
        assert NUM_FEATURES == 20

    def test_feature_names_length_matches(self):
        from learning.feature_extractor import FEATURE_NAMES, NUM_FEATURES
        assert len(FEATURE_NAMES) == NUM_FEATURES

    def test_feature_names_contain_lifecycle(self):
        from learning.feature_extractor import FEATURE_NAMES
        assert "lifecycle_state" in FEATURE_NAMES

    def test_feature_names_contain_outcome(self):
        from learning.feature_extractor import FEATURE_NAMES
        assert "outcome_success_rate" in FEATURE_NAMES

    def test_feature_names_contain_behavioral(self):
        from learning.feature_extractor import FEATURE_NAMES
        assert "behavioral_match" in FEATURE_NAMES

    def test_feature_names_contain_cross_project(self):
        from learning.feature_extractor import FEATURE_NAMES
        assert "cross_project_score" in FEATURE_NAMES

    def test_feature_names_contain_retention(self):
        from learning.feature_extractor import FEATURE_NAMES
        assert "retention_priority" in FEATURE_NAMES

    def test_feature_names_contain_trust(self):
        from learning.feature_extractor import FEATURE_NAMES
        assert "trust_at_creation" in FEATURE_NAMES

    def test_feature_names_contain_lifecycle_decay(self):
        from learning.feature_extractor import FEATURE_NAMES
        assert "lifecycle_aware_decay" in FEATURE_NAMES

    def test_mcp_tools_report_outcome_available(self):
        import mcp_tools_v28
        assert hasattr(mcp_tools_v28, 'report_outcome')

    def test_mcp_tools_get_lifecycle_status_available(self):
        import mcp_tools_v28
        assert hasattr(mcp_tools_v28, 'get_lifecycle_status')

    def test_mcp_tools_set_retention_policy_available(self):
        import mcp_tools_v28
        assert hasattr(mcp_tools_v28, 'set_retention_policy')

    def test_mcp_tools_compact_memories_available(self):
        import mcp_tools_v28
        assert hasattr(mcp_tools_v28, 'compact_memories')

    def test_mcp_tools_get_behavioral_patterns_available(self):
        import mcp_tools_v28
        assert hasattr(mcp_tools_v28, 'get_behavioral_patterns')

    def test_mcp_tools_audit_trail_available(self):
        import mcp_tools_v28
        assert hasattr(mcp_tools_v28, 'audit_trail')

    def test_mcp_tools_are_async(self):
        import mcp_tools_v28
        import inspect
        assert inspect.iscoroutinefunction(mcp_tools_v28.report_outcome)
        assert inspect.iscoroutinefunction(mcp_tools_v28.get_lifecycle_status)
        assert inspect.iscoroutinefunction(mcp_tools_v28.audit_trail)

    def test_lifecycle_get_singleton_function(self):
        from lifecycle import get_lifecycle_engine
        assert callable(get_lifecycle_engine)

    def test_behavioral_get_singleton_function(self):
        from behavioral import get_outcome_tracker
        assert callable(get_outcome_tracker)

    def test_compliance_get_singleton_function(self):
        from compliance import get_abac_engine
        assert callable(get_abac_engine)

    def test_lifecycle_status_keys(self):
        from lifecycle import get_status
        status = get_status()
        assert "lifecycle_available" in status
        assert "init_error" in status
        assert "engine_active" in status

    def test_behavioral_status_keys(self):
        from behavioral import get_status
        status = get_status()
        assert "behavioral_available" in status
        assert "init_error" in status
        assert "tracker_active" in status

    def test_compliance_status_keys(self):
        from compliance import get_status
        status = get_status()
        assert "compliance_available" in status
        assert "init_error" in status
        assert "abac_active" in status


# ============================================================================
# 4. LIFECYCLE STATE MACHINE TESTS
# ============================================================================

