#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""SuperLocalMemory V2 - Edge Case Tests (v2.7)

Tests boundary conditions, malformed input, Unicode handling, SQL injection
attempts, corrupt data recovery, concurrency under stress, and other edge
cases that could crash the learning system in production. All tests use
temporary databases -- NEVER touches production ~/.claude-memory/.

Run with:
    pytest tests/test_edge_cases.py -v
"""
import json
import sqlite3
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_memory_db(db_path: Path) -> None:
    """Create a memory.db with full v2.6 schema."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            summary TEXT,
            tags TEXT DEFAULT '[]',
            category TEXT,
            memory_type TEXT DEFAULT 'general',
            importance INTEGER DEFAULT 5,
            project_name TEXT,
            project_path TEXT,
            profile TEXT DEFAULT 'default',
            parent_id INTEGER,
            cluster_id INTEGER,
            tier INTEGER DEFAULT 1,
            entity_vector TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_accessed TIMESTAMP,
            access_count INTEGER DEFAULT 0,
            created_by TEXT,
            source_protocol TEXT,
            trust_score REAL DEFAULT 1.0,
            provenance_chain TEXT
        )
    ''')
    cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
        USING fts5(content, summary, tags, content='memories', content_rowid='id')
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS identity_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_type TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            confidence REAL DEFAULT 0.0,
            evidence_count INTEGER DEFAULT 0,
            category TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS graph_clusters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            summary TEXT,
            member_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset all singletons between tests."""
    from src.learning.learning_db import LearningDB
    LearningDB.reset_instance()
    yield
    LearningDB.reset_instance()


@pytest.fixture
def env(tmp_path):
    """Create isolated test environment."""
    memory_db = tmp_path / "memory.db"
    learning_db_path = tmp_path / "learning.db"

    _create_memory_db(memory_db)

    from src.learning.learning_db import LearningDB
    ldb = LearningDB(db_path=learning_db_path)

    return {
        "memory_db": memory_db,
        "learning_db": learning_db_path,
        "ldb": ldb,
        "tmp_path": tmp_path,
    }


# ============================================================================
# Edge Case Test Scenarios
# ============================================================================


class TestCorruptJsonTags:
    """Scenario 12: Memory with invalid JSON in tags -> handled gracefully."""

    def test_corrupt_json_tags(self, env):
        """Invalid JSON in tags field -> feature extractor handles it."""
        from src.learning.feature_extractor import FeatureExtractor

        extractor = FeatureExtractor()
        extractor.set_context(
            tech_preferences={"python": {"confidence": 0.9}},
        )

        # Memory with corrupt tags
        memory = {
            "id": 1,
            "content": "python fastapi test",
            "importance": 5,
            "score": 0.5,
            "match_type": "keyword",
            "created_at": datetime.now().isoformat(),
            "access_count": 0,
            "tags": "{invalid json[",  # corrupt
        }
        features = extractor.extract_features(memory, "python")
        assert len(features) == 20
        # Should not crash, all features should be valid floats
        for f in features:
            assert isinstance(f, float)

    def test_corrupt_tags_workflow_miner(self, env):
        """Corrupt tags -> workflow miner still classifies content."""
        from src.learning.workflow_pattern_miner import WorkflowPatternMiner

        miner = WorkflowPatternMiner(
            memory_db_path=env["memory_db"],
            learning_db=env["ldb"],
        )

        memories = [
            {
                "content": "Writing pytest unit tests for authentication",
                "created_at": "2026-02-16 10:00:00",
                "tags": "not valid json at all!!!",
            },
        ]
        # Should not crash
        sequences = miner.mine_sequences(memories=memories)
        assert isinstance(sequences, list)


class TestVeryManyFeedbackSignals:
    """Scenario 13: 10,000 feedback records -> get_feedback_for_training works."""

    def test_very_many_feedback_signals(self, env):
        """10,000 feedback records -> retrieval works correctly."""
        ldb = env["ldb"]

        # Insert 10,000 records (batch for speed)
        conn = ldb._get_connection()
        cursor = conn.cursor()
        for i in range(10000):
            cursor.execute('''
                INSERT INTO ranking_feedback
                    (query_hash, memory_id, signal_type, signal_value, channel)
                VALUES (?, ?, 'mcp_used_high', 1.0, 'mcp')
            ''', (f"hash_{i % 200}", i + 1))
        conn.commit()
        conn.close()

        assert ldb.get_feedback_count() == 10000

        # Default limit is 10000 -- should return all
        training_data = ldb.get_feedback_for_training(limit=10000)
        assert len(training_data) == 10000

        # With lower limit
        limited = ldb.get_feedback_for_training(limit=100)
        assert len(limited) == 100

        # Unique query count
        unique = ldb.get_unique_query_count()
        assert unique == 200  # hash_0 through hash_199


class TestConcurrentLearningDbAccess:
    """Scenario 14: 10 threads reading + writing simultaneously -> zero errors."""

    def test_concurrent_learning_db_access(self, env):
        """10 threads mixed read/write on learning.db -> zero errors."""
        ldb = env["ldb"]
        errors = []

        def writer(thread_id: int):
            try:
                for i in range(50):
                    ldb.store_feedback(
                        query_hash=f"conc_hash_t{thread_id}_{i}",
                        memory_id=(thread_id * 1000) + i,
                        signal_type="mcp_used_high",
                        signal_value=1.0,
                        channel="mcp",
                    )
            except Exception as e:
                errors.append(("writer", thread_id, str(e)))

        def reader(thread_id: int):
            try:
                for _ in range(50):
                    _ = ldb.get_feedback_count()
                    _ = ldb.get_stats()
                    _ = ldb.get_transferable_patterns()
                    _ = ldb.get_source_scores()
            except Exception as e:
                errors.append(("reader", thread_id, str(e)))

        threads = []
        # 5 writers + 5 readers = 10 threads
        for tid in range(5):
            t = threading.Thread(target=writer, args=(tid,))
            threads.append(t)
        for tid in range(5):
            t = threading.Thread(target=reader, args=(tid,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert len(errors) == 0, f"Concurrent access errors: {errors}"

        # All writes should have completed: 5 threads * 50 = 250
        assert ldb.get_feedback_count() == 250


class TestLearningDbWalMode:
    """Scenario 15: Verify learning.db uses WAL journal mode."""

    def test_learning_db_wal_mode(self, env):
        """learning.db should be configured for WAL journal mode."""
        ldb = env["ldb"]
        conn = ldb._get_connection()
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        conn.close()

        assert mode.lower() == "wal", (
            f"Expected WAL journal mode, got '{mode}'"
        )


