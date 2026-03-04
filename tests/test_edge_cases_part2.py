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


class TestMaxImportance:
    """Scenario 7: Importance = 10 -> normalized to 1.0."""

    def test_max_importance(self, env):
        """Importance = 10 -> feature extraction normalizes to 1.0."""
        from src.learning.feature_extractor import FeatureExtractor

        extractor = FeatureExtractor()
        memory = {
            "id": 1,
            "content": "test memory",
            "importance": 10,
            "score": 0.5,
            "match_type": "keyword",
            "created_at": datetime.now().isoformat(),
            "access_count": 0,
        }
        features = extractor.extract_features(memory, "test")
        importance_norm = features[6]  # index 6 = importance_norm
        assert importance_norm == 1.0


class TestZeroImportance:
    """Scenario 8: Importance = 0 -> normalized to 0.1 (clamped to min 1)."""

    def test_zero_importance(self, env):
        """Importance = 0 -> clamped to 1, normalized to 0.1."""
        from src.learning.feature_extractor import FeatureExtractor

        extractor = FeatureExtractor()
        memory = {
            "id": 1,
            "content": "test memory",
            "importance": 0,
            "score": 0.5,
            "match_type": "keyword",
            "created_at": datetime.now().isoformat(),
            "access_count": 0,
        }
        features = extractor.extract_features(memory, "test")
        importance_norm = features[6]
        # Clamped to max(1, min(0, 10)) = 1, then 1/10 = 0.1
        assert importance_norm == 0.1


class TestFutureTimestamp:
    """Scenario 9: Memory with future created_at -> recency score handles it."""

    def test_future_timestamp(self, env):
        """Future created_at -> recency score doesn't crash or go > 1.0."""
        from src.learning.feature_extractor import FeatureExtractor

        extractor = FeatureExtractor()
        future_date = (datetime.now() + timedelta(days=365)).isoformat()

        memory = {
            "id": 1,
            "content": "future memory",
            "importance": 5,
            "score": 0.5,
            "match_type": "keyword",
            "created_at": future_date,
            "access_count": 0,
        }
        features = extractor.extract_features(memory, "test")
        recency_score = features[7]  # index 7 = recency_score

        # Should still be in [0, 1] range
        assert 0.0 <= recency_score <= 1.0


class TestNullFields:
    """Scenario 10: Memory with NULL fields -> all features handle gracefully."""

    def test_null_fields(self, env):
        """NULL project_name, project_path, created_by -> no crash."""
        from src.learning.feature_extractor import FeatureExtractor

        extractor = FeatureExtractor()
        memory = {
            "id": 1,
            "content": "test memory",
            "importance": None,
            "score": None,
            "match_type": None,
            "created_at": None,
            "access_count": None,
            "project_name": None,
            "project_path": None,
            "created_by": None,
            "tags": None,
        }
        features = extractor.extract_features(memory, "test")
        assert len(features) == 20
        # All features should be in [0, 1] range
        for i, f in enumerate(features):
            assert 0.0 <= f <= 1.0, (
                f"Feature {i} out of range: {f}"
            )

    def test_null_fields_project_detection(self, env):
        """NULL project fields in recent memories -> detect returns None."""
        from src.learning.project_context_manager import ProjectContextManager

        pcm = ProjectContextManager(memory_db_path=env["memory_db"])

        # Pass memories with all NULL project fields
        memories = [
            {"id": 1, "project_name": None, "project_path": None,
             "cluster_id": None, "content": "test"},
            {"id": 2, "project_name": "", "project_path": "",
             "cluster_id": None, "content": "another"},
        ]
        project = pcm.detect_current_project(recent_memories=memories)
        assert project is None


class TestMissingColumnsOldDb:
    """Scenario 11: memory.db without v2.5 columns -> learning still works."""

    def test_missing_columns_old_db(self, tmp_path):
        """Pre-v2.5 memory.db (no created_by, source_protocol) -> works."""
        # Create a minimal pre-v2.5 database
        old_db = tmp_path / "old_memory.db"
        conn = sqlite3.connect(str(old_db))
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                summary TEXT,
                tags TEXT DEFAULT '[]',
                category TEXT,
                importance INTEGER DEFAULT 5,
                project_name TEXT,
                profile TEXT DEFAULT 'default',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0
            )
        ''')
        # Insert some memories
        for i in range(10):
            cursor.execute(
                "INSERT INTO memories (content, created_at) VALUES (?, ?)",
                (f"Old memory {i}", datetime.now().isoformat()),
            )
        conn.commit()
        conn.close()

        # EngagementTracker should handle missing columns gracefully
        from src.learning.learning_db import LearningDB
        ldb = LearningDB(db_path=tmp_path / "learning.db")

        from src.learning.engagement_tracker import EngagementTracker
        tracker = EngagementTracker(
            memory_db_path=old_db,
            learning_db=ldb,
        )

        stats = tracker.get_engagement_stats()
        assert stats["total_memories"] == 10
        assert stats["active_sources"] == []  # No created_by column

    def test_missing_columns_source_scorer(self, tmp_path):
        """Pre-v2.5 DB -> SourceQualityScorer groups all as 'unknown'."""
        old_db = tmp_path / "old_memory.db"
        conn = sqlite3.connect(str(old_db))
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                importance INTEGER DEFAULT 5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0
            )
        ''')
        for i in range(10):
            cursor.execute(
                "INSERT INTO memories (content) VALUES (?)",
                (f"Memory {i}",)
            )
        conn.commit()
        conn.close()

        from src.learning.learning_db import LearningDB
        ldb = LearningDB(db_path=tmp_path / "learning.db")

        from src.learning.source_quality_scorer import SourceQualityScorer
        scorer = SourceQualityScorer(
            memory_db_path=old_db,
            learning_db=ldb,
        )
        scores = scorer.compute_source_scores()

        # All memories grouped as 'unknown'
        assert isinstance(scores, dict)
        if scores:
            assert "unknown" in scores


