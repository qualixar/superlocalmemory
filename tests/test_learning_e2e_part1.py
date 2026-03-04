#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""SuperLocalMemory V2 - Learning System End-to-End Tests (v2.7)

Full pipeline E2E tests that exercise the complete learning system from
memory seeding through feedback collection, pattern aggregation, workflow
mining, and adaptive ranking. All tests use temporary databases -- NEVER
touches production ~/.claude-memory/.

Run with:
    pytest tests/test_learning_e2e.py -v
"""
import hashlib
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

def _create_memory_db(db_path: Path) -> sqlite3.Connection:
    """Create a memory.db with the full v2.6 schema including v2.5 columns."""
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

    # FTS5 virtual table for full-text search
    cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
        USING fts5(content, summary, tags, content='memories', content_rowid='id')
    ''')

    # Identity patterns table (used by pattern_learner)
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

    # Graph clusters table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS graph_clusters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            summary TEXT,
            member_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Creator metadata (required by system)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS creator_metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    cursor.execute(
        "INSERT OR REPLACE INTO creator_metadata (key, value) "
        "VALUES ('creator', 'Varun Pratap Bhardwaj')"
    )

    conn.commit()
    return conn


def seed_memories(
    db_path: Path,
    count: int = 100,
    profile: str = "default",
    with_patterns: bool = False,
    with_timestamps: bool = False,
    source: str = None,
    base_date: datetime = None,
):
    """
    Seed test memories with realistic content.

    Returns list of inserted memory IDs.
    """
    if base_date is None:
        base_date = datetime.now() - timedelta(days=30)

    tech_topics = [
        ("Implemented FastAPI endpoint for user authentication using OAuth2",
         "python,fastapi,auth", "code", "MyProject"),
        ("Wrote pytest fixtures for database integration tests",
         "python,pytest,testing", "test", "MyProject"),
        ("Configured Docker compose for local development environment",
         "docker,devops,config", "config", "MyProject"),
        ("Designed REST API schema for payment processing service",
         "architecture,api,design", "docs", "PaymentService"),
        ("Debugged race condition in WebSocket handler for real-time updates",
         "python,websocket,debug", "debug", "MyProject"),
        ("Set up CI/CD pipeline with GitHub Actions for automated deployment",
         "ci/cd,github,deploy", "deploy", "MyProject"),
        ("Refactored database connection pool to use async context managers",
         "python,database,refactor", "code", "PaymentService"),
        ("Created React component for user dashboard with real-time charts",
         "react,frontend,component", "code", "Dashboard"),
        ("Wrote comprehensive documentation for the API endpoints",
         "documentation,api,docs", "docs", "MyProject"),
        ("Analyzed performance bottleneck in search query optimization",
         "performance,database,optimization", "debug", "MyProject"),
    ]

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    inserted_ids = []

    for i in range(count):
        topic = tech_topics[i % len(tech_topics)]
        content, tags, category, project = topic

        # Vary content slightly
        content = f"{content} (iteration {i})"
        importance = min(10, max(1, 5 + (i % 6) - 3))
        access_count = (i % 8)

        if with_timestamps:
            created_at = (base_date + timedelta(hours=i * 2)).isoformat()
        else:
            created_at = (base_date + timedelta(days=i % 30)).isoformat()

        created_by = source or ("mcp:claude-desktop" if i % 3 == 0 else
                                "cli:terminal" if i % 3 == 1 else
                                "mcp:cursor")

        cursor.execute('''
            INSERT INTO memories
                (content, tags, category, importance, project_name,
                 profile, created_at, access_count, created_by,
                 source_protocol, trust_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            content, json.dumps(tags.split(",")), category, importance,
            project, profile, created_at, access_count, created_by,
            created_by.split(":")[0] if ":" in created_by else created_by,
            1.0,
        ))
        inserted_ids.append(cursor.lastrowid)

        # Sync FTS5
        cursor.execute('''
            INSERT INTO memories_fts(rowid, content, summary, tags)
            VALUES (?, ?, ?, ?)
        ''', (cursor.lastrowid, content, "", json.dumps(tags.split(","))))

    if with_patterns:
        patterns = [
            ("preference", "python_framework", "FastAPI", 0.85, 12),
            ("preference", "test_framework", "pytest", 0.78, 8),
            ("preference", "frontend_framework", "React", 0.65, 5),
        ]
        for ptype, key, value, confidence, evidence in patterns:
            cursor.execute('''
                INSERT INTO identity_patterns
                    (pattern_type, key, value, confidence, evidence_count, category)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (ptype, key, value, confidence, evidence, "tech"))

    conn.commit()
    conn.close()
    return inserted_ids


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
def test_env(tmp_path):
    """Create isolated test environment with memory.db and learning.db."""
    memory_db = tmp_path / "memory.db"
    learning_db_path = tmp_path / "learning.db"

    # Create memory.db with full schema
    conn = _create_memory_db(memory_db)
    conn.close()

    # Create LearningDB
    from src.learning.learning_db import LearningDB
    ldb = LearningDB(db_path=learning_db_path)

    return {
        "memory_db": memory_db,
        "learning_db": learning_db_path,
        "ldb": ldb,
        "tmp_path": tmp_path,
    }


# ============================================================================
# E2E Test Scenarios
# ============================================================================


class TestFreshInstallZeroMemories:
    """Scenario 1: Empty memory.db + no learning.db -> install -> verify."""

    def test_fresh_install_zero_memories(self, test_env):
        """Empty DB -> learning.db created with all 6 tables -> no crash."""
        ldb = test_env["ldb"]

        # Verify learning.db was created
        assert test_env["learning_db"].exists()

        # Verify all 6 tables exist
        conn = ldb._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        expected_tables = {
            "transferable_patterns",
            "workflow_patterns",
            "ranking_feedback",
            "ranking_models",
            "source_quality",
            "engagement_metrics",
        }
        assert expected_tables.issubset(tables), (
            f"Missing tables: {expected_tables - tables}"
        )

        # Verify recall (empty) returns nothing, no crash
        stats = ldb.get_stats()
        assert stats["feedback_count"] == 0
        assert stats["transferable_patterns"] == 0
        assert stats["workflow_patterns"] == 0
        assert stats["tracked_sources"] == 0
        assert stats["models_trained"] == 0


class TestFreshInstallWithMemories:
    """Scenario 2: Pre-seeded memories -> first recall -> learning init."""

    def test_fresh_install_with_memories(self, test_env):
        """100 pre-seeded memories -> learning system initializes."""
        memory_db = test_env["memory_db"]
        ldb = test_env["ldb"]

        # Seed 100 memories
        ids = seed_memories(memory_db, count=100, with_patterns=True)
        assert len(ids) == 100

        # Verify memory.db has data
        conn = sqlite3.connect(str(memory_db))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM memories")
        assert cursor.fetchone()[0] == 100
        conn.close()

        # Learning DB starts empty
        stats = ldb.get_stats()
        assert stats["feedback_count"] == 0

        # Engagement tracker can read from memory.db
        from src.learning.engagement_tracker import EngagementTracker
        tracker = EngagementTracker(
            memory_db_path=memory_db,
            learning_db=ldb,
        )
        eng_stats = tracker.get_engagement_stats()
        assert eng_stats["total_memories"] == 100
        assert eng_stats["days_active"] >= 1


