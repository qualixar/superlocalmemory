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


class TestFullFeedbackLoop:
    """Scenario 3: Remember -> recall -> feedback -> verify ranking phase."""

    def test_full_feedback_loop(self, test_env):
        """Remember 20 memories -> recall -> feedback -> verify data stored."""
        memory_db = test_env["memory_db"]
        ldb = test_env["ldb"]

        # Seed 20 memories
        ids = seed_memories(memory_db, count=20)

        # Create feedback collector
        from src.learning.feedback_collector import FeedbackCollector
        collector = FeedbackCollector(learning_db=ldb)

        # Simulate MCP recall -> memory_used feedback
        for i, mid in enumerate(ids[:10]):
            collector.record_memory_used(
                memory_id=mid,
                query="deploy FastAPI application",
                usefulness="high" if i < 5 else "medium",
                source_tool="claude-desktop",
                rank_position=i + 1,
            )

        # Simulate CLI feedback
        collector.record_cli_useful(ids[10:15], "pytest fixtures")

        # Simulate dashboard click
        collector.record_dashboard_click(
            memory_id=ids[15],
            query="docker compose setup",
            dwell_time=8.5,
        )

        # Verify feedback stored in learning.db
        assert ldb.get_feedback_count() == 16  # 10 mcp + 5 cli + 1 dashboard

        # Verify summary
        summary = collector.get_feedback_summary()
        assert summary["total_signals"] == 16
        assert summary["by_channel"]["mcp"] == 10
        assert summary["by_channel"]["cli"] == 5
        assert summary["by_channel"]["dashboard"] == 1

        # Check adaptive ranker phase (should be baseline with 16 signals)
        from src.learning.adaptive_ranker import AdaptiveRanker
        ranker = AdaptiveRanker(learning_db=ldb)
        assert ranker.get_phase() == "baseline"

        # Add more feedback to reach rule_based threshold (20+)
        for mid in ids[16:20]:
            collector.record_memory_used(
                memory_id=mid,
                query="authentication setup",
                usefulness="high",
            )
        assert ldb.get_feedback_count() == 20
        assert ranker.get_phase() == "rule_based"


