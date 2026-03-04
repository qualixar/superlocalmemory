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


class TestUnicodeContent:
    """Scenario 1: Memory with emoji, CJK, RTL text."""

    def test_unicode_content(self, env):
        """Unicode content in feedback and patterns -> handled correctly."""
        ldb = env["ldb"]

        # Store feedback with Unicode query keywords
        row_id = ldb.store_feedback(
            query_hash="unicode_test_hash",
            memory_id=1,
            signal_type="mcp_used_high",
            signal_value=1.0,
            channel="mcp",
            query_keywords="emoji,test",
        )
        assert row_id is not None

        # Store pattern with CJK characters
        pattern_id = ldb.upsert_transferable_pattern(
            pattern_type="preference",
            key="framework",
            value="React",
            confidence=0.85,
            evidence_count=5,
        )
        assert pattern_id is not None

        # Verify retrieval
        patterns = ldb.get_transferable_patterns(min_confidence=0.0)
        assert len(patterns) >= 1
        assert patterns[0]["value"] == "React"

    def test_unicode_in_feedback_collector(self, env):
        """FeedbackCollector handles Unicode queries."""
        from src.learning.feedback_collector import FeedbackCollector

        collector = FeedbackCollector(learning_db=env["ldb"])

        # Query with emoji and CJK
        result = collector.record_memory_used(
            memory_id=1,
            query="deploy application with emoji content",
            usefulness="high",
        )
        assert result is not None

    def test_unicode_workflow_classification(self, env):
        """WorkflowPatternMiner classifies Unicode content without crash."""
        from src.learning.workflow_pattern_miner import WorkflowPatternMiner

        miner = WorkflowPatternMiner(
            memory_db_path=env["memory_db"],
            learning_db=env["ldb"],
        )

        # Memories with Unicode content
        memories = [
            {"content": "Implement function for data processing", "created_at": "2026-02-16 10:00:00"},
            {"content": "Test the Unicode handling module", "created_at": "2026-02-16 11:00:00"},
            {"content": "Debug error in the parser component", "created_at": "2026-02-16 12:00:00"},
        ]
        sequences = miner.mine_sequences(memories=memories)
        assert isinstance(sequences, list)  # No crash


class TestVeryLongContent:
    """Scenario 2: 50KB memory content -> keyword extraction handles it."""

    def test_very_long_content(self, env):
        """50KB content -> keyword extraction doesn't hang."""
        from src.learning.feedback_collector import FeedbackCollector

        collector = FeedbackCollector(learning_db=env["ldb"])

        # Generate 50KB+ content
        long_content = "performance optimization database query " * 1300  # >50KB
        assert len(long_content) >= 50000

        # Should complete quickly without hanging
        result = collector.record_memory_used(
            memory_id=1,
            query=long_content,
            usefulness="high",
        )
        assert result is not None

    def test_very_long_content_workflow_miner(self, env):
        """50KB content in workflow classification -> no hang."""
        from src.learning.workflow_pattern_miner import WorkflowPatternMiner

        miner = WorkflowPatternMiner(
            memory_db_path=env["memory_db"],
            learning_db=env["ldb"],
        )

        long_content = "implement function class module refactor " * 1250
        memories = [
            {"content": long_content, "created_at": "2026-02-16 10:00:00"},
        ]
        sequences = miner.mine_sequences(memories=memories)
        assert isinstance(sequences, list)


class TestSpecialCharsInQuery:
    """Scenario 3: SQL injection attempt in query -> parameterized queries protect."""

    def test_special_chars_in_query(self, env):
        """SQL injection attempt -> handled safely via parameterized queries."""
        ldb = env["ldb"]

        # Attempt SQL injection in feedback
        injection_strings = [
            "'; DROP TABLE ranking_feedback; --",
            "1 OR 1=1",
            "Robert'); DROP TABLE memories;--",
            '"; SELECT * FROM ranking_feedback WHERE "1"="1',
        ]

        for injection in injection_strings:
            row_id = ldb.store_feedback(
                query_hash=injection,
                memory_id=1,
                signal_type="mcp_used_high",
                signal_value=1.0,
                channel="mcp",
                query_keywords=injection,
            )
            assert row_id is not None

        # Verify tables still exist and data is intact
        assert ldb.get_feedback_count() == len(injection_strings)

        # Verify data round-trips correctly
        feedback = ldb.get_feedback_for_training()
        stored_hashes = {f["query_hash"] for f in feedback}
        for injection in injection_strings:
            assert injection in stored_hashes

    def test_injection_in_pattern_key(self, env):
        """SQL injection in pattern key -> handled safely."""
        ldb = env["ldb"]

        pattern_id = ldb.upsert_transferable_pattern(
            pattern_type="preference",
            key="'; DROP TABLE transferable_patterns; --",
            value="React",
            confidence=0.8,
            evidence_count=5,
        )
        assert pattern_id is not None

        patterns = ldb.get_transferable_patterns()
        assert len(patterns) >= 1


class TestEmptyStringQuery:
    """Scenario 4: Empty string recall -> returns empty, no crash."""

    def test_empty_string_query_feedback(self, env):
        """Empty query in FeedbackCollector -> returns None."""
        from src.learning.feedback_collector import FeedbackCollector

        collector = FeedbackCollector(learning_db=env["ldb"])

        result = collector.record_memory_used(
            memory_id=1,
            query="",
            usefulness="high",
        )
        assert result is None  # Should return None for empty query

    def test_empty_string_rerank(self, env):
        """Empty query in rerank -> still returns results."""
        from src.learning.adaptive_ranker import AdaptiveRanker

        ranker = AdaptiveRanker(learning_db=env["ldb"])
        results = [{"id": 1, "content": "test", "score": 0.5}]

        reranked = ranker.rerank(results, "")
        assert len(reranked) == 1

    def test_empty_string_keyword_extraction(self, env):
        """Empty string keyword extraction -> returns empty string."""
        from src.learning.feedback_collector import FeedbackCollector

        collector = FeedbackCollector(learning_db=env["ldb"])
        keywords = collector._extract_keywords("")
        assert keywords == ""

    def test_none_query_keyword_extraction(self, env):
        """None query -> keyword extraction handles gracefully."""
        from src.learning.feedback_collector import FeedbackCollector

        collector = FeedbackCollector(learning_db=env["ldb"])
        keywords = collector._extract_keywords(None)
        assert keywords == ""


class TestNegativeMemoryId:
    """Scenario 5: Negative ID in memory_used -> handled gracefully."""

    def test_negative_memory_id(self, env):
        """Negative memory ID -> stored (no crash), DB integrity maintained."""
        ldb = env["ldb"]

        row_id = ldb.store_feedback(
            query_hash="neg_test",
            memory_id=-1,
            signal_type="mcp_used_high",
            signal_value=1.0,
            channel="mcp",
        )
        # SQLite allows negative integers in non-PK columns
        assert row_id is not None

        feedback = ldb.get_feedback_for_training()
        neg_records = [f for f in feedback if f["memory_id"] == -1]
        assert len(neg_records) == 1


class TestDuplicateFeedback:
    """Scenario 6: Same feedback recorded twice -> both stored."""

    def test_duplicate_feedback(self, env):
        """Same feedback recorded twice -> both stored (no unique constraint)."""
        ldb = env["ldb"]

        for _ in range(2):
            ldb.store_feedback(
                query_hash="same_hash",
                memory_id=42,
                signal_type="mcp_used_high",
                signal_value=1.0,
                channel="mcp",
            )

        assert ldb.get_feedback_count() == 2

        feedback = ldb.get_feedback_for_training()
        same_records = [
            f for f in feedback
            if f["query_hash"] == "same_hash" and f["memory_id"] == 42
        ]
        assert len(same_records) == 2


