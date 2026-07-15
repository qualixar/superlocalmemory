# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for the outcome -> TrustScorer feedback loop added to
``EngagementRewardModel.finalize_outcome`` (belief-update framework
alignment): "feedback: updates tau(s) for next input".
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.learning.reward import EngagementRewardModel
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, MemoryRecord
from superlocalmemory.trust.scorer import TrustScorer


@pytest.fixture()
def memory_db(tmp_path: Path) -> Path:
    """Full schema (atomic_facts + trust_scores) via the real bootstrap,
    plus the pending_outcomes table and M006's action_outcomes reward
    columns that finalize_outcome/record_recall need (deferred migrations
    that ``DatabaseManager.initialize()`` alone does not apply)."""
    db_path = tmp_path / "memory.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize(real_schema)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('default', 'default')"
        )
        conn.executescript(
            """
            ALTER TABLE action_outcomes ADD COLUMN reward REAL;
            ALTER TABLE action_outcomes ADD COLUMN settled INTEGER NOT NULL DEFAULT 0;
            ALTER TABLE action_outcomes ADD COLUMN settled_at TEXT;
            ALTER TABLE action_outcomes ADD COLUMN recall_query_id TEXT;

            CREATE TABLE IF NOT EXISTS pending_outcomes (
                outcome_id       TEXT PRIMARY KEY,
                profile_id       TEXT NOT NULL,
                session_id       TEXT NOT NULL,
                recall_query_id  TEXT NOT NULL,
                fact_ids_json    TEXT NOT NULL,
                query_text_hash  TEXT NOT NULL,
                created_at_ms    INTEGER NOT NULL,
                expires_at_ms    INTEGER NOT NULL,
                signals_json     TEXT NOT NULL DEFAULT '{}',
                status           TEXT NOT NULL DEFAULT 'pending'
            );
            """
        )
    return db_path


def _store_fact_with_source(db_path: Path, fact_id: str, agent_id: str) -> None:
    mgr = DatabaseManager(db_path)
    mgr.store_memory(MemoryRecord(memory_id=f"m_{fact_id}", content="parent"))
    mgr.store_fact(AtomicFact(
        fact_id=fact_id, memory_id=f"m_{fact_id}",
        content="some fact", source_agent_id=agent_id,
    ))


class TestTrustFeedbackLoop:
    def test_positive_outcome_boosts_source_trust(self, memory_db: Path) -> None:
        _store_fact_with_source(memory_db, "f1", "agent_good")
        model = EngagementRewardModel(memory_db)
        outcome_id = model.record_recall(
            profile_id="default", session_id="s", recall_query_id="q1",
            fact_ids=["f1"], query_text="x",
        )
        model.register_signal(outcome_id=outcome_id, signal_name="cite", signal_value=True)
        reward = model.finalize_outcome(outcome_id=outcome_id)
        assert reward > 0.55  # positive engagement -> recall_hit signal

        scorer = TrustScorer(DatabaseManager(memory_db))
        trust = scorer.get_agent_trust("agent_good", "default")
        assert trust > 0.5  # boosted above the neutral prior

    def test_negative_outcome_lowers_source_trust(self, memory_db: Path) -> None:
        _store_fact_with_source(memory_db, "f2", "agent_bad")
        model = EngagementRewardModel(memory_db)
        outcome_id = model.record_recall(
            profile_id="default", session_id="s", recall_query_id="q2",
            fact_ids=["f2"], query_text="x",
        )
        model.register_signal(outcome_id=outcome_id, signal_name="requery", signal_value=True)
        reward = model.finalize_outcome(outcome_id=outcome_id)
        assert reward < 0.45  # requery -> contradiction signal

        scorer = TrustScorer(DatabaseManager(memory_db))
        trust = scorer.get_agent_trust("agent_bad", "default")
        assert trust < 0.5  # lowered below the neutral prior

    def test_missing_atomic_facts_table_is_non_fatal(self, tmp_path: Path) -> None:
        """No atomic_facts table (e.g. minimal test DBs) must not break finalize_outcome."""
        db_path = tmp_path / "minimal.db"
        with sqlite3.connect(db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE action_outcomes (
                    outcome_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL DEFAULT 'default',
                    query TEXT NOT NULL DEFAULT '', fact_ids_json TEXT NOT NULL DEFAULT '[]',
                    outcome TEXT NOT NULL DEFAULT '', context_json TEXT NOT NULL DEFAULT '{}',
                    timestamp TEXT NOT NULL DEFAULT (datetime('now')), reward REAL,
                    settled INTEGER NOT NULL DEFAULT 0, settled_at TEXT, recall_query_id TEXT
                );
                CREATE TABLE pending_outcomes (
                    outcome_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL, session_id TEXT NOT NULL,
                    recall_query_id TEXT NOT NULL, fact_ids_json TEXT NOT NULL,
                    query_text_hash TEXT NOT NULL, created_at_ms INTEGER NOT NULL,
                    expires_at_ms INTEGER NOT NULL, signals_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'pending'
                );
                """
            )
        model = EngagementRewardModel(db_path)
        outcome_id = model.record_recall(
            profile_id="default", session_id="s", recall_query_id="q3",
            fact_ids=["ghost"], query_text="x",
        )
        model.register_signal(outcome_id=outcome_id, signal_name="cite", signal_value=True)
        reward = model.finalize_outcome(outcome_id=outcome_id)
        assert reward == pytest.approx(0.9)  # unaffected by the missing table
