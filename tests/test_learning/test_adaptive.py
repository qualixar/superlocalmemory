# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.learning.adaptive — Adaptive Learning (3-Phase).

Covers:
  Phase 1 — Collect feedback:
    - record_feedback: stores FeedbackRecord in DB
    - get_feedback_count: returns correct count
  Phase 2 — Train:
    - train with insufficient data: returns empty dict
    - train with enough positive data: returns default weights
    - train with enough negative data: returns boosted precision weights
  Phase 3 — Apply weights:
    - get_weights with no training: triggers lazy train, returns defaults
    - get_weights after training: returns learned weights
    - get_weights for specific query_type vs fallback to "general"
  Misc:
    - is_trained: threshold check
    - Profile isolation: feedback is profile-scoped
"""

from __future__ import annotations

from pathlib import Path

import pytest

from superlocalmemory.learning.adaptive import _MIN_FEEDBACK_FOR_TRAINING, AdaptiveLearner
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, FeedbackRecord, MemoryRecord

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    db_path = tmp_path / "adaptive_test.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize(real_schema)
    # Create a parent memory and fact for FK satisfaction
    mgr.store_memory(MemoryRecord(memory_id="m0", content="parent"))
    for i in range(30):
        mgr.store_fact(AtomicFact(
            fact_id=f"f{i}", memory_id="m0", content=f"fact {i}"
        ))
    return mgr


@pytest.fixture()
def learner(db: DatabaseManager) -> AdaptiveLearner:
    return AdaptiveLearner(db)


def _fill_feedback(
    learner: AdaptiveLearner,
    n: int,
    feedback_type: str = "relevant",
    profile_id: str = "default",
) -> None:
    """Helper to insert n feedback records."""
    for i in range(n):
        learner.record_feedback(
            query=f"query_{i}",
            fact_id=f"f{i % 30}",
            feedback_type=feedback_type,
            profile_id=profile_id,
        )


# ---------------------------------------------------------------------------
# Phase 1: Collect feedback
# ---------------------------------------------------------------------------

class TestRecordFeedback:
    def test_creates_feedback_record(self, learner: AdaptiveLearner) -> None:
        rec = learner.record_feedback(
            "What is X?", "f0", "relevant", "default", dwell_time_ms=1500
        )
        assert isinstance(rec, FeedbackRecord)
        assert rec.query == "What is X?"
        assert rec.fact_id == "f0"
        assert rec.feedback_type == "relevant"
        assert rec.dwell_time_ms == 1500

    def test_feedback_persists_to_db(
        self, learner: AdaptiveLearner, db: DatabaseManager
    ) -> None:
        learner.record_feedback("q", "f0", "irrelevant", "default")
        rows = db.execute(
            "SELECT * FROM feedback_records WHERE profile_id = 'default'"
        )
        assert len(rows) == 1
        assert dict(rows[0])["feedback_type"] == "irrelevant"


class TestGetFeedbackCount:
    def test_zero_initially(self, learner: AdaptiveLearner) -> None:
        assert learner.get_feedback_count("default") == 0

    def test_counts_correctly(self, learner: AdaptiveLearner) -> None:
        _fill_feedback(learner, 5)
        assert learner.get_feedback_count("default") == 5

    def test_counts_per_profile(self, learner: AdaptiveLearner, db: DatabaseManager) -> None:
        db.execute(
            "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('work', 'Work')"
        )
        _fill_feedback(learner, 3, profile_id="default")
        _fill_feedback(learner, 7, profile_id="work")
        assert learner.get_feedback_count("default") == 3
        assert learner.get_feedback_count("work") == 7


# ---------------------------------------------------------------------------
# Phase 2: Train
# ---------------------------------------------------------------------------

class TestTrain:
    def test_insufficient_data_returns_empty(self, learner: AdaptiveLearner) -> None:
        _fill_feedback(learner, _MIN_FEEDBACK_FOR_TRAINING - 1)
        result = learner.train("default")
        assert result == {}

    def test_positive_feedback_returns_default_weights(
        self, learner: AdaptiveLearner
    ) -> None:
        _fill_feedback(learner, _MIN_FEEDBACK_FOR_TRAINING, "relevant")
        result = learner.train("default")
        assert "general" in result
        assert result["general"]["semantic"] == 1.5
        assert result["general"]["bm25"] == 1.0

    def test_negative_feedback_boosts_precision_channels(
        self, learner: AdaptiveLearner
    ) -> None:
        _fill_feedback(learner, _MIN_FEEDBACK_FOR_TRAINING, "irrelevant")
        result = learner.train("default")
        assert "general" in result
        assert result["general"]["bm25"] == 1.5
        assert result["general"]["entity_graph"] == 1.3

    def test_mixed_feedback_below_50_percent(
        self, learner: AdaptiveLearner
    ) -> None:
        """With more irrelevant than relevant, should boost precision."""
        _fill_feedback(learner, 5, "relevant")
        _fill_feedback(learner, 16, "irrelevant")
        result = learner.train("default")
        assert result["general"]["bm25"] == 1.5

    def test_mixed_feedback_above_70_percent(
        self, learner: AdaptiveLearner
    ) -> None:
        """With >70% relevant, should return default weights."""
        _fill_feedback(learner, 18, "relevant")
        _fill_feedback(learner, 3, "irrelevant")
        result = learner.train("default")
        assert result["general"]["semantic"] == 1.5

    def test_all_partial_feedback_returns_empty(
        self, learner: AdaptiveLearner
    ) -> None:
        """Only 'partial' feedback means 0 positive + 0 negative = empty."""
        _fill_feedback(learner, _MIN_FEEDBACK_FOR_TRAINING, "partial")
        result = learner.train("default")
        assert result == {}


# ---------------------------------------------------------------------------
# Phase 3: Get weights
# ---------------------------------------------------------------------------

class TestGetWeights:
    def test_defaults_when_no_training_data(
        self, learner: AdaptiveLearner
    ) -> None:
        weights = learner.get_weights("factual", "default")
        assert weights["semantic"] == 1.5
        assert weights["bm25"] == 1.0

    def test_returns_learned_general_weights(
        self, learner: AdaptiveLearner
    ) -> None:
        _fill_feedback(learner, _MIN_FEEDBACK_FOR_TRAINING, "irrelevant")
        learner.train("default")
        weights = learner.get_weights("factual", "default")
        assert weights["bm25"] == 1.5

    def test_specific_query_type_falls_back_to_general(
        self, learner: AdaptiveLearner
    ) -> None:
        _fill_feedback(learner, _MIN_FEEDBACK_FOR_TRAINING, "relevant")
        learner.train("default")
        weights = learner.get_weights("temporal", "default")
        # Falls back to "general"
        assert "semantic" in weights

    def test_lazy_train_triggers_on_get_weights(
        self, learner: AdaptiveLearner
    ) -> None:
        """get_weights calls train internally if no learned weights."""
        _fill_feedback(learner, _MIN_FEEDBACK_FOR_TRAINING, "relevant")
        # No explicit train call
        weights = learner.get_weights("factual", "default")
        assert "semantic" in weights


# ---------------------------------------------------------------------------
# is_trained
# ---------------------------------------------------------------------------

class TestIsTrained:
    def test_not_trained_initially(self, learner: AdaptiveLearner) -> None:
        assert not learner.is_trained("default")

    def test_trained_after_enough_feedback(
        self, learner: AdaptiveLearner
    ) -> None:
        _fill_feedback(learner, _MIN_FEEDBACK_FOR_TRAINING)
        assert learner.is_trained("default")

    def test_not_trained_one_below_threshold(
        self, learner: AdaptiveLearner
    ) -> None:
        _fill_feedback(learner, _MIN_FEEDBACK_FOR_TRAINING - 1)
        assert not learner.is_trained("default")
