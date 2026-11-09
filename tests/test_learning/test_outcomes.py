# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.learning.outcomes — Outcome Tracking.

Covers:
  - record_outcome: stores ActionOutcome in DB with JSON serialization
  - get_outcomes: retrieves recent outcomes with limit
  - get_success_rate: computes correct rate (success=1.0, partial=0.5)
  - get_fact_success_rate: per-fact success rate, neutral 0.5 default
  - Profile isolation: outcomes scoped to profile_id
  - Edge cases: no outcomes, empty fact_ids, special characters
  - ActionOutcome field correctness (JSON round-trip)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from superlocalmemory.learning.outcomes import OutcomeTracker
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import ActionOutcome

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    db_path = tmp_path / "outcomes_test.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize(real_schema)
    return mgr


@pytest.fixture()
def tracker(db: DatabaseManager) -> OutcomeTracker:
    return OutcomeTracker(db)


# ---------------------------------------------------------------------------
# record_outcome
# ---------------------------------------------------------------------------

class TestRecordOutcome:
    def test_creates_outcome_record(self, tracker: OutcomeTracker) -> None:
        rec = tracker.record_outcome(
            query="What is X?",
            fact_ids=["f1", "f2"],
            outcome="success",
            profile_id="default",
            context={"source": "test"},
        )
        assert isinstance(rec, ActionOutcome)
        assert rec.query == "What is X?"
        assert rec.fact_ids == ["f1", "f2"]
        assert rec.outcome == "success"
        assert rec.context == {"source": "test"}
        assert rec.outcome_id  # non-empty

    def test_outcome_persists_to_db(
        self, tracker: OutcomeTracker, db: DatabaseManager
    ) -> None:
        tracker.record_outcome("q", ["f1"], "failure", "default")
        rows = db.execute(
            "SELECT * FROM action_outcomes WHERE profile_id = 'default'"
        )
        assert len(rows) == 1
        d = dict(rows[0])
        assert d["outcome"] == "failure"
        assert json.loads(d["fact_ids_json"]) == ["f1"]

    def test_default_context_is_empty_dict(self, tracker: OutcomeTracker) -> None:
        rec = tracker.record_outcome("q", [], "success", "default")
        assert rec.context == {}

    def test_empty_fact_ids(self, tracker: OutcomeTracker) -> None:
        rec = tracker.record_outcome("q", [], "failure", "default")
        assert rec.fact_ids == []


# ---------------------------------------------------------------------------
# get_outcomes
# ---------------------------------------------------------------------------

class TestGetOutcomes:
    def test_returns_recent_outcomes(self, tracker: OutcomeTracker) -> None:
        tracker.record_outcome("q1", ["f1"], "success", "default")
        tracker.record_outcome("q2", ["f2"], "failure", "default")
        outcomes = tracker.get_outcomes("default")
        assert len(outcomes) == 2
        assert all(isinstance(o, ActionOutcome) for o in outcomes)

    def test_limit_respected(self, tracker: OutcomeTracker) -> None:
        for i in range(10):
            tracker.record_outcome(f"q{i}", [f"f{i}"], "success", "default")
        outcomes = tracker.get_outcomes("default", limit=3)
        assert len(outcomes) == 3

    def test_empty_for_unknown_profile(self, tracker: OutcomeTracker) -> None:
        assert tracker.get_outcomes("ghost") == []

    def test_json_roundtrip_fact_ids(self, tracker: OutcomeTracker) -> None:
        tracker.record_outcome("q", ["a", "b", "c"], "success", "default")
        outcomes = tracker.get_outcomes("default")
        assert outcomes[0].fact_ids == ["a", "b", "c"]

    def test_json_roundtrip_context(self, tracker: OutcomeTracker) -> None:
        ctx = {"model": "v3", "mode": "a", "channels": [1, 2]}
        tracker.record_outcome("q", [], "partial", "default", context=ctx)
        outcomes = tracker.get_outcomes("default")
        assert outcomes[0].context == ctx


# ---------------------------------------------------------------------------
# get_success_rate
# ---------------------------------------------------------------------------

class TestGetSuccessRate:
    def test_zero_when_no_outcomes(self, tracker: OutcomeTracker) -> None:
        assert tracker.get_success_rate("default") == 0.0

    def test_all_success(self, tracker: OutcomeTracker) -> None:
        for _ in range(5):
            tracker.record_outcome("q", ["f"], "success", "default")
        assert tracker.get_success_rate("default") == 1.0

    def test_all_failure(self, tracker: OutcomeTracker) -> None:
        for _ in range(5):
            tracker.record_outcome("q", ["f"], "failure", "default")
        assert tracker.get_success_rate("default") == 0.0

    def test_partial_counts_as_half(self, tracker: OutcomeTracker) -> None:
        tracker.record_outcome("q1", ["f"], "success", "default")  # 1.0
        tracker.record_outcome("q2", ["f"], "partial", "default")  # 0.5
        # Total: 2, success value: 1.0 + 0.5 = 1.5
        # Rate: 1.5 / 2 = 0.75
        assert tracker.get_success_rate("default") == pytest.approx(0.75)

    def test_mixed_outcomes(self, tracker: OutcomeTracker) -> None:
        tracker.record_outcome("q1", ["f"], "success", "default")   # 1.0
        tracker.record_outcome("q2", ["f"], "failure", "default")   # 0.0
        tracker.record_outcome("q3", ["f"], "partial", "default")   # 0.5
        tracker.record_outcome("q4", ["f"], "failure", "default")   # 0.0
        # Total: 4, success value: 1.0 + 0.0 + 0.5 + 0.0 = 1.5
        # Rate: 1.5 / 4 = 0.375
        assert tracker.get_success_rate("default") == pytest.approx(0.375)


# ---------------------------------------------------------------------------
# get_fact_success_rate
# ---------------------------------------------------------------------------

class TestGetFactSuccessRate:
    def test_neutral_when_no_data(self, tracker: OutcomeTracker) -> None:
        assert tracker.get_fact_success_rate("unknown_fact", "default") == 0.5

    def test_fact_in_successful_outcomes(self, tracker: OutcomeTracker) -> None:
        tracker.record_outcome("q1", ["f_target"], "success", "default")
        tracker.record_outcome("q2", ["f_target"], "success", "default")
        rate = tracker.get_fact_success_rate("f_target", "default")
        assert rate == 1.0

    def test_fact_in_mixed_outcomes(self, tracker: OutcomeTracker) -> None:
        tracker.record_outcome("q1", ["f_mix"], "success", "default")
        tracker.record_outcome("q2", ["f_mix"], "failure", "default")
        rate = tracker.get_fact_success_rate("f_mix", "default")
        assert rate == 0.5

    def test_fact_among_other_facts(self, tracker: OutcomeTracker) -> None:
        tracker.record_outcome("q1", ["f_a", "f_target", "f_b"], "success", "default")
        rate = tracker.get_fact_success_rate("f_target", "default")
        assert rate == 1.0


# ---------------------------------------------------------------------------
# Profile isolation
# ---------------------------------------------------------------------------

class TestProfileIsolation:
    def test_outcomes_separated_by_profile(
        self, tracker: OutcomeTracker, db: DatabaseManager
    ) -> None:
        db.execute(
            "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('work', 'Work')"
        )
        tracker.record_outcome("q1", ["f1"], "success", "default")
        tracker.record_outcome("q2", ["f2"], "failure", "work")

        assert tracker.get_success_rate("default") == 1.0
        assert tracker.get_success_rate("work") == 0.0

    def test_get_outcomes_isolated(
        self, tracker: OutcomeTracker, db: DatabaseManager
    ) -> None:
        db.execute(
            "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('work', 'Work')"
        )
        tracker.record_outcome("qd", ["f"], "success", "default")
        tracker.record_outcome("qw", ["f"], "failure", "work")

        default_outcomes = tracker.get_outcomes("default")
        work_outcomes = tracker.get_outcomes("work")

        assert len(default_outcomes) == 1
        assert default_outcomes[0].query == "qd"
        assert len(work_outcomes) == 1
        assert work_outcomes[0].query == "qw"
