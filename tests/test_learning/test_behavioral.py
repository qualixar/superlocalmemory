# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.learning.behavioral — Behavioral Pattern Detection.

Covers:
  - record_query: creates time_of_day, query_type, entity_pref patterns
  - get_patterns: filter by pattern_type, profile, min_confidence
  - get_entity_preferences: top-K entities by confidence
  - get_active_hours: returns top 5 active hours
  - get_query_type_distribution: proportional distribution
  - _upsert_pattern: observation_count increments, confidence saturates at 100
  - Profile isolation: patterns are profile-scoped
  - Edge cases: empty patterns, no queries, invalid hour parsing
"""

from __future__ import annotations

from pathlib import Path

import pytest

from superlocalmemory.learning.behavioral import BehavioralTracker
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    db_path = tmp_path / "behavioral_test.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize(real_schema)
    return mgr


@pytest.fixture()
def tracker(db: DatabaseManager) -> BehavioralTracker:
    return BehavioralTracker(db)


# ---------------------------------------------------------------------------
# record_query
# ---------------------------------------------------------------------------

class TestRecordQuery:
    def test_creates_time_of_day_pattern(self, tracker: BehavioralTracker) -> None:
        tracker.record_query("What is X?", "factual", [], "default")
        patterns = tracker.get_patterns("time_of_day", "default")
        assert len(patterns) >= 1
        assert all(p.pattern_key.startswith("hour_") for p in patterns)

    def test_creates_query_type_pattern(self, tracker: BehavioralTracker) -> None:
        tracker.record_query("When did X?", "temporal", [], "default")
        patterns = tracker.get_patterns("query_type", "default")
        assert len(patterns) == 1
        assert patterns[0].pattern_key == "temporal"

    def test_creates_entity_pref_patterns(self, tracker: BehavioralTracker) -> None:
        tracker.record_query("Tell about Alice", "factual", ["Alice", "Bob"], "default")
        patterns = tracker.get_patterns("entity_pref", "default")
        keys = {p.pattern_key for p in patterns}
        assert "alice" in keys  # lowercased
        assert "bob" in keys

    def test_limits_entities_to_five(self, tracker: BehavioralTracker) -> None:
        entities = [f"entity_{i}" for i in range(10)]
        tracker.record_query("q", "factual", entities, "default")
        patterns = tracker.get_patterns("entity_pref", "default")
        assert len(patterns) == 5

    def test_empty_query_type_skipped(self, tracker: BehavioralTracker) -> None:
        tracker.record_query("q", "", [], "default")
        patterns = tracker.get_patterns("query_type", "default")
        assert len(patterns) == 0

    def test_observation_count_increments(self, tracker: BehavioralTracker) -> None:
        for _ in range(5):
            tracker.record_query("q", "factual", ["alice"], "default")
        patterns = tracker.get_patterns("entity_pref", "default")
        alice = [p for p in patterns if p.pattern_key == "alice"]
        assert len(alice) == 1
        assert alice[0].observation_count == 5

    def test_confidence_saturates_at_100(self, tracker: BehavioralTracker) -> None:
        for _ in range(150):
            tracker.record_query("q", "factual", ["alice"], "default")
        patterns = tracker.get_patterns("entity_pref", "default")
        alice = [p for p in patterns if p.pattern_key == "alice"]
        assert alice[0].confidence == 1.0


# ---------------------------------------------------------------------------
# get_patterns
# ---------------------------------------------------------------------------

class TestGetPatterns:
    def test_filter_by_type(self, tracker: BehavioralTracker) -> None:
        tracker.record_query("q1", "factual", ["alice"], "default")
        tracker.record_query("q2", "temporal", ["bob"], "default")

        entity_pats = tracker.get_patterns("entity_pref", "default")
        type_pats = tracker.get_patterns("query_type", "default")
        assert len(entity_pats) == 2  # alice, bob
        assert len(type_pats) == 2    # factual, temporal

    def test_min_confidence_filter(self, tracker: BehavioralTracker) -> None:
        # Record once -> confidence = 0.01
        tracker.record_query("q", "factual", ["alice"], "default")
        # Record many times -> confidence approaches 1.0
        for _ in range(99):
            tracker.record_query("q", "factual", ["bob"], "default")

        high_conf = tracker.get_patterns("entity_pref", "default", min_confidence=0.5)
        # Only bob should pass (100 observations / 100 = 1.0)
        assert all(p.pattern_key == "bob" for p in high_conf)

    def test_empty_for_unknown_type(self, tracker: BehavioralTracker) -> None:
        assert tracker.get_patterns("nonexistent_type", "default") == []


# ---------------------------------------------------------------------------
# get_entity_preferences
# ---------------------------------------------------------------------------

class TestGetEntityPreferences:
    def test_returns_top_k_entities(self, tracker: BehavioralTracker) -> None:
        for _ in range(10):
            tracker.record_query("q", "factual", ["alice"], "default")
        for _ in range(5):
            tracker.record_query("q", "factual", ["bob"], "default")
        for _ in range(1):
            tracker.record_query("q", "factual", ["charlie"], "default")

        prefs = tracker.get_entity_preferences("default", top_k=2)
        assert len(prefs) == 2
        assert prefs[0] == "alice"  # highest confidence

    def test_empty_when_no_queries(self, tracker: BehavioralTracker) -> None:
        assert tracker.get_entity_preferences("default") == []


# ---------------------------------------------------------------------------
# get_active_hours
# ---------------------------------------------------------------------------

class TestGetActiveHours:
    def test_returns_hours(self, tracker: BehavioralTracker) -> None:
        # Record queries at fixed hours
        tracker.record_query("q", "", [], "default")  # current hour
        hours = tracker.get_active_hours("default")
        assert len(hours) >= 1
        assert all(0 <= h <= 23 for h in hours)

    def test_max_five_hours(self, tracker: BehavioralTracker, db: DatabaseManager) -> None:
        # Manually insert 7 different hour patterns
        for h in range(7):
            db.execute(
                "INSERT INTO behavioral_patterns "
                "(pattern_id, profile_id, pattern_type, pattern_key, "
                "pattern_value, confidence, observation_count, last_updated) "
                "VALUES (?, 'default', 'time_of_day', ?, '', 0.5, 10, '2026-01-01')",
                (f"pid_{h}", f"hour_{h}"),
            )
        hours = tracker.get_active_hours("default")
        assert len(hours) <= 5


# ---------------------------------------------------------------------------
# get_query_type_distribution
# ---------------------------------------------------------------------------

class TestGetQueryTypeDistribution:
    def test_proportional_distribution(self, tracker: BehavioralTracker) -> None:
        for _ in range(3):
            tracker.record_query("q", "factual", [], "default")
        for _ in range(7):
            tracker.record_query("q", "temporal", [], "default")

        dist = tracker.get_query_type_distribution("default")
        assert "factual" in dist
        assert "temporal" in dist
        assert dist["factual"] == pytest.approx(0.3, abs=0.01)
        assert dist["temporal"] == pytest.approx(0.7, abs=0.01)

    def test_empty_distribution(self, tracker: BehavioralTracker) -> None:
        dist = tracker.get_query_type_distribution("default")
        assert dist == {}


# ---------------------------------------------------------------------------
# Profile isolation
# ---------------------------------------------------------------------------

class TestProfileIsolation:
    def test_patterns_separated_by_profile(
        self, tracker: BehavioralTracker, db: DatabaseManager
    ) -> None:
        db.execute(
            "INSERT OR IGNORE INTO profiles (profile_id, name) "
            "VALUES ('work', 'Work')"
        )
        tracker.record_query("q", "factual", ["alice"], "default")
        tracker.record_query("q", "temporal", ["bob"], "work")

        default_entities = tracker.get_entity_preferences("default")
        work_entities = tracker.get_entity_preferences("work")

        assert "alice" in default_entities
        assert "alice" not in work_entities
        assert "bob" in work_entities
        assert "bob" not in default_entities
