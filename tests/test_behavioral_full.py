"""Tests for Behavioral System — Task 13 of V3 build."""
import pytest
from pathlib import Path
from superlocalmemory.learning.behavioral import BehavioralPatternStore
from superlocalmemory.learning.outcomes import OutcomeTracker


@pytest.fixture
def pattern_store(tmp_path):
    return BehavioralPatternStore(tmp_path / "patterns.db")

@pytest.fixture
def outcomes(tmp_path):
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage import schema as real_schema
    db = DatabaseManager(tmp_path / "outcomes.db")
    db.initialize(real_schema)
    return OutcomeTracker(db)


# -- Pattern Store --
def test_record_and_get_patterns(pattern_store):
    pattern_store.record_pattern("default", "refinement", {"topic": "auth"})
    patterns = pattern_store.get_patterns("default")
    assert len(patterns) >= 1

def test_pattern_summary(pattern_store):
    pattern_store.record_pattern("default", "refinement", {})
    pattern_store.record_pattern("default", "refinement", {})
    pattern_store.record_pattern("default", "interest", {})
    summary = pattern_store.get_summary("default")
    assert summary.get("refinement", 0) >= 1
    assert summary.get("interest", 0) >= 1

def test_transfer_patterns(pattern_store):
    pattern_store.record_pattern("default", "refinement", {"topic": "auth"})
    count = pattern_store.transfer_patterns("default", "p2")
    assert count >= 1
    p2_patterns = pattern_store.get_patterns("p2")
    assert len(p2_patterns) >= 1

# -- Outcomes --
def test_record_outcome(outcomes):
    outcomes.record_outcome("q1", ["f1", "f2"], "success", "default")
    results = outcomes.get_outcomes("default")
    assert len(results) >= 1

def test_success_rate(outcomes):
    outcomes.record_outcome("q1", ["f1"], "success", "default")
    outcomes.record_outcome("q2", ["f2"], "success", "default")
    outcomes.record_outcome("q3", ["f3"], "failure", "default")
    rate = outcomes.get_success_rate("default")
    assert 0.5 < rate < 0.8  # 2/3 = 0.667

def test_success_rate_empty(outcomes):
    rate = outcomes.get_success_rate("default")
    assert rate == 0.0
