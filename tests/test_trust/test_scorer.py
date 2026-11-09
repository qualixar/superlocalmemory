# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.trust.scorer — Bayesian Trust Scoring.

Covers:
  - get_trust: returns default (0.5) when no score exists
  - update_on_confirmation: trust increases toward 1.0
  - update_on_contradiction: trust decreases toward 0.0
  - update_on_access: small trust boost per access
  - get_entity_trust / get_fact_trust: convenience wrappers
  - get_all_scores: returns all TrustScore objects for a profile
  - Bayesian bounds: trust never exceeds 1.0 or drops below 0.0
  - Profile isolation: scores are scoped to profile_id
  - Multiple updates: repeated confirmations accumulate
  - New target auto-creation via _get_or_create
"""

from __future__ import annotations

from pathlib import Path

import pytest

from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.trust.scorer import TrustScorer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    """DatabaseManager with schema applied."""
    db_path = tmp_path / "trust_test.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize(real_schema)
    return mgr


@pytest.fixture()
def scorer(db: DatabaseManager) -> TrustScorer:
    return TrustScorer(db)


# ---------------------------------------------------------------------------
# get_trust — default behaviour
# ---------------------------------------------------------------------------

class TestGetTrust:
    def test_returns_default_for_unknown_target(self, scorer: TrustScorer) -> None:
        score = scorer.get_trust("entity", "nonexistent", "default")
        assert score == 0.5

    def test_returns_stored_score(self, scorer: TrustScorer) -> None:
        scorer.update_on_confirmation("entity", "e1", "default")
        score = scorer.get_trust("entity", "e1", "default")
        assert score > 0.5  # confirmation boosts above default


# ---------------------------------------------------------------------------
# Confirmation updates
# ---------------------------------------------------------------------------

class TestConfirmation:
    def test_single_confirmation_increases_trust(self, scorer: TrustScorer) -> None:
        new = scorer.update_on_confirmation("entity", "e1", "default")
        assert new > 0.5
        assert new <= 1.0

    def test_repeated_confirmations_approach_one(self, scorer: TrustScorer) -> None:
        for _ in range(100):
            score = scorer.update_on_confirmation("entity", "e_reps", "default")
        assert score > 0.95
        assert score <= 1.0

    def test_confirmation_never_exceeds_one(self, scorer: TrustScorer) -> None:
        for _ in range(500):
            score = scorer.update_on_confirmation("fact", "f_max", "default")
        assert score <= 1.0


# ---------------------------------------------------------------------------
# Contradiction updates
# ---------------------------------------------------------------------------

class TestContradiction:
    def test_single_contradiction_decreases_trust(self, scorer: TrustScorer) -> None:
        new = scorer.update_on_contradiction("entity", "e2", "default")
        assert new < 0.5

    def test_repeated_contradictions_approach_zero(self, scorer: TrustScorer) -> None:
        for _ in range(100):
            score = scorer.update_on_contradiction("entity", "e_bad", "default")
        assert score < 0.05

    def test_contradiction_never_below_zero(self, scorer: TrustScorer) -> None:
        for _ in range(500):
            score = scorer.update_on_contradiction("source", "s_floor", "default")
        assert score >= 0.0


# ---------------------------------------------------------------------------
# Access updates
# ---------------------------------------------------------------------------

class TestAccess:
    def test_access_boost_is_small(self, scorer: TrustScorer) -> None:
        new = scorer.update_on_access("entity", "e3", "default")
        # Beta(1,1) + access adds 0.5 to alpha => Beta(1.5, 1) => trust = 1.5/2.5 = 0.6
        assert new == pytest.approx(0.6, abs=0.001)

    def test_access_never_exceeds_one(self, scorer: TrustScorer) -> None:
        for _ in range(200):
            score = scorer.update_on_access("entity", "e_acc_max", "default")
        assert score <= 1.0


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

class TestConvenience:
    def test_get_entity_trust(self, scorer: TrustScorer) -> None:
        assert scorer.get_entity_trust("eid_1", "default") == 0.5

    def test_get_fact_trust(self, scorer: TrustScorer) -> None:
        assert scorer.get_fact_trust("fid_1", "default") == 0.5


# ---------------------------------------------------------------------------
# get_all_scores
# ---------------------------------------------------------------------------

class TestGetAllScores:
    def test_empty_profile_returns_empty_list(self, scorer: TrustScorer) -> None:
        assert scorer.get_all_scores("default") == []

    def test_returns_all_scores_for_profile(self, scorer: TrustScorer) -> None:
        scorer.update_on_confirmation("entity", "e1", "default")
        scorer.update_on_confirmation("fact", "f1", "default")
        scores = scorer.get_all_scores("default")
        assert len(scores) == 2
        assert all(isinstance(s, dict) for s in scores)

    def test_profile_isolation(self, scorer: TrustScorer, db: DatabaseManager) -> None:
        db.execute(
            "INSERT OR IGNORE INTO profiles (profile_id, name) "
            "VALUES ('work', 'Work')"
        )
        scorer.update_on_confirmation("entity", "e1", "default")
        scorer.update_on_confirmation("entity", "e2", "work")

        default_scores = scorer.get_all_scores("default")
        work_scores = scorer.get_all_scores("work")

        assert len(default_scores) == 1
        assert len(work_scores) == 1
        assert default_scores[0]["target_id"] == "e1"
        assert work_scores[0]["target_id"] == "e2"


# ---------------------------------------------------------------------------
# Mixed operations
# ---------------------------------------------------------------------------

class TestMixedOperations:
    def test_confirm_then_contradict(self, scorer: TrustScorer) -> None:
        scorer.update_on_confirmation("fact", "f_mix", "default")
        after_confirm = scorer.get_trust("fact", "f_mix", "default")

        scorer.update_on_contradiction("fact", "f_mix", "default")
        after_contradict = scorer.get_trust("fact", "f_mix", "default")

        assert after_contradict < after_confirm

    def test_evidence_count_increments(self, scorer: TrustScorer) -> None:
        scorer.update_on_confirmation("entity", "e_ev", "default")
        scorer.update_on_confirmation("entity", "e_ev", "default")
        scorer.update_on_contradiction("entity", "e_ev", "default")

        scores = scorer.get_all_scores("default")
        match = [s for s in scores if s["target_id"] == "e_ev"]
        assert len(match) == 1
        # 2 confirms (alpha +2) + 1 contradiction (beta +3):
        # alpha=3, beta=4 => evidence = round(3+4-2) = 5
        assert match[0]["evidence_count"] == 5

    def test_different_target_types_independent(self, scorer: TrustScorer) -> None:
        scorer.update_on_confirmation("entity", "x", "default")
        scorer.update_on_contradiction("fact", "x", "default")

        entity_score = scorer.get_trust("entity", "x", "default")
        fact_score = scorer.get_trust("fact", "x", "default")

        assert entity_score > 0.5
        assert fact_score < 0.5
