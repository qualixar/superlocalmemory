# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.retrieval.fusion — Weighted Reciprocal Rank Fusion.

Covers:
  - Single channel, multiple channels, empty channels
  - Weight application, rank ordering
  - k parameter validation
  - max_rank_penalty for missing documents
  - FusionResult structure (channel_ranks, channel_scores)
  - Deduplication across channels
"""

from __future__ import annotations

import pytest

from superlocalmemory.retrieval.fusion import FusionResult, weighted_rrf

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ids(results: list[FusionResult]) -> list[str]:
    """Extract fact_ids in order from fusion results."""
    return [r.fact_id for r in results]


# ---------------------------------------------------------------------------
# Basic fusion
# ---------------------------------------------------------------------------

class TestWeightedRRFBasic:
    def test_single_channel(self) -> None:
        channels = {"sem": [("f1", 0.9), ("f2", 0.7), ("f3", 0.5)]}
        weights = {"sem": 1.0}
        results = weighted_rrf(channels, weights, k=20)
        assert _ids(results) == ["f1", "f2", "f3"]

    def test_empty_channels(self) -> None:
        results = weighted_rrf({}, {}, k=20)
        assert results == []

    def test_single_channel_empty_list(self) -> None:
        channels = {"sem": []}
        results = weighted_rrf(channels, {"sem": 1.0}, k=20)
        assert results == []

    def test_two_channels_same_docs(self) -> None:
        channels = {
            "sem": [("f1", 0.9), ("f2", 0.5)],
            "bm25": [("f2", 0.8), ("f1", 0.3)],
        }
        weights = {"sem": 1.0, "bm25": 1.0}
        results = weighted_rrf(channels, weights, k=20)
        # Both docs appear in both channels — should be 2 results
        assert len(results) == 2
        # f1 has rank 1 in sem, rank 2 in bm25 -> RRF = 1/(20+1) + 1/(20+2)
        # f2 has rank 2 in sem, rank 1 in bm25 -> RRF = 1/(20+2) + 1/(20+1)
        # They should be equal, order may vary
        assert set(_ids(results)) == {"f1", "f2"}

    def test_two_channels_disjoint_docs(self) -> None:
        channels = {
            "sem": [("f1", 0.9)],
            "bm25": [("f2", 0.8)],
        }
        weights = {"sem": 1.0, "bm25": 1.0}
        results = weighted_rrf(channels, weights, k=20)
        assert len(results) == 2
        # f1 is in sem only, f2 in bm25 only
        # f1: sem rank=1 -> 1/(20+1)=0.0476, bm25 rank=1000 -> 1/(20+1000)≈0.001
        # f2: bm25 rank=1 -> 0.0476, sem rank=1000 -> ≈0.001
        # Both should have similar scores
        ids = _ids(results)
        assert set(ids) == {"f1", "f2"}


# ---------------------------------------------------------------------------
# Weight influence
# ---------------------------------------------------------------------------

class TestWeightInfluence:
    def test_higher_weight_boosts_channel(self) -> None:
        channels = {
            "sem": [("f_sem", 0.9)],
            "bm25": [("f_bm", 0.9)],
        }
        # Give semantic 10x the weight
        weights = {"sem": 10.0, "bm25": 1.0}
        results = weighted_rrf(channels, weights, k=20)
        # f_sem should be ranked first due to higher weight
        assert results[0].fact_id == "f_sem"

    def test_zero_weight_still_counted(self) -> None:
        channels = {"sem": [("f1", 0.9)]}
        weights = {"sem": 0.0}
        results = weighted_rrf(channels, weights, k=20)
        # Weight 0 gives 0 contribution
        assert len(results) == 1
        assert results[0].fused_score == 0.0

    def test_missing_weight_defaults_to_one(self) -> None:
        channels = {"sem": [("f1", 0.9)]}
        weights = {}  # No weight for "sem"
        results = weighted_rrf(channels, weights, k=20)
        assert len(results) == 1
        assert results[0].fused_score == pytest.approx(1.0 / (20 + 1))


# ---------------------------------------------------------------------------
# k parameter
# ---------------------------------------------------------------------------

class TestKParameter:
    def test_k_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            weighted_rrf({}, {}, k=0)

    def test_k_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            weighted_rrf({}, {}, k=-5)

    def test_small_k_increases_discrimination(self) -> None:
        channels = {"sem": [("f1", 0.9), ("f2", 0.5)]}
        weights = {"sem": 1.0}
        # Small k -> bigger gap between rank 1 and rank 2
        results_k1 = weighted_rrf(channels, weights, k=1)
        results_k100 = weighted_rrf(channels, weights, k=100)
        gap_k1 = results_k1[0].fused_score - results_k1[1].fused_score
        gap_k100 = results_k100[0].fused_score - results_k100[1].fused_score
        assert gap_k1 > gap_k100


# ---------------------------------------------------------------------------
# FusionResult structure
# ---------------------------------------------------------------------------

class TestFusionResultStructure:
    def test_channel_ranks_populated(self) -> None:
        channels = {
            "sem": [("f1", 0.9), ("f2", 0.5)],
            "bm25": [("f2", 0.8)],
        }
        weights = {"sem": 1.0, "bm25": 1.0}
        results = weighted_rrf(channels, weights, k=20)
        f1_result = next(r for r in results if r.fact_id == "f1")
        assert f1_result.channel_ranks["sem"] == 1
        assert f1_result.channel_ranks["bm25"] == 1000  # max_rank_penalty

    def test_channel_scores_populated(self) -> None:
        channels = {"sem": [("f1", 0.95)]}
        weights = {"sem": 1.0}
        results = weighted_rrf(channels, weights, k=20)
        assert results[0].channel_scores["sem"] == pytest.approx(0.95)

    def test_frozen_dataclass(self) -> None:
        fr = FusionResult(fact_id="x", fused_score=0.5)
        with pytest.raises(AttributeError):
            fr.fact_id = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# max_rank_penalty
# ---------------------------------------------------------------------------

class TestMaxRankPenalty:
    def test_custom_max_rank_penalty(self) -> None:
        channels = {
            "sem": [("f1", 0.9)],
            "bm25": [("f2", 0.8)],
        }
        weights = {"sem": 1.0, "bm25": 1.0}
        results = weighted_rrf(channels, weights, k=20, max_rank_penalty=10)
        # f1 missing from bm25 -> penalty rank = 10
        f1 = next(r for r in results if r.fact_id == "f1")
        assert f1.channel_ranks["bm25"] == 10


# ---------------------------------------------------------------------------
# Sorting stability
# ---------------------------------------------------------------------------

class TestSorting:
    def test_results_sorted_descending(self) -> None:
        channels = {
            "sem": [("f1", 0.9), ("f2", 0.7), ("f3", 0.3)],
        }
        weights = {"sem": 1.0}
        results = weighted_rrf(channels, weights, k=20)
        scores = [r.fused_score for r in results]
        assert scores == sorted(scores, reverse=True)
