# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.retrieval.semantic_channel — Fisher-aware semantic search.

Covers:
  - Cosine similarity fallback (no fisher_variance)
  - Fisher-Rao similarity (with fisher_variance)
  - Empty query embedding
  - Mismatched dimensions skipped
  - Facts without embeddings skipped
  - Top-k limiting
  - Score ordering (descending)
  - Zero-vector handling
  - Module-level helper functions
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from superlocalmemory.retrieval.semantic_channel import (
    SemanticChannel,
    _cosine_similarity,
    _fisher_rao_similarity,
)
from superlocalmemory.storage.models import AtomicFact


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fact(
    fact_id: str,
    embedding: list[float] | None = None,
    fisher_variance: list[float] | None = None,
) -> AtomicFact:
    return AtomicFact(
        fact_id=fact_id,
        memory_id="m0",
        content=f"fact {fact_id}",
        embedding=embedding,
        fisher_variance=fisher_variance,
    )


def _mock_db(facts: list[AtomicFact]) -> MagicMock:
    db = MagicMock()
    db.get_all_facts.return_value = facts
    return db


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        assert _cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-5)

    def test_opposite_vectors(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert _cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-5)

    def test_zero_vector_returns_zero(self) -> None:
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 2.0])
        assert _cosine_similarity(a, b) == 0.0

    def test_both_zero_returns_zero(self) -> None:
        a = np.array([0.0, 0.0])
        assert _cosine_similarity(a, a) == 0.0


# ---------------------------------------------------------------------------
# _fisher_rao_similarity
# ---------------------------------------------------------------------------

class TestFisherRaoSimilarity:
    def test_identical_means_returns_one(self) -> None:
        mu = np.array([1.0, 2.0, 3.0])
        var = np.array([1.0, 1.0, 1.0])
        sim = _fisher_rao_similarity(mu, mu, var, temperature=15.0)
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_different_means_lower_score(self) -> None:
        mu_q = np.array([1.0, 0.0])
        mu_f = np.array([0.0, 1.0])
        var = np.array([1.0, 1.0])
        sim = _fisher_rao_similarity(mu_q, mu_f, var, temperature=15.0)
        assert 0.0 < sim < 1.0

    def test_low_variance_increases_distance(self) -> None:
        mu_q = np.array([1.0, 0.0])
        mu_f = np.array([0.5, 0.0])
        var_high = np.array([10.0, 10.0])
        var_low = np.array([0.01, 0.01])
        sim_high_var = _fisher_rao_similarity(mu_q, mu_f, var_high, temperature=15.0)
        sim_low_var = _fisher_rao_similarity(mu_q, mu_f, var_low, temperature=15.0)
        # Lower variance -> more confident -> bigger distance for same delta
        assert sim_high_var > sim_low_var

    def test_temperature_effect(self) -> None:
        mu_q = np.array([1.0, 0.0])
        mu_f = np.array([0.0, 1.0])
        var = np.array([1.0, 1.0])
        sim_low_t = _fisher_rao_similarity(mu_q, mu_f, var, temperature=1.0)
        sim_high_t = _fisher_rao_similarity(mu_q, mu_f, var, temperature=100.0)
        assert sim_high_t > sim_low_t  # Higher temp = smoother = higher sim

    def test_variance_floor_prevents_division_by_zero(self) -> None:
        mu_q = np.array([1.0, 0.0])
        mu_f = np.array([0.0, 1.0])
        var = np.array([0.0, 0.0])  # Zero variance
        sim = _fisher_rao_similarity(mu_q, mu_f, var, temperature=15.0)
        assert math.isfinite(sim)


# ---------------------------------------------------------------------------
# SemanticChannel.search
# ---------------------------------------------------------------------------

class TestSemanticChannelSearch:
    def test_empty_query_returns_empty(self) -> None:
        db = _mock_db([])
        channel = SemanticChannel(db)
        results = channel.search([], "default")
        assert results == []

    def test_no_facts_returns_empty(self) -> None:
        db = _mock_db([])
        channel = SemanticChannel(db)
        results = channel.search([1.0, 0.0, 0.0], "default")
        assert results == []

    def test_fact_without_embedding_skipped(self) -> None:
        facts = [_make_fact("f1", embedding=None)]
        db = _mock_db(facts)
        channel = SemanticChannel(db)
        results = channel.search([1.0, 0.0], "default")
        assert results == []

    def test_dimension_mismatch_skipped(self) -> None:
        facts = [_make_fact("f1", embedding=[1.0, 0.0, 0.0])]
        db = _mock_db(facts)
        channel = SemanticChannel(db)
        results = channel.search([1.0, 0.0], "default")  # 2-dim vs 3-dim
        assert results == []

    def test_cosine_fallback_when_no_variance(self) -> None:
        facts = [
            _make_fact("f1", embedding=[1.0, 0.0]),
            _make_fact("f2", embedding=[0.0, 1.0]),
        ]
        db = _mock_db(facts)
        channel = SemanticChannel(db)
        results = channel.search([1.0, 0.0], "default", top_k=10)
        assert len(results) == 2
        assert results[0][0] == "f1"  # Cosine: (1+1)/2 = 1.0
        assert results[0][1] > results[1][1]

    def test_fisher_rao_used_when_variance_present(self) -> None:
        facts = [
            _make_fact(
                "f1",
                embedding=[1.0, 0.0],
                fisher_variance=[0.1, 0.1],
            ),
        ]
        db = _mock_db(facts)
        channel = SemanticChannel(db, fisher_temperature=15.0)
        results = channel.search([1.0, 0.0], "default")
        assert len(results) == 1
        assert results[0][0] == "f1"
        # Fisher-Rao with identical means -> exp(0) = 1.0
        assert results[0][1] == pytest.approx(1.0, abs=1e-3)

    def test_top_k_limits_results(self) -> None:
        facts = [
            _make_fact(f"f{i}", embedding=[float(i), 0.0]) for i in range(1, 20)
        ]
        db = _mock_db(facts)
        channel = SemanticChannel(db)
        results = channel.search([1.0, 0.0], "default", top_k=5)
        assert len(results) <= 5

    def test_results_sorted_descending(self) -> None:
        facts = [
            _make_fact("close", embedding=[0.9, 0.1]),
            _make_fact("far", embedding=[0.1, 0.9]),
        ]
        db = _mock_db(facts)
        channel = SemanticChannel(db)
        results = channel.search([1.0, 0.0], "default")
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_profile_passed_to_db(self) -> None:
        db = _mock_db([])
        channel = SemanticChannel(db)
        channel.search([1.0], "work_profile")
        # multi-scope: profile is still threaded; scope flags default to include-all
        db.get_all_facts.assert_called_once_with(
            "work_profile", include_global=True, include_shared=True
        )

    def test_fisher_variance_wrong_length_falls_back(self) -> None:
        facts = [
            _make_fact(
                "f1",
                embedding=[1.0, 0.0],
                fisher_variance=[0.1],  # 1-dim vs 2-dim embedding
            ),
        ]
        db = _mock_db(facts)
        channel = SemanticChannel(db)
        results = channel.search([1.0, 0.0], "default")
        # Should fall back to cosine, not crash
        assert len(results) == 1
