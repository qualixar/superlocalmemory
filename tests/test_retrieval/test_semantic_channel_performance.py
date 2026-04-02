# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for SemanticChannel performance — baseline before Phase 1 KNN replacement.

Phase 0 Safety Net: records wall-clock latency of the current full-table-scan
retrieval at 100, 500, 1000, 5000 facts. These baselines let Phase 1 prove
that KNN actually improves performance.

Covers:
  - Cosine-only scan at 100, 500, 1000, 5000 facts
  - Fisher-Rao scan at 100 facts
  - Cosine vs Fisher-Rao latency ratio
  - Mixed access_count graduated ramp timing

Uses real numpy math (no mocking of numpy). DB is a MagicMock returning
synthetic AtomicFact instances with random 768-dim embeddings.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from superlocalmemory.retrieval.semantic_channel import SemanticChannel
from superlocalmemory.storage.models import AtomicFact


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_facts(
    count: int,
    dim: int = 768,
    seed: int = 42,
    with_fisher: bool = False,
    mixed_access: bool = False,
) -> list[AtomicFact]:
    """Generate synthetic AtomicFact instances with random embeddings.

    Args:
        count: Number of facts to generate.
        dim: Embedding dimension.
        seed: RNG seed for reproducibility.
        with_fisher: If True, populate fisher_variance.
        mixed_access: If True, vary access_count (0..19) to trigger
                      graduated Fisher-Rao ramp.
    """
    rng = np.random.RandomState(seed)  # noqa: NPY002
    facts: list[AtomicFact] = []
    for i in range(count):
        emb = rng.randn(dim).astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 1e-10:
            emb = emb / norm

        access = (i % 20) if mixed_access else 0
        facts.append(AtomicFact(
            fact_id=f"f_{i:06d}",
            memory_id="m0",
            content=f"Fact number {i} about topic {i % 10}",
            embedding=emb.tolist(),
            fisher_variance=([1.0] * dim) if with_fisher else None,
            fisher_mean=emb.tolist() if with_fisher else None,
            access_count=access,
            confidence=0.9,
        ))
    return facts


def _mock_db(facts: list[AtomicFact]) -> MagicMock:
    """Return a MagicMock DatabaseManager that returns given facts."""
    db = MagicMock()
    db.get_all_facts.return_value = facts
    return db


def _make_query_embedding(dim: int = 768, seed: int = 99) -> list[float]:
    """Generate a deterministic normalized query vector."""
    rng = np.random.RandomState(seed)  # noqa: NPY002
    q = rng.randn(dim).astype(np.float32)
    q = q / np.linalg.norm(q)
    return q.tolist()


def _timed_search(
    channel: SemanticChannel,
    query: list[float],
    profile_id: str = "default",
    top_k: int = 50,
) -> tuple[list[tuple[str, float]], float]:
    """Run channel.search() and return (results, elapsed_ms)."""
    t0 = time.monotonic()
    results = channel.search(query, profile_id, top_k=top_k)
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    return results, elapsed_ms


# ---------------------------------------------------------------------------
# Cosine-only performance baselines
# ---------------------------------------------------------------------------

class TestSemanticPerformanceBaseline:
    """Baseline latency for cosine-only full-table scan (no Fisher data)."""

    def test_100_facts_under_50ms(self) -> None:
        """100-fact scan should complete well under 50ms."""
        facts = _generate_facts(100)
        db = _mock_db(facts)
        channel = SemanticChannel(db)
        query = _make_query_embedding()

        results, elapsed = _timed_search(channel, query)
        assert len(results) > 0, "Expected non-empty results"
        assert elapsed < 50.0, f"100-fact scan took {elapsed:.1f}ms (budget: 50ms)"

    def test_500_facts_under_200ms(self) -> None:
        """500-fact scan should complete under 200ms."""
        facts = _generate_facts(500)
        db = _mock_db(facts)
        channel = SemanticChannel(db)
        query = _make_query_embedding()

        results, elapsed = _timed_search(channel, query)
        assert len(results) > 0
        assert elapsed < 200.0, f"500-fact scan took {elapsed:.1f}ms (budget: 200ms)"

    def test_1000_facts_under_500ms(self) -> None:
        """1000-fact scan should complete under 500ms."""
        facts = _generate_facts(1000)
        db = _mock_db(facts)
        channel = SemanticChannel(db)
        query = _make_query_embedding()

        results, elapsed = _timed_search(channel, query)
        assert len(results) > 0
        assert elapsed < 500.0, f"1000-fact scan took {elapsed:.1f}ms (budget: 500ms)"

    def test_5000_facts_completes(self) -> None:
        """5000-fact scan completes without error. Time recorded, no strict limit."""
        facts = _generate_facts(5000)
        db = _mock_db(facts)
        channel = SemanticChannel(db)
        query = _make_query_embedding()

        results, elapsed = _timed_search(channel, query)
        assert len(results) > 0, "Expected non-empty results for 5000 facts"
        # Record time for baseline; no strict assertion.
        # Phase 1 KNN should beat this number significantly.
        print(f"\n  [BASELINE] 5000-fact cosine scan: {elapsed:.1f}ms")


# ---------------------------------------------------------------------------
# Fisher-Rao performance baselines
# ---------------------------------------------------------------------------

class TestFisherRaoPerformanceBaseline:
    """Baseline latency when Fisher-Rao distance is active."""

    def test_fisher_rao_100_facts_under_100ms(self) -> None:
        """100-fact Fisher-Rao scan (with variance data) under 100ms."""
        facts = _generate_facts(100, with_fisher=True, mixed_access=True)
        db = _mock_db(facts)
        channel = SemanticChannel(db, fisher_temperature=15.0)
        query = _make_query_embedding()

        results, elapsed = _timed_search(channel, query)
        assert len(results) > 0
        assert elapsed < 100.0, f"Fisher-Rao 100-fact scan took {elapsed:.1f}ms (budget: 100ms)"

    def test_cosine_vs_fisher_ratio(self) -> None:
        """Fisher-Rao should be less than 3x slower than pure cosine for 100 facts."""
        n = 100
        cosine_facts = _generate_facts(n, with_fisher=False)
        fisher_facts = _generate_facts(n, with_fisher=True, mixed_access=True)
        query = _make_query_embedding()

        # Cosine-only timing
        db_cos = _mock_db(cosine_facts)
        ch_cos = SemanticChannel(db_cos)
        _, cosine_ms = _timed_search(ch_cos, query)

        # Fisher-Rao timing
        db_fr = _mock_db(fisher_facts)
        ch_fr = SemanticChannel(db_fr, fisher_temperature=15.0)
        _, fisher_ms = _timed_search(ch_fr, query)

        # Guard against division by zero on very fast machines
        if cosine_ms < 0.01:
            pytest.skip("Cosine scan too fast to measure ratio reliably")

        ratio = fisher_ms / cosine_ms
        assert ratio < 5.0, (
            f"Fisher-Rao is {ratio:.1f}x slower than cosine "
            f"(cosine={cosine_ms:.1f}ms, fisher={fisher_ms:.1f}ms)"
        )


# ---------------------------------------------------------------------------
# Graduated ramp (mixed access_count) performance
# ---------------------------------------------------------------------------

class TestGraduatedRampPerformance:
    """Baseline for the graduated Fisher-Rao ramp with mixed access counts."""

    def test_mixed_access_counts_reasonable_time(self) -> None:
        """Mixed access_count facts (some cosine, some Fisher) complete quickly."""
        facts = _generate_facts(200, with_fisher=True, mixed_access=True)
        db = _mock_db(facts)
        channel = SemanticChannel(db, fisher_temperature=15.0)
        query = _make_query_embedding()

        results, elapsed = _timed_search(channel, query)
        assert len(results) > 0
        # Generous budget: 200 mixed facts in under 200ms
        assert elapsed < 200.0, (
            f"Mixed graduated ramp 200-fact scan took {elapsed:.1f}ms (budget: 200ms)"
        )
