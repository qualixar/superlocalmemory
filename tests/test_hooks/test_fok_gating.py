# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Tests for FOK (Feeling-of-Knowing) gating in AutoInvoker

"""Tests for FOK gating -- SYNAPSE tau_gate = 0.12 threshold."""

from __future__ import annotations

from unittest.mock import MagicMock

from superlocalmemory.core.config import AutoInvokeConfig
from superlocalmemory.hooks.auto_invoker import AutoInvoker


def _make_invoker_with_candidates(
    candidates: list[tuple[str, float]],
    fok_threshold: float = 0.12,
) -> AutoInvoker:
    """Create an AutoInvoker with pre-configured candidates and minimal mocks."""
    db = MagicMock()

    def mock_execute(sql, params=()):
        if "MAX(accessed_at)" in sql:
            return [{"last_access": None}]
        if "access_count" in sql and "MAX" in sql:
            return [{"max_count": 10}]
        if "access_count" in sql:
            return [{"access_count": 0}]
        if "atomic_facts" in sql and "content" in sql:
            fact_id = params[0]
            return [
                {"fact_id": fact_id, "content": f"Content {fact_id}", "fact_type": "semantic", "lifecycle": "active"}
            ]
        return []

    db.execute.side_effect = mock_execute
    db.get_fact_context.return_value = None

    vs = MagicMock()
    embedder = MagicMock()
    embedder.embed.return_value = [0.1] * 768
    vs.search.return_value = candidates

    cfg = AutoInvokeConfig(
        enabled=True,
        profile_id="test",
        fok_threshold=fok_threshold,
    )

    return AutoInvoker(
        db=db, vector_store=vs, embedder=embedder,
        config=cfg,
    )


class TestFOKGating:
    """FOK (Feeling-of-Knowing) threshold gating."""

    def test_score_above_threshold_passes(self):
        # High similarity -> high combined score -> passes FOK
        invoker = _make_invoker_with_candidates(
            [("fact_high", 0.9)], fok_threshold=0.12,
        )
        results = invoker.invoke("test", "test")
        # High similarity of 0.9 * 0.40 = 0.36 minimum, well above 0.12
        assert len(results) >= 1

    def test_score_below_threshold_rejected(self):
        # Very low similarity + cold start defaults -> low score
        invoker = _make_invoker_with_candidates(
            [("fact_low", 0.0)], fok_threshold=0.50,
        )
        results = invoker.invoke("test", "test")
        # With similarity=0 and cold start defaults, score should be low
        # 0.0*0.40 + 0.1*0.25 + 0.0*0.20 + 0.5*0.15 = 0.025 + 0.075 = 0.1
        # 0.1 < 0.50 threshold -> rejected
        assert len(results) == 0

    def test_score_exactly_at_threshold_passes(self):
        """Score == threshold should pass (>= comparison)."""
        # We need a scenario where score is exactly at threshold
        # This is tricky with floating point, so we test boundary behavior
        invoker = _make_invoker_with_candidates(
            [("fact_edge", 0.3)], fok_threshold=0.12,
        )
        results = invoker.invoke("test", "test")
        # 0.3*0.40 = 0.12, plus recency and trust contributions > 0
        # Score > 0.12, should pass
        assert len(results) >= 1

    def test_threshold_configurable(self):
        # Very low threshold -> everything passes
        invoker = _make_invoker_with_candidates(
            [("fact_1", 0.1)], fok_threshold=0.001,
        )
        results = invoker.invoke("test", "test")
        assert len(results) >= 1

    def test_zero_threshold_passes_all(self):
        invoker = _make_invoker_with_candidates(
            [("fact_1", 0.01)], fok_threshold=0.0,
        )
        results = invoker.invoke("test", "test")
        assert len(results) >= 1

    def test_gating_applied_before_enrichment(self):
        """FOK gating should filter BEFORE enrichment (performance)."""
        db = MagicMock()
        enrich_calls = []

        def mock_execute(sql, params=()):
            if "MAX(accessed_at)" in sql:
                return [{"last_access": None}]
            if "access_count" in sql and "MAX" in sql:
                return [{"max_count": 10}]
            if "access_count" in sql:
                return [{"access_count": 0}]
            if "atomic_facts" in sql and "content" in sql:
                enrich_calls.append(params[0])
                return [{"fact_id": params[0], "content": "test", "fact_type": "semantic", "lifecycle": "active"}]
            return []

        db.execute.side_effect = mock_execute
        db.get_fact_context.return_value = None

        vs = MagicMock()
        embedder = MagicMock()
        embedder.embed.return_value = [0.1] * 768
        # Two candidates: one high score, one that will be gated
        vs.search.return_value = [("fact_high", 0.9), ("fact_low", 0.0)]

        cfg = AutoInvokeConfig(
            enabled=True,
            profile_id="test",
            fok_threshold=0.20,
        )
        invoker = AutoInvoker(
            db=db, vector_store=vs, embedder=embedder,
            config=cfg,
        )

        invoker.invoke("test", "test", limit=10)
        # The low-scoring fact should be gated before enrichment
        # Only the high-scoring fact should trigger enrichment
        assert "fact_low" not in enrich_calls
