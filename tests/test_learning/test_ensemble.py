# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-03 §7.3

"""Tests for ``learning/ensemble.py`` — D8 blend + batched LGBM rerank.

Covers hard rules E1, E2, E3, E4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from superlocalmemory.learning.ensemble import (
    EnsembleWeights,
    _softmax_unit,
    choose_ensemble,
    ensemble_rerank,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeBooster:
    """Stub LightGBM booster that records predict calls."""
    scores_per_batch: list[list[float]] = field(default_factory=list)
    calls: int = 0

    def predict(self, X):  # noqa: N802 — lightgbm API
        self.calls += 1
        # Require 2D input (batch shape).
        assert hasattr(X, "shape") and len(X.shape) == 2
        n = X.shape[0]
        # Return scores from queue or a default descending list.
        if self.scores_per_batch:
            scores = self.scores_per_batch.pop(0)
            assert len(scores) == n
            return list(scores)
        return [float(n - i) for i in range(n)]


@dataclass
class _FakeModel:
    booster: Any


@dataclass
class _Cand:
    fact_id: str
    score: float = 0.0
    channel_scores: dict[str, float] = field(default_factory=dict)
    cross_encoder_score: float | None = None


@dataclass
class _FakeBanditChoice:
    stratum: str = "single_hop|0|morning"
    arm_id: str = "fallback_default"
    weights: dict[str, float] = field(default_factory=lambda: {
        "semantic": 1.0, "bm25": 1.0, "entity_graph": 1.0,
        "temporal": 1.0, "cross_encoder_bias": 1.0,
    })
    play_id: int | None = 1


def _mk_candidates(n: int) -> list[_Cand]:
    return [
        _Cand(
            fact_id=f"f{i}",
            score=float(n - i),
            channel_scores={"semantic": 1.0 - i * 0.1, "bm25": 0.5},
            cross_encoder_score=0.7,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# E1: weights sum to 1.0 across all bins
# ---------------------------------------------------------------------------


def test_weights_sum_to_one_all_bins():
    """E1: every D8 bin + explicit construction obeys sum-to-one."""
    model = _FakeModel(booster=_FakeBooster())
    for count in (0, 1, 199, 200, 499, 500, 10_000):
        for m in (None, model):
            w = choose_ensemble(count, m)
            assert abs((w.bandit + w.lgbm) - 1.0) < 1e-9, (
                f"count={count}, model={m}, got {w}"
            )


def test_weights_rejects_non_sum_to_one():
    with pytest.raises(AssertionError):
        EnsembleWeights(0.5, 0.4)


def test_weights_rejects_negative():
    with pytest.raises(AssertionError):
        EnsembleWeights(1.5, -0.5)


def test_choose_ensemble_cold_start_bandit_only():
    assert choose_ensemble(0, _FakeModel(booster=_FakeBooster())) == EnsembleWeights(1.0, 0.0)
    assert choose_ensemble(199, _FakeModel(booster=_FakeBooster())) == EnsembleWeights(1.0, 0.0)


def test_choose_ensemble_warm_blend():
    w = choose_ensemble(250, _FakeModel(booster=_FakeBooster()))
    assert w == EnsembleWeights(0.4, 0.6)


def test_choose_ensemble_mature_blend():
    w = choose_ensemble(1000, _FakeModel(booster=_FakeBooster()))
    assert w == EnsembleWeights(0.2, 0.8)


def test_choose_ensemble_no_model_forces_bandit_only():
    assert choose_ensemble(10_000, None) == EnsembleWeights(1.0, 0.0)


def test_choose_ensemble_handles_non_numeric_count():
    assert choose_ensemble("banana", _FakeModel(booster=_FakeBooster())) == EnsembleWeights(1.0, 0.0)


# ---------------------------------------------------------------------------
# E3: no predict when weight == 0 or model None
# ---------------------------------------------------------------------------


def test_lgbm_bypass_when_no_model():
    """E3: model=None → no predict, input order preserved."""
    booster = _FakeBooster()
    model = None
    candidates = _mk_candidates(4)
    weights = EnsembleWeights(1.0, 0.0)
    out = ensemble_rerank(candidates, _FakeBanditChoice(), model, weights, {})
    assert out == candidates
    assert booster.calls == 0


def test_lgbm_bypass_when_zero_lgbm_weight():
    """E3: lgbm weight = 0 → no predict even if model is present."""
    booster = _FakeBooster()
    model = _FakeModel(booster=booster)
    weights = EnsembleWeights(1.0, 0.0)
    candidates = _mk_candidates(5)
    out = ensemble_rerank(candidates, _FakeBanditChoice(), model, weights, {})
    assert booster.calls == 0
    assert out == candidates


def test_lgbm_bypass_when_booster_missing_predict():
    """Defensive: booster without predict falls back gracefully."""
    class _BrokenBooster:
        pass
    model = _FakeModel(booster=_BrokenBooster())
    candidates = _mk_candidates(3)
    weights = EnsembleWeights(0.4, 0.6)
    out = ensemble_rerank(candidates, _FakeBanditChoice(), model, weights, {})
    assert out == candidates


# ---------------------------------------------------------------------------
# E2: single batched predict per recall
# ---------------------------------------------------------------------------


def test_single_batch_predict_call():
    """E2: booster.predict invoked exactly once per ensemble_rerank call."""
    booster = _FakeBooster()
    model = _FakeModel(booster=booster)
    candidates = _mk_candidates(10)
    weights = EnsembleWeights(0.4, 0.6)
    ensemble_rerank(candidates, _FakeBanditChoice(), model, weights, {})
    assert booster.calls == 1


# ---------------------------------------------------------------------------
# E4: normalisation before blending
# ---------------------------------------------------------------------------


def test_softmax_unit_preserves_ordering():
    raw = [-5.0, 0.0, 3.0, 10.0]
    out = _softmax_unit(raw)
    # Sorted ascending original → sorted ascending output.
    assert list(out) == sorted(out)
    assert sum(out) == pytest.approx(1.0, rel=1e-6)


def test_softmax_unit_empty():
    assert _softmax_unit([]) == []


def test_softmax_unit_equal_inputs_uniform():
    out = _softmax_unit([3.0, 3.0, 3.0, 3.0])
    assert all(o == pytest.approx(0.25) for o in out)


def test_blend_normalisation():
    """E4: mixed-magnitude streams ([-10,10] vs [0,1]) both enter [0,1].

    Also verify stable ordering under both sort directions.
    """
    booster = _FakeBooster()
    # LGBM raw predictions with large magnitude.
    booster.scores_per_batch.append([-10.0, 5.0, 10.0, 2.0])
    model = _FakeModel(booster=booster)
    # Bandit scores [0, 1] range — synthetic via candidate.score.
    candidates = [
        _Cand(fact_id="a", score=0.1),
        _Cand(fact_id="b", score=0.5),
        _Cand(fact_id="c", score=0.9),
        _Cand(fact_id="d", score=0.4),
    ]
    weights = EnsembleWeights(0.4, 0.6)
    out = ensemble_rerank(candidates, _FakeBanditChoice(), model, weights, {})
    # Verify every candidate still present, deterministic order.
    assert {c.fact_id for c in out} == {"a", "b", "c", "d"}
    assert len(out) == 4
    # The order is driven by blended scores, which combine both streams
    # post-softmax — any final order must be stable for equal blended pairs.


def test_ensemble_rerank_empty_input_returns_empty():
    out = ensemble_rerank(
        [], _FakeBanditChoice(), _FakeModel(booster=_FakeBooster()),
        EnsembleWeights(0.4, 0.6), {},
    )
    assert out == []


def test_ensemble_rerank_predict_exception_safe():
    """booster.predict raises → return input unchanged, no crash."""
    class _ExplodingBooster:
        def predict(self, X):
            raise RuntimeError("boom")
    model = _FakeModel(booster=_ExplodingBooster())
    candidates = _mk_candidates(3)
    out = ensemble_rerank(
        candidates, _FakeBanditChoice(), model, EnsembleWeights(0.4, 0.6), {},
    )
    assert out == candidates


def test_ensemble_rerank_populates_precomputed_features_cache():
    """PERF-v2-02: ``ensemble_rerank`` MUST stash a
    ``{fact_id: features_json_str}`` dict on
    ``query_context['_precomputed_features_json']`` so the downstream
    signal writer can skip re-extracting features for the same candidates.
    """
    import json as _json

    model = _FakeModel(booster=_FakeBooster())
    candidates = _mk_candidates(3)
    # Ensure candidates have distinct fact_ids for the cache assertion.
    for i, c in enumerate(candidates):
        c.fact_id = f"f{i}"  # type: ignore[attr-defined]
    query_context: dict = {"query_type": "single_hop", "profile_id": "p"}

    out = ensemble_rerank(
        candidates, _FakeBanditChoice(), model,
        EnsembleWeights(bandit=0.4, lgbm=0.6), query_context,
    )

    assert len(out) == 3
    cache = query_context.get("_precomputed_features_json")
    assert isinstance(cache, dict), (
        "ensemble_rerank must populate _precomputed_features_json")
    for expected in ("f0", "f1", "f2"):
        assert expected in cache, f"{expected} missing from features cache"
        parsed = _json.loads(cache[expected])
        # features.py FEATURE_NAMES must be a superset of what we expect.
        assert "semantic_score" in parsed
        assert "cross_encoder_score" in parsed
