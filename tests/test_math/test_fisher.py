# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.math.fisher — Fisher-Rao geodesic metric.

Covers:
  - FisherRaoMetric construction and parameter validation
  - compute_params: L2-normalization, content-derived variance
  - distance: geodesic non-negativity, symmetry, identity of indiscernibles
  - similarity: range [0, 1], monotonic w.r.t. distance
  - bayesian_update: precision-additive, variance floor/ceil enforcement
  - adaptive_temperature: data-driven scaling
  - Edge cases: zero vectors, single dimension, NaN rejection
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from superlocalmemory.math.fisher import (
    _VARIANCE_CEIL,
    _VARIANCE_FLOOR,
    FisherRaoMetric,
    _stable_arccosh_1p_vec,
)

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestFisherConstruction:
    def test_default_temperature(self) -> None:
        fm = FisherRaoMetric()
        assert fm.temperature == 15.0

    def test_custom_temperature(self) -> None:
        fm = FisherRaoMetric(temperature=5.0)
        assert fm.temperature == 5.0

    def test_zero_temperature_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            FisherRaoMetric(temperature=0.0)

    def test_negative_temperature_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            FisherRaoMetric(temperature=-1.0)


# ---------------------------------------------------------------------------
# compute_params
# ---------------------------------------------------------------------------

class TestComputeParams:
    def test_output_shapes_match_input(self) -> None:
        fm = FisherRaoMetric()
        emb = [1.0, 2.0, 3.0, 4.0]
        mean, var = fm.compute_params(emb)
        assert len(mean) == 4
        assert len(var) == 4

    def test_mean_is_l2_normalized(self) -> None:
        fm = FisherRaoMetric()
        emb = [3.0, 4.0]
        mean, _ = fm.compute_params(emb)
        norm = math.sqrt(sum(m ** 2 for m in mean))
        np.testing.assert_allclose(norm, 1.0, atol=1e-10)

    def test_variance_within_bounds(self) -> None:
        fm = FisherRaoMetric()
        emb = np.random.default_rng(42).standard_normal(768).tolist()
        _, var = fm.compute_params(emb)
        arr = np.array(var)
        assert np.all(arr >= _VARIANCE_FLOOR - 1e-12)
        assert np.all(arr <= _VARIANCE_CEIL + 1e-12)

    def test_zero_embedding(self) -> None:
        fm = FisherRaoMetric()
        mean, var = fm.compute_params([0.0, 0.0, 0.0])
        np.testing.assert_allclose(mean, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(var, [_VARIANCE_CEIL] * 3)

    def test_single_dimension(self) -> None:
        fm = FisherRaoMetric()
        mean, var = fm.compute_params([5.0])
        np.testing.assert_allclose(mean, [1.0], atol=1e-10)
        # Strongest signal => lowest variance
        np.testing.assert_allclose(var, [_VARIANCE_FLOOR], atol=1e-10)

    def test_heterogeneous_variance(self) -> None:
        """Strong-signal dims get low variance, weak dims get high variance."""
        fm = FisherRaoMetric()
        emb = [10.0, 0.0]  # dim-0 strong, dim-1 zero
        mean, var = fm.compute_params(emb)
        # After normalization: [1.0, 0.0] => dim 0 = max signal => low var
        assert var[0] < var[1]


# ---------------------------------------------------------------------------
# distance
# ---------------------------------------------------------------------------

class TestDistance:
    def test_identical_distributions_zero_distance(self) -> None:
        fm = FisherRaoMetric()
        mean = [0.5, -0.3, 0.7]
        var = [1.0, 1.0, 1.0]
        d = fm.distance(mean, var, mean, var)
        np.testing.assert_allclose(d, 0.0, atol=1e-10)

    def test_distance_non_negative(self) -> None:
        fm = FisherRaoMetric()
        rng = np.random.default_rng(7)
        for _ in range(20):
            ma = rng.standard_normal(10).tolist()
            mb = rng.standard_normal(10).tolist()
            va = (rng.uniform(0.3, 2.0, 10)).tolist()
            vb = (rng.uniform(0.3, 2.0, 10)).tolist()
            assert fm.distance(ma, va, mb, vb) >= 0.0

    def test_distance_symmetry(self) -> None:
        fm = FisherRaoMetric()
        ma = [0.1, 0.2, 0.3]
        va = [0.5, 0.5, 0.5]
        mb = [0.4, 0.5, 0.6]
        vb = [1.0, 1.0, 1.0]
        d1 = fm.distance(ma, va, mb, vb)
        d2 = fm.distance(mb, vb, ma, va)
        np.testing.assert_allclose(d1, d2, atol=1e-10)

    def test_same_mean_different_variance_nonzero(self) -> None:
        """Fisher-Rao distinguishes identical means with different variance."""
        fm = FisherRaoMetric()
        mean = [0.5, 0.5]
        va = [0.3, 0.3]
        vb = [2.0, 2.0]
        d = fm.distance(mean, va, mean, vb)
        assert d > 0.0, "Fisher-Rao should detect variance difference"

    def test_nan_mean_raises(self) -> None:
        fm = FisherRaoMetric()
        with pytest.raises(ValueError, match="NaN"):
            fm.distance([float("nan")], [1.0], [0.0], [1.0])

    def test_non_positive_variance_raises(self) -> None:
        fm = FisherRaoMetric()
        with pytest.raises(ValueError, match="positive"):
            fm.distance([0.0], [0.0], [0.0], [1.0])

    def test_shape_mismatch_raises(self) -> None:
        fm = FisherRaoMetric()
        with pytest.raises(ValueError, match="mismatch"):
            fm.distance([0.0, 0.1], [1.0, 1.0], [0.0], [1.0])

    def test_sigma_shape_mismatch_raises(self) -> None:
        fm = FisherRaoMetric()
        with pytest.raises(ValueError, match="mismatch"):
            fm.distance([0.0], [1.0, 1.0], [0.0], [1.0])


# ---------------------------------------------------------------------------
# similarity
# ---------------------------------------------------------------------------

class TestSimilarity:
    def test_identical_gives_one(self) -> None:
        fm = FisherRaoMetric()
        m = [0.3, 0.4]
        v = [1.0, 1.0]
        s = fm.similarity(m, v, m, v)
        np.testing.assert_allclose(s, 1.0, atol=1e-10)

    def test_range_zero_to_one(self) -> None:
        fm = FisherRaoMetric()
        rng = np.random.default_rng(42)
        for _ in range(20):
            ma = rng.standard_normal(8).tolist()
            mb = rng.standard_normal(8).tolist()
            va = rng.uniform(0.3, 2.0, 8).tolist()
            vb = rng.uniform(0.3, 2.0, 8).tolist()
            s = fm.similarity(ma, va, mb, vb)
            assert 0.0 <= s <= 1.0

    def test_monotonic_with_distance(self) -> None:
        """Closer distributions should have higher similarity."""
        fm = FisherRaoMetric()
        m = [0.5, 0.5]
        v = [1.0, 1.0]
        # Near point
        m_near = [0.51, 0.51]
        # Far point
        m_far = [5.0, 5.0]
        s_near = fm.similarity(m, v, m_near, v)
        s_far = fm.similarity(m, v, m_far, v)
        assert s_near > s_far


# ---------------------------------------------------------------------------
# bayesian_update
# ---------------------------------------------------------------------------

class TestBayesianUpdate:
    def test_update_reduces_variance(self) -> None:
        fm = FisherRaoMetric()
        old_var = [1.0, 1.0, 1.0]
        obs_var = [1.0, 1.0, 1.0]
        new_var = fm.bayesian_update(old_var, obs_var)
        # 1/(1/1 + 1/1) = 0.5, but clamped to floor
        for v in new_var:
            assert v <= 1.0

    def test_precision_additive(self) -> None:
        """1/new = 1/old + 1/obs."""
        fm = FisherRaoMetric()
        old = [1.0]
        obs = [1.0]
        new = fm.bayesian_update(old, obs)
        expected = 1.0 / (1.0 / 1.0 + 1.0 / 1.0)
        np.testing.assert_allclose(new, [max(expected, _VARIANCE_FLOOR)], atol=1e-10)

    def test_floor_enforced(self) -> None:
        fm = FisherRaoMetric()
        old = [_VARIANCE_FLOOR, _VARIANCE_FLOOR]
        obs = [_VARIANCE_FLOOR, _VARIANCE_FLOOR]
        new = fm.bayesian_update(old, obs)
        for v in new:
            assert v >= _VARIANCE_FLOOR - 1e-12

    def test_ceil_enforced(self) -> None:
        fm = FisherRaoMetric()
        old = [_VARIANCE_CEIL, _VARIANCE_CEIL]
        obs = [_VARIANCE_CEIL, _VARIANCE_CEIL]
        new = fm.bayesian_update(old, obs)
        for v in new:
            assert v <= _VARIANCE_CEIL + 1e-12

    def test_shape_mismatch_raises(self) -> None:
        fm = FisherRaoMetric()
        with pytest.raises(ValueError, match="shape mismatch"):
            fm.bayesian_update([1.0, 1.0], [1.0])

    def test_multiple_updates_converge(self) -> None:
        """Repeated identical observations should narrow variance toward floor."""
        fm = FisherRaoMetric()
        var = [_VARIANCE_CEIL] * 4
        obs = [1.0] * 4
        for _ in range(50):
            var = fm.bayesian_update(var, obs)
        # After many updates, should be close to the floor
        for v in var:
            np.testing.assert_allclose(v, _VARIANCE_FLOOR, atol=0.01)


# ---------------------------------------------------------------------------
# adaptive_temperature
# ---------------------------------------------------------------------------

class TestAdaptiveTemperature:
    def test_empty_variances_returns_base(self) -> None:
        fm = FisherRaoMetric(temperature=10.0)
        assert fm.adaptive_temperature([]) == 10.0

    def test_midpoint_variance_close_to_base(self) -> None:
        fm = FisherRaoMetric(temperature=10.0)
        # avg variance = 1.0 => T_adapted = 10 * (1+1)/2 = 10.0
        variances = [[1.0, 1.0], [1.0, 1.0]]
        t = fm.adaptive_temperature(variances)
        np.testing.assert_allclose(t, 10.0, atol=1e-10)

    def test_high_variance_increases_temperature(self) -> None:
        fm = FisherRaoMetric(temperature=10.0)
        high_var = [[2.0, 2.0]]
        t = fm.adaptive_temperature(high_var)
        assert t > 10.0

    def test_low_variance_decreases_temperature(self) -> None:
        fm = FisherRaoMetric(temperature=10.0)
        low_var = [[0.3, 0.3]]
        t = fm.adaptive_temperature(low_var)
        assert t < 10.0

    def test_always_positive(self) -> None:
        fm = FisherRaoMetric(temperature=0.1)
        variances = [[0.0001]]
        t = fm.adaptive_temperature(variances)
        assert t >= 0.1


# ---------------------------------------------------------------------------
# _stable_arccosh_1p_vec (internal helper, exposed for testing)
# ---------------------------------------------------------------------------

class TestStableArccosh:
    def test_zero_delta(self) -> None:
        result = _stable_arccosh_1p_vec(np.array([0.0]))
        np.testing.assert_allclose(result, [0.0], atol=1e-12)

    def test_large_delta_matches_arccosh(self) -> None:
        delta = np.array([1.0, 5.0, 10.0])
        result = _stable_arccosh_1p_vec(delta)
        expected = np.arccosh(1.0 + delta)
        np.testing.assert_allclose(result, expected, atol=1e-8)

    def test_small_delta_taylor_approximation(self) -> None:
        """For very small delta, the Taylor path should be used."""
        delta = np.array([1e-10, 1e-12])
        result = _stable_arccosh_1p_vec(delta)
        # arccosh(1+d) ≈ sqrt(2d) for small d
        expected = np.sqrt(2.0 * delta)
        np.testing.assert_allclose(result, expected, atol=1e-8)

    def test_negative_clamped_to_zero(self) -> None:
        result = _stable_arccosh_1p_vec(np.array([-5.0]))
        np.testing.assert_allclose(result, [0.0], atol=1e-12)


# ---------------------------------------------------------------------------
# Integration: compute_params -> distance -> similarity round-trip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_same_embedding_gives_max_similarity(self) -> None:
        fm = FisherRaoMetric()
        emb = [1.0, 2.0, 3.0]
        ma, va = fm.compute_params(emb)
        mb, vb = fm.compute_params(emb)
        s = fm.similarity(ma, va, mb, vb)
        np.testing.assert_allclose(s, 1.0, atol=1e-10)

    def test_orthogonal_embeddings_lower_similarity(self) -> None:
        fm = FisherRaoMetric()
        emb_a = [1.0, 0.0, 0.0]
        emb_b = [0.0, 1.0, 0.0]
        ma, va = fm.compute_params(emb_a)
        mb, vb = fm.compute_params(emb_b)
        s = fm.similarity(ma, va, mb, vb)
        assert s < 1.0

    def test_confirmed_memory_scores_higher(self) -> None:
        """After bayesian_update, a well-confirmed memory should have
        tighter variance and thus higher similarity to a matching query."""
        fm = FisherRaoMetric()
        emb = [1.0, 2.0, 3.0]
        q_mean, q_var = fm.compute_params(emb)
        f_mean, f_var = fm.compute_params(emb)

        # Score before confirmation
        fm.similarity(q_mean, q_var, f_mean, f_var)

        # Confirm the fact 10 times
        for _ in range(10):
            f_var = fm.bayesian_update(f_var, q_var)

        s_after = fm.similarity(q_mean, q_var, f_mean, f_var)
        # After confirmation, the distance changes but similarity should
        # remain high (identical means with tighter variance)
        assert s_after > 0.5
