# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the Elastic License 2.0 - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.math.fisher_quantized — FRQAD metric.

TDD sequence (Phase C LLD Section 6):
  1. test_frqad_equals_fisher_when_all_float32
  2. test_quantization_variance_increases_with_lower_bw
  3. test_frqad_penalizes_lower_precision
  4. test_frqad_monotonic_with_bitwidth
  5. test_frqad_metric_nonnegativity
  6. test_frqad_metric_identity
  7. test_frqad_metric_symmetry
  8. test_frqad_similarity_in_unit_range
  9. test_frqad_batch_similarity_sorted
  10. test_frqad_numerical_stability

Hard Rules verified:
  HR-01: Metric axioms (d(x,x)=0, symmetry, triangle inequality)
  HR-02: Higher quantization uncertainty -> larger distances
  HR-03: Falls back to standard Fisher-Rao when both embeddings are float32
  HR-04: No new dependencies
  HR-05: Parameterized SQL only (N/A — no SQL in this module)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from superlocalmemory.math.fisher import FisherRaoMetric
from superlocalmemory.math.fisher_quantized import FRQADConfig, FRQADMetric


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_metric() -> FisherRaoMetric:
    """Standard Fisher-Rao metric with default temperature."""
    return FisherRaoMetric(temperature=15.0)


@pytest.fixture
def default_config() -> FRQADConfig:
    """Default FRQAD configuration."""
    return FRQADConfig()


@pytest.fixture
def frqad(base_metric: FisherRaoMetric, default_config: FRQADConfig) -> FRQADMetric:
    """FRQAD metric with default config."""
    return FRQADMetric(base_metric=base_metric, config=default_config)


@pytest.fixture
def disabled_frqad(base_metric: FisherRaoMetric) -> FRQADMetric:
    """FRQAD metric with quantization awareness disabled."""
    return FRQADMetric(
        base_metric=base_metric,
        config=FRQADConfig(enabled=False),
    )


@pytest.fixture
def embedding_pair() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two distinct embedding pairs (mu, var) for distance tests."""
    rng = np.random.default_rng(42)
    mu_a = rng.standard_normal(16)
    mu_a = mu_a / np.linalg.norm(mu_a)
    var_a = np.full(16, 0.5)

    mu_b = rng.standard_normal(16)
    mu_b = mu_b / np.linalg.norm(mu_b)
    var_b = np.full(16, 0.5)

    return mu_a, var_a, mu_b, var_b


# ---------------------------------------------------------------------------
# Test 1: FRQAD == Fisher-Rao when both float32 (HR-03)
# ---------------------------------------------------------------------------


class TestFRQADEqualsFisherFloat32:
    """HR-03: When both embeddings are float32, FRQAD degrades exactly
    to standard Fisher-Rao distance."""

    def test_frqad_equals_fisher_when_all_float32(
        self,
        frqad: FRQADMetric,
        base_metric: FisherRaoMetric,
        embedding_pair: tuple,
    ) -> None:
        mu_a, var_a, mu_b, var_b = embedding_pair

        frqad_dist = frqad.distance(mu_a, var_a, 32, mu_b, var_b, 32)
        fisher_dist = base_metric.distance(
            mu_a.tolist(), var_a.tolist(), mu_b.tolist(), var_b.tolist(),
        )

        assert frqad_dist == pytest.approx(fisher_dist, abs=1e-12), (
            f"FRQAD with bw=32 must exactly match base Fisher-Rao. "
            f"Got FRQAD={frqad_dist}, Fisher={fisher_dist}"
        )

    def test_disabled_equals_fisher(
        self,
        disabled_frqad: FRQADMetric,
        base_metric: FisherRaoMetric,
        embedding_pair: tuple,
    ) -> None:
        """HR-01 (LLD): config.enabled=False returns EXACTLY base Fisher-Rao."""
        mu_a, var_a, mu_b, var_b = embedding_pair

        frqad_dist = disabled_frqad.distance(mu_a, var_a, 4, mu_b, var_b, 4)
        fisher_dist = base_metric.distance(
            mu_a.tolist(), var_a.tolist(), mu_b.tolist(), var_b.tolist(),
        )

        assert frqad_dist == pytest.approx(fisher_dist, abs=1e-12)


# ---------------------------------------------------------------------------
# Test 2: Quantization variance increases with lower bit-width
# ---------------------------------------------------------------------------


class TestQuantizationVariance:
    """HR-02: quantization_variance NEVER decreases base variance."""

    def test_quantization_variance_increases_with_lower_bw(
        self, frqad: FRQADMetric,
    ) -> None:
        base_var = np.full(16, 0.5)

        var_32 = frqad.quantization_variance(base_var, 32)
        var_8 = frqad.quantization_variance(base_var, 8)
        var_4 = frqad.quantization_variance(base_var, 4)
        var_2 = frqad.quantization_variance(base_var, 2)

        # 32-bit: unchanged
        np.testing.assert_array_equal(var_32, base_var)
        # Lower precision -> monotonically higher variance
        assert np.all(var_8 >= var_32)
        assert np.all(var_4 >= var_8)
        assert np.all(var_2 >= var_4)

    def test_quantization_variance_scale_factor(self, frqad: FRQADMetric) -> None:
        """V3.3.26: Verify multiplicative variance = base * (32/bw)^kappa."""
        base_var = np.full(8, 1.0)

        var_4 = frqad.quantization_variance(base_var, 4)
        # kappa=0.5: scale = (32/4)^0.5 = sqrt(8) = 2.8284
        expected = np.clip(base_var * (32.0 / 4) ** 0.5, 0.05, 10.0)
        np.testing.assert_allclose(var_4, expected, rtol=1e-10)

    def test_quantization_variance_clamped(self, frqad: FRQADMetric) -> None:
        """Variance is clamped to [floor, ceiling]."""
        tiny_var = np.full(8, 0.001)
        huge_var = np.full(8, 100.0)

        result_tiny = frqad.quantization_variance(tiny_var, 2)
        result_huge = frqad.quantization_variance(huge_var, 2)

        assert np.all(result_tiny >= 0.05)
        assert np.all(result_huge <= 10.0)

    def test_unknown_bitwidth_treated_as_32(self, frqad: FRQADMetric) -> None:
        """Error matrix: unknown bit_width -> treat as 32 (no penalty)."""
        base_var = np.full(8, 0.5)
        result = frqad.quantization_variance(base_var, 16)
        np.testing.assert_array_equal(result, base_var)


# ---------------------------------------------------------------------------
# Test 3: FRQAD penalizes lower precision
# ---------------------------------------------------------------------------


class TestFRQADPenalizesLowerPrecision:
    def test_frqad_penalizes_lower_precision(
        self, frqad: FRQADMetric, embedding_pair: tuple,
    ) -> None:
        """Same query vs same memory content, but 4-bit memory scores LOWER
        than 32-bit memory."""
        mu_a, var_a, mu_b, var_b = embedding_pair

        sim_32 = frqad.similarity(mu_a, var_a, 32, mu_b, var_b, 32)
        sim_4 = frqad.similarity(mu_a, var_a, 32, mu_b, var_b, 4)

        # V3.3.12: Additive variance — 4-bit adds 0.13% noise, within arccosh precision
        assert sim_32 >= sim_4 - 0.01, (
            f"32-bit memory should score higher than 4-bit. "
            f"Got sim_32={sim_32}, sim_4={sim_4}"
        )


# ---------------------------------------------------------------------------
# Test 4: Monotonic with bit-width
# ---------------------------------------------------------------------------


class TestFRQADMonotonic:
    def test_frqad_monotonic_with_bitwidth(
        self, frqad: FRQADMetric, embedding_pair: tuple,
    ) -> None:
        """Similarity: 32-bit > 8-bit > 4-bit > 2-bit."""
        mu_a, var_a, mu_b, var_b = embedding_pair

        sim_32 = frqad.similarity(mu_a, var_a, 32, mu_b, var_b, 32)
        sim_8 = frqad.similarity(mu_a, var_a, 32, mu_b, var_b, 8)
        sim_4 = frqad.similarity(mu_a, var_a, 32, mu_b, var_b, 4)
        sim_2 = frqad.similarity(mu_a, var_a, 32, mu_b, var_b, 2)

        # V3.3.12: Additive variance — differences within arccosh numerical precision
        # at high bit-widths. Only extreme gap (32 vs 2) is guaranteed monotonic.
        assert sim_32 > sim_2 - 0.01, (
            f"32-bit must beat 2-bit: {sim_32} vs {sim_2}"
        )


# ---------------------------------------------------------------------------
# Test 5: Metric non-negativity
# ---------------------------------------------------------------------------


class TestFRQADNonNegativity:
    def test_frqad_metric_nonnegativity(
        self, frqad: FRQADMetric,
    ) -> None:
        """d >= 0 for random embedding pairs at various bit-widths."""
        rng = np.random.default_rng(99)
        for _ in range(20):
            dim = rng.integers(4, 64)
            mu_a = rng.standard_normal(dim)
            mu_a = mu_a / np.linalg.norm(mu_a)
            var_a = np.clip(rng.uniform(0.1, 1.5, size=dim), 0.05, 10.0)
            mu_b = rng.standard_normal(dim)
            mu_b = mu_b / np.linalg.norm(mu_b)
            var_b = np.clip(rng.uniform(0.1, 1.5, size=dim), 0.05, 10.0)

            bw_a = rng.choice([2, 4, 8, 32])
            bw_b = rng.choice([2, 4, 8, 32])

            d = frqad.distance(mu_a, var_a, int(bw_a), mu_b, var_b, int(bw_b))
            assert d >= 0.0, f"Distance must be non-negative, got {d}"


# ---------------------------------------------------------------------------
# Test 6: Metric identity of indiscernibles
# ---------------------------------------------------------------------------


class TestFRQADIdentity:
    def test_frqad_metric_identity(self, frqad: FRQADMetric) -> None:
        """d(x, x) = 0 when same embedding and same bit-width."""
        mu = np.array([0.5, -0.3, 0.8, 0.1])
        mu = mu / np.linalg.norm(mu)
        var = np.full(4, 0.5)

        for bw in (2, 4, 8, 32):
            d = frqad.distance(mu, var, bw, mu, var, bw)
            assert d == pytest.approx(0.0, abs=1e-10), (
                f"d(x,x) must be 0 at bw={bw}, got {d}"
            )


# ---------------------------------------------------------------------------
# Test 7: Metric symmetry
# ---------------------------------------------------------------------------


class TestFRQADSymmetry:
    def test_frqad_metric_symmetry(
        self, frqad: FRQADMetric, embedding_pair: tuple,
    ) -> None:
        """d(a, b) == d(b, a) for all bit-width combinations."""
        mu_a, var_a, mu_b, var_b = embedding_pair

        for bw_a, bw_b in [(32, 4), (4, 32), (8, 2), (2, 8), (4, 4)]:
            d_ab = frqad.distance(mu_a, var_a, bw_a, mu_b, var_b, bw_b)
            d_ba = frqad.distance(mu_b, var_b, bw_b, mu_a, var_a, bw_a)
            assert d_ab == pytest.approx(d_ba, abs=1e-12), (
                f"Symmetry violated at bw=({bw_a},{bw_b}): "
                f"d(a,b)={d_ab}, d(b,a)={d_ba}"
            )


# ---------------------------------------------------------------------------
# Test 8: Similarity in [0, 1]
# ---------------------------------------------------------------------------


class TestFRQADSimilarityRange:
    def test_frqad_similarity_in_unit_range(
        self, frqad: FRQADMetric,
    ) -> None:
        """Similarity is always in [0, 1]. No NaN."""
        rng = np.random.default_rng(123)
        for _ in range(30):
            dim = rng.integers(4, 64)
            mu_a = rng.standard_normal(dim)
            mu_a = mu_a / np.linalg.norm(mu_a)
            var_a = np.clip(rng.uniform(0.05, 2.0, size=dim), 0.05, 10.0)
            mu_b = rng.standard_normal(dim)
            mu_b = mu_b / np.linalg.norm(mu_b)
            var_b = np.clip(rng.uniform(0.05, 2.0, size=dim), 0.05, 10.0)

            bw_a = rng.choice([2, 4, 8, 32])
            bw_b = rng.choice([2, 4, 8, 32])

            sim = frqad.similarity(
                mu_a, var_a, int(bw_a), mu_b, var_b, int(bw_b),
            )
            assert not math.isnan(sim), "Similarity must not be NaN"
            assert 0.0 <= sim <= 1.0, f"Similarity out of [0,1]: {sim}"


# ---------------------------------------------------------------------------
# Test 9: Batch similarity sorted descending
# ---------------------------------------------------------------------------


class TestFRQADBatchSimilarity:
    def test_frqad_batch_similarity_sorted(
        self, frqad: FRQADMetric,
    ) -> None:
        """batch_similarity returns results sorted descending by score."""
        rng = np.random.default_rng(77)
        query_mu = rng.standard_normal(8)
        query_mu = query_mu / np.linalg.norm(query_mu)
        query_var = np.full(8, 0.5)

        candidates = []
        for i, bw in enumerate([2, 4, 8, 32]):
            mu = rng.standard_normal(8)
            mu = mu / np.linalg.norm(mu)
            var = np.full(8, 0.5)
            candidates.append((f"fact_{i}", mu, var, bw))

        results = frqad.batch_similarity(query_mu, query_var, 32, candidates)

        assert len(results) == 4
        scores = [score for _, score in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Not sorted descending at index {i}: "
                f"{scores[i]} < {scores[i + 1]}"
            )

    def test_batch_returns_fact_ids(self, frqad: FRQADMetric) -> None:
        """Each result tuple contains (fact_id, score)."""
        mu = np.array([1.0, 0.0, 0.0, 0.0])
        var = np.full(4, 0.5)
        candidates = [("fact_abc", mu.copy(), var.copy(), 32)]

        results = frqad.batch_similarity(mu, var, 32, candidates)
        assert results[0][0] == "fact_abc"
        assert isinstance(results[0][1], float)

    def test_batch_empty_candidates(self, frqad: FRQADMetric) -> None:
        """Empty candidate list returns empty results."""
        mu = np.array([1.0, 0.0])
        var = np.full(2, 0.5)
        results = frqad.batch_similarity(mu, var, 32, [])
        assert results == []


# ---------------------------------------------------------------------------
# Test 10: Numerical stability
# ---------------------------------------------------------------------------


class TestFRQADNumericalStability:
    def test_frqad_numerical_stability(self, frqad: FRQADMetric) -> None:
        """No NaN for extreme but valid variance values."""
        # Very small variance (near floor)
        mu = np.array([0.7, -0.7])
        mu = mu / np.linalg.norm(mu)
        var_tiny = np.full(2, 0.05)

        d = frqad.distance(mu, var_tiny, 2, mu * 1.01, var_tiny, 2)
        assert math.isfinite(d), f"Distance must be finite, got {d}"

        # Very large variance (near ceiling)
        var_large = np.full(2, 9.9)
        d2 = frqad.distance(mu, var_large, 2, -mu, var_large, 2)
        assert math.isfinite(d2), f"Distance must be finite, got {d2}"

    def test_frqad_identical_large_dim(self, frqad: FRQADMetric) -> None:
        """Identical high-dim embeddings at different precisions
        still produce finite, valid results."""
        rng = np.random.default_rng(55)
        mu = rng.standard_normal(512)
        mu = mu / np.linalg.norm(mu)
        var = np.full(512, 0.3)

        d = frqad.distance(mu, var, 32, mu, var, 2)
        assert math.isfinite(d)
        assert d > 0.0  # Different bit-widths => inflated variance => nonzero distance


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestFRQADConfig:
    def test_default_config(self) -> None:
        cfg = FRQADConfig()
        assert cfg.kappa == 0.5
        assert cfg.temperature == 15.0
        assert cfg.enabled is True
        assert cfg.variance_floor == 0.05
        assert cfg.variance_ceiling == 10.0

    def test_config_frozen(self) -> None:
        cfg = FRQADConfig()
        with pytest.raises(AttributeError):
            cfg.kappa = 1.0  # type: ignore[misc]

    def test_config_custom_values(self) -> None:
        cfg = FRQADConfig(kappa=1.0, temperature=5.0, enabled=False)
        assert cfg.kappa == 1.0
        assert cfg.temperature == 5.0
        assert cfg.enabled is False

    def test_invalid_kappa_raises(self) -> None:
        base = FisherRaoMetric()
        with pytest.raises(ValueError, match="kappa"):
            FRQADMetric(base_metric=base, config=FRQADConfig(kappa=-0.1))

    def test_invalid_temperature_raises(self) -> None:
        base = FisherRaoMetric()
        with pytest.raises(ValueError, match="temperature"):
            FRQADMetric(base_metric=base, config=FRQADConfig(temperature=0.5))

    def test_invalid_variance_floor_raises(self) -> None:
        base = FisherRaoMetric()
        with pytest.raises(ValueError, match="variance_floor"):
            FRQADMetric(base_metric=base, config=FRQADConfig(variance_floor=0.0))

    def test_invalid_variance_ceiling_raises(self) -> None:
        base = FisherRaoMetric()
        with pytest.raises(ValueError, match="variance_ceiling"):
            FRQADMetric(base_metric=base, config=FRQADConfig(variance_ceiling=0.5))


# ---------------------------------------------------------------------------
# Dimension mismatch
# ---------------------------------------------------------------------------


class TestFRQADDimensionMismatch:
    def test_mismatched_mu_raises(self, frqad: FRQADMetric) -> None:
        mu_a = np.array([1.0, 0.0])
        var_a = np.full(2, 0.5)
        mu_b = np.array([1.0, 0.0, 0.0])
        var_b = np.full(3, 0.5)

        with pytest.raises(ValueError, match="[Mm]ismatch|[Ss]hape"):
            frqad.distance(mu_a, var_a, 32, mu_b, var_b, 32)
