# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Fisher-Rao geodesic metric with Bayesian variance update.

Information-geometric similarity on the statistical manifold of diagonal
Gaussians.  Two embeddings with the same mean but different uncertainty
have zero cosine distance but nonzero Fisher distance --- this is the
core insight that differentiates Fisher-Rao from cosine similarity.

Geodesic distance (Atkinson & Mitchell 1981, Pinele et al. 2020):

    Univariate component i:
        delta_i = ((mu1_i - mu2_i)^2 + 2*(sigma1_i - sigma2_i)^2)
                  / (4 * sigma1_i * sigma2_i)
        d_i     = sqrt(2) * arccosh(1 + delta_i)

    Diagonal multivariate (product-manifold decomposition):
        d_FR(p, q) = sqrt( sum_i  d_i^2 )

Bayesian variance update (NEW in Innovation Wave 4):

    V1 bug: query always received UNIFORM variance, so Fisher degenerated
    to a monotonic transform of cosine.  FIX: every fact maintains its own
    variance vector that narrows on each observation.

        new_var_i = 1 / (1/old_var_i + 1/obs_var_i)

    Multiple consistent observations reduce variance, increasing confidence.
    This gives Fisher-Rao a genuine ranking advantage over cosine: well-
    confirmed memories have tighter distributions and score higher when the
    query matches.

References:
    Rao C R (1945). Information and the accuracy attainable in the
        estimation of statistical parameters. Bull. Calcutta Math. Soc.
    Atkinson C & Mitchell A (1981). Rao's distance measure.
        Sankhya: The Indian Journal of Statistics, Series A.
    Pinele J, Strapasson J E & Costa S I R (2020). The Fisher-Rao
        distance between multivariate normal distributions: special
        cases, bounds and applications. Entropy 22(4):404.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VARIANCE_FLOOR: float = 0.05  # Minimum per-dimension variance (40x dynamic range)
_VARIANCE_CEIL: float = 2.0    # Maximum per-dimension variance (initial uncertainty)
_DEFAULT_TEMPERATURE: float = 15.0
_SMALL_DELTA_THRESHOLD: float = 1e-7


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class FisherRaoMetric:
    """Fisher-Rao geodesic metric with Bayesian variance tracking.

    Each fact is modelled as a diagonal Gaussian N(mu, diag(sigma^2)).
    Distance is the Fisher-Rao geodesic on this statistical manifold.
    Variance starts high (uncertain) and NARROWS on repeated access via
    Bayesian update, giving well-confirmed memories a ranking advantage.

    Attributes:
        temperature: Softmax scaling for similarity conversion.
    """

    __slots__ = ("temperature",)

    def __init__(self, temperature: float = _DEFAULT_TEMPERATURE) -> None:
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Parameter derivation
    # ------------------------------------------------------------------

    def compute_params(
        self,
        embedding: list[float],
    ) -> tuple[list[float], list[float]]:
        """Derive Fisher parameters (mean, variance) from a raw embedding.

        Mean is the L2-normalised embedding.  Variance is content-derived
        and *heterogeneous*: dimensions with strong signal get low variance,
        weak-signal dimensions get high variance.

        Mapping: ``var_i = CEIL - (CEIL - FLOOR) * |normed_i| / max_abs``
        Strong signal (large |normed_i|) -> low variance (high confidence).

        Args:
            embedding: Raw embedding vector, any dimensionality.

        Returns:
            (mean, variance) as plain Python lists, same length as input.
        """
        arr = np.asarray(embedding, dtype=np.float64)
        norm = np.linalg.norm(arr)
        if norm < 1e-12:
            mean = np.zeros_like(arr)
            variance = np.full_like(arr, _VARIANCE_CEIL)
            return mean.tolist(), variance.tolist()

        mean = arr / norm

        abs_vals = np.abs(mean)
        max_abs = float(np.max(abs_vals))
        if max_abs < 1e-12:
            max_abs = 1.0

        normalised_signal = abs_vals / max_abs  # in [0, 1]
        variance = _VARIANCE_CEIL - (_VARIANCE_CEIL - _VARIANCE_FLOOR) * normalised_signal
        variance = np.clip(variance, _VARIANCE_FLOOR, _VARIANCE_CEIL)

        return mean.tolist(), variance.tolist()

    # ------------------------------------------------------------------
    # Geodesic distance
    # ------------------------------------------------------------------

    def distance(
        self,
        mean_a: list[float],
        var_a: list[float],
        mean_b: list[float],
        var_b: list[float],
    ) -> float:
        """Exact Fisher-Rao geodesic distance between diagonal Gaussians.

        Uses Atkinson & Mitchell (1981) per-component closed form with
        Pinele et al. (2020) product-manifold decomposition.

        Args:
            mean_a:  Mean of distribution A.
            var_a:   Per-dimension variance of A (strictly positive).
            mean_b:  Mean of distribution B.
            var_b:   Per-dimension variance of B (strictly positive).

        Returns:
            Non-negative geodesic distance.

        Raises:
            ValueError: On NaN, non-positive variance, or length mismatch.
        """
        mu1 = np.asarray(mean_a, dtype=np.float64)
        sig1 = np.asarray(var_a, dtype=np.float64)
        mu2 = np.asarray(mean_b, dtype=np.float64)
        sig2 = np.asarray(var_b, dtype=np.float64)

        _validate(mu1, sig1, mu2, sig2)

        # Per-component: delta_i = ((mu1-mu2)^2 + 2*(sigma1-sigma2)^2) / (4*s1*s2)
        mu_diff_sq = (mu1 - mu2) ** 2
        sig_diff_sq = (sig1 - sig2) ** 2
        product = sig1 * sig2

        delta = (mu_diff_sq + 2.0 * sig_diff_sq) / (4.0 * product)

        # Per-component squared distance: 2 * arccosh(1 + delta_i)^2
        acosh_vals = _stable_arccosh_1p_vec(delta)
        total_dist_sq = float(np.sum(2.0 * acosh_vals ** 2))

        return math.sqrt(max(total_dist_sq, 0.0))

    # ------------------------------------------------------------------
    # Similarity (exponential kernel)
    # ------------------------------------------------------------------

    def similarity(
        self,
        mean_a: list[float],
        var_a: list[float],
        mean_b: list[float],
        var_b: list[float],
    ) -> float:
        """Convert Fisher-Rao distance to similarity in [0, 1].

        sim = exp(-distance / temperature)

        Args:
            mean_a:  Mean of distribution A.
            var_a:   Per-dimension variance of A.
            mean_b:  Mean of distribution B.
            var_b:   Per-dimension variance of B.

        Returns:
            Similarity in [0, 1].  1 = identical, 0 = maximally different.
        """
        d = self.distance(mean_a, var_a, mean_b, var_b)
        return float(np.exp(-d / self.temperature))

    # ------------------------------------------------------------------
    # Bayesian variance update (THE key V1 fix)
    # ------------------------------------------------------------------

    def bayesian_update(
        self,
        old_var: list[float],
        observation_var: list[float],
    ) -> list[float]:
        """Bayesian precision-additive variance update.

        Each observation tightens the posterior variance:

            1/new_var_i = 1/old_var_i + 1/obs_var_i

        This is the standard conjugate update for a Gaussian likelihood
        with known mean and unknown precision.  After *k* observations
        with identical variance sigma^2:

            var_k = sigma^2 / k

        So variance shrinks as 1/k --- giving well-confirmed memories
        significantly tighter distributions and higher Fisher similarity
        to matching queries.

        Args:
            old_var:         Current per-dimension variance.
            observation_var: Variance of the new observation (derived from
                             the embedding of the confirming content).

        Returns:
            Updated variance (strictly within [FLOOR, CEIL]).
        """
        old = np.asarray(old_var, dtype=np.float64)
        obs = np.asarray(observation_var, dtype=np.float64)

        if old.shape != obs.shape:
            raise ValueError(
                f"Variance shape mismatch: {old.shape} vs {obs.shape}"
            )

        # Clamp inputs to valid range before update
        old = np.clip(old, _VARIANCE_FLOOR, _VARIANCE_CEIL)
        obs = np.clip(obs, _VARIANCE_FLOOR, _VARIANCE_CEIL)

        # Precision-additive update: 1/new = 1/old + 1/obs
        new_precision = (1.0 / old) + (1.0 / obs)
        new_var = 1.0 / new_precision

        new_var = np.clip(new_var, _VARIANCE_FLOOR, _VARIANCE_CEIL)
        return new_var.tolist()

    # ------------------------------------------------------------------
    # Adaptive temperature
    # ------------------------------------------------------------------

    def adaptive_temperature(
        self,
        variances: list[list[float]],
    ) -> float:
        """Compute data-driven temperature from corpus variance statistics.

        Instead of a fixed temperature, adapt to the actual spread of
        variances in the memory store.  High average variance (uncertain
        corpus) -> higher temperature (softer discrimination).  Low average
        variance (well-confirmed corpus) -> lower temperature (sharper).

        Formula:
            T = base_T * (1 + avg_variance) / 2

        This ensures T stays close to base_T when avg_variance ~ 1.0
        (the typical midpoint) and scales proportionally.

        Args:
            variances: List of per-fact variance vectors.

        Returns:
            Adapted temperature (always positive).
        """
        if not variances:
            return self.temperature

        all_vars = np.array(variances, dtype=np.float64)
        avg = float(np.mean(all_vars))

        adapted = self.temperature * (1.0 + avg) / 2.0
        return max(adapted, 0.1)  # floor to prevent division issues


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(
    mu1: np.ndarray,
    sig1: np.ndarray,
    mu2: np.ndarray,
    sig2: np.ndarray,
) -> None:
    """Validate Fisher metric inputs."""
    for name, arr in [("mu1", mu1), ("sig1", sig1), ("mu2", mu2), ("sig2", sig2)]:
        if np.any(np.isnan(arr)):
            raise ValueError(f"{name} contains NaN")

    if mu1.shape != mu2.shape:
        raise ValueError(f"Mean shape mismatch: {mu1.shape} vs {mu2.shape}")
    if sig1.shape != sig2.shape:
        raise ValueError(f"Sigma shape mismatch: {sig1.shape} vs {sig2.shape}")
    if mu1.shape[0] != sig1.shape[0]:
        raise ValueError(
            f"Mean/sigma length mismatch: {mu1.shape[0]} vs {sig1.shape[0]}"
        )

    if np.any(sig1 <= 0):
        raise ValueError("sig1 must be strictly positive")
    if np.any(sig2 <= 0):
        raise ValueError("sig2 must be strictly positive")


# ---------------------------------------------------------------------------
# Numerically stable arccosh
# ---------------------------------------------------------------------------

def _stable_arccosh_1p_vec(delta: np.ndarray) -> np.ndarray:
    """Vectorised stable arccosh(1 + delta).

    For small delta, uses Taylor expansion to avoid catastrophic
    cancellation: arccosh(1+d) ~ sqrt(2d) * (1 - d/12).
    For larger delta, uses the identity:
        arccosh(1+d) = log(1 + d + sqrt(d*(d+2))).

    Args:
        delta: Non-negative array.

    Returns:
        arccosh(1 + delta) element-wise.
    """
    delta = np.maximum(delta, 0.0)
    result = np.empty_like(delta)

    small = delta < _SMALL_DELTA_THRESHOLD
    large = ~small

    if np.any(small):
        d_s = delta[small]
        result[small] = np.sqrt(2.0 * d_s) * (1.0 - d_s / 12.0)

    if np.any(large):
        d_l = delta[large]
        result[large] = np.log1p(d_l + np.sqrt(d_l * (d_l + 2.0)))

    return result
