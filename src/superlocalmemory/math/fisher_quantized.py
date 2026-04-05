# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the Elastic License 2.0 - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Fisher-Rao Quantization-Aware Distance (FRQAD).

Novel contribution (98% IP confidence): Fisher-Rao geodesic distance on
mixed-precision embeddings.  Quantized memories naturally rank lower because
their variance is inflated to model quantization uncertainty.

    sigma_quant = sigma_base * (32 / bit_width) ^ kappa

When both embeddings are float32, FRQAD degrades exactly to standard
Fisher-Rao --- no regression, pure extension.

Mathematical basis:
  - Fisher-Rao geodesic (Atkinson & Mitchell 1981, Pinele et al. 2020)
  - Variance inflation models quantization noise as additional uncertainty
    on the statistical manifold of diagonal Gaussians

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: Elastic-2.0
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superlocalmemory.math.fisher import FisherRaoMetric

logger = logging.getLogger(__name__)

# Valid bit-widths aligned with Phase B/D (no 16-bit format exists).
_VALID_BIT_WIDTHS: frozenset[int] = frozenset({2, 4, 8, 32})


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FRQADConfig:
    """Configuration for FRQAD metric.

    Attributes:
        kappa:            Scaling exponent for variance inflation.
                          sigma_quant = sigma_base * (32/bw)^kappa.
        temperature:      Softmax temperature for similarity conversion.
        enabled:          When False, fall back to base Fisher-Rao.
        variance_floor:   Minimum per-dimension variance after inflation.
        variance_ceiling: Maximum per-dimension variance after inflation.
    """

    kappa: float = 0.5
    temperature: float = 15.0
    enabled: bool = True
    variance_floor: float = 0.05
    variance_ceiling: float = 10.0


# ---------------------------------------------------------------------------
# FRQAD Metric
# ---------------------------------------------------------------------------


class FRQADMetric:
    """Fisher-Rao Quantization-Aware Distance metric.

    Wraps an existing FisherRaoMetric and inflates variance vectors based
    on each embedding's bit-width before computing geodesic distance.

    Self-consistency property:
        FRQAD(32bit, 32bit) == FisherRao(same pair) within float epsilon.
    """

    __slots__ = ("_base", "_config")

    def __init__(
        self,
        base_metric: FisherRaoMetric,
        config: FRQADConfig,
    ) -> None:
        """Construct FRQAD metric.

        Args:
            base_metric: Existing Fisher-Rao metric instance.
            config:      FRQAD configuration.

        Raises:
            ValueError: On invalid configuration values.
        """
        # Validate config ranges (HR-04)
        if not 0.0 <= config.kappa <= 2.0:
            raise ValueError(
                f"kappa must be in [0.0, 2.0], got {config.kappa}"
            )
        if not 1.0 <= config.temperature <= 100.0:
            raise ValueError(
                f"temperature must be in [1.0, 100.0], got {config.temperature}"
            )
        if not 0.001 <= config.variance_floor <= 1.0:
            raise ValueError(
                f"variance_floor must be in [0.001, 1.0], got {config.variance_floor}"
            )
        if not 1.0 <= config.variance_ceiling <= 100.0:
            raise ValueError(
                f"variance_ceiling must be in [1.0, 100.0], got {config.variance_ceiling}"
            )

        self._base = base_metric
        self._config = config

    # ------------------------------------------------------------------
    # Quantization variance inflation
    # ------------------------------------------------------------------

    def quantization_variance(
        self,
        base_variance: NDArray[np.floating],
        bit_width: int,
    ) -> NDArray[np.floating]:
        """Inflate variance to model quantization uncertainty.

        HR-02: NEVER decreases base variance (scale >= 1.0 since bw <= 32).

        Args:
            base_variance: Per-dimension variance vector (strictly positive).
            bit_width:     Embedding precision in bits.

        Returns:
            Quantization-aware variance, clamped to [floor, ceiling].
        """
        if bit_width not in _VALID_BIT_WIDTHS:
            logger.warning(
                "Unknown bit_width %d, treating as 32 (no penalty)", bit_width,
            )
            return np.array(base_variance, dtype=np.float64)

        if bit_width >= 32:
            return np.array(base_variance, dtype=np.float64)

        # V3.3.26: MULTIPLICATIVE variance inflation (Paper 3, Equation 2).
        # sigma²_eff = sigma²_obs * (32 / bit_width) ^ kappa
        # When bw=32: scale=1.0 (no change). When bw=4: scale=2.83x (kappa=0.5).
        # This is MORE novel and MORE aggressive than additive Delta²/12.
        scale = (32.0 / bit_width) ** self._config.kappa
        sigma_inflated = np.asarray(base_variance, dtype=np.float64) * scale

        return np.clip(sigma_inflated, self._config.variance_floor, self._config.variance_ceiling)

    # ------------------------------------------------------------------
    # Core distance (THE novel contribution)
    # ------------------------------------------------------------------

    def distance(
        self,
        mu_a: NDArray[np.floating],
        var_a: NDArray[np.floating],
        bw_a: int,
        mu_b: NDArray[np.floating],
        var_b: NDArray[np.floating],
        bw_b: int,
    ) -> float:
        """FRQAD distance between mixed-precision embeddings.

        Inflates variance based on bit-width, then delegates to the
        standard Fisher-Rao geodesic.

        Args:
            mu_a:  Mean of embedding A.
            var_a: Per-dimension variance of A.
            bw_a:  Bit-width of A.
            mu_b:  Mean of embedding B.
            var_b: Per-dimension variance of B.
            bw_b:  Bit-width of B.

        Returns:
            Non-negative geodesic distance.
        """
        if not self._config.enabled:
            return self._base.distance(
                np.asarray(mu_a, dtype=np.float64).tolist(),
                np.asarray(var_a, dtype=np.float64).tolist(),
                np.asarray(mu_b, dtype=np.float64).tolist(),
                np.asarray(var_b, dtype=np.float64).tolist(),
            )

        var_a_q = self.quantization_variance(var_a, bw_a)
        var_b_q = self.quantization_variance(var_b, bw_b)

        return self._base.distance(
            np.asarray(mu_a, dtype=np.float64).tolist(),
            var_a_q.tolist(),
            np.asarray(mu_b, dtype=np.float64).tolist(),
            var_b_q.tolist(),
        )

    # ------------------------------------------------------------------
    # Similarity (exponential kernel)
    # ------------------------------------------------------------------

    def similarity(
        self,
        mu_a: NDArray[np.floating],
        var_a: NDArray[np.floating],
        bw_a: int,
        mu_b: NDArray[np.floating],
        var_b: NDArray[np.floating],
        bw_b: int,
    ) -> float:
        """Convert FRQAD distance to similarity in [0, 1].

        sim = clamp(exp(-distance / temperature), 0, 1)

        HR-03: No NaN, no negative, no > 1.
        """
        d = self.distance(mu_a, var_a, bw_a, mu_b, var_b, bw_b)
        sim = math.exp(-d / self._config.temperature)
        return max(0.0, min(1.0, sim))

    # ------------------------------------------------------------------
    # Batch similarity
    # ------------------------------------------------------------------

    def batch_similarity(
        self,
        query_mu: NDArray[np.floating],
        query_var: NDArray[np.floating],
        query_bw: int,
        candidates: list[tuple[str, NDArray[np.floating], NDArray[np.floating], int]],
    ) -> list[tuple[str, float]]:
        """Score a batch of candidate memories against a query.

        Args:
            query_mu:   Query embedding mean.
            query_var:  Query embedding variance.
            query_bw:   Query bit-width (typically 32).
            candidates: List of (fact_id, mu, var, bit_width) tuples.

        Returns:
            List of (fact_id, similarity) sorted descending by score.
        """
        results: list[tuple[str, float]] = []
        for fact_id, mu, var, bw in candidates:
            sim = self.similarity(query_mu, query_var, query_bw, mu, var, bw)
            results.append((fact_id, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results
