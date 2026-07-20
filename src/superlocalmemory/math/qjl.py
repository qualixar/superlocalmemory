# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""QJL (Quantized Johnson-Lindenstrauss) 1-bit residual correction.

Encodes the residual between original and PolarQuant-decoded embedding
as 1-bit sign projections. At query time, the asymmetric estimator
provides an unbiased correction to improve approximate similarity.

Pipeline:
  1. Random projection: projected = R @ residual  (m x d matrix)
  2. Sign quantize: bits = (projected > 0)
  3. Pack bits into bytes

Query-time correction:
  1. Project query: q_proj = R @ query
  2. Unpack signs to +/-1
  3. Estimate: correction = dot(q_proj, signs) / m

HR-07: QJL is OPTIONAL -- system works without it.
HR-08: No new pip dependencies (numpy + stdlib only).

References:
  - Zandieh A et al. (2025). QJL: 1-Bit Quantized JL Transform for KNN.
    AAAI 2025, arXiv 2406.03482.
  - TurboQuant (ICLR 2026, arXiv 2504.19874).

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from superlocalmemory.core.config import QJLConfig


class QJLEncoder:
    """1-bit random projection encoder for residual correction.

    The projection matrix R is (m x d) where m = projection_dim.
    Entries are i.i.d. N(0, 1/m) -- the Johnson-Lindenstrauss property
    preserves inner products in expectation.

    Lazy initialization: the projection matrix is created on first use
    to avoid allocating memory if QJL is never invoked.
    """

    __slots__ = ("_config", "_R")

    def __init__(self, config: QJLConfig) -> None:
        self._config = config
        self._R: NDArray | None = None

    def _ensure_projection(self, d: int) -> None:
        """Create or verify the random projection matrix.

        Matrix shape: (projection_dim, d).
        Scaling: 1/sqrt(m) for unbiased estimation.
        """
        if self._R is not None and self._R.shape[1] == d:
            return
        rng = np.random.default_rng(self._config.seed)
        m = self._config.projection_dim
        self._R = rng.standard_normal((m, d)) / math.sqrt(m)

    def encode_residual(self, residual: NDArray) -> bytes:
        """Encode a residual vector into 1-bit sign projections.

        Args:
            residual: Difference between original and decoded embedding.

        Returns:
            Packed bits as bytes (ceil(projection_dim / 8) bytes).
        """
        self._ensure_projection(len(residual))
        projected = self._R @ residual
        bits = (projected > 0).astype(np.uint8)
        return np.packbits(bits).tobytes()

    def estimate_correction(self, query: NDArray, qjl_bits: bytes) -> float:
        """Estimate inner product correction from QJL bits.

        This is the asymmetric estimator from the QJL paper:
          correction = dot(R @ query, signs) / m

        Args:
            query:     Query embedding.
            qjl_bits:  Packed sign bits from encode_residual().

        Returns:
            Estimated inner product correction (float).
        """
        self._ensure_projection(len(query))
        m = self._config.projection_dim

        # Unpack bits
        bits = np.unpackbits(
            np.frombuffer(qjl_bits, dtype=np.uint8),
        )[:m]

        # Convert 0/1 to -1/+1
        signs = 2.0 * bits.astype(np.float64) - 1.0

        # Project query
        q_proj = self._R @ query

        # Asymmetric estimator
        correction = float(np.dot(q_proj, signs)) / m
        return correction
