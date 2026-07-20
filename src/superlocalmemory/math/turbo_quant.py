# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""TurboQuant embedding quantization (ICLR 2026).

Per-coordinate Lloyd-Max scalar quantization after random orthogonal rotation.
D_mse <= sqrt(3*pi/2) / 4^b. No scipy (HR-SCIPY-01). 2-byte "TQ" prefix on
all BLOBs (HR-MIG-02). Bit-widths: 2, 4, 8 only (HR-3BIT-01).

References: TurboQuant (arXiv 2504.19874), PolarQuant (arXiv 2502.02617).
Part of Qualixar | Author: Varun Pratap Bhardwaj | License: AGPL-3.0-or-later
"""

from __future__ import annotations

import logging
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from superlocalmemory.core.config import PolarQuantConfig
from superlocalmemory.infra.data_root import canonical_data_root

logger = logging.getLogger(__name__)

TQ_MAGIC = b"\x54\x51"  # 2-byte prefix for TurboQuant BLOBs (HR-MIG-02)
SUPPORTED_BIT_WIDTHS: frozenset[int] = frozenset({2, 4, 8})

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurboQuantResult:
    """Immutable TurboQuant-encoded embedding. radius=float16, indices=TQ-prefixed."""

    radius: float
    indices: bytes  # TQ_MAGIC + packed codebook indices
    bit_width: int


# ---------------------------------------------------------------------------
# Lloyd-Max codebook (HR-SCIPY-01: math.erf + math.exp only)
# ---------------------------------------------------------------------------

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_SQRT_2 = math.sqrt(2.0)


def _std_normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def _std_normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT_2))


def _compute_lloyd_max_gaussian(
    sigma: float, n_levels: int, max_iter: int = 100, tol: float = 1e-10,
) -> NDArray:
    """Lloyd-Max optimal codebook for N(0, sigma^2) on [-1, 1]. Deterministic (HR-CB-01).

    The codebook boundaries extend to [-1, 1] (full unit-sphere coordinate range)
    rather than [-5*sigma, 5*sigma], because after rotation, unit vector coordinates
    CAN have extreme values (up to ±1). The Gaussian distribution determines
    centroid placement, but the boundary range must cover all possible values.
    """
    lo, hi = -1.0, 1.0  # Full unit-sphere coordinate range
    boundaries = np.linspace(lo, hi, n_levels + 1)
    centroids = np.zeros(n_levels)
    for k in range(n_levels):
        centroids[k] = 0.5 * (boundaries[k] + boundaries[k + 1])

    for _ in range(max_iter):
        old = centroids.copy()
        for k in range(n_levels):
            a_k = float(boundaries[k]) / sigma
            b_k = float(boundaries[k + 1]) / sigma
            denom = _std_normal_cdf(b_k) - _std_normal_cdf(a_k)
            if denom > 1e-15:
                centroids[k] = sigma * (_std_normal_pdf(a_k) - _std_normal_pdf(b_k)) / denom
            else:
                # Tail region: use midpoint (values here are rare but must be handled)
                centroids[k] = 0.5 * (boundaries[k] + boundaries[k + 1])
        for k in range(1, n_levels):
            boundaries[k] = 0.5 * (centroids[k - 1] + centroids[k])
        if float(np.max(np.abs(centroids - old))) < tol:
            break

    return np.sort(centroids)


# ---------------------------------------------------------------------------
# Bit packing
# ---------------------------------------------------------------------------


def _pack_8bit(indices: NDArray) -> bytes:
    return indices.astype(np.uint8).tobytes()


def _unpack_8bit(data: bytes, length: int) -> NDArray:
    return np.frombuffer(data, dtype=np.uint8)[:length].copy()


def _pack_4bit(indices: NDArray) -> bytes:
    n = len(indices)
    padded = np.zeros(n + (n % 2), dtype=np.uint8)
    padded[:n] = np.clip(indices, 0, 15)
    return ((padded[0::2] << 4) | padded[1::2]).tobytes()


def _unpack_4bit(data: bytes, length: int) -> NDArray:
    packed = np.frombuffer(data, dtype=np.uint8)
    result = np.empty(len(packed) * 2, dtype=np.uint8)
    result[0::2] = packed >> 4
    result[1::2] = packed & 0x0F
    return result[:length]


def _pack_2bit(indices: NDArray) -> bytes:
    n = len(indices)
    padded = np.zeros(n + (4 - n % 4) % 4, dtype=np.uint8)
    padded[:n] = np.clip(indices, 0, 3)
    return (
        (padded[0::4] << 6) | (padded[1::4] << 4)
        | (padded[2::4] << 2) | padded[3::4]
    ).tobytes()


def _unpack_2bit(data: bytes, length: int) -> NDArray:
    packed = np.frombuffer(data, dtype=np.uint8)
    result = np.empty(len(packed) * 4, dtype=np.uint8)
    result[0::4] = (packed >> 6) & 0x03
    result[1::4] = (packed >> 4) & 0x03
    result[2::4] = (packed >> 2) & 0x03
    result[3::4] = packed & 0x03
    return result[:length]


_PACKERS: dict[int, tuple] = {
    8: (_pack_8bit, _unpack_8bit),
    4: (_pack_4bit, _unpack_4bit),
    2: (_pack_2bit, _unpack_2bit),
}

# ---------------------------------------------------------------------------
# TurboQuantEncoder
# ---------------------------------------------------------------------------


class TurboQuantEncoder:
    """Per-coordinate Lloyd-Max quantizer with random rotation.

    HR-ROT-01: Same rotation matrix for encode/decode.
    HR-CB-02: Codebooks computed ONCE at __init__.
    HR-SCIPY-01: No scipy dependency.
    """

    __slots__ = ("_config", "_d", "_S", "_codebooks")

    def __init__(self, config: PolarQuantConfig) -> None:
        self._config = config
        self._d = config.dimension
        self._S = self._load_or_create_rotation_matrix()
        self._codebooks = self._compute_codebooks()

    def _load_or_create_rotation_matrix(self) -> NDArray:
        """Load/create rotation matrix with copy-on-detect (AUDIT C4-MED-01)."""
        d = self._d
        slm_dir = canonical_data_root()

        turbo_path_str = self._config.rotation_matrix_path
        if not turbo_path_str:
            turbo_path_str = str(slm_dir / f"turbo_rotation_{d}.npy")
        turbo_path = Path(turbo_path_str)

        if turbo_path.exists():
            try:
                S = np.load(str(turbo_path))
                if S.shape == (d, d):
                    return S
                logger.warning("Turbo rotation shape %s != (%d,%d)", S.shape, d, d)
            except Exception as exc:
                logger.warning("Corrupt turbo rotation: %s", exc)

        # Copy-on-detect: reuse existing polar rotation matrix
        polar_path = slm_dir / f"polar_rotation_{d}.npy"
        if polar_path.exists() and not turbo_path.exists():
            try:
                S = np.load(str(polar_path))
                if S.shape == (d, d):
                    turbo_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(polar_path), str(turbo_path))
                    logger.info("Copied polar rotation matrix for TurboQuant compatibility")
                    return S
            except Exception as exc:
                logger.warning("Could not copy polar rotation: %s", exc)

        # Generate new via Mezzadri-corrected QR
        rng = np.random.default_rng(self._config.seed)
        H = rng.standard_normal((d, d))
        Q, R = np.linalg.qr(H)
        S = Q @ np.diag(np.sign(np.diag(R)))

        turbo_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(turbo_path), S)
        logger.info("Generated TurboQuant rotation (%d x %d) at %s", d, d, turbo_path)
        return S

    def _compute_codebooks(self) -> dict[int, NDArray]:
        """Pre-compute Lloyd-Max codebooks for 2/4/8-bit."""
        sigma = 1.0 / math.sqrt(self._d)
        codebooks: dict[int, NDArray] = {}
        for bw in sorted(SUPPORTED_BIT_WIDTHS):
            centroids = _compute_lloyd_max_gaussian(sigma, 2 ** bw)
            assert len(centroids) == 2 ** bw
            assert np.all(centroids[1:] >= centroids[:-1])
            codebooks[bw] = centroids
        return codebooks

    def encode(self, embedding: NDArray, bit_width: int = 4) -> TurboQuantResult:
        """Encode embedding. HR-ENC-01: pure. HR-ENC-02: radius=float16."""
        if bit_width not in SUPPORTED_BIT_WIDTHS:
            raise ValueError(f"bit_width must be 2, 4, or 8, got {bit_width}")
        if embedding.shape != (self._d,):
            raise ValueError(f"shape mismatch: expected ({self._d},), got {embedding.shape}")

        y = self._S @ embedding
        r = float(np.linalg.norm(y))

        if r < 1e-12:
            pack_fn, _ = _PACKERS[bit_width]
            packed = TQ_MAGIC + pack_fn(np.zeros(self._d, dtype=np.uint8))
            return TurboQuantResult(radius=0.0, indices=packed, bit_width=bit_width)

        y_unit = y / r
        centroids = self._codebooks[bit_width]
        idx = np.searchsorted(centroids, y_unit)
        idx = np.clip(idx, 0, len(centroids) - 1)
        left = np.clip(idx - 1, 0, len(centroids) - 1)
        use_left = np.abs(y_unit - centroids[left]) < np.abs(y_unit - centroids[idx])
        idx = np.where(use_left, left, idx).astype(np.uint8)

        pack_fn, _ = _PACKERS[bit_width]
        packed = TQ_MAGIC + pack_fn(idx)

        return TurboQuantResult(
            radius=float(np.float16(r)), indices=packed, bit_width=bit_width,
        )

    def decode(self, result: TurboQuantResult) -> NDArray:
        """Decode with format detection: TQ prefix -> turbo, else -> legacy polar."""
        blob = result.indices

        if blob[:2] == TQ_MAGIC:
            data = blob[2:]
        else:
            return self._decode_legacy_polar(result)

        _, unpack_fn = _PACKERS[result.bit_width]
        indices = unpack_fn(data, self._d)
        centroids = self._codebooks[result.bit_width]
        y_unit_approx = centroids[np.clip(indices, 0, len(centroids) - 1)]
        return self._S.T @ (y_unit_approx * result.radius)

    def _decode_legacy_polar(self, result: TurboQuantResult) -> NDArray:
        """Decode legacy PolarQuant BLOB (no TQ prefix) for SLM <= 3.3.6."""
        from superlocalmemory.math.polar_quant import PolarQuantEncoder, _polar_to_cartesian

        n_angles = self._d - 1
        if result.bit_width == 8:
            indices = np.frombuffer(result.indices, dtype=np.uint8).copy()
        elif result.bit_width == 4:
            indices = PolarQuantEncoder.unpack_4bit(result.indices, n_angles)
        else:
            indices = PolarQuantEncoder.unpack_2bit(result.indices, n_angles)

        levels = 2 ** result.bit_width
        boundaries = np.linspace(0.0, math.pi, levels + 1)
        centroids = (boundaries[:-1] + boundaries[1:]) / 2.0
        angles = centroids[np.clip(indices, 0, len(centroids) - 1)]

        v_unit = _polar_to_cartesian(angles, self._d)
        return self._S.T @ (v_unit * result.radius)

    def approximate_similarity(self, query: NDArray, result: TurboQuantResult) -> float:
        """Cosine similarity via decode. Returns 0.0 on degenerate inputs."""
        decoded = self.decode(result)
        denom = np.linalg.norm(query) * np.linalg.norm(decoded)
        if denom < 1e-12:
            return 0.0
        sim = float(np.dot(query, decoded) / denom)
        return 0.0 if (math.isnan(sim) or math.isinf(sim)) else sim

    # Static pack/unpack (backward compat with PolarQuantEncoder API)

    @staticmethod
    def pack_4bit(indices: NDArray) -> bytes:
        return _pack_4bit(indices)

    @staticmethod
    def unpack_4bit(data: bytes, length: int) -> NDArray:
        return _unpack_4bit(data, length)

    @staticmethod
    def pack_2bit(indices: NDArray) -> bytes:
        return _pack_2bit(indices)

    @staticmethod
    def unpack_2bit(data: bytes, length: int) -> NDArray:
        return _unpack_2bit(data, length)
