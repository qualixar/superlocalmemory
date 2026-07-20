# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""PolarQuant embedding quantization.

Implements random orthogonal rotation + recursive polar coordinate
transform + scalar quantization for ultra-compact embedding storage.

Pipeline:
  1. Random rotation (Mezzadri-corrected QR) -- preserves angles
  2. Cartesian -> hyperspherical polar coordinates
  3. Scalar quantization of angles (uniform codebook)
  4. Byte packing (8/4/2-bit)

Reconstruction:
  1. Unpack bytes -> indices
  2. Map indices -> centroid angles via codebook
  3. Polar -> Cartesian
  4. Inverse rotation (S^T since S is orthogonal)

References:
  - PolarQuant (arXiv 2502.02617)
  - TurboQuant (ICLR 2026, arXiv 2504.19874)
  - Mezzadri F (2007). How to generate random matrices from
    the classical compact groups.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from superlocalmemory.core.config import PolarQuantConfig
from superlocalmemory.infra.data_root import state_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantizedEmbedding:
    """Immutable container for a quantized embedding.

    Fields:
      fact_id:       Linked atomic fact (empty string during encode).
      radius:        float16-precision L2 norm (sole norm for storage + reconstruct).
      angle_indices: Packed quantized angle indices as bytes.
      bit_width:     Quantization level (2, 4, or 8).
      qjl_bits:      Optional QJL residual correction bits.
    """

    fact_id: str
    radius: float
    angle_indices: bytes
    bit_width: int
    qjl_bits: bytes | None


# ---------------------------------------------------------------------------
# PolarQuantEncoder
# ---------------------------------------------------------------------------


class PolarQuantEncoder:
    """Random-rotation + polar-coordinate embedding quantizer.

    HR-01: Rotation matrix generated ONCE and reused for ALL embeddings.
    HR-02: Same rotation matrix for all embeddings in a profile.
    HR-08: No new pip dependencies (numpy + stdlib only).
    HR-09: Angle indices as uint8, packed into bytes.
    """

    __slots__ = ("_config", "_d", "_S", "_codebooks", "_turbo", "_use_turbo")

    def __init__(self, config: PolarQuantConfig) -> None:
        self._config = config
        self._d = config.dimension
        codebook_method = getattr(config, "codebook_method", "turbo")
        if codebook_method == "turbo":
            from superlocalmemory.math.turbo_quant import TurboQuantEncoder
            self._turbo = TurboQuantEncoder(config)
            self._S = self._turbo._S
            self._codebooks = self._generate_uniform_codebooks()  # for legacy decode
            self._use_turbo = True
        else:
            self._turbo = None
            self._S = self._load_or_create_rotation_matrix()
            self._codebooks = self._generate_uniform_codebooks()
            self._use_turbo = False

    # -- Rotation matrix (HR-01, HR-02) ------------------------------------

    def _load_or_create_rotation_matrix(self) -> NDArray:
        """Load or create Mezzadri-corrected random orthogonal matrix.

        Mezzadri correction (B-CRIT-01):
          S = Q @ diag(sign(diag(R)))
        ensures UNIFORM sampling from O(d). Plain QR gives Haar-random
        only up to sign flips.
        """
        path_str = self._config.rotation_matrix_path
        if not path_str:
            path_str = str(state_path(f"polar_rotation_{self._d}.npy"))

        path = Path(path_str)

        if path.exists():
            try:
                S = np.load(str(path))
                if S.shape == (self._d, self._d):
                    return S
                logger.warning(
                    "Rotation matrix shape %s != expected (%d,%d), regenerating",
                    S.shape, self._d, self._d,
                )
            except Exception as exc:
                logger.warning("Corrupt rotation matrix, regenerating: %s", exc)

        # Generate new rotation matrix with Mezzadri correction
        rng = np.random.default_rng(self._config.seed)
        H = rng.standard_normal((self._d, self._d))
        Q, R = np.linalg.qr(H)
        # Mezzadri correction: ensures uniform sampling from O(d)
        S = Q @ np.diag(np.sign(np.diag(R)))

        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), S)
        logger.info("Generated rotation matrix (%d x %d) at %s", self._d, self._d, path)
        return S

    # -- Codebook generation -----------------------------------------------

    def _generate_uniform_codebooks(self) -> dict[int, dict[str, NDArray]]:
        """Generate uniform codebooks for 2/4/8-bit quantization.

        Uniform approximation is justified because at d=768, the
        Beta(d/2-k, 1/2) angle distribution concentrates around pi/2
        so tightly that uniform and Lloyd-Max converge (KL < 0.01 bits).
        """
        codebooks: dict[int, dict[str, NDArray]] = {}
        for bit_width in (2, 4, 8):
            levels = 2 ** bit_width
            boundaries = np.linspace(0.0, math.pi, levels + 1)
            centroids = (boundaries[:-1] + boundaries[1:]) / 2.0
            codebooks[bit_width] = {
                "boundaries": boundaries,
                "centroids": centroids,
            }
        return codebooks

    # -- Encode ------------------------------------------------------------

    def encode(self, embedding: NDArray, bit_width: int = 4) -> QuantizedEmbedding:
        """Encode a float32 embedding into quantized representation.

        Args:
            embedding: 1-D float vector of dimension self._d.
            bit_width:  2, 4, or 8.

        Returns:
            QuantizedEmbedding with packed indices.

        Raises:
            ValueError: Invalid bit_width or dimension mismatch.
        """
        if bit_width not in (2, 4, 8):
            raise ValueError(
                f"bit_width must be 2, 4, or 8, got {bit_width}"
            )
        if embedding.shape != (self._d,):
            raise ValueError(
                f"shape mismatch: expected ({self._d},), got {embedding.shape}"
            )

        # V3.3.8: TurboQuant path (default)
        if self._use_turbo:
            result = self._turbo.encode(embedding, bit_width)
            return QuantizedEmbedding(
                fact_id="",
                radius=result.radius,
                angle_indices=result.indices,
                bit_width=result.bit_width,
                qjl_bits=None,
            )

        # Legacy PolarQuant path
        return self._encode_polar(embedding, bit_width)

    def _encode_polar(self, embedding: NDArray, bit_width: int) -> QuantizedEmbedding:
        """Legacy PolarQuant encode (polar coordinate transform)."""
        v_rot = self._S @ embedding
        r = float(np.linalg.norm(v_rot))

        if r < 1e-12:
            zero_angles = np.zeros(self._d - 1, dtype=np.uint8)
            if bit_width == 8:
                packed = zero_angles.tobytes()
            elif bit_width == 4:
                packed = self.pack_4bit(zero_angles)
            else:
                packed = self.pack_2bit(zero_angles)
            return QuantizedEmbedding(
                fact_id="",
                radius=0.0,
                angle_indices=packed,
                bit_width=bit_width,
                qjl_bits=None,
            )

        v_unit = v_rot / r
        angles = _cartesian_to_polar_angles(v_unit)
        cb = self._codebooks[bit_width]
        indices = np.digitize(angles, cb["boundaries"][1:-1]).astype(np.uint8)

        if bit_width == 8:
            packed = indices.tobytes()
        elif bit_width == 4:
            packed = self.pack_4bit(indices)
        else:
            packed = self.pack_2bit(indices)

        return QuantizedEmbedding(
            fact_id="",
            radius=float(np.float16(r)),
            angle_indices=packed,
            bit_width=bit_width,
            qjl_bits=None,
        )

    # -- Decode ------------------------------------------------------------

    # TQ magic prefix for format detection (HR-MIG-02)
    _TQ_MAGIC = b"\x54\x51"

    def decode(self, qe: QuantizedEmbedding) -> NDArray:
        """Decode a QuantizedEmbedding back to float64 vector.

        V3.3.8: Detects "TQ" prefix (0x54, 0x51) to route between
        TurboQuant and legacy PolarQuant decode paths.

        Args:
            qe: Quantized embedding produced by encode().

        Returns:
            Reconstructed vector of dimension self._d.
        """
        # Format detection: TQ prefix = TurboQuant, else legacy polar
        if qe.angle_indices[:2] == self._TQ_MAGIC:
            return self._decode_turbo(qe)
        return self._decode_polar(qe)

    def _decode_turbo(self, qe: QuantizedEmbedding) -> NDArray:
        """Decode TurboQuant-encoded BLOB (has TQ prefix)."""
        if self._turbo is None:
            from superlocalmemory.math.turbo_quant import TurboQuantEncoder
            self._turbo = TurboQuantEncoder(self._config)
        from superlocalmemory.math.turbo_quant import TurboQuantResult
        result = TurboQuantResult(
            radius=qe.radius,
            indices=qe.angle_indices,
            bit_width=qe.bit_width,
        )
        return self._turbo.decode(result)

    def _decode_polar(self, qe: QuantizedEmbedding) -> NDArray:
        """Decode legacy PolarQuant BLOB (no TQ prefix)."""
        n_angles = self._d - 1

        if qe.bit_width == 8:
            indices = np.frombuffer(qe.angle_indices, dtype=np.uint8).copy()
        elif qe.bit_width == 4:
            indices = self.unpack_4bit(qe.angle_indices, n_angles)
        else:
            indices = self.unpack_2bit(qe.angle_indices, n_angles)

        centroids = self._codebooks[qe.bit_width]["centroids"]
        indices = np.clip(indices, 0, len(centroids) - 1)
        angles = centroids[indices]

        v_unit = _polar_to_cartesian(angles, self._d)
        v_rot = v_unit * qe.radius
        v_orig = self._S.T @ v_rot

        return v_orig

    # -- Similarity --------------------------------------------------------

    def approximate_similarity(
        self, query: NDArray, qe: QuantizedEmbedding,
    ) -> float:
        """Compute approximate cosine similarity via decode.

        Args:
            query: Query vector (float32/64).
            qe:    Quantized embedding.

        Returns:
            Cosine similarity in [-1, 1]. Returns 0.0 on degenerate inputs.
        """
        v_decoded = self.decode(qe)
        denom = np.linalg.norm(query) * np.linalg.norm(v_decoded)
        if denom < 1e-12:
            return 0.0
        sim = float(np.dot(query, v_decoded) / denom)
        # NaN guard
        if math.isnan(sim) or math.isinf(sim):
            return 0.0
        return sim

    # -- Bit packing (static methods) --------------------------------------

    @staticmethod
    def pack_4bit(indices: NDArray) -> bytes:
        """Pack uint8 indices (0-15) into 4-bit pairs.

        Two indices per byte: high nibble | low nibble.
        Pads to even length if needed.
        """
        n = len(indices)
        padded = np.zeros(n + (n % 2), dtype=np.uint8)
        padded[:n] = np.clip(indices, 0, 15)
        packed = (padded[0::2] << 4) | padded[1::2]
        return packed.tobytes()

    @staticmethod
    def unpack_4bit(data: bytes, length: int) -> NDArray:
        """Unpack 4-bit pairs back to uint8 indices."""
        packed = np.frombuffer(data, dtype=np.uint8)
        high = packed >> 4
        low = packed & 0x0F
        result = np.empty(len(packed) * 2, dtype=np.uint8)
        result[0::2] = high
        result[1::2] = low
        return result[:length]

    @staticmethod
    def pack_2bit(indices: NDArray) -> bytes:
        """Pack uint8 indices (0-3) into 2-bit quads.

        Four indices per byte: [b7b6 | b5b4 | b3b2 | b1b0].
        Pads to multiple of 4 if needed.
        """
        n = len(indices)
        pad_len = (4 - n % 4) % 4
        padded = np.zeros(n + pad_len, dtype=np.uint8)
        padded[:n] = np.clip(indices, 0, 3)
        packed = (
            (padded[0::4] << 6)
            | (padded[1::4] << 4)
            | (padded[2::4] << 2)
            | padded[3::4]
        )
        return packed.tobytes()

    @staticmethod
    def unpack_2bit(data: bytes, length: int) -> NDArray:
        """Unpack 2-bit quads back to uint8 indices."""
        packed = np.frombuffer(data, dtype=np.uint8)
        result = np.empty(len(packed) * 4, dtype=np.uint8)
        result[0::4] = (packed >> 6) & 0x03
        result[1::4] = (packed >> 4) & 0x03
        result[2::4] = (packed >> 2) & 0x03
        result[3::4] = packed & 0x03
        return result[:length]


# ---------------------------------------------------------------------------
# Coordinate conversion helpers (module-level for reuse)
# ---------------------------------------------------------------------------


def _cartesian_to_polar_angles(v_unit: NDArray) -> NDArray:
    """Convert unit vector to d-1 polar angles. O(d) time.

    Uses the recursive polar decomposition:
      v[i] = cos(theta_i) * product(sin(theta_j) for j < i)
    """
    d = len(v_unit)
    angles = np.empty(d - 1)
    for i in range(d - 1):
        remaining_norm = float(np.linalg.norm(v_unit[i:]))
        if remaining_norm < 1e-12:
            angles[i:] = math.pi / 2
            break
        angles[i] = math.acos(np.clip(v_unit[i] / remaining_norm, -1.0, 1.0))
    return angles


def _polar_to_cartesian(angles: NDArray, d: int) -> NDArray:
    """Convert d-1 polar angles to d-dimensional unit vector.

    Inverse of _cartesian_to_polar_angles.
    """
    v = np.empty(d)
    sin_product = 1.0
    for i in range(d - 1):
        v[i] = math.cos(angles[i]) * sin_product
        sin_product *= math.sin(angles[i])
    v[d - 1] = sin_product
    return v
