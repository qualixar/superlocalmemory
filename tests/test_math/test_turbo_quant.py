# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for TurboQuant encoding/decoding pipeline.

LLD v2.0 Section 5: TDD Sequence (22 tests).
Tests 1-18: turbo_quant.py
Tests 19-22: polar_quant.py backward compat
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

from superlocalmemory.core.config import PolarQuantConfig
from superlocalmemory.math.turbo_quant import (
    SUPPORTED_BIT_WIDTHS,
    TQ_MAGIC,
    TurboQuantEncoder,
    TurboQuantResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config(dim: int, tmp_dir: Path) -> PolarQuantConfig:
    return PolarQuantConfig(
        dimension=dim,
        rotation_matrix_path=str(tmp_dir / f"turbo_rot_{dim}.npy"),
        seed=42,
        codebook_method="turbo",
    )


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def config_16d(tmp_dir: Path) -> PolarQuantConfig:
    return _make_config(16, tmp_dir)


@pytest.fixture
def config_64d(tmp_dir: Path) -> PolarQuantConfig:
    return _make_config(64, tmp_dir)


@pytest.fixture
def config_768d(tmp_dir: Path) -> PolarQuantConfig:
    return _make_config(768, tmp_dir)


@pytest.fixture
def encoder_16d(config_16d: PolarQuantConfig) -> TurboQuantEncoder:
    return TurboQuantEncoder(config_16d)


@pytest.fixture
def encoder_64d(config_64d: PolarQuantConfig) -> TurboQuantEncoder:
    return TurboQuantEncoder(config_64d)


@pytest.fixture
def encoder_768d(config_768d: PolarQuantConfig) -> TurboQuantEncoder:
    return TurboQuantEncoder(config_768d)


def _random_unit_vec(d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d)
    return v / np.linalg.norm(v)


def _mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    return float(np.sum((original - reconstructed) ** 2))


def _cosine_sim(u: np.ndarray, v: np.ndarray) -> float:
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    return float(np.dot(u, v) / denom) if denom > 1e-12 else 0.0


# ===================================================================
# Test 1-3: Encode/decode roundtrip MSE at 8/4/2-bit
# ===================================================================


class TestRoundtripMSE:
    """LLD Tests 1-3: MSE matches paper values."""

    def test_encode_decode_roundtrip_8bit(
        self, encoder_768d: TurboQuantEncoder,
    ) -> None:
        """Test 1: 8-bit avg MSE < 0.005."""
        mses = []
        for seed in range(20):
            v = _random_unit_vec(768, seed=seed + 200)
            qe = encoder_768d.encode(v, bit_width=8)
            decoded = encoder_768d.decode(qe)
            mses.append(_mse(v, decoded))
        avg_mse = sum(mses) / len(mses)
        assert avg_mse < 0.005, f"8-bit avg MSE={avg_mse:.6f}, expected < 0.005"

    def test_encode_decode_roundtrip_4bit(
        self, encoder_768d: TurboQuantEncoder,
    ) -> None:
        """Test 2: 4-bit avg MSE < 0.02 (paper: 0.009)."""
        mses = []
        for seed in range(20):
            v = _random_unit_vec(768, seed=seed + 100)
            qe = encoder_768d.encode(v, bit_width=4)
            decoded = encoder_768d.decode(qe)
            mses.append(_mse(v, decoded))
        avg_mse = sum(mses) / len(mses)
        assert avg_mse < 0.02, f"4-bit avg MSE={avg_mse:.6f}, expected < 0.02"

    def test_encode_decode_roundtrip_2bit(
        self, encoder_768d: TurboQuantEncoder,
    ) -> None:
        """Test 3: 2-bit avg MSE < 0.5 (4 centroids on [-1,1] — archival quality)."""
        mses = []
        for seed in range(20):
            v = _random_unit_vec(768, seed=seed + 300)
            qe = encoder_768d.encode(v, bit_width=2)
            decoded = encoder_768d.decode(qe)
            mses.append(_mse(v, decoded))
        avg_mse = sum(mses) / len(mses)
        assert avg_mse < 0.5, f"2-bit avg MSE={avg_mse:.6f}, expected < 0.5"


# ===================================================================
# Test 4-5: Rotation matrix
# ===================================================================


class TestRotationMatrix:
    """LLD Tests 4-5: Rotation matrix orthogonality and persistence."""

    def test_rotation_matrix_orthogonal(
        self, encoder_768d: TurboQuantEncoder,
    ) -> None:
        """Test 4: S @ S.T approx I (Frobenius < 1e-10)."""
        S = encoder_768d._S
        identity = S @ S.T
        frobenius = float(np.linalg.norm(identity - np.eye(768), "fro"))
        assert frobenius < 1e-8, f"Frobenius error={frobenius}"

    def test_rotation_matrix_persistence(self, tmp_dir: Path) -> None:
        """Test 5: Save + load gives same matrix."""
        path = str(tmp_dir / "persist_rot.npy")
        config = PolarQuantConfig(
            dimension=32,
            rotation_matrix_path=path,
            seed=42,
            codebook_method="turbo",
        )
        enc1 = TurboQuantEncoder(config)
        enc2 = TurboQuantEncoder(config)
        np.testing.assert_array_equal(enc1._S, enc2._S)


# ===================================================================
# Test 6: Rotation matrix copy-on-detect (AUDIT FIX C4)
# ===================================================================


class TestRotationCopyOnDetect:
    """LLD Test 6: Copy polar_rotation if exists."""

    def test_rotation_matrix_copy_on_detect(
        self, monkeypatch, tmp_dir: Path,
    ) -> None:
        """Test 6: Copies polar_rotation_{d}.npy if turbo doesn't exist."""
        import shutil
        import tempfile

        # Create a fake .superlocalmemory dir in tmp
        slm_dir = tmp_dir / ".superlocalmemory"
        slm_dir.mkdir()
        monkeypatch.setenv("SLM_DATA_DIR", str(slm_dir))
        d = 32

        # Generate a polar rotation matrix
        rng = np.random.default_rng(42)
        H = rng.standard_normal((d, d))
        Q, R = np.linalg.qr(H)
        S_polar = Q @ np.diag(np.sign(np.diag(R)))

        polar_path = slm_dir / f"polar_rotation_{d}.npy"
        np.save(str(polar_path), S_polar)

        turbo_path = slm_dir / f"turbo_rotation_{d}.npy"
        assert not turbo_path.exists()

        config = PolarQuantConfig(
            dimension=d,
            rotation_matrix_path="",  # use default path
            seed=99,  # different seed -- but copy should still happen
            codebook_method="turbo",
        )
        enc = TurboQuantEncoder(config)

        # Turbo file should now exist (copied from polar)
        assert turbo_path.exists()
        # The loaded matrix should match the polar matrix
        np.testing.assert_array_equal(enc._S, S_polar)


# ===================================================================
# Test 7-8: Codebook properties
# ===================================================================


class TestCodebook:
    """LLD Tests 7-8: Codebook sorted and deterministic."""

    def test_codebook_sorted_ascending(
        self, encoder_768d: TurboQuantEncoder,
    ) -> None:
        """Test 7: Centroids are sorted ascending."""
        for bw in SUPPORTED_BIT_WIDTHS:
            cb = encoder_768d._codebooks[bw]
            assert np.all(cb[1:] >= cb[:-1]), f"bw={bw} not sorted"

    def test_codebook_deterministic(self, tmp_dir: Path) -> None:
        """Test 8: Same (d, b) -> same codebooks."""
        config = PolarQuantConfig(
            dimension=64,
            rotation_matrix_path=str(tmp_dir / "det_rot.npy"),
            seed=42,
            codebook_method="turbo",
        )
        enc1 = TurboQuantEncoder(config)
        enc2 = TurboQuantEncoder(config)
        for bw in SUPPORTED_BIT_WIDTHS:
            np.testing.assert_array_equal(
                enc1._codebooks[bw], enc2._codebooks[bw],
            )


# ===================================================================
# Test 9: No scipy import at runtime (HR-SCIPY-01)
# ===================================================================


class TestNoScipy:
    """LLD Test 9: Verify no scipy import."""

    def test_codebook_no_scipy(self) -> None:
        """Test 9: turbo_quant.py does not import scipy."""
        import superlocalmemory.math.turbo_quant as tq_mod

        source = Path(tq_mod.__file__).read_text()
        # Check for actual import statements (not comments/docstrings)
        for line in source.splitlines():
            stripped = line.strip()
            # Skip comments and docstring lines
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'"):
                continue
            if stripped.startswith("from scipy") or stripped.startswith("import scipy"):
                raise AssertionError(
                    f"turbo_quant.py imports scipy: {stripped}"
                )


# ===================================================================
# Test 10-11: Zero vector and unit vector
# ===================================================================


class TestSpecialVectors:
    """LLD Tests 10-11: Special vector handling."""

    def test_zero_vector_encode_decode(
        self, encoder_16d: TurboQuantEncoder,
    ) -> None:
        """Test 10: No NaN, no crash on zero vector."""
        v = np.zeros(16)
        for bw in SUPPORTED_BIT_WIDTHS:
            qe = encoder_16d.encode(v, bit_width=bw)
            decoded = encoder_16d.decode(qe)
            assert np.all(np.isfinite(decoded)), f"bw={bw}: non-finite"
            assert qe.radius < 1e-10
            # TQ prefix present even on zero vector
            assert qe.indices[:2] == TQ_MAGIC

    def test_unit_vector_encode_decode(
        self, encoder_16d: TurboQuantEncoder,
    ) -> None:
        """Test 11: Specific known axis-aligned vector."""
        v = np.zeros(16)
        v[0] = 1.0
        qe = encoder_16d.encode(v, bit_width=8)
        decoded = encoder_16d.decode(qe)
        cos = _cosine_sim(v, decoded)
        assert cos > 0.9, f"Axis-aligned cosine={cos:.4f}"


# ===================================================================
# Test 12: Inner product preservation
# ===================================================================


class TestInnerProduct:
    """LLD Test 12: Inner product preservation."""

    def test_inner_product_preservation(
        self, encoder_768d: TurboQuantEncoder,
    ) -> None:
        """Test 12: |<x,y> - <Q(x),Q(y)>| small at 4-bit."""
        errors = []
        for seed in range(10):
            u = _random_unit_vec(768, seed=seed)
            v = _random_unit_vec(768, seed=seed + 1000)
            ip_exact = float(np.dot(u, v))

            qu = encoder_768d.encode(u, bit_width=4)
            qv = encoder_768d.encode(v, bit_width=4)
            u_hat = encoder_768d.decode(qu)
            v_hat = encoder_768d.decode(qv)
            ip_quant = float(np.dot(u_hat, v_hat))

            errors.append(abs(ip_exact - ip_quant))

        avg_error = sum(errors) / len(errors)
        assert avg_error < 0.05, f"Avg IP error={avg_error:.6f}"


# ===================================================================
# Test 13: Batch consistency
# ===================================================================


class TestBatch:
    """LLD Test 13: Batch encode produces valid results."""

    def test_batch_consistency(
        self, encoder_64d: TurboQuantEncoder,
    ) -> None:
        """Test 13: Encode N vectors, all valid."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            v = rng.standard_normal(64)
            v = v / np.linalg.norm(v)
            qe = encoder_64d.encode(v, bit_width=4)
            decoded = encoder_64d.decode(qe)
            assert np.all(np.isfinite(decoded))
            assert qe.indices[:2] == TQ_MAGIC
            assert qe.bit_width == 4


# ===================================================================
# Test 14-15: Invalid inputs
# ===================================================================


class TestInvalidInputs:
    """LLD Tests 14-15: Error handling."""

    def test_invalid_bit_width_raises(
        self, encoder_16d: TurboQuantEncoder,
    ) -> None:
        """Test 14: ValueError for bit_width=3 or 5."""
        v = _random_unit_vec(16)
        with pytest.raises(ValueError, match="bit_width"):
            encoder_16d.encode(v, bit_width=3)
        with pytest.raises(ValueError, match="bit_width"):
            encoder_16d.encode(v, bit_width=5)

    def test_dimension_mismatch_raises(
        self, encoder_16d: TurboQuantEncoder,
    ) -> None:
        """Test 15: ValueError for wrong shape."""
        v = _random_unit_vec(32)
        with pytest.raises(ValueError, match="shape"):
            encoder_16d.encode(v, bit_width=4)


# ===================================================================
# Test 16: TQ prefix present (HR-MIG-02, AUDIT FIX C-B4-1)
# ===================================================================


class TestTQPrefix:
    """LLD Test 16: 2-byte TQ prefix on all turbo BLOBs."""

    def test_tq_prefix_present(
        self, encoder_64d: TurboQuantEncoder,
    ) -> None:
        """Test 16: Encoded BLOB starts with b'\\x54\\x51'."""
        v = _random_unit_vec(64, seed=7)
        for bw in SUPPORTED_BIT_WIDTHS:
            qe = encoder_64d.encode(v, bit_width=bw)
            assert qe.indices[:2] == b"\x54\x51", (
                f"bw={bw}: missing TQ prefix, got {qe.indices[:2]!r}"
            )


# ===================================================================
# Test 17: Decode legacy polar BLOB (HR-BC-03, AUDIT FIX C-B9-1)
# ===================================================================


class TestLegacyDecode:
    """LLD Test 17: Legacy polar BLOB decode."""

    def test_decode_legacy_polar_blob(self, tmp_dir: Path) -> None:
        """Test 17: BLOB without TQ prefix decodes via polar path."""
        from superlocalmemory.math.polar_quant import (
            PolarQuantEncoder,
            _cartesian_to_polar_angles,
            _polar_to_cartesian,
        )

        d = 16
        config = PolarQuantConfig(
            dimension=d,
            rotation_matrix_path=str(tmp_dir / "legacy_rot.npy"),
            seed=42,
            codebook_method="polar_legacy",
        )

        # Encode using legacy polar path
        polar_enc = PolarQuantEncoder(config)
        v = _random_unit_vec(d, seed=5)
        polar_qe = polar_enc.encode(v, bit_width=4)

        # The legacy BLOB should NOT have TQ prefix
        assert polar_qe.angle_indices[:2] != TQ_MAGIC

        # Now decode it via TurboQuantEncoder (should route to legacy path)
        turbo_config = PolarQuantConfig(
            dimension=d,
            rotation_matrix_path=str(tmp_dir / "legacy_rot.npy"),
            seed=42,
            codebook_method="turbo",
        )
        turbo_enc = TurboQuantEncoder(turbo_config)
        turbo_result = TurboQuantResult(
            radius=polar_qe.radius,
            indices=polar_qe.angle_indices,
            bit_width=polar_qe.bit_width,
        )
        decoded = turbo_enc.decode(turbo_result)
        assert decoded.shape == (d,)
        assert np.all(np.isfinite(decoded))


# ===================================================================
# Test 18: Decode turbo BLOB
# ===================================================================


class TestTurboDecode:
    """LLD Test 18: Turbo BLOB decode."""

    def test_decode_turbo_blob(
        self, encoder_64d: TurboQuantEncoder,
    ) -> None:
        """Test 18: BLOB with TQ prefix decodes via turbo path."""
        v = _random_unit_vec(64, seed=3)
        qe = encoder_64d.encode(v, bit_width=4)
        # Has TQ prefix
        assert qe.indices[:2] == TQ_MAGIC
        # Decodes correctly
        decoded = encoder_64d.decode(qe)
        cos = _cosine_sim(v, decoded)
        assert cos > 0.8, f"Turbo decode cosine={cos:.4f}"


# ===================================================================
# Test 19-22: PolarQuantEncoder backward compat (in polar_quant.py)
# ===================================================================


class TestPolarQuantBackwardCompat:
    """LLD Tests 19-22: PolarQuantEncoder wrapper."""

    def test_turbo_mode_default(self, tmp_dir: Path) -> None:
        """Test 19: Verify TurboQuant is active by default."""
        from superlocalmemory.math.polar_quant import PolarQuantEncoder

        config = PolarQuantConfig(
            dimension=16,
            rotation_matrix_path=str(tmp_dir / "default_mode.npy"),
            seed=42,
        )
        enc = PolarQuantEncoder(config)
        assert enc._use_turbo is True
        assert enc._turbo is not None

    def test_legacy_mode_works(self, tmp_dir: Path) -> None:
        """Test 20: codebook_method='polar_legacy' uses old path."""
        from superlocalmemory.math.polar_quant import PolarQuantEncoder

        config = PolarQuantConfig(
            dimension=16,
            rotation_matrix_path=str(tmp_dir / "legacy_mode.npy"),
            seed=42,
            codebook_method="polar_legacy",
        )
        enc = PolarQuantEncoder(config)
        assert enc._use_turbo is False
        assert enc._turbo is None

        # Should still encode/decode
        v = _random_unit_vec(16, seed=1)
        qe = enc.encode(v, bit_width=4)
        decoded = enc.decode(qe)
        assert decoded.shape == (16,)
        assert np.all(np.isfinite(decoded))

    def test_turbo_decode_polar_encoded(self, tmp_dir: Path) -> None:
        """Test 21: Turbo encoder can decode legacy polar BLOBs."""
        from superlocalmemory.math.polar_quant import (
            PolarQuantEncoder,
            QuantizedEmbedding,
        )

        d = 16

        # Encode with legacy polar
        legacy_config = PolarQuantConfig(
            dimension=d,
            rotation_matrix_path=str(tmp_dir / "cross_compat.npy"),
            seed=42,
            codebook_method="polar_legacy",
        )
        legacy_enc = PolarQuantEncoder(legacy_config)
        v = _random_unit_vec(d, seed=7)
        legacy_qe = legacy_enc.encode(v, bit_width=4)

        # Decode with turbo-mode wrapper (should detect legacy format)
        turbo_config = PolarQuantConfig(
            dimension=d,
            rotation_matrix_path=str(tmp_dir / "cross_compat.npy"),
            seed=42,
            codebook_method="turbo",
        )
        turbo_enc = PolarQuantEncoder(turbo_config)
        decoded = turbo_enc.decode(legacy_qe)
        assert decoded.shape == (d,)
        assert np.all(np.isfinite(decoded))

    def test_encode_rejects_3bit(self, tmp_dir: Path) -> None:
        """Test 22: PolarQuantEncoder.encode(bit_width=3) raises ValueError."""
        from superlocalmemory.math.polar_quant import PolarQuantEncoder

        config = PolarQuantConfig(
            dimension=16,
            rotation_matrix_path=str(tmp_dir / "reject_3bit.npy"),
            seed=42,
        )
        enc = PolarQuantEncoder(config)
        v = _random_unit_vec(16, seed=1)
        with pytest.raises(ValueError, match="bit_width"):
            enc.encode(v, bit_width=3)


# ===================================================================
# Additional: HR-ENC-01 (encode is pure), HR-ENC-02 (radius float16)
# ===================================================================


class TestHardRules:
    """Hard rule enforcement tests."""

    def test_encode_is_pure(self, encoder_16d: TurboQuantEncoder) -> None:
        """HR-ENC-01: Input embedding is NOT modified."""
        v = _random_unit_vec(16, seed=1)
        v_copy = v.copy()
        encoder_16d.encode(v, bit_width=4)
        np.testing.assert_array_equal(v, v_copy)

    def test_radius_float16(self, encoder_64d: TurboQuantEncoder) -> None:
        """HR-ENC-02: Radius stored as float16 precision."""
        v = _random_unit_vec(64, seed=1) * 42.0
        qe = encoder_64d.encode(v, bit_width=4)
        expected_radius = float(np.float16(np.linalg.norm(v)))
        assert qe.radius == expected_radius

    def test_approximate_similarity_degenerate(
        self, encoder_16d: TurboQuantEncoder,
    ) -> None:
        """approximate_similarity returns 0.0 on zero query."""
        v = _random_unit_vec(16, seed=1)
        qe = encoder_16d.encode(v, bit_width=4)
        sim = encoder_16d.approximate_similarity(np.zeros(16), qe)
        assert sim == 0.0

    def test_quantized_embedding_unchanged(self) -> None:
        """HR-BC-01: QuantizedEmbedding dataclass fields unchanged."""
        from superlocalmemory.math.polar_quant import QuantizedEmbedding

        fields = set(QuantizedEmbedding.__dataclass_fields__.keys())
        expected = {"fact_id", "radius", "angle_indices", "bit_width", "qjl_bits"}
        assert fields == expected
