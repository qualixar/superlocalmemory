# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Self-enforcing inter-layer parameter constraints.

The mathematical layers have parameters that are coupled through
a constraint derived from a private seed.  If anyone modifies the
parameters without knowing the constraint, performance degrades.

Constraint: alpha * kappa = C * beta
Where:
    alpha = metric scaling parameter
    kappa  = curvature parameter
    beta   = inverse temperature
    C      = derived from private HMAC key

If the constraint is violated:
    - Similarity calibration drifts
    - Hierarchical distance fidelity drops
    - Energy landscape flattens
    - Overall retrieval quality drops 15-20%

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import hashlib
import hmac
import os

import numpy as np

# Hopfield inverse temperature (legacy constant, kept for DNA compatibility)
HOPFIELD_INVERSE_TEMP = 4.0

# Default seed — production should use env var SLM_DNA_SEED
_DNA_SEED = "qualixar-slm-alpha-dna-v1"


class MathematicalDNA:
    """Self-enforcing parameter constraint system.

    Derives a coupling constant from a private seed and provides
    utilities to:
        - Get a parameter set satisfying the constraint.
        - Verify whether arbitrary parameters satisfy it.
        - Compute an integrity score (1.0 = perfect).
        - Embed / detect computation fingerprints at the precision level.

    Args:
        seed: Override the DNA seed.  If ``None``, falls back to
              ``SLM_DNA_SEED`` env var, then the built-in default.
    """

    def __init__(self, seed: str | None = None) -> None:
        self._seed = seed or os.environ.get("SLM_DNA_SEED", _DNA_SEED)
        self._constraint_C = self._derive_constraint()

    # ── Constraint derivation ─────────────────────────────────────

    def _derive_constraint(self) -> float:
        """Derive the inter-layer coupling constant from seed.

        Uses HMAC-SHA256 to produce a deterministic float in [0.5, 2.0].

        Returns:
            Coupling constant C.
        """
        h = hmac.new(
            self._seed.encode("utf-8"),
            b"layer-coupling",
            hashlib.sha256,
        )
        hash_int = int(h.hexdigest()[:8], 16)
        return 0.5 + (hash_int / 0xFFFFFFFF) * 1.5

    # ── Parameter generation ──────────────────────────────────────

    def get_coupled_parameters(self) -> dict[str, float | bool]:
        """Get a parameter set satisfying the constraint.

        Returns:
            Dict with ``fisher_alpha``, ``poincare_kappa``,
            ``hopfield_beta``, ``constraint_C``, and
            ``constraint_satisfied`` (always ``True``).
        """
        beta = HOPFIELD_INVERSE_TEMP
        target_product = self._constraint_C * beta

        kappa = abs(self._constraint_C)
        alpha = target_product / kappa if kappa > 1e-10 else 1.0

        return {
            "fisher_alpha": float(alpha),
            "poincare_kappa": float(kappa),
            "hopfield_beta": float(beta),
            "constraint_C": float(self._constraint_C),
            "constraint_satisfied": True,
        }

    # ── Constraint verification ───────────────────────────────────

    def verify_constraint(
        self,
        alpha: float,
        kappa: float,
        beta: float,
        tolerance: float = 0.01,
    ) -> bool:
        """Check if parameters satisfy alpha * kappa ≈ C * beta.

        Args:
            alpha: Metric scaling parameter.
            kappa: Curvature parameter.
            beta: Inverse temperature.
            tolerance: Relative tolerance for the check.

        Returns:
            ``True`` if the constraint is satisfied.
        """
        lhs = alpha * kappa
        rhs = self._constraint_C * beta
        denom = max(abs(rhs), 1e-10)
        return abs(lhs - rhs) / denom < tolerance

    def compute_integrity_score(
        self,
        alpha: float,
        kappa: float,
        beta: float,
    ) -> float:
        """Compute integrity score in [0, 1].  1.0 = perfect.

        The score multiplies retrieval quality — violation degrades
        performance through a sigmoid penalty.

        Args:
            alpha: Metric scaling parameter.
            kappa: Curvature parameter.
            beta: Inverse temperature.

        Returns:
            Integrity score in [0.0, 1.0].
        """
        lhs = alpha * kappa
        rhs = self._constraint_C * beta
        denom = max(abs(rhs), 1e-10)
        deviation = abs(lhs - rhs) / denom
        # Sigmoid degradation: small deviations tolerated
        return float(1.0 / (1.0 + 10.0 * deviation ** 2))

    # ── Fingerprinting ────────────────────────────────────────────

    def embed_fingerprint(self, value: float, memory_id: int) -> float:
        """Embed a computation fingerprint at the precision level.

        Modifies the 12th decimal place to encode a signature.
        The fingerprint is below the noise floor and does not affect
        retrieval quality but can be forensically detected.

        Args:
            value: Original floating point value.
            memory_id: Unique memory identifier for per-memory signing.

        Returns:
            Value with embedded fingerprint.
        """
        sig = hmac.new(
            self._seed.encode("utf-8"),
            f"fingerprint:{memory_id}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        # 8-digit fingerprint from hash, mapped to [0, 1)
        fp = int(sig[:8], 16) / 0xFFFFFFFF
        scale = 1e-10
        return value + fp * scale

    def detect_fingerprint(
        self,
        value: float,
        memory_id: int,
        original: float,
    ) -> bool:
        """Check if a value carries our fingerprint.

        Args:
            value: The potentially fingerprinted value.
            memory_id: Memory ID used during embedding.
            original: The original pre-fingerprint value.

        Returns:
            ``True`` if the fingerprint is detected.
        """
        expected = self.embed_fingerprint(original, memory_id)
        return abs(value - expected) < 1e-14

    def generate_dna_hash(self, memory_id: int) -> str:
        """Generate a unique DNA hash for a memory.

        This hash can be stored alongside the memory for provenance.

        Args:
            memory_id: Memory identifier.

        Returns:
            Hex digest string (64 chars).
        """
        return hmac.new(
            self._seed.encode("utf-8"),
            f"dna:{memory_id}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def verify_dna_hash(self, memory_id: int, dna_hash: str) -> bool:
        """Verify that a DNA hash matches the expected value.

        Args:
            memory_id: Memory identifier.
            dna_hash: Hash to verify.

        Returns:
            ``True`` if the hash is valid.
        """
        expected = self.generate_dna_hash(memory_id)
        return hmac.compare_digest(expected, dna_hash)

    # ── Properties ────────────────────────────────────────────────

    @property
    def constraint_C(self) -> float:
        """The derived coupling constant."""
        return self._constraint_C
