# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Riemannian Langevin dynamics with persistence for memory lifecycle.

Evolves memory positions via a discretised Langevin SDE driven by an
access-aware potential function.  Frequently accessed memories are
confined near the origin (ACTIVE, high retrieval weight) by a strong
potential, while neglected memories diffuse toward the boundary
(ARCHIVED, weight approaches 0) as their confinement weakens.

Core SDE (Girolami & Calderhead 2011):

    xi_{t+1} = xi_t - g^{-1}(xi_t) * grad U(xi_t) * dt
             + sqrt(2 * T * dt) * g^{-1/2}(xi_t) * eta

Potential function:

    U(xi) = alpha * ||xi||^2
           - beta  * log(access_count + 1)
           + gamma * age_days
           - delta * importance

    - Accessed memories:  beta term lowers potential  -> drift to origin
    - Old / unused:       gamma term raises potential -> drift to boundary
    - Important memories: delta term lowers potential -> retain near origin

V1 bugs fixed in Innovation Wave 4:
    1. Positions were computed per-recall then DISCARDED.  Now ``step()``
       and ``batch_step()`` return new positions for the caller to persist.
    2. Weight range was [0.7, 1.0] --- too narrow to change rankings.
       Now [0.0, 1.0] so archived memories can be effectively suppressed.

References:
    Girolami M & Calderhead B (2011). Riemann manifold Langevin and
        Hamiltonian Monte Carlo methods. JRSS-B, 73(2), 123--214.
    Roberts G O & Tweedie R L (1996). Exponential convergence of Langevin
        distributions. Bernoulli, 2(4), 341--363.
    Xifara T, Sherlock C, Livingstone S, Byrne S & Girolami M (2014).
        Langevin diffusions and the MALA algorithm. Statistics &
        Probability Letters, 91, 59--67.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from superlocalmemory.storage.models import MemoryLifecycle

# ---------------------------------------------------------------------------
# Lifecycle zone boundaries (radius thresholds on the unit ball)
# ---------------------------------------------------------------------------

_RADIUS_ACTIVE: float = 0.3    # [0, 0.3)    -> ACTIVE
_RADIUS_WARM: float = 0.55     # [0.3, 0.55) -> WARM
_RADIUS_COLD: float = 0.8      # [0.55, 0.8) -> COLD
                                # [0.8, 1.0)  -> ARCHIVED

# Potential coefficients (defaults)
# Alpha must be > 0.5*T*(d-2)/lam_inv ≈ 1.8 at T=0.3 for confinement
# to counter the Riemannian curvature correction near the origin.
_ALPHA: float = 3.0   # Base confinement (strong enough to counter correction)
_BETA: float = 0.8    # Access STRENGTHENS confinement (keeps active facts near origin)
_GAMMA: float = 0.005 # Age WEAKENS confinement (old facts drift to boundary)
_DELTA: float = 0.5   # Importance STRENGTHENS confinement

# Safety
_MAX_NORM: float = 0.99   # Clamp radius < 1 (open ball)
_EPS: float = 1e-8


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class LangevinDynamics:
    """Riemannian Langevin dynamics with persistence for memory lifecycle.

    Positions live on the open unit ball B^d.  The potential function
    encodes access frequency, age, and importance as competing forces.
    The lifecycle state (ACTIVE / WARM / COLD / ARCHIVED) is determined
    purely by radial position.

    Attributes:
        dt:           Time-step size for Euler-Maruyama integration.
        temperature:  Boltzmann temperature controlling diffusion spread.
        weight_range: (min_weight, max_weight) for lifecycle weighting.
        dim:          Dimensionality of the position space.
    """

    __slots__ = (
        "dt", "temperature", "weight_range", "dim",
        "_alpha", "_beta", "_gamma", "_delta",
    )

    def __init__(
        self,
        dt: float = 0.01,
        temperature: float = 1.0,
        weight_range: tuple[float, float] = (0.0, 1.0),
        dim: int = 8,
    ) -> None:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if weight_range[0] > weight_range[1]:
            raise ValueError(
                f"weight_range min > max: {weight_range[0]} > {weight_range[1]}"
            )

        self.dt = dt
        self.temperature = temperature
        self.weight_range = weight_range
        self.dim = dim

        # Potential coefficients
        self._alpha = _ALPHA
        self._beta = _BETA
        self._gamma = _GAMMA
        self._delta = _DELTA

    # ------------------------------------------------------------------
    # Single-fact step
    # ------------------------------------------------------------------

    def step(
        self,
        position: list[float],
        access_count: int,
        age_days: float,
        importance: float,
        seed: int | None = None,
    ) -> tuple[list[float], float]:
        """One Euler-Maruyama step of the Langevin SDE.

        Computes the gradient of the potential at the current position,
        applies drift + stochastic diffusion, and returns the new position
        plus the resulting lifecycle weight.

        The caller is responsible for PERSISTING the returned position
        back to the database.  This is the key V1 fix: positions must
        evolve across recalls, not be recomputed from scratch each time.

        Args:
            position:     Current position on the unit ball, length ``dim``.
            access_count: Total number of times this fact has been accessed.
            age_days:     Days since the fact was first stored.
            importance:   Fact importance score in [0, 1].
            seed:         Optional RNG seed for reproducible diffusion.

        Returns:
            (new_position, lifecycle_weight) where new_position is a list
            of floats inside the unit ball and lifecycle_weight is in
            [weight_range[0], weight_range[1]].
        """
        xi = np.asarray(position, dtype=np.float64)
        if xi.shape[0] != self.dim:
            # Graceful resize: pad or truncate
            xi = _resize_position(xi, self.dim)

        # --- Gradient of potential ---
        grad = self._potential_gradient(xi, access_count, age_days, importance)

        # --- Conformal factor on Poincare ball: lambda = 2 / (1 - ||x||^2) ---
        norm_sq = float(np.dot(xi, xi))
        norm_sq = min(norm_sq, _MAX_NORM ** 2)
        lam = 2.0 / (1.0 - norm_sq + _EPS)
        lam_inv = 1.0 / lam

        # --- Drift: -lambda^{-2} * grad_U * dt (Eq. 5 term 1) ---
        drift = -(lam_inv ** 2) * grad * self.dt

        # --- V3.3.12: Ebbinghaus forgetting drift (Eq. 6 in Paper 3) ---
        # λ(m) = 1/S(m) pushes toward boundary (forgetting) based on memory strength.
        # S(m) is computed from access_count + importance. Higher S → less drift.
        strength = max(0.5, 0.3 * math.log(1.0 + access_count) + 0.4 * importance)
        forget_rate = 1.0 / strength  # λ(m)
        # F(ξ) = ξ/||ξ|| points outward (toward boundary = archived zone)
        xi_norm = float(np.linalg.norm(xi))
        if xi_norm > _EPS:
            forget_direction = xi / xi_norm
        else:
            forget_direction = np.zeros(self.dim)
        forgetting_drift = forget_rate * forget_direction * self.dt * 0.1  # Scaled down to prevent instability

        # --- Curvature correction: 0.5 * T * (d-2) * lambda^{-1} * xi * dt (Eq. 5 term 3) ---
        correction = 0.5 * self.temperature * (self.dim - 2) * lam_inv * xi * self.dt

        # --- Diffusion: sqrt(2T * dt) * lambda^{-1} * noise (Eq. 5 term 2) ---
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(self.dim)
        diffusion = math.sqrt(2.0 * self.temperature * self.dt) * lam_inv * noise

        # --- Full Euler-Maruyama update with forgetting (Eq. 6, Girolami & Calderhead 2011) ---
        new_xi = xi + drift + forgetting_drift + correction + diffusion

        # --- Project back into the open ball ---
        new_xi = _project_to_ball(new_xi)

        weight = self.compute_lifecycle_weight(new_xi.tolist())
        return new_xi.tolist(), weight

    # ------------------------------------------------------------------
    # Lifecycle weight from position
    # ------------------------------------------------------------------

    def compute_lifecycle_weight(self, position: list[float]) -> float:
        """Compute retrieval weight from radial position.

        Weight decreases linearly from max to min as radius moves from
        origin to boundary.  This ensures archived memories (near boundary)
        are strongly suppressed in retrieval rankings.

        Args:
            position: Current position, length ``dim``.

        Returns:
            Weight in [weight_range[0], weight_range[1]].
        """
        xi = np.asarray(position, dtype=np.float64)
        radius = float(np.linalg.norm(xi))
        radius = min(radius, _MAX_NORM)

        # Linear interpolation: radius 0 -> max weight, radius 1 -> min weight
        lo, hi = self.weight_range
        weight = hi - (hi - lo) * (radius / _MAX_NORM)
        return max(lo, min(hi, weight))

    # ------------------------------------------------------------------
    # Lifecycle state classification
    # ------------------------------------------------------------------

    def get_lifecycle_state(self, weight: float) -> MemoryLifecycle:
        """Classify lifecycle state from weight.

        Uses the weight (derived from radius) rather than raw radius,
        so the caller doesn't need to know zone boundaries.

        Args:
            weight: Lifecycle weight in [0, 1].

        Returns:
            MemoryLifecycle enum value.
        """
        # Map weight back to approximate radius for zone classification
        lo, hi = self.weight_range
        span = hi - lo
        if span < _EPS:
            return MemoryLifecycle.ACTIVE

        # radius_approx: weight=hi -> radius=0, weight=lo -> radius=MAX_NORM
        radius_approx = (hi - weight) / span * _MAX_NORM

        if radius_approx < _RADIUS_ACTIVE:
            return MemoryLifecycle.ACTIVE
        if radius_approx < _RADIUS_WARM:
            return MemoryLifecycle.WARM
        if radius_approx < _RADIUS_COLD:
            return MemoryLifecycle.COLD
        return MemoryLifecycle.ARCHIVED

    # ------------------------------------------------------------------
    # Batch step (background maintenance)
    # ------------------------------------------------------------------

    def batch_step(
        self,
        facts: list[dict[str, Any]],
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """Batch Langevin step for background maintenance.

        Evolves all fact positions in a single pass.  Designed to be
        called periodically (e.g. every N hours) to let the memory store
        self-organise over time.

        Each input dict must contain:
            - ``fact_id``:      str
            - ``position``:     list[float]
            - ``access_count``: int
            - ``age_days``:     float
            - ``importance``:   float

        Args:
            facts: List of fact dicts with position and metadata.
            seed:  Optional base seed (incremented per fact for variety).

        Returns:
            List of dicts with ``fact_id``, ``position`` (updated),
            ``weight``, and ``lifecycle`` fields.
        """
        results: list[dict[str, Any]] = []
        base_seed = seed

        for i, fact in enumerate(facts):
            fact_seed = (base_seed + i) if base_seed is not None else None

            new_pos, weight = self.step(
                position=fact["position"],
                access_count=fact["access_count"],
                age_days=fact["age_days"],
                importance=fact["importance"],
                seed=fact_seed,
            )

            results.append({
                "fact_id": fact["fact_id"],
                "position": new_pos,
                "weight": weight,
                "lifecycle": self.get_lifecycle_state(weight).value,
            })

        return results

    # ------------------------------------------------------------------
    # Potential gradient (private)
    # ------------------------------------------------------------------

    def _potential_gradient(
        self,
        xi: np.ndarray,
        access_count: int,
        age_days: float,
        importance: float,
    ) -> np.ndarray:
        """Metadata-modulated confinement gradient .

        Potential:
            U(xi) = effective_alpha * ||xi||^2

        Where effective_alpha is modulated by metadata:
            effective_alpha = alpha
                + beta * log(access+1)/10   (frequent access → stronger confinement → ACTIVE)
                - gamma * min(age, 365)/365  (aging → weaker confinement → drifts to ARCHIVED)
                + delta * importance          (important → stronger confinement → stays ACTIVE)

        Gradient:
            grad U = 2 * effective_alpha * xi

        This ensures metadata drives correct lifecycle dynamics:
        - Frequently accessed facts stay near origin (ACTIVE, high weight)
        - Old unused facts drift toward boundary (ARCHIVED, low weight)
        - Important facts resist archival (persistent)

        Args:
            xi:           Position array.
            access_count: Total access count.
            age_days:     Days since creation.
            importance:   Importance score [0, 1].

        Returns:
            Gradient vector, same shape as xi.
        """
        import math as _math
        effective_alpha = (
            self._alpha
            + self._beta * _math.log(access_count + 1) / 10.0
            - self._gamma * min(age_days, 365.0) / 365.0
            + self._delta * importance
        )
        effective_alpha = max(0.1, effective_alpha)
        return 2.0 * effective_alpha * xi


# ---------------------------------------------------------------------------
# Helpers (module-private)
# ---------------------------------------------------------------------------

def _project_to_ball(xi: np.ndarray) -> np.ndarray:
    """Project point back into the open unit ball.

    If ||xi|| >= MAX_NORM, rescale to sit at MAX_NORM.

    Args:
        xi: Position array.

    Returns:
        Projected position with ||result|| < 1.
    """
    norm = float(np.linalg.norm(xi))
    if norm >= _MAX_NORM:
        return xi * (_MAX_NORM / (norm + _EPS))
    return xi


def _resize_position(xi: np.ndarray, target_dim: int) -> np.ndarray:
    """Resize a position vector to target dimensionality.

    Pads with zeros or truncates as needed.

    Args:
        xi:         Source position.
        target_dim: Desired dimensionality.

    Returns:
        Resized position array.
    """
    current = xi.shape[0]
    if current == target_dim:
        return xi
    if current < target_dim:
        return np.pad(xi, (0, target_dim - current), constant_values=0.0)
    return xi[:target_dim]
