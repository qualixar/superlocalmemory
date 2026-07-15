# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Ebbinghaus-Langevin coupling — forgetting drift in dynamics.

Combines the Ebbinghaus forgetting curve with Fisher-Langevin coupling
to create a three-way information-dynamic lifecycle:

  1. Fisher confidence  -> Langevin temperature (existing)
  2. Ebbinghaus retention -> forgetting drift coefficient (new)
  3. Combined temperature = T_fisher * (1 + lambda_forget)

The forgetting drift pushes low-retention memories toward the Langevin
boundary faster, while high-retention memories resist drift. This creates
a thermodynamically grounded forgetting process.

Mathematical formulation:
  lambda_forget = (1 - R) * forgetting_drift_scale
  T_combined = T_fisher * (1 + lambda_forget)
  weight_combined = w_fisher * w_ebbinghaus

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from superlocalmemory.core.config import ForgettingConfig
from superlocalmemory.dynamics.fisher_langevin_coupling import (
    FisherLangevinCoupling,
)
from superlocalmemory.math.ebbinghaus import EbbinghausCurve
from superlocalmemory.math.langevin import LangevinDynamics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Coupling state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EbbinghausCouplingState:
    """State of the Ebbinghaus-Langevin coupling for a single memory.

    Attributes:
        fact_id: Fact identifier.
        retention_score: R(t) in [0, 1].
        memory_strength: S(m) in [S_MIN, S_MAX].
        lifecycle_zone: One of active/warm/cold/archive/forgotten.
        effective_temperature: Combined T (Fisher + forgetting).
        lifecycle_weight: Combined weight [0, 1].
        forgetting_drift: lambda_forget coefficient.
        is_forgotten: Below forget_threshold.
        consolidation_signal: From Fisher coupling.
    """

    fact_id: str
    retention_score: float
    memory_strength: float
    lifecycle_zone: str
    effective_temperature: float
    lifecycle_weight: float
    forgetting_drift: float
    is_forgotten: bool
    consolidation_signal: bool


# ---------------------------------------------------------------------------
# Coupling class
# ---------------------------------------------------------------------------

class EbbinghausLangevinCoupling:
    """Couples Ebbinghaus forgetting to Fisher-Langevin dynamics.

    Creates a three-way feedback loop:
    1. Fisher confidence modulates Langevin temperature (existing).
    2. Ebbinghaus retention adds forgetting drift.
    3. Combined effect determines lifecycle zone and weight.
    """

    __slots__ = ("_ebbinghaus", "_langevin", "_fisher_coupling", "_config")

    def __init__(
        self,
        ebbinghaus: EbbinghausCurve,
        langevin: LangevinDynamics,
        fisher_coupling: FisherLangevinCoupling,
        config: ForgettingConfig,
    ) -> None:
        self._ebbinghaus = ebbinghaus
        self._langevin = langevin
        self._fisher_coupling = fisher_coupling
        self._config = config

    def compute_coupled_state(
        self,
        fact_id: str,
        fisher_variance: np.ndarray,
        langevin_radius: float,
        access_count: int,
        importance: float,
        confirmation_count: int,
        emotional_salience: float,
        hours_since_last_access: float,
    ) -> EbbinghausCouplingState:
        """Compute the full coupled state for a single memory.

        Combines Fisher-Langevin coupling with Ebbinghaus forgetting
        to produce a unified lifecycle state.

        Args:
            fact_id: Fact identifier.
            fisher_variance: Fisher variance vector (diagonal).
            langevin_radius: Current Langevin position radius [0, 1).
            access_count: Total access count.
            importance: PageRank importance score.
            confirmation_count: Evidence/confirmation count.
            emotional_salience: Emotional strength.
            hours_since_last_access: Hours since last access.

        Returns:
            EbbinghausCouplingState with all computed fields.
        """
        # Step 1: Fisher-Langevin coupling (existing system)
        fl_state = self._fisher_coupling.compute_coupling(
            fisher_variance, langevin_radius, access_count,
        )

        # Step 2: Ebbinghaus strength
        strength = self._ebbinghaus.memory_strength(
            access_count, importance, confirmation_count, emotional_salience,
        )

        # Step 3: Ebbinghaus retention
        retention = self._ebbinghaus.retention(hours_since_last_access, strength)

        # Step 4: Lifecycle zone
        zone = self._ebbinghaus.lifecycle_zone(retention)

        # Step 5: Forgetting drift coefficient
        # Higher forgetting (lower R) -> stronger drift toward boundary
        lambda_forget = (1.0 - retention) * self._config.forgetting_drift_scale

        # Step 6: Combined temperature
        # Forgetting increases effective temperature -> more noise -> faster drift
        t_combined = fl_state.langevin_temperature * (1.0 + lambda_forget)

        # Step 7: Combined weight
        weight = fl_state.lifecycle_weight * self._ebbinghaus.lifecycle_weight(zone)

        # Step 8: Is forgotten?
        is_forgotten = zone == "forgotten"

        return EbbinghausCouplingState(
            fact_id=fact_id,
            retention_score=retention,
            memory_strength=strength,
            lifecycle_zone=zone,
            effective_temperature=t_combined,
            lifecycle_weight=weight,
            forgetting_drift=lambda_forget,
            is_forgotten=is_forgotten,
            consolidation_signal=fl_state.consolidation_signal,
        )
