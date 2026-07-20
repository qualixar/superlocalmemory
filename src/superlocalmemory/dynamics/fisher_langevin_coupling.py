# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Fisher-Langevin Coupling: information-dynamic lifecycle for memory.

THE CORE INVENTION — Paper Headline 2 (STRONGEST NOVELTY):
Fisher information confidence directly drives Langevin dynamics,
creating self-organizing memory where:

  - High Fisher confidence (low variance) -> memory stabilizes
    (Langevin drift toward origin -> ACTIVE state)
  - Low Fisher confidence (high variance) -> memory fades
    (Langevin drift toward boundary -> ARCHIVED state)
  - Evidence accumulation -> Fisher variance decreases -> consolidation

Mathematical formulation:
  Coupled SDE with information-modulated temperature:
    T_eff = T_0 / (fisher_confidence + epsilon)

  High confidence -> low T_eff -> less noise -> position stabilizes.
  Low confidence  -> high T_eff -> more noise -> position drifts to boundary.

  This produces a provable Fokker-Planck steady-state distribution where
  the memory system self-organizes without external scheduling.

Adapted: uses Langevin position radius for lifecycle zone classification.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default coupling parameters
DEFAULT_CONFIDENCE_SCALE = 1.0
DEFAULT_BASE_TEMPERATURE = 0.3    # Matches Langevin MathConfig default
DEFAULT_EPSILON = 0.1
DEFAULT_CONSOLIDATION_THRESHOLD = 0.8


@dataclass
class CouplingState:
    """State of the Fisher-Langevin coupling for a single memory.

    Attributes:
        fact_id: Fact identifier.
        fisher_confidence: Current Fisher confidence [0, 1].
        langevin_temperature: Derived effective temperature.
        drift_direction: "stabilize" | "neutral" | "fade".
        lifecycle_weight: Score multiplier from coupling [0.3, 1.0].
        consolidation_signal: Whether to consolidate this memory.
    """

    fact_id: str = ""
    fisher_confidence: float = 0.5
    langevin_temperature: float = 0.3
    drift_direction: str = "neutral"
    lifecycle_weight: float = 1.0
    consolidation_signal: bool = False


class FisherLangevinCoupling:
    """Couples Fisher information to Langevin dynamics for self-organizing memory.

    The coupling creates a feedback loop:
    1. Fisher confidence measures how CERTAIN a memory's embedding is.
    2. Confidence drives Langevin TEMPERATURE (high conf -> low noise).
    3. Low noise -> memory position stabilizes near origin (ACTIVE).
    4. High noise -> memory drifts toward boundary (ARCHIVED).
    5. Evidence accumulation -> Fisher variance decreases -> confidence rises.

    This is a provable self-organizing system with Fokker-Planck steady state.
    """

    def __init__(
        self,
        confidence_scale: float = DEFAULT_CONFIDENCE_SCALE,
        base_temperature: float = DEFAULT_BASE_TEMPERATURE,
        epsilon: float = DEFAULT_EPSILON,
        consolidation_threshold: float = DEFAULT_CONSOLIDATION_THRESHOLD,
    ) -> None:
        self._scale = confidence_scale
        self._base_temp = base_temperature
        self._epsilon = epsilon
        self._consolidation_threshold = consolidation_threshold

    def compute_coupling(
        self,
        fisher_variance: np.ndarray | list[float],
        langevin_radius: float,
        access_count: int = 0,
    ) -> CouplingState:
        """Compute the coupled state for a single memory.

        Args:
            fisher_variance: Fisher variance vector (diagonal).
            langevin_radius: Current Langevin position radius [0, 1).
            access_count: Number of times memory has been accessed.

        Returns:
            CouplingState with derived temperature, direction, and weight.
        """
        var_arr = np.asarray(fisher_variance, dtype=np.float64)
        if var_arr.size == 0:
            return CouplingState()

        # Step 1: Fisher confidence from variance
        # Low variance = high confidence (memory is well-characterized)
        avg_var = float(np.mean(np.clip(var_arr, 1e-8, None)))
        fisher_conf = 1.0 / (1.0 + avg_var)

        # Evidence reinforcement from repeated access
        evidence_boost = min(access_count * 0.02, 0.2)
        fisher_conf = min(1.0, fisher_conf + evidence_boost)

        # Step 2: Derive Langevin effective temperature
        # T_eff = T_0 / (confidence + epsilon)
        # High confidence -> low temperature -> less noise -> stabilizes
        temperature = self._base_temp / (fisher_conf + self._epsilon)

        # Step 3: Determine drift direction and lifecycle weight
        if fisher_conf >= self._consolidation_threshold:
            direction = "stabilize"
            weight = 0.9 + 0.1 * fisher_conf
        elif fisher_conf >= 0.5:
            direction = "neutral"
            weight = 0.7 + 0.3 * fisher_conf
        elif fisher_conf >= 0.3:
            direction = "fade"
            weight = 0.4 + 0.6 * fisher_conf
        else:
            direction = "fade"
            weight = max(0.3, fisher_conf)

        # Step 4: Langevin radius modulates (near-boundary penalty)
        if langevin_radius > 0.85:
            weight *= 0.8

        return CouplingState(
            fisher_confidence=fisher_conf,
            langevin_temperature=temperature,
            drift_direction=direction,
            lifecycle_weight=weight,
            consolidation_signal=fisher_conf >= self._consolidation_threshold,
        )

    def apply_to_facts(
        self,
        facts: list[dict[str, Any]],
    ) -> list[CouplingState]:
        """Apply Fisher-Langevin coupling to a list of fact dicts.

        Each dict must have: fact_id, fisher_variance (or None),
        langevin_position (or None), access_count.

        Returns:
            List of CouplingState for each fact.
        """
        states: list[CouplingState] = []

        for fact in facts:
            fid = fact.get("fact_id", "")
            f_var = fact.get("fisher_variance")
            l_pos = fact.get("langevin_position")
            access = fact.get("access_count", 0)

            if f_var is None:
                # No Fisher data — use default coupling
                states.append(CouplingState(fact_id=fid))
                continue

            # Compute Langevin radius from position vector
            radius = 0.5
            if l_pos is not None:
                pos = np.asarray(l_pos, dtype=np.float64)
                radius = float(np.linalg.norm(pos))

            state = self.compute_coupling(f_var, radius, access)
            state.fact_id = fid
            states.append(state)

        return states

    def get_effective_temperature(
        self,
        fisher_variance: np.ndarray | list[float] | None,
        access_count: int = 0,
    ) -> float:
        """Quick helper: get effective Langevin temperature from Fisher variance.

        Used by maintenance.py when running Langevin batch_step with
        Fisher-coupled temperature.
        """
        if fisher_variance is None:
            return self._base_temp

        var_arr = np.asarray(fisher_variance, dtype=np.float64)
        if var_arr.size == 0:
            return self._base_temp
        avg_var = float(np.mean(np.clip(var_arr, 1e-8, None)))
        fisher_conf = min(1.0, 1.0 / (1.0 + avg_var) + min(access_count * 0.02, 0.2))
        return self._base_temp / (fisher_conf + self._epsilon)

    def get_consolidation_candidates(
        self, states: list[CouplingState],
    ) -> list[str]:
        """Return fact IDs that should be consolidated (high confidence)."""
        return [s.fact_id for s in states if s.consolidation_signal]

    def get_decay_candidates(
        self, states: list[CouplingState],
    ) -> list[str]:
        """Return fact IDs that are fading (low confidence + drifting)."""
        return [
            s.fact_id for s in states
            if s.drift_direction == "fade" and s.fisher_confidence < 0.3
        ]
