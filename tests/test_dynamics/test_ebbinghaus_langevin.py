# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for Ebbinghaus-Langevin coupling — Phase A.

TDD: 5 tests covering temperature coupling, stabilization,
fading, core memory immunity, and forgetting drift formula.
"""

from __future__ import annotations

import numpy as np
import pytest

from superlocalmemory.core.config import ForgettingConfig
from superlocalmemory.dynamics.ebbinghaus_langevin_coupling import (
    EbbinghausLangevinCoupling,
)
from superlocalmemory.dynamics.fisher_langevin_coupling import (
    FisherLangevinCoupling,
)
from superlocalmemory.math.ebbinghaus import EbbinghausCurve
from superlocalmemory.math.langevin import LangevinDynamics


@pytest.fixture
def config() -> ForgettingConfig:
    return ForgettingConfig()


@pytest.fixture
def coupling(config: ForgettingConfig) -> EbbinghausLangevinCoupling:
    ebbinghaus = EbbinghausCurve(config)
    langevin = LangevinDynamics()
    fisher_coupling = FisherLangevinCoupling()
    return EbbinghausLangevinCoupling(ebbinghaus, langevin, fisher_coupling, config)


# ---- Test 9: Forgetting increases temperature ----

def test_forgetting_increases_temperature(coupling: EbbinghausLangevinCoupling) -> None:
    """Low retention fact should produce higher effective temperature
    than high retention fact."""
    fisher_var = np.array([1.0] * 8)

    # High retention: recently accessed, high access count
    state_high = coupling.compute_coupled_state(
        fact_id="fact_high",
        fisher_variance=fisher_var,
        langevin_radius=0.2,
        access_count=50,
        importance=0.9,
        confirmation_count=10,
        emotional_salience=0.5,
        hours_since_last_access=1.0,
    )

    # Low retention: never accessed, long time since
    state_low = coupling.compute_coupled_state(
        fact_id="fact_low",
        fisher_variance=fisher_var,
        langevin_radius=0.2,
        access_count=0,
        importance=0.0,
        confirmation_count=0,
        emotional_salience=0.0,
        hours_since_last_access=720.0,
    )

    assert state_low.effective_temperature > state_high.effective_temperature, (
        "Low retention should produce higher effective temperature"
    )


# ---- Test 10: High strength stabilizes ----

def test_high_strength_stabilizes(coupling: EbbinghausLangevinCoupling) -> None:
    """access_count=100, importance=0.9 -> zone='active', weight > 0.5."""
    fisher_var = np.array([0.5] * 8)

    state = coupling.compute_coupled_state(
        fact_id="fact_stable",
        fisher_variance=fisher_var,
        langevin_radius=0.1,
        access_count=100,
        importance=0.9,
        confirmation_count=20,
        emotional_salience=0.8,
        hours_since_last_access=1.0,
    )

    assert state.lifecycle_zone == "active", f"Expected 'active', got '{state.lifecycle_zone}'"
    assert state.lifecycle_weight > 0.5, f"Expected weight > 0.5, got {state.lifecycle_weight}"
    assert not state.is_forgotten


# ---- Test 11: Low strength fades ----

def test_low_strength_fades(coupling: EbbinghausLangevinCoupling) -> None:
    """access_count=0, importance=0, 720h since access -> demoted below warm.

    Store-scaled clock (GitHub #136, aligned here 2026-09-23): 30 days
    unaccessed at zero importance computes R ~= 0.37 -> 'cold'. The old
    assertion ('archive'/'forgotten') pinned the hours-tuned curve fed
    store-time — the exact bug #136 fixed in the batch path.
    """
    fisher_var = np.array([5.0] * 8)

    state = coupling.compute_coupled_state(
        fact_id="fact_fading",
        fisher_variance=fisher_var,
        langevin_radius=0.8,
        access_count=0,
        importance=0.0,
        confirmation_count=0,
        emotional_salience=0.0,
        hours_since_last_access=720.0,
    )

    assert state.lifecycle_zone in ("cold", "archive", "forgotten"), (
        f"Expected at most 'cold', got '{state.lifecycle_zone}'"
    )
    assert state.lifecycle_zone == "cold", (
        f"30d unaccessed zero-importance on the store clock is 'cold', "
        f"got '{state.lifecycle_zone}'"
    )


# ---- Test 12 (A-HIGH-01): Core memory immune ----
# Note: Core memory immunity is enforced by the scheduler (not coupling).
# This test validates that even a forgotten-zone fact produces correct state.

def test_core_memory_immune_via_scheduler(coupling: EbbinghausLangevinCoupling) -> None:
    """Validate that the coupling itself produces a forgotten state
    for a fact that would be forgotten — the scheduler is responsible
    for skipping core memory facts before they reach this point."""
    fisher_var = np.array([10.0] * 8)

    state = coupling.compute_coupled_state(
        fact_id="fact_should_forget",
        fisher_variance=fisher_var,
        langevin_radius=0.9,
        access_count=0,
        importance=0.0,
        confirmation_count=0,
        emotional_salience=0.0,
        hours_since_last_access=5000.0,
    )

    # The coupling computes zone honestly; the scheduler is what skips core memory
    assert state.retention_score < 0.1, (
        f"Expected very low retention, got {state.retention_score}"
    )


# ---- Test 18 (A-HIGH-01): Forgetting drift in Langevin ----

def test_forgetting_drift_in_langevin(coupling: EbbinghausLangevinCoupling) -> None:
    """Low-retention fact produces forgetting_drift > 0 and higher
    effective_temperature. Verify lambda_forget = (1 - R) * drift_scale."""
    config = coupling._config
    fisher_var = np.array([1.0] * 8)

    # Low retention fact
    state_low = coupling.compute_coupled_state(
        fact_id="fact_drift",
        fisher_variance=fisher_var,
        langevin_radius=0.5,
        access_count=0,
        importance=0.0,
        confirmation_count=0,
        emotional_salience=0.0,
        hours_since_last_access=500.0,
    )

    # High retention fact
    state_high = coupling.compute_coupled_state(
        fact_id="fact_no_drift",
        fisher_variance=fisher_var,
        langevin_radius=0.5,
        access_count=50,
        importance=0.9,
        confirmation_count=10,
        emotional_salience=0.5,
        hours_since_last_access=0.5,
    )

    # Low retention should have higher forgetting drift
    assert state_low.forgetting_drift > state_high.forgetting_drift, (
        "Low retention should produce higher forgetting drift"
    )
    assert state_low.forgetting_drift > 0.0, "Forgetting drift should be positive"

    # Verify formula: lambda_forget = (1 - R) * drift_scale
    expected_drift = (1.0 - state_low.retention_score) * config.forgetting_drift_scale
    assert state_low.forgetting_drift == pytest.approx(expected_drift, abs=1e-9), (
        f"Expected drift {expected_drift}, got {state_low.forgetting_drift}"
    )


class TestStoreScaledClock:
    """The ELC Ebbinghaus step must use the store-scaled clock
    (the #136 clock fix landed only in the batch path; the maintenance ELC
    step still used raw retention, so the overnight maintenance window filed
    memories unaccessed for 3 months (curve truth = warm) as forgotten ->
    archive in bulk)."""

    def test_three_months_unaccessed_is_warm_not_forgotten(self):
        import numpy as np
        from superlocalmemory.core.config import ForgettingConfig
        from superlocalmemory.dynamics.ebbinghaus_langevin_coupling import (
            EbbinghausLangevinCoupling,
        )

        cfg = ForgettingConfig()
        coupling = _make_elc(cfg)
        state = coupling.compute_coupled_state(
            fact_id="f",
            fisher_variance=np.ones(8) * 0.1,
            langevin_radius=0.5,
            access_count=0,
            importance=0.5,
            confirmation_count=0,
            emotional_salience=0.0,
            hours_since_last_access=3 * 30 * 24.0,  # 3 months
        )
        assert state.lifecycle_zone == "warm"

    def test_matches_batch_path_zone(self):
        """ELC and batch_compute_retention (the corrected clock) give the same zone for the same input."""
        import numpy as np
        from superlocalmemory.core.config import ForgettingConfig
        from superlocalmemory.dynamics.ebbinghaus_langevin_coupling import (
            EbbinghausLangevinCoupling,
        )
        from superlocalmemory.math.ebbinghaus import EbbinghausCurve

        cfg = ForgettingConfig()
        curve = EbbinghausCurve(cfg)
        coupling = _make_elc(cfg)
        for hours in [24.0, 24 * 30, 24 * 90, 24 * 365]:
            state = coupling.compute_coupled_state(
                fact_id="f",
                fisher_variance=np.ones(8) * 0.1, langevin_radius=0.5,
                access_count=0, importance=0.5, confirmation_count=0,
                emotional_salience=0.0, hours_since_last_access=hours,
            )
            batch = curve.batch_compute_retention([{
                "fact_id": "f", "access_count": 0, "importance": 0.5,
                "confirmation_count": 0, "emotional_salience": 0.0,
                "last_accessed_at": _iso_hours_ago(hours),
            }])[0]
            assert state.lifecycle_zone == batch["zone"], (
                f"hours={hours}: ELC={state.lifecycle_zone} batch={batch['zone']}"
            )


def _make_curve(cfg):
    from superlocalmemory.math.ebbinghaus import EbbinghausCurve
    return EbbinghausCurve(cfg)


def _make_elc(cfg):
    from superlocalmemory.dynamics.ebbinghaus_langevin_coupling import (
        EbbinghausLangevinCoupling,
    )
    from superlocalmemory.math.langevin import LangevinDynamics
    from superlocalmemory.dynamics.fisher_langevin_coupling import (
        FisherLangevinCoupling,
    )
    return EbbinghausLangevinCoupling(
        _make_curve(cfg), LangevinDynamics(), FisherLangevinCoupling(), cfg,
    )


def _make_fl(cfg):
    from superlocalmemory.dynamics.fisher_langevin_coupling import (
        FisherLangevinCoupling,
    )
    return FisherLangevinCoupling()


def _iso_hours_ago(hours: float) -> str:
    from datetime import datetime, timedelta, timezone
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
