# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Every pass that can archive a memory gives the same zone for the same facts
and the same settings: the decay pass, the ELC coupled step, and the
maintenance Langevin pass.

The drift this guards against, measured on a real store (Oct 2026): the
Langevin pass computed retention with ``ForgettingConfig()`` defaults while the
decay pass used the person's own ``forgetting`` settings, so 630 memories were
archived by one pass that the other called warm or cold. Before that, the ELC
step used the hours clock while the batch path used the store clock (#136,
#266). Each fix landed in one place; the next drift came from another. One
function now decides, ``EbbinghausCurve.retention_and_zone``, and this test
fails if any pass stops using it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from superlocalmemory.core.config import ForgettingConfig
from superlocalmemory.core.langevin_tier import _langevin_pass, _seed_langevin_position
from superlocalmemory.core.lifecycle_state import atomic_lifecycle_for
from superlocalmemory.dynamics.ebbinghaus_langevin_coupling import EbbinghausLangevinCoupling
from superlocalmemory.dynamics.fisher_langevin_coupling import FisherLangevinCoupling
from superlocalmemory.math.ebbinghaus import EbbinghausCurve
from superlocalmemory.math.langevin import LangevinDynamics

#: The person's own settings as one real store has them, and the defaults.
CONFIGS = {
    "defaults": ForgettingConfig(),
    "personal": ForgettingConfig(alpha=50.0, beta=100.0, gamma=30.0, delta=10.0,
                                 min_strength=720.0, max_strength=87600.0,
                                 archive_threshold=0.05, forget_threshold=0.01,
                                 learning_rate=50.0, forgetting_drift_scale=0.05),
}
FACTS = [(access, age, importance)
         for access in (0, 1, 6)
         for age in (0.5, 9.0, 33.0, 75.0, 140.0, 260.0, 520.0, 1500.0)
         for importance in (0.0, 0.5, 1.0)]


def _decay_zone(curve: EbbinghausCurve, access: int, age: float, importance: float) -> str:
    seen = (datetime.now(UTC) - timedelta(days=age)).isoformat()
    (row,) = curve.batch_compute_retention([{
        "fact_id": "f", "access_count": access, "importance": importance,
        "confirmation_count": 0, "emotional_salience": 0.0, "memory_strength": 0.0,
        "last_accessed_at": seen, "trust_score": 1.0}])
    return atomic_lifecycle_for(row["zone"])


def _elc_zone(curve: EbbinghausCurve, config: ForgettingConfig, access: int, age: float,
              importance: float) -> str:
    coupling = EbbinghausLangevinCoupling(
        curve, LangevinDynamics(dim=8), FisherLangevinCoupling(base_temperature=0.3), config)
    state = coupling.compute_coupled_state(
        fact_id="f", fisher_variance=np.ones(8), langevin_radius=0.5, access_count=access,
        importance=importance, confirmation_count=0, emotional_salience=0.0,
        hours_since_last_access=age * 24.0)
    return atomic_lifecycle_for(state.lifecycle_zone)


def _langevin_zone(curve: EbbinghausCurve, access: int, age: float, importance: float) -> str:
    ld = LangevinDynamics(dim=8, dt=0.005, temperature=0.3)
    position = _seed_langevin_position(access, age, importance, fact_id="f", curve=curve)
    (row,) = _langevin_pass(ld, [{"fact_id": "f", "position": position,
                                  "access_count": access, "age_days": age,
                                  "importance": importance}], curve=curve)
    return row["lifecycle"]


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_every_pass_that_can_archive_agrees(name: str) -> None:
    config = CONFIGS[name]
    curve = EbbinghausCurve(config)
    disagree = []
    for access, age, importance in FACTS:
        _, _, zone = curve.retention_and_zone(
            access_count=access, importance=importance, confirmation_count=0,
            emotional_salience=0.0, hours_since=age * 24.0)
        shared = atomic_lifecycle_for(zone)
        found = {"decay": _decay_zone(curve, access, age, importance),
                 "elc": _elc_zone(curve, config, access, age, importance),
                 "langevin": _langevin_zone(curve, access, age, importance)}
        if set(found.values()) != {shared}:
            disagree.append(((access, age, importance), shared, found))
    assert disagree == []


def test_the_persons_settings_decide_not_the_defaults() -> None:
    """A memory unused for 140 days: the defaults archive it, these settings keep
    it. The Langevin pass must follow the settings it is given."""
    personal = EbbinghausCurve(CONFIGS["personal"])
    assert _langevin_zone(EbbinghausCurve(ForgettingConfig()), 0, 140.0, 0.0) == "archived"
    assert _langevin_zone(personal, 0, 140.0, 0.0) != "archived"
    assert _langevin_zone(personal, 0, 140.0, 0.0) == _decay_zone(personal, 0, 140.0, 0.0)
