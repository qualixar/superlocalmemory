# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Regression tests for FisherLangevinCoupling with empty fisher_variance.

Background: when storage._jl(raw, None) erroneously returned [] for NULL DB
columns, facts with no Fisher data reached apply_to_facts() with
fisher_variance=[] instead of None. The `if f_var is None` guard never
fired, np.mean was called on an empty array, and the daemon emitted:

    numpy/_core/fromnumeric.py:3824: RuntimeWarning: Mean of empty slice
    numpy/_core/_methods.py:142:    RuntimeWarning: invalid value encountered in scalar divide

These tests lock down two invariants regardless of which layer leaked the
empty array:

  1. apply_to_facts() never emits RuntimeWarnings for None/empty variance.
  2. The returned CouplingState has finite (non-NaN) numeric fields.
"""

from __future__ import annotations

import math
import warnings

import pytest

from superlocalmemory.dynamics.fisher_langevin_coupling import (
    CouplingState,
    FisherLangevinCoupling,
)


@pytest.fixture
def coupling() -> FisherLangevinCoupling:
    return FisherLangevinCoupling()


@pytest.mark.parametrize(
    "fisher_variance",
    [None, [], [0.0] * 0],
    ids=["none", "empty-list", "zero-length-vector"],
)
def test_apply_to_facts_emits_no_runtime_warnings(
    coupling: FisherLangevinCoupling, fisher_variance: object,
) -> None:
    facts = [{"fact_id": "f1", "fisher_variance": fisher_variance, "access_count": 0}]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        coupling.apply_to_facts(facts)
    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert not runtime, f"unexpected RuntimeWarnings: {[str(w.message) for w in runtime]}"


@pytest.mark.parametrize(
    "fisher_variance", [None, []], ids=["none", "empty-list"],
)
def test_apply_to_facts_returns_finite_state(
    coupling: FisherLangevinCoupling, fisher_variance: object,
) -> None:
    facts = [{"fact_id": "f1", "fisher_variance": fisher_variance, "access_count": 0}]
    [state] = coupling.apply_to_facts(facts)
    assert isinstance(state, CouplingState)
    assert math.isfinite(state.fisher_confidence)
    assert math.isfinite(state.langevin_temperature)
    assert math.isfinite(state.lifecycle_weight)


def test_compute_coupling_empty_variance_returns_default_state(
    coupling: FisherLangevinCoupling,
) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        state = coupling.compute_coupling([], langevin_radius=0.5)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert isinstance(state, CouplingState)


def test_get_effective_temperature_empty_variance_falls_back_to_base(
    coupling: FisherLangevinCoupling,
) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        temp = coupling.get_effective_temperature([])
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert math.isfinite(temp)
    # Same path None takes — base_temp, not NaN.
    assert temp == coupling.get_effective_temperature(None)
