# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Fable audit A-04: "it helps" / "it hurts" under no effect, and the claim rule is pinned.

The audit's null simulation (outcome independent of the arm, ε = 0.10, looks every few
deliveries up to 400): the old code claimed in 8.7% (base rate 0.5), 9.0% (0.1) and 17.0%
(0.9) of runs against a stated 5%, with first claims at n = 5 on one treated unit. Swapping
the anytime interval for the fixed one in ``_claim`` passed all 270 tests.

Guarantee now: at most ``ALPHA`` of runs ever claim under no effect (tolerance: ALPHA + 3
binomial SEs), no claim before ``MIN_ARM_CLUSTERS`` sessions in each arm, and the claim rule
reads the anytime interval built on the larger of the clustered and the bounded-outcome SE.
"""

from __future__ import annotations

import math
import random

import pytest

from superlocalmemory.outcomes import effects as fx
from superlocalmemory.outcomes import memory_report as mr
from superlocalmemory.outcomes import randomiser as rz


def _unit(i, arm, y, *, k=None, cluster=None):
    k = k if k is not None else 1 + i % 10
    return fx.Unit(f"d{i}", cluster or f"s{i}", "session_context", arm, 1 - rz.EPSILON,
                   rz.EPSILON, 0.0, k, k + 3, y)


def _null_run(rng, rate, max_n, step):
    """True if any look claims. Y is independent of the arm."""
    units = []
    for i in range(max_n):
        arm = fx.CONTROL if rng.random() >= rz.EPSILON else fx.MEMORY
        units.append(_unit(i, arm, 1.0 if rng.random() < rate else 0.0))
        if (i + 1) % step == 0 and fx.estimate(units, "session_context", fx.MEMORY, fx.USED).claim:
            return True
    return False


@pytest.mark.parametrize("rate", [0.5, 0.1, 0.9])
def test_no_effect_means_a_claim_in_at_most_alpha_of_runs(rate) -> None:
    """150 seeded runs per base rate, a look every 5 deliveries up to 250. Tolerance
    ALPHA + 3 binomial SEs = 0.105. Old code on this design: 0.06 / 0.06 / 0.11 (rate
    0.9 is above the line; the audit's longer 300-run design gave 8.7% / 9.0% / 17.0%,
    all above 5%). Observed now: 0. The floor, not the minimum, is what carries this:
    with ``MIN_ARM_CLUSTERS`` = 1 it is still 0."""
    reps = 150
    rng = random.Random(5)
    claims = sum(_null_run(rng, rate, 250, 5) for _ in range(reps))
    assert claims / reps <= fx.ALPHA + 3 * math.sqrt(fx.ALPHA * (1 - fx.ALPHA) / reps)


def test_a_true_effect_is_still_claimed_when_the_data_can_show_it() -> None:
    """The floor costs power only where the design never promised any: at the planned size
    the plan's own variance bound is the floor. Control 0.5, treated 0.1 (a 0.4 drop), 2,000
    deliveries: claimed, in the right direction (including helps)."""
    rng = random.Random(2)
    units = []
    for i in range(2000):
        arm = fx.CONTROL if rng.random() >= rz.EPSILON else fx.MEMORY
        units.append(_unit(i, arm, 1.0 if rng.random() < (0.5 if arm == fx.CONTROL else 0.1)
                           else 0.0))
    est = fx.estimate(units, "session_context", fx.MEMORY, fx.USED)
    assert est.claim == "helps" and est.aipw == pytest.approx(0.4, abs=0.1)
    assert min(est.clusters_treated, est.clusters_control) >= fx.MIN_ARM_CLUSTERS


# -- the rule itself, pinned ------------------------------------------------------------------------

def test_the_claim_reads_the_anytime_interval_not_the_fixed_one() -> None:
    """Seeded data on which the fixed 95% interval excludes 0 and the anytime one does
    not (the audit's surviving mutation: ``_claim(interval, ...)``)."""
    rng = random.Random(0)
    units = []
    for i in range(700):
        arm = fx.CONTROL if rng.random() >= rz.EPSILON else fx.MEMORY
        units.append(_unit(i, arm, 1.0 if rng.random() < (0.5 if arm == fx.CONTROL else 0.33)
                           else 0.0))
    est = fx.estimate(units, "session_context", fx.MEMORY, fx.USED)
    assert min(est.clusters_treated, est.clusters_control) >= fx.MIN_ARM_CLUSTERS
    assert est.interval[0] > 0                          # the fixed interval would claim
    assert est.anytime[0] <= 0 <= est.anytime[1] and est.claim is None


def test_no_claim_before_each_arm_has_its_minimum_sessions() -> None:
    n = fx.MIN_ARM_CLUSTERS
    def build(treated):
        control = [_unit(i, fx.CONTROL, 1.0) for i in range(2000)]
        arm = [_unit(5000 + i, fx.MEMORY, 0.0) for i in range(treated)]   # the largest effect
        return control + arm
    short = fx.estimate(build(n - 1), "session_context", fx.MEMORY, fx.USED)
    assert short.clusters_treated == n - 1 and short.claim is None
    # same data with the minimum met: the effect is huge, so it is claimed (no other reason
    # for the earlier None than the minimum itself)
    enough = fx.estimate(build(n), "session_context", fx.MEMORY, fx.USED)
    assert enough.clusters_treated == n and enough.claim == "helps"


def test_the_control_arm_needs_its_minimum_sessions_too() -> None:
    n = fx.MIN_ARM_CLUSTERS

    def build(control):
        treated = [_unit(i, fx.MEMORY, 0.0) for i in range(300)]
        return treated + [_unit(5000 + i, fx.CONTROL, 1.0) for i in range(control)]
    short = fx.estimate(build(n - 1), "session_context", fx.MEMORY, fx.USED)
    assert short.clusters_control == n - 1 and short.clusters_treated == 300 and short.claim is None
    assert fx.estimate(build(n), "session_context", fx.MEMORY, fx.USED).claim == "helps"


def test_the_interval_is_never_narrower_than_the_bounded_outcome_floor() -> None:
    """Treated units that all share one outcome give a tiny plug-in standard error; the
    interval is read at the floor sqrt(0.25 (1/s_t + 1/s_c)) instead."""
    units = [_unit(i, fx.CONTROL, float(i % 2)) for i in range(180)] + \
        [_unit(1000 + i, fx.MEMORY, 1.0) for i in range(20)]
    est = fx.estimate(units, "session_context", fx.MEMORY, fx.USED)
    floor = fx.variance_floor(est.clusters_treated, est.clusters_control)
    assert est.aipw_se < floor                                  # the plug-in is fooled
    half = (est.anytime[1] - est.anytime[0]) / 2
    sigma2 = fx.contrast_variance(*fx.design_probabilities("session_context", fx.MEMORY))
    assert half == pytest.approx(fx.anytime_half_width(floor, sigma2, est.planned))
    assert half > (est.interval[1] - est.interval[0]) / 2        # wider than the fixed one


def test_the_floor_is_the_bounded_outcome_standard_error() -> None:
    assert fx.variance_floor(20, 180) == pytest.approx(math.sqrt(0.25 * (1 / 20 + 1 / 180)))
    assert fx.variance_floor(0, 10) == math.inf
    assert fx.anytime_half_width(0.0, 1.0, 100) == math.inf


def test_the_report_line_says_what_a_claim_still_needs() -> None:
    e = {"path": "session_context", "contrast": "memory", "outcome": "used", "aipw": 0.2,
         "anytime": (-0.1, 0.5), "claim": None, "n": 50, "planned": 400,
         "clusters_treated": 5, "clusters_control": 45}
    line = mr.describe_effect(e)
    assert f"a claim needs {fx.MIN_ARM_CLUSTERS} sessions in each arm; has 5 and 45" in line
    done = mr.describe_effect({**e, "clusters_treated": 30, "claim": "helps"})
    assert "a claim needs" not in done and "it helps" in done
