# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Fable audit A-05: the offline gate's boundary is anytime and spent across candidates.

(a) The boundary was tuned at the current sample size on every look, which made its
multiplier the constant 3.279 for every size: a fixed z-threshold read repeatedly. The
audit's null simulation (paired gains with mean 0, looks at 2..5,000 sessions, 4,000
runs) certified 4.05% against a one-sided 1.25%.
(b) Every retrain is a new candidate on the same holdout. Level is spent across them.

All simulations are seeded numpy; tolerances are three binomial SEs above the stated
level (``GATE_ALPHA`` = 0.025).
"""

from __future__ import annotations

import json
import math
import random
import sqlite3

import numpy as np
import pytest

from superlocalmemory.learning import offline_gate as og
from superlocalmemory.learning import position_bias as pb
from superlocalmemory.outcomes import effects

from .test_offline_gate import TRUE_P, _by, _holdout, _model


def _tolerance(level: float, reps: int) -> float:
    return level + 3 * math.sqrt(level * (1 - level) / reps)


def _certified_at_any_look(x: np.ndarray, alpha: float) -> np.ndarray:
    """For each row of ``x`` (one null run): does the lower bound exceed 0 at any look
    from GATE_MIN_CLUSTERS on, using the gate's own multiplier and minimum?"""
    n_max = x.shape[1]
    n = np.arange(1, n_max + 1)
    g = np.array([og.boundary_multiplier(int(k), alpha) for k in n])
    cs = np.cumsum(x, axis=1)
    mean = cs / n
    var = (np.cumsum(x * x, axis=1) - n * mean * mean) * n / np.maximum(n - 1, 1)
    se = np.sqrt(np.maximum(var, 0)) / n
    ok = (n >= og.GATE_MIN_CLUSTERS) & (se > 0) & (mean - g * se > 0)
    return ok.any(axis=1)


# -- (a) the multiplier and the null simulation ------------------------------------------------------

def test_the_multiplier_is_not_a_constant_of_the_sample_size() -> None:
    """The old call ``cs_multiplier(s, s, a)`` is 3.279 for every s. Tuned at a fixed
    size, it varies with s and is smallest near the planned size."""
    old = {round(effects.cs_multiplier(s, s, og.GATE_ALPHA), 6) for s in (2, 10, 100, 10_000)}
    assert len(old) == 1                                   # the audit's finding, reproduced
    new = [og.boundary_multiplier(s, og.GATE_ALPHA) for s in (30, 200, 10_000)]
    assert new[1] < new[0] and new[1] < new[2] and len({round(v, 6) for v in new}) == 3
    assert og.boundary_multiplier(500, 0.01) == pytest.approx(
        effects.cs_multiplier(500, og.GATE_PLANNED_CLUSTERS, 0.01))


@pytest.mark.parametrize("kind", ["symmetric", "skewed", "sparse"])
def test_repeated_looks_under_no_gain_certify_at_most_the_gate_level(kind) -> None:
    """3,000 seeded null runs, a look after every session from 30 to 3,000. First
    candidate (level GATE_ALPHA / 2). The old tuning certified 4.05% in the audit's
    symmetric setting. Observed here: 0.5% symmetric, under 0.1% skewed."""
    rng = np.random.default_rng(5)
    reps, n = 3000, 3000
    if kind == "symmetric":
        x = rng.choice([-1.0, 0.0, 0.0, 1.0], size=(reps, n))
    elif kind == "skewed":
        x = rng.exponential(1.0, size=(reps, n)) - 1.0
    else:
        x = np.where(rng.random((reps, n)) < 0.1, rng.exponential(3.0, size=(reps, n)), 0.0) - 0.3
    rate = _certified_at_any_look(x, og.candidate_alpha(1)).mean()
    assert rate <= _tolerance(og.GATE_ALPHA, reps)


def test_the_old_tuning_would_fail_this_null_simulation() -> None:
    """Teeth: the same simulation with the boundary tuned at the current size (planned = s)
    and no minimum lands at the audit's 4.05%, above the stated level plus tolerance."""
    rng = np.random.default_rng(5)
    reps, n_max = 3000, 5000
    x = rng.choice([-1.0, 0.0, 0.0, 1.0], size=(reps, n_max))
    n = np.arange(1, n_max + 1)
    g = np.array([effects.cs_multiplier(int(k), int(k), og.GATE_ALPHA) for k in n])
    mean = np.cumsum(x, axis=1) / n
    var = (np.cumsum(x * x, axis=1) - n * mean * mean) * n / np.maximum(n - 1, 1)
    se = np.sqrt(np.maximum(var, 0)) / n
    ok = ((n >= 2) & (se > 0) & (mean - g * se > 0)).any(axis=1)
    assert ok.mean() > _tolerance(og.GATE_ALPHA / 2, reps)


def test_no_bound_is_computed_below_the_minimum_number_of_sessions() -> None:
    few = [1.0] * (og.GATE_MIN_CLUSTERS - 1)
    mean, lb, s = og._bound(few, [f"s{i}" for i in range(len(few))], og.GATE_ALPHA)
    assert lb is None and s == og.GATE_MIN_CLUSTERS - 1 and mean == 1.0
    ok = [1.0, 2.0] * (og.GATE_MIN_CLUSTERS // 2)
    assert og._bound(ok, [f"s{i}" for i in range(len(ok))], og.GATE_ALPHA)[1] is not None


# -- (b) alpha spending across candidates -------------------------------------------------------------

def test_the_levels_sum_to_at_most_the_gate_alpha_however_many_candidates() -> None:
    assert og.candidate_alpha(1) == pytest.approx(og.GATE_ALPHA / 2)
    assert og.candidate_alpha(10) == pytest.approx(og.GATE_ALPHA / 110)
    total = sum(og.candidate_alpha(j) for j in range(1, 100_001))
    assert total < og.GATE_ALPHA and total == pytest.approx(og.GATE_ALPHA * (1 - 1 / 100_001))
    with pytest.raises(ValueError):
        og.candidate_alpha(0)


def test_twenty_candidates_on_one_holdout_certify_a_null_one_at_most_at_the_gate_level() -> None:
    """Twenty null candidates (independent mean-zero gains, standing in for retrains no
    better than the served order), each tested at its own spent level: the chance that
    ANY is certified stays within GATE_ALPHA. Without spending (every candidate at
    GATE_ALPHA) it does not."""
    rng = np.random.default_rng(21)
    reps, n = 1500, 1500
    spent = np.zeros(reps, dtype=bool)
    flat = np.zeros(reps, dtype=bool)
    for j in range(1, 21):
        x = rng.choice([-1.0, 0.0, 0.0, 1.0], size=(reps, n))
        spent |= _certified_at_any_look(x, og.candidate_alpha(j))
        flat |= _certified_at_any_look(x, og.GATE_ALPHA)
    assert spent.mean() <= _tolerance(og.GATE_ALPHA, reps)
    assert flat.mean() > spent.mean() * 3                  # spending is what makes the difference


def test_the_candidate_counter_is_per_profile_atomic_and_fails_closed(tmp_path) -> None:
    assert og.next_index(tmp_path, "p") == 1
    og.record_test(tmp_path, "p", 1)
    og.record_test(tmp_path, "q", 1)
    og.record_test(tmp_path, "p", 2)
    assert og.next_index(tmp_path, "p") == 3 and og.next_index(tmp_path, "q") == 2
    og.record_test(tmp_path, "p", 1)                       # never goes back
    assert og.next_index(tmp_path, "p") == 3
    assert json.loads((tmp_path / og.LEDGER_FILE).read_text()) == {"p": 2, "q": 1}
    assert not list(tmp_path.glob("*.tmp"))
    (tmp_path / og.LEDGER_FILE).write_text("not json")
    with pytest.raises(ValueError):
        og.next_index(tmp_path, "p")


def test_a_verdict_says_which_candidate_it_was_and_the_level_it_spent() -> None:
    v = og.evaluate(_by(lambda r: r["features"]["rel"]), _holdout(random.Random(5), 1500),
                    _model(), candidate_index=3)
    assert v.tested and v.candidate_index == 3 and v.alpha == pytest.approx(og.candidate_alpha(3))
    thin = og.evaluate(_by(lambda r: r["features"]["rel"]),
                       _holdout(random.Random(5), og.GATE_MIN_CLUSTERS - 5), _model())
    assert not thin.tested and not thin.certified and "needs" in thin.reason
    unmeasured = pb.PositionModel(tuple(pb.Ratio(k, None, None, None, ()) for k in range(1, 5)))
    assert not og.evaluate(_by(lambda r: 0.0), [], unmeasured).tested


def test_a_later_candidate_faces_a_stricter_bound() -> None:
    rows = _holdout(random.Random(7), 120)                 # a modest holdout: a real, small edge
    first = og.evaluate(_by(lambda r: r["features"]["rel"]), rows, _model(), candidate_index=1)
    tenth = og.evaluate(_by(lambda r: r["features"]["rel"]), rows, _model(), candidate_index=10)
    assert tenth.lower["estimate"] < first.lower["estimate"]


# -- the retrain cycle spends one index per tested candidate -------------------------------------------

def test_the_cycle_counts_a_candidate_only_when_a_bound_was_computed(tmp_path, monkeypatch) -> None:
    from superlocalmemory.learning import consolidation_worker as cw

    memory = tmp_path / "memory.db"
    sqlite3.connect(str(memory)).close()
    monkeypatch.setattr(pb, "estimate", lambda conn, profile: _model())

    class Booster:
        def predict(self, x):
            return x[:, 0]

    rows = _holdout(random.Random(5), 400)
    for r in rows:
        r["features"] = {"rel": r["features"]["rel"]}
    first = cw._offline_gate(Booster(), rows, ["rel"], str(memory), "p", ledger_dir=str(tmp_path))
    second = cw._offline_gate(Booster(), rows, ["rel"], str(memory), "p", ledger_dir=str(tmp_path))
    assert first["candidate_index"] == 1 and second["candidate_index"] == 2
    assert second["alpha"] == pytest.approx(og.candidate_alpha(2))
    assert og.next_index(tmp_path, "p") == 3
    thin = cw._offline_gate(Booster(), rows[:40], ["rel"], str(memory), "p",
                            ledger_dir=str(tmp_path))
    assert thin["tested"] is False and og.next_index(tmp_path, "p") == 3   # nothing spent
    no_counter = cw._offline_gate(Booster(), rows, ["rel"], str(memory), "p")
    assert no_counter["certified"] is False and "counter" in no_counter["reason"]
    (tmp_path / og.LEDGER_FILE).write_text("{broken")
    broken = cw._offline_gate(Booster(), rows, ["rel"], str(memory), "p",
                              ledger_dir=str(tmp_path))
    assert broken["certified"] is False and "gate error" in broken["reason"]


def test_the_true_attention_curve_fixture_is_the_one_the_gate_tests_use() -> None:
    assert TRUE_P[0] == 1.0
