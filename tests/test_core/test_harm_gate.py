# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W3-9 (D-15, M-14): the harm gate in shadow names memories whose inclusion makes
the agent worse, with anytime-valid bounds and false-discovery control."""

from __future__ import annotations

import math
import random
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from superlocalmemory.core import harm_gate as hg
from superlocalmemory.outcomes import effects
from tests.test_outcomes.test_injection_receipts import _fact


@pytest.fixture
def store(tmp_path):
    from tests.test_core.test_coverage_backfill import _db

    return _db(tmp_path / "memory.db")


def _units(n, *, p, harm, seed=1, sessions=None):
    """(included, p, session, used): leaving the fact out raises use by ``harm``."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        inc = int(rng.random() < p)
        rate = 0.3 if inc else 0.3 + harm
        out.append((inc, p, f"s{i if sessions is None else i % sessions}",
                    float(rng.random() < rate)))
    return out


# -- one test -------------------------------------------------------------------------------------

def test_a_harmful_memory_has_a_negative_uplift_bounded_below_zero() -> None:
    t = hg.check_fact("f", effects.USED, _units(6000, p=0.5, harm=0.3))
    assert t.uplift == pytest.approx(-0.3, abs=0.06)
    assert t.upper < 0 and t.p_value < hg.DELTA_H


def test_a_harmless_memory_is_not_flagged() -> None:
    t = hg.check_fact("f", effects.USED, _units(6000, p=0.5, harm=0.0))
    assert t.upper > 0 and t.p_value > hg.DELTA_H


def test_reask_counts_the_other_way() -> None:
    # leaving it out raises re-asking: including it helps, so it is not harmful
    t = hg.check_fact("f", effects.REASK, _units(6000, p=0.5, harm=0.3))
    assert t.uplift > 0 and t.p_value == 1.0


def test_the_contrast_is_exactly_the_stated_two_arm_hajek_formula() -> None:
    units = [(1, 0.8, "a", 1.0), (0, 0.8, "b", 1.0), (1, 0.8, "c", 0.0)]
    delta, _se, k_in, k_out = hg.contrast(units)
    # included: weights 1.25, 1.25 -> mean (1.25 * 1 + 0) / 2.5 = 0.5; left out: weight 5 -> 1.0
    assert delta == pytest.approx(0.5 - 1.0)
    assert k_in == pytest.approx(2.0) and k_out == pytest.approx(1.0)
    assert hg.contrast([(1, 0.8, "a", 1.0), (1, 0.8, "b", 0.0)]) is None   # one arm: no contrast


def test_the_standard_error_is_the_stated_clustered_linearisation() -> None:
    """Two sessions, each with one delivery with the fact and one without, so a session's
    total mixes both arms (the sign of the left-out term matters), and unequal inclusion
    probabilities in the left-out arm (its weights matter)."""
    units = [(1, 0.8, "s1", 1.0), (0, 0.5, "s1", 1.0),
             (1, 0.8, "s2", 0.0), (0, 0.9, "s2", 0.0)]
    delta, se, k_in, k_out = hg.contrast(units)
    a = [1.25, 0.0, 1.25, 0.0]
    b = [0.0, 2.0, 0.0, 10.0]
    mu1 = (1.25 * 1 + 1.25 * 0) / 2.5                                # 0.5
    mu0 = (2.0 * 1 + 10.0 * 0) / 12.0                                # 1/6
    assert delta == pytest.approx(mu1 - mu0)
    y = [1.0, 1.0, 0.0, 0.0]
    psi = [a[i] * (y[i] - mu1) / 2.5 - b[i] * (y[i] - mu0) / 12.0 for i in range(4)]
    totals = [psi[0] + psi[1], psi[2] + psi[3]]
    expected = math.sqrt(sum(t * t for t in totals) * 2 / (2 - 1))
    assert se == pytest.approx(expected)
    assert k_in == pytest.approx(2.0) and k_out == pytest.approx(144 / (4 + 100))


def test_one_session_is_not_enough_to_test() -> None:
    t = hg.check_fact("f", effects.USED, _units(50, p=0.5, harm=0.3, sessions=1))
    assert t.clusters == 1 and t.upper is None and t.p_value == 1.0


def test_each_arm_needs_its_minimum_of_sessions() -> None:
    n = hg.MIN_ARM_CLUSTERS
    enough = [(i % 2, 0.5, f"s{i}", float(i % 3 == 0)) for i in range(4 * n)]
    assert hg.check_fact("f", effects.USED, enough).upper is not None
    short = [(0 if i < n - 1 else 1, 0.5, f"s{i}", float(i % 2)) for i in range(4 * n)]
    short = short[:n - 1] + [(1, 0.5, f"t{i}", 1.0) for i in range(3 * n)]   # n - 1 left out
    t = hg.check_fact("f", effects.USED, short)
    assert t.upper is None and t.clusters_left_out == n - 1 and t.clusters_included == 3 * n
    exact = [(0, 0.5, f"o{i}", float(i % 2)) for i in range(n)] + \
        [(1, 0.5, f"t{i}", float(i % 3 == 0)) for i in range(n)]
    assert hg.check_fact("f", effects.USED, exact).upper is not None     # exactly n each: tested


def test_the_p_value_is_the_level_where_the_anytime_bound_crosses_zero() -> None:
    units = _units(3000, p=0.5, harm=0.06)
    t = hg.check_fact("f", effects.USED, units)
    delta, se_plugin, k_in, k_out = hg.contrast(units)
    se = max(se_plugin, effects.variance_floor(k_in, k_out))
    n_eff = effects.VARIANCE_BOUND / se ** 2
    assert t.upper == pytest.approx(t.uplift + effects.cs_multiplier(
        n_eff, hg.TUNED_CLUSTERS, 2 * hg.DELTA_H) * se)
    at = effects.cs_multiplier(n_eff, hg.TUNED_CLUSTERS, 2 * t.p_value)
    assert t.uplift + at * se == pytest.approx(0, abs=1e-6)


# -- false discovery across facts -----------------------------------------------------------------

def test_benjamini_yekutieli_rejects_by_its_step_up_rule() -> None:
    m, h = 4, 1 + 1 / 2 + 1 / 3 + 1 / 4
    ps = [0.001, 0.2, 0.0001, 0.9]
    assert hg.benjamini_yekutieli(ps, 0.05) == [True, False, True, False]
    edge = 2 * 0.05 / (m * h)
    assert hg.benjamini_yekutieli([0.0001, edge * 0.999, 0.5, 0.5], 0.05) == [True, True, False,
                                                                              False]
    assert hg.benjamini_yekutieli([0.0001, edge * 1.01, 0.5, 0.5], 0.05) == [True, False, False,
                                                                             False]
    assert hg.benjamini_yekutieli([]) == []


def test_many_harmless_memories_flag_almost_none() -> None:
    tests = [hg.check_fact(f"f{i}", effects.USED, _units(400, p=0.5, harm=0.0, seed=i))
             for i in range(40)]
    flags = hg.benjamini_yekutieli([t.p_value for t in tests])
    assert sum(flags) == 0


# -- on a real store, through the receipts --------------------------------------------------------

def _delivery(conn, did, session, at, fact, included, p):
    conn.execute(
        "INSERT INTO injection_deliveries (delivery_id, profile_id, session_id, client, path,"
        " design, holdout_state, epsilon, eligible_count, withhold_draw, pick_draw, key_id, candidates,"
        " delivered_at) VALUES (?, 'default', ?, 'hook', 'session_context', 'one-out-v2', 'on',"
        " 0.1, 4, 0.5, 0.5, 'k1', 4, ?)", (did, session, at.isoformat()))
    conn.execute(
        "INSERT INTO injection_receipts (delivery_id, fact_id, profile_id, role, position,"
        " score, eligible, exclusion, propensity, included, delivered) VALUES"
        " (?, ?, 'default', 'recall', 1, 0.5, 1, NULL, ?, ?, ?)",
        (did, fact, p, included, included))


def test_the_report_reads_randomised_receipts_and_writes_nothing(store) -> None:
    fact = _fact(store, "the deploy key rotates every Monday")
    now = datetime(2026, 11, 18, 12, tzinfo=UTC)
    rng = random.Random(7)
    conn = sqlite3.connect(str(store.db_path))
    conn.execute(  # W0-17b: evidence starts at the profile's first pull delivery
        "INSERT INTO injection_deliveries (delivery_id, profile_id, session_id, client, path,"
        " design, holdout_state, epsilon, eligible_count, candidates, delivered_at) VALUES"
        " ('first-pull', 'default', 'boot', 'mcp', 'mcp_recall', 'one-out-v2', 'off', 0.0, 0, 0,"
        " ?)", ((now - timedelta(days=30)).isoformat(),))
    for i in range(600):
        at = now - timedelta(days=3, minutes=i)
        inc = int(rng.random() < 0.5)
        _delivery(conn, f"d{i}", f"s{i}", at, fact, inc, 0.5)
        if not inc and rng.random() < 0.6:   # without it the agent uses memory more
            conn.execute("INSERT INTO injection_usage VALUES (?, ?, 'default', 'marker_edit',"
                         " NULL, ?)", (f"d{i}", fact, (at + timedelta(minutes=5)).isoformat()))
    conn.commit()
    before = conn.total_changes
    section = hg.report(conn, "default", as_of=now, state="shadow")
    assert conn.total_changes == before
    conn.close()
    assert section["acting"] is False and section["facts_seen"] == 1
    assert section["would_withhold"] == [fact]
    assert "would withhold 1" in hg.describe(section)[0]


def test_untreated_and_ineligible_receipts_are_not_evidence(store) -> None:
    fact = _fact(store, "a pinned rule")
    now = datetime(2026, 11, 18, 12, tzinfo=UTC)
    conn = sqlite3.connect(str(store.db_path))
    for i in range(50):
        _delivery(conn, f"d{i}", f"s{i}", now - timedelta(days=2), fact, 1, 1.0)
    conn.commit()
    assert hg.evaluate(conn, "default", as_of=now) == []
    conn.close()
    assert hg.report(sqlite3.connect(":memory:"), "default")["tests"] == 0
