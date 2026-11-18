# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Fable audit A-01 (BLOCKER): the harm gate flagged almost every memory under no effect.

The audit's null simulation, kept as seeded tests: the outcome is independent of
whether the memory was included. The old one-armed test flagged a memory harmful in
78.3% of runs at k = 3 and 93.0% at k = 10 (re-ask rate 0.5; level 5%). The guarantee
now: at most ``DELTA_H`` of runs flag it, within a stated binomial tolerance, and a
memory that was never left out is "not testable", never "harmful".
"""

from __future__ import annotations

import math
import random
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from superlocalmemory.core import harm_gate as hg
from superlocalmemory.outcomes import effects
from tests.test_outcomes.test_injection_receipts import _fact


def _flagged_in_a_run(rng, p, rate, max_sessions, step, outcome, *, per_session=1,
                      session_effect=0.0):
    """Read the gate every ``step`` sessions; True if it ever flags the fact (BY at DELTA_H)."""
    units = []
    for s in range(max_sessions):
        base = min(max(rate + rng.uniform(-session_effect, session_effect), 0.0), 1.0)
        for _ in range(per_session):
            inc = 1 if rng.random() < p else 0
            units.append((inc, p, f"s{s}", 1.0 if rng.random() < base else 0.0))
        if (s + 1) % step == 0:
            t = hg.check_fact("f", outcome, units)
            if t.upper is not None and hg.benjamini_yekutieli([t.p_value])[0]:
                return True
    return False


def _tolerance(level: float, reps: int) -> float:
    """Three binomial standard errors above the nominal level."""
    return level + 3 * math.sqrt(level * (1 - level) / reps)


def test_a_memory_never_left_out_is_not_testable_and_never_flagged() -> None:
    """The audit's deterministic case: 30 sessions, never left out, a re-ask in half
    of them. The old test returned uplift -0.505, upper -0.229, p = 3.8e-06 and BY rejected."""
    units = [(1, 0.99, f"s{i}", float(i % 2)) for i in range(30)]
    t = hg.check_fact("f", effects.REASK, units)
    assert t.upper is None and t.p_value == 1.0 and t.uplift is None and not t.harmful
    assert t.clusters_included == 30 and t.clusters_left_out == 0


@pytest.mark.parametrize(("k", "outcome"), [(3, effects.REASK), (10, effects.REASK),
                                            (10, effects.USED)])
def test_the_audits_k_3_and_k_10_nulls_flag_nothing(k, outcome) -> None:
    """ε = 0.1 over k eligible memories: inclusion 1 - 0.1/k. Up to 200 sessions read
    every 5; the old gate flagged 78% / 93%. Here the left-out arm never reaches
    MIN_ARM_CLUSTERS, so nothing is testable, and 0 of 100 runs are flagged."""
    rng = random.Random(7)
    p = 1 - 0.1 / k
    flagged = sum(_flagged_in_a_run(rng, p, 0.5, 200, 5, outcome) for _ in range(100))
    assert flagged == 0


@pytest.mark.parametrize(("p", "rate", "outcome"), [(0.5, 0.5, effects.REASK),
                                                    (0.9, 0.5, effects.REASK),
                                                    (0.9, 0.2, effects.REASK),
                                                    (0.9, 0.5, effects.USED),
                                                    (0.7, 0.1, effects.USED)])
def test_when_both_arms_are_populated_the_false_flag_rate_is_at_most_delta_h(p, rate,
                                                                             outcome) -> None:
    """Left out in 10-50% of sessions so the test actually runs: 300 seeded runs, read
    every 20 sessions up to 600. Tolerance: DELTA_H + 3 binomial SEs = 0.088. Observed 0 to
    0.01 on these settings, and 0.04 in a longer run (read every 10 sessions up to 1,500,
    p = 0.9, rate 0.5, 150 runs), the worst seen; the old gate flagged 92%."""
    reps = 300
    rng = random.Random(11)
    flagged = sum(_flagged_in_a_run(rng, p, rate, 600, 20, outcome) for _ in range(reps))
    assert flagged / reps <= _tolerance(hg.DELTA_H, reps)


def test_session_clustered_outcomes_do_not_inflate_the_false_flag_rate() -> None:
    """Three deliveries per session sharing a session-level effect (±0.3 on the rate)."""
    reps = 200
    rng = random.Random(13)
    flagged = sum(_flagged_in_a_run(rng, 0.8, 0.5, 400, 20, effects.REASK, per_session=3,
                                    session_effect=0.3) for _ in range(reps))
    assert flagged / reps <= _tolerance(hg.DELTA_H, reps)


def test_mixed_inclusion_probabilities_keep_the_rate_at_delta_h() -> None:
    """One-out-v2 gives each delivery its own p = 1 - ε/k with k between 1 and 10."""
    reps = 100
    rng = random.Random(17)
    flagged = 0
    for _ in range(reps):
        units, hit = [], False
        for s in range(900):
            p = 1 - 0.5 / rng.randint(1, 10)       # ε exaggerated to 0.5 so the arm fills
            inc = 1 if rng.random() < p else 0
            units.append((inc, p, f"s{s}", 1.0 if rng.random() < 0.4 else 0.0))
            if (s + 1) % 60 == 0:
                t = hg.check_fact("f", effects.REASK, units)
                hit |= t.upper is not None and hg.benjamini_yekutieli([t.p_value])[0]
        flagged += hit
    assert flagged / reps <= _tolerance(hg.DELTA_H, reps)


def test_a_real_harm_is_still_found_once_both_arms_have_enough_sessions() -> None:
    rng = random.Random(3)
    units = []
    for s in range(3000):
        inc = int(rng.random() < 0.5)
        units.append((inc, 0.5, f"s{s}", float(rng.random() < (0.3 if inc else 0.6))))
    t = hg.check_fact("f", effects.USED, units)
    assert t.uplift == pytest.approx(-0.3, abs=0.06) and t.upper < 0 and t.p_value < hg.DELTA_H


def test_the_floor_protects_a_small_left_out_arm() -> None:
    """MIN_ARM_CLUSTERS sessions left out, all re-asking, the rest never: the plug-in
    standard error is tiny, but the bounded-outcome floor keeps the interval wide."""
    n = hg.MIN_ARM_CLUSTERS
    units = [(0, 0.9, f"o{i}", 1.0) for i in range(n)] + \
        [(1, 0.9, f"i{i}", 0.0) for i in range(n * 9)]
    t = hg.check_fact("f", effects.USED, units)           # left out -> used; including -> not
    got = hg.contrast(units)
    assert got is not None and got[1] < 0.05              # plug-in SE is small
    floor = effects.variance_floor(got[2], got[3])
    assert t.upper - t.uplift >= 3.0 * floor - 1e-9       # the interval is read at the floor


# -- the report says how many facts had both arms --------------------------------------------------

def _delivery(conn, did, session, at, fact, included, p):
    conn.execute(
        "INSERT INTO injection_deliveries (delivery_id, profile_id, session_id, client, path,"
        " design, holdout_state, epsilon, eligible_count, withhold_draw, pick_draw, key_id,"
        " candidates, delivered_at) VALUES (?, 'default', ?, 'hook', 'session_context',"
        " 'one-out-v2', 'on', 0.1, 4, 0.5, 0.5, 'k1', 4, ?)", (did, session, at.isoformat()))
    conn.execute(
        "INSERT INTO injection_receipts (delivery_id, fact_id, profile_id, role, position,"
        " score, eligible, exclusion, propensity, included, delivered) VALUES"
        " (?, ?, 'default', 'recall', 1, 0.5, 1, NULL, ?, ?, ?)",
        (did, fact, p, included, included))


def test_the_report_counts_the_facts_that_had_both_arms(tmp_path) -> None:
    from tests.test_core.test_coverage_backfill import _db

    store = _db(tmp_path / "memory.db")
    both = _fact(store, "the deploy key rotates every Monday")
    never_out = _fact(store, "the staging cluster has three nodes")
    now = datetime(2026, 11, 18, 12, tzinfo=UTC)
    rng = random.Random(5)
    conn = sqlite3.connect(str(store.db_path))
    conn.execute(
        "INSERT INTO injection_deliveries (delivery_id, profile_id, session_id, client, path,"
        " design, holdout_state, epsilon, eligible_count, candidates, delivered_at) VALUES"
        " ('first-pull', 'default', 'boot', 'mcp', 'mcp_recall', 'one-out-v2', 'off', 0.0, 0, 0,"
        " ?)", ((now - timedelta(days=30)).isoformat(),))
    for i in range(200):
        at = now - timedelta(days=3, minutes=i)
        _delivery(conn, f"a{i}", f"sa{i}", at, both, int(rng.random() < 0.5), 0.5)
        _delivery(conn, f"b{i}", f"sb{i}", at, never_out, 1, 0.99)
    conn.commit()
    section = hg.report(conn, "default", as_of=now, state="shadow")
    conn.close()
    assert section["facts_seen"] == 2 and section["facts_both_arms"] == 1
    assert section["facts_ever_left_out"] == 1 and section["min_arm_clusters"] == hg.MIN_ARM_CLUSTERS
    assert never_out not in section["would_withhold"]
    text = hg.describe(section)[0]
    assert "2 memories seen" in text and "1 of them with both arms" in text


def test_with_no_fact_testable_the_report_prints_no_would_withhold_count() -> None:
    section = {"facts_seen": 40, "facts_both_arms": 0, "tests": 0, "would_withhold": [],
               "min_arm_clusters": 20}
    (line,) = hg.describe(section)
    assert "Not testable yet" in line and "would withhold" not in line
    assert "none shown to make the agent worse" not in line
