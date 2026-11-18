# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Fable audit A-02, A-03, A-18, A-19: the need gate controls what it claims.

The claim (M-16): the share of the turns that NEEDED memory which the gate
skips, P(skipped | needed), is at most NEED_ALPHA. The audit showed the old
code bounded P(needed and skipped) over all turns, which a gate that skips
everything meets whenever fewer than 10% of turns need memory.

Every simulation is seeded. Tolerances are stated where they are used.
"""

from __future__ import annotations

import math
import random
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from superlocalmemory.core import need_gate as ng


@pytest.fixture
def store(tmp_path):
    from tests.test_core.test_coverage_backfill import _db

    return _db(tmp_path / "memory.db")


def _population(rng: random.Random, n: int, need_rate: float, shift: float):
    """(score, needed): a mildly informative score (shift 0 = no information)."""
    out = []
    for _ in range(n):
        need = rng.random() < need_rate
        score = min(1.0, max(0.0, rng.gauss(0.5 + shift if need else 0.5 - shift, 0.2)))
        out.append((round(score, 3), need))
    return out


def _true_risk(lam: float | None, population) -> float:
    """P(score < lam | needed) on a big fresh population."""
    needed = [s for s, n in population if n]
    return 0.0 if lam is None else sum(s < lam for s in needed) / len(needed)


# -- A-02: the estimand ---------------------------------------------------------------------------

def test_the_audits_8_percent_case_no_longer_skips_every_prompt() -> None:
    """1,000 labelled turns, 8% needed, need independent of score. The old gate's
    threshold went above the highest score: 100% of needed turns skipped while the
    report said 'under 10%'. Now 80 needed turns weigh less than the 180 any bound
    under 10% requires, so the score skips nothing."""
    rng = random.Random(3)
    rows = [ng.Labelled(None, round(rng.random(), 3), rng.random() < 0.08, 1.0)
            for _ in range(1000)]
    assert 60 < sum(r.needed for r in rows) < 100
    assert ng.calibrate(rows) is None


def test_the_share_of_needed_turns_skipped_stays_under_alpha() -> None:
    """Two rounds the way production runs: round 1 recalls everything (weight 1);
    round 2 runs under round 1's threshold, recalling skipped turns with probability
    EXPLORE (weight 20). Re-calibrate on both, then measure P(skipped | needed) on
    20,000 fresh turns. Mean over 60 seeded runs must be at most alpha (observed
    0.078-0.086 across the four settings; the standard error of the mean is about
    0.004, so the 0.10 line is more than 4 SE away)."""
    for need_rate, shift in ((0.08, 0.0), (0.3, 0.0), (0.3, 0.1), (0.5, 0.15)):
        rng = random.Random(1)
        fresh = _population(random.Random(99), 20_000, need_rate, shift)
        risks = []
        for _ in range(60):
            r1 = [ng.Labelled(None, s, n, 1.0) for s, n in _population(rng, 1500, need_rate, shift)]
            lam1 = ng.calibrate(r1)
            r2 = []
            for s, n in _population(rng, 1500, need_rate, shift):
                if lam1 is None or s >= lam1:
                    r2.append(ng.Labelled(None, s, n, 1.0))
                elif rng.random() < ng.EXPLORE:
                    r2.append(ng.Labelled(None, s, n, 1 / ng.EXPLORE))
            risks.append(_true_risk(ng.calibrate(r1 + r2), fresh))
        assert sum(risks) / len(risks) <= ng.NEED_ALPHA, (need_rate, shift)


def test_an_informative_score_still_earns_a_threshold() -> None:
    """The gate is not vacuous: with a score that separates needed turns, it skips
    a real share of the turns that did not need memory."""
    rng = random.Random(5)
    rows = [ng.Labelled(None, s, n, 1.0) for s, n in _population(rng, 4000, 0.3, 0.25)]
    lam = ng.calibrate(rows)
    assert lam is not None and lam > 0.3
    unneeded = [r for r in rows if not r.needed]
    assert sum(r.score < lam for r in unneeded) / len(unneeded) > 0.2


def test_calibrate_matches_a_brute_force_walk_over_the_grid() -> None:
    """The single sorted pass is the old grid walk, written once more slowly."""
    rng = random.Random(8)
    for trial in range(30):
        rows = [ng.Labelled(rng.choice([None, None, None, ng.RULE_ACK]),
                            round(rng.random(), 2), rng.random() < 0.4,
                            rng.choice([1.0, 1.0, 1 / ng.EXPLORE]))
                for _ in range(rng.randint(150, 600))]
        off = frozenset({ng.RULE_ACK}) if trial % 2 else frozenset()
        best = None
        if ng.bound(rows, None, off) <= ng.NEED_ALPHA:
            for lam in sorted({r.score for r in rows
                               if not ng._skipped(r.rule, r.score, None, off)}):  # noqa: SLF001
                nxt = math.nextafter(lam, math.inf)
                if ng.bound(rows, nxt, off) > ng.NEED_ALPHA:
                    break
                best = nxt
        assert ng.calibrate(rows, suspended=off) == best


# -- A-03: the test weight and the rules' bound ----------------------------------------------------

def test_the_unseen_turn_gets_the_largest_possible_weight_not_the_largest_seen() -> None:
    """Weights all 1 (nothing explored yet): the old test weight was the largest
    seen, 1, giving 1 / (n + 1). The unseen turn can weigh 1 / EXPLORE = 20."""
    rows = [ng.Labelled(None, 0.5, True, 1.0) for _ in range(100)]
    assert ng.W_TEST == pytest.approx(20.0)
    assert ng.bound(rows, None) == pytest.approx(20 / 120)
    assert ng.bound(rows, None) > ng.NEED_ALPHA            # 100 needed turns prove nothing yet


def test_a_rule_that_never_had_a_labelled_turn_is_not_measured(store) -> None:
    """The audit's case: 1,000 let-through rows, none from a rule. The report printed
    'Rules alone: needed turns skipped <= 0.1%' with 0 recalled to learn."""
    now = datetime(2026, 11, 18, 12, tzinfo=UTC)
    rows = [ng.Labelled(None, 0.5, False, 1.0) for _ in range(1000)]
    assert ng.rule_bounds(rows) == {}
    text = "\n".join(ng.describe({"turns": 1500, "skipped": 500, "explored": 0, "labelled": 1000,
                                  "threshold": None, "rules_bound": None, "caveat": "x"}))
    assert "Rules: not measured yet" in text and "<= 0.1%" not in text
    conn = sqlite3.connect(str(store.db_path))
    assert ng.report(conn, "default", now=now)["rules_bound"] is None
    conn.close()


def test_a_rules_bound_needs_enough_needed_turns_to_mean_anything() -> None:
    few = [ng.Labelled(ng.RULE_ACK, 0.1, False, 20.0)] + \
        [ng.Labelled(None, 0.6, True, 1.0) for _ in range(100)]
    got = ng.rule_bounds(few)[ng.RULE_ACK]
    assert got.state == "not_measured" and got.bound is None and got.labelled == 1
    many = [ng.Labelled(ng.RULE_ACK, 0.1, False, 20.0)] + \
        [ng.Labelled(None, 0.6, True, 1.0) for _ in range(400)]
    ok = ng.rule_bounds(many)[ng.RULE_ACK]
    assert ok.state == "ok" and ok.bound == pytest.approx(20 / 420)


# -- A-18: rules ----------------------------------------------------------------------------------

@pytest.mark.parametrize("prompt", ["continue", "proceed", "go ahead", "Go ahead!", "resume",
                                    "yes please continue", "ok proceed", "ok go ahead"])
def test_a_request_to_carry_on_is_not_an_acknowledgement(prompt) -> None:
    assert ng.rule(prompt) is None


def test_the_acknowledgement_rule_still_skips_plain_acknowledgements() -> None:
    assert all(ng.rule(p) == ng.RULE_ACK for p in ("yes", "ok thanks", "sure", "7", "got it", "go"))


def test_a_rule_whose_measured_bound_exceeds_alpha_stops_skipping() -> None:
    misses = [ng.Labelled(ng.RULE_ACK, 0.1, True, 20.0) for _ in range(10)]    # 10 explored misses
    others = [ng.Labelled(None, 0.6, True, 1.0) for _ in range(300)]
    rows = misses + others
    b = ng.rule_bounds(rows)[ng.RULE_ACK]
    assert b.state == "exceeded" and b.bound == pytest.approx((200 + 20) / (500 + 20))
    assert ng.suspended_rules(rows) == frozenset({ng.RULE_ACK})
    assert ng.suspended_rules(others) == frozenset()
    skip = ng.decide("yes", threshold=None, key=None)
    assert skip.rule == ng.RULE_ACK and not skip.recall
    kept = ng.decide("yes", threshold=None, key=None, suspended=frozenset({ng.RULE_ACK}))
    assert kept.recall and not kept.explored and kept.propensity == 1.0
    assert kept.rule == ng.RULE_ACK and kept.score > 0   # still logged, with a real score


def test_a_switched_off_rule_is_measured_and_comes_back_when_it_recovers() -> None:
    rows = [ng.Labelled(ng.RULE_ACK, 0.1, True, 1.0) for _ in range(3)] + \
        [ng.Labelled(None, 0.6, True, 1.0) for _ in range(900)]
    assert ng.suspended_rules(rows) == frozenset()          # 3 of 903 needed turns: under 10%
    worse = rows + [ng.Labelled(ng.RULE_ACK, 0.1, True, 1.0) for _ in range(200)]
    assert ng.suspended_rules(worse) == frozenset({ng.RULE_ACK})


def test_the_threshold_is_calibrated_with_the_switched_off_rules_not_skipping() -> None:
    rows = [ng.Labelled(ng.RULE_ACK, 0.9, True, 1.0) for _ in range(40)] + \
        [ng.Labelled(None, 0.2, False, 1.0) for _ in range(10)] + \
        [ng.Labelled(None, 0.7, True, 1.0) for _ in range(500)]
    assert ng.calibrate(rows) is None        # while the rule skips, its 40 misses cost (60/560 > 10%)
    off = frozenset({ng.RULE_ACK})           # switched off, those turns are ordinary high-score turns
    assert ng.calibrate(rows, suspended=off) == math.nextafter(0.2, math.inf)


def test_the_gate_applies_the_suspended_rules_it_computed(store, tmp_path, monkeypatch) -> None:
    from superlocalmemory.core.feature_flags import FlagState, write_flag

    gate = ng.Gate(tmp_path, store.db_path)
    monkeypatch.setattr(gate, "calibration",
                        lambda profile: ng.Calibration(None, frozenset({ng.RULE_ACK})))
    write_flag(tmp_path, "delivery.need_gate", FlagState.ON)
    assert gate.decide("yes", "default").recall
    assert gate.suspended("default") == frozenset({ng.RULE_ACK})
    assert gate.threshold("default") is None


def test_shadow_never_skips_by_score_and_says_what_it_does(store, tmp_path, monkeypatch) -> None:
    """Audit A-18 (c): shadow injects (labels need real deliveries); the score never
    skips in it, and the module text states the departure from 'changes nothing'."""
    from superlocalmemory.core.feature_flags import FlagState, write_flag

    gate = ng.Gate(tmp_path, store.db_path)
    monkeypatch.setattr(gate, "calibration", lambda profile: ng.Calibration(0.99, frozenset()))
    write_flag(tmp_path, "delivery.need_gate", FlagState.SHADOW)
    d = gate.decide("write a haiku", "default")
    assert d.recall and d.acting is False and d.threshold == 0.99
    assert "memories ARE injected" in (ng.__doc__ or "")


# -- A-19: dropped turns are counted ----------------------------------------------------------------

def _decision(at):
    return ng.Decision("turn_drop", True, None, 0.5, None, False, 1.0, at)


def test_a_turn_the_busy_store_refuses_is_counted_not_lost_silently(store) -> None:
    now = datetime(2026, 11, 18, 12, tzinfo=UTC)
    blocker = sqlite3.connect(str(store.db_path), timeout=0.1, isolation_level=None)
    blocker.execute("BEGIN EXCLUSIVE")
    try:
        assert ng.log(store.db_path, _decision(now), profile_id="default", session_id=None,
                      prompt_digest=None) is False
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
    assert ng.log(store.db_path, ng.Decision("turn_ok", True, None, 0.5, None, False, 1.0, now),
                  profile_id="default", session_id=None, prompt_digest=None) is True
    assert ng.dropped_since(store.db_path, now - timedelta(days=1)) == 1
    assert ng.dropped_since(store.db_path, now + timedelta(days=1)) == 0
    assert ng.dropped_since(store.db_path, now - timedelta(days=1), "other") == 0
    conn = sqlite3.connect(str(store.db_path))
    section = ng.report(conn, "default", now=now + timedelta(hours=1))
    conn.close()
    assert section["turns_dropped"] == 1
    assert "1 prompts could not be logged" in "\n".join(ng.describe(section))


def test_counting_a_drop_takes_no_lock_and_survives_a_missing_folder(tmp_path) -> None:
    assert ng.dropped_since(tmp_path / "nowhere" / "memory.db", datetime.now(UTC)) == 0
    ng._note_drop(tmp_path / "nowhere" / "memory.db", "p", datetime.now(UTC))  # noqa: SLF001
    assert not (tmp_path / "nowhere").exists()                                  # never raises


def test_the_report_states_its_estimand_and_what_it_needs(store) -> None:
    conn = sqlite3.connect(str(store.db_path))
    ng.log(store.db_path, _decision(datetime.now(UTC)), profile_id="default", session_id=None,
           prompt_digest=None)
    section = ng.report(conn, "default")
    conn.close()
    assert section["estimand"] == ng.ESTIMAND and "P(skipped | needed)" in section["estimand"]
    assert section["min_needed_weight"] == pytest.approx(180.0)
    text = "\n".join(ng.describe(section))
    assert "weigh at least 180" in text and "skips nothing yet" in text


def test_the_decision_comment_matches_what_is_stored() -> None:
    """A-21 (part): the propensity of a skipped turn is EXPLORE if explored, else 1 - EXPLORE."""
    drawn = [ng.decide("yes", threshold=None, key=b"k" * 32) for _ in range(400)]
    assert {d.propensity for d in drawn if not d.recall} == {1 - ng.EXPLORE}
    assert {d.propensity for d in drawn if d.recall} <= {ng.EXPLORE}
    assert ng.decide("yes", threshold=None, key=None).propensity == 0.0
