# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W3-4 (M-12): IPW and doubly robust effects of delivered memory, with intervals.

The estimators are checked by simulation against a known truth (unbiased,
covering at the stated level, the anytime interval holding over repeated
looks where a fixed one does not), and the ledger reading against hand-built
rows (closed windows only, the outcome window, the arms and their exact
probabilities, sessions as clusters).
"""

from __future__ import annotations

import itertools
import math
import random
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from superlocalmemory.outcomes import effects as fx
from superlocalmemory.outcomes import randomiser as rz
from superlocalmemory.storage.migrations import M075_injection_receipts as M075

T0 = datetime(2026, 11, 16, 9, 0, tzinfo=UTC)
LATER = T0 + timedelta(days=3)


# -- the power line -------------------------------------------------------------------------------

def test_the_anytime_multiplier_at_its_tuning_point_is_the_known_constant() -> None:
    x = -2 * math.log(0.05) + math.log(-2 * math.log(0.05) + 1)
    by_hand = math.sqrt(2 * (1 + x) / x * math.log(math.sqrt(1 + x) / 0.05))
    assert fx.cs_multiplier(1, 1) == pytest.approx(by_hand) == pytest.approx(3.0352, abs=1e-4)
    assert fx.cs_multiplier(500, 500) == pytest.approx(by_hand)  # depends on n / planned only


@pytest.mark.parametrize("n", [50, 800, 20_000])
def test_the_boundary_is_tightest_near_where_it_is_tuned(n) -> None:
    tuned = fx.cs_multiplier(n, n)
    others = [fx.cs_multiplier(n, round(n * f)) for f in (0.1, 0.3, 0.6, 1.7, 3, 10)]
    assert tuned <= min(others) * 1.01


def test_the_anytime_interval_is_wider_than_a_fixed_one_and_shrinks_like_root_n() -> None:
    z = 1.959964
    assert all(fx.cs_multiplier(n, 1000) > z for n in (10, 1000, 10 ** 6))
    # In standard errors the width grows only like sqrt(log n): its margin over z stays small.
    assert fx.cs_multiplier(10 ** 6, 1000) < 2.5 * z


def test_the_planned_size_is_the_textbook_formula() -> None:
    sigma2 = 0.25 * (1 / 0.05 + 1 / 0.95)
    fixed = math.ceil(sigma2 * (1.959964 + 0.841621) ** 2 / 0.1 ** 2)
    assert fx.planned_deliveries(0.05, 0.95, anytime=False) == pytest.approx(fixed, abs=1)
    anytime = fx.planned_deliveries(0.05, 0.95)
    assert anytime / fixed == pytest.approx(((3.0352 + 0.8416) / (1.96 + 0.8416)) ** 2, rel=1e-3)
    assert fx.planned_deliveries(0.05, 0.95, mde=0.05) == pytest.approx(4 * anytime, rel=1e-3)
    assert fx.planned_deliveries(0.05, 0.95, design_effect=2) == pytest.approx(2 * anytime,
                                                                             rel=1e-3)


def test_the_design_probabilities_follow_one_out_v2() -> None:
    eps = rz.EPSILON
    pt, pc = fx.design_probabilities("session_init", fx.MEMORY)
    assert pt == pytest.approx(eps / 2 / (eps / 2 + 1 - eps)) and pt + pc == pytest.approx(1)
    pt, _ = fx.design_probabilities("session_context", fx.MEMORY)
    assert pt == pytest.approx(eps)  # no channel arm: the whole budget withholds memory
    pt, _ = fx.design_probabilities("session_init", "channel:bm25")
    each = eps / 2 / len(rz.DROPPABLE_CHANNELS)
    assert pt == pytest.approx(each / (each + 1 - eps))
    with pytest.raises(ValueError):
        fx.design_probabilities("session_context", fx.CHANNEL)
    assert fx.contrasts("session_context") == [fx.MEMORY]
    assert len(fx.contrasts("session_init")) == 2 + len(rz.DROPPABLE_CHANNELS)


# -- the estimators, by simulation ----------------------------------------------------------------

def _units(n: int, rng: random.Random, *, effect: float, p_mem: float = 0.3,
           base: float = 0.3, slope: float = 0.0, per_session: int = 1,
           path: str = "session_context") -> list[fx.Unit]:
    """Deliveries with a known effect of including the memory: Y ~ Bernoulli(
    base + slope * k/10 + effect * included)."""
    out = []
    for i in range(n):
        k = rng.randint(1, 10)
        treated = rng.random() < p_mem
        p = min(max(base + slope * k / 10 + (0.0 if treated else effect), 0.0), 1.0)
        out.append(fx.Unit(f"d{i}", f"s{i // per_session}", path,
                           fx.MEMORY if treated else fx.CONTROL, 1 - p_mem, p_mem, 0.0,
                           k, k + 2, float(rng.random() < p)))
    return out


def test_ipw_and_doubly_robust_are_unbiased_and_their_intervals_cover() -> None:
    rng = random.Random(4230)
    truth, reps = 0.10, 300
    ipw, dr, cover = [], [], 0
    for _ in range(reps):
        est = fx.estimate(_units(400, rng, effect=truth, slope=0.4), "session_context",
                          fx.MEMORY, fx.USED)
        ipw.append(est.ipw)
        dr.append(est.aipw)
        cover += est.interval[0] <= truth <= est.interval[1]
    for xs in (ipw, dr):
        mean = sum(xs) / reps
        sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (reps - 1))
        assert abs(mean - truth) < 3 * sd / math.sqrt(reps)
    assert 0.91 <= cover / reps <= 0.99  # 95% +- about 3 binomial SDs at 300 reps
    # The outcome model earns its keep: same truth, smaller spread.
    assert _sd(dr) < 0.8 * _sd(ipw)


def _sd(xs) -> float:
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def test_the_standard_error_is_clustered_by_session() -> None:
    """Deliveries of one conversation share its outcome: copies add no information."""
    rng = random.Random(7)
    single = _units(300, rng, effect=0.1)
    tripled = [fx.Unit(f"{u.delivery_id}-{j}", u.cluster, u.path, u.arm, u.p_control,
                       u.p_memory, 0.0, u.eligible_count, u.candidates, u.y)
               for u in single for j in range(3)]
    a = fx.estimate(single, "session_context", fx.MEMORY, fx.USED)
    b = fx.estimate(tripled, "session_context", fx.MEMORY, fx.USED)
    assert b.ipw == pytest.approx(a.ipw) and b.clusters == a.clusters == 300
    assert b.ipw_se == pytest.approx(a.ipw_se, rel=0.01)  # not a third of it


def test_the_anytime_interval_holds_over_repeated_looks_where_the_fixed_one_does_not() -> None:
    """Read the report after every 25 deliveries up to 1,000: under no effect the
    anytime interval almost never excludes 0 at any look; a fixed 95% one,
    re-read 40 times, does far more often than 5%."""
    rng = random.Random(11)
    reps, any_false, fixed_false = 120, 0, 0
    for _ in range(reps):
        units = _units(1000, rng, effect=0.0, p_mem=0.3)
        hit_any = hit_fixed = False
        for n in range(100, 1001, 25):
            est = fx.estimate(units[:n], "session_context", fx.MEMORY, fx.USED)
            hit_any |= not est.anytime[0] <= 0 <= est.anytime[1]
            hit_fixed |= not est.interval[0] <= 0 <= est.interval[1]
        any_false += hit_any
        fixed_false += hit_fixed
    assert any_false / reps <= 0.08
    assert fixed_false / reps >= 0.12


def test_a_helpful_memory_is_claimed_as_helping_on_both_outcomes() -> None:
    rng = random.Random(3)
    used = fx.estimate(_units(4000, rng, effect=0.25), "session_context", fx.MEMORY, fx.USED)
    assert used.aipw > 0 and used.claim == "helps"
    # reask: withholding raises re-asking, so including lowers it: also "helps".
    reask = fx.estimate(_units(4000, rng, effect=-0.25), "session_context", fx.MEMORY,
                        fx.REASK)
    assert reask.aipw < 0 and reask.claim == "helps"
    harmful = fx.estimate(_units(4000, rng, effect=-0.25), "session_context", fx.MEMORY,
                          fx.USED)
    assert harmful.claim == "hurts"
    null = fx.estimate(_units(300, rng, effect=0.0), "session_context", fx.MEMORY, fx.USED)
    assert null.claim is None


def test_no_estimate_without_both_arms_or_a_second_cluster() -> None:
    rng = random.Random(1)
    only_control = [u for u in _units(50, rng, effect=0.1) if u.arm == fx.CONTROL]
    est = fx.estimate(only_control, "session_context", fx.MEMORY, fx.USED)
    assert est.aipw is None and est.claim is None and est.n_treated == 0
    one_session = [fx.Unit(u.delivery_id, "s", u.path, u.arm, u.p_control, u.p_memory, 0.0,
                           u.eligible_count, u.candidates, u.y)
                   for u in _units(50, rng, effect=0.1)]
    est = fx.estimate(one_session, "session_context", fx.MEMORY, fx.USED)
    assert est.aipw is not None and est.aipw_se is None and est.interval is None


# -- contrasts: who is compared with whom --------------------------------------------------------

_IDS = itertools.count()


def _u(arm: str, *, k=3, y=0.0, path="session_init", eps=0.1, share=0.5) -> fx.Unit:
    i = next(_IDS)
    return fx.Unit(f"d{i}", f"s{i}", path, arm, 1 - eps, eps * share,
                   eps * (1 - share) / len(rz.DROPPABLE_CHANNELS), k, k, y)


def test_a_contrast_compares_two_arms_with_probabilities_conditional_on_them() -> None:
    units = [_u(fx.CONTROL), _u(fx.MEMORY), _u("channel:bm25"), _u("channel:semantic"),
             _u(fx.MEMORY, k=0), _u(fx.CONTROL, k=0)]
    mem = fx._rows(units, fx.MEMORY)
    assert [(r[0].arm, r[1]) for r in mem] == [(fx.CONTROL, "control"), (fx.MEMORY, "treated")]
    assert mem[1][2] == pytest.approx(0.05 / 0.95)  # P(memory | memory or control)
    chans = fx._rows(units, fx.CHANNEL)
    assert {r[0].arm for r in chans if r[1] == "treated"} == {"channel:bm25", "channel:semantic"}
    assert len(chans) == 4  # both controls: a channel drop does not need a candidate
    one = fx._rows(units, "channel:bm25")
    assert [r[0].arm for r in one if r[1] == "treated"] == ["channel:bm25"]
    each = 0.05 / len(rz.DROPPABLE_CHANNELS)
    assert one[-1][2] == pytest.approx(each / (each + 0.9))


def test_a_channel_contrast_never_adjusts_for_what_the_drop_changed() -> None:
    """Dropping a channel lowers the candidate count, which lowers use: that is
    the channel's value. Adjusting for the count would hide it."""
    rng = random.Random(5)
    units = []
    for i in range(6000):
        dropped = rng.random() < 0.3
        k = rng.randint(1, 4) if dropped else rng.randint(6, 9)
        units.append(fx.Unit(f"d{i}", f"s{i}", "session_init",
                             "channel:bm25" if dropped else fx.CONTROL, 0.7, 0.0, 0.3, k, k,
                             float(rng.random() < k / 10)))
    est = fx.estimate(units, "session_init", "channel:bm25", fx.USED)
    assert est.aipw == pytest.approx(0.5, abs=0.04)  # E[k/10]: 0.75 vs 0.25
    assert fx._covariates(units[0], "channel:bm25") == [1.0]


# -- reading the ledger ---------------------------------------------------------------------------

@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    M075.apply(c)
    # W0-17b: deliveries before the profile's first pull delivery are pilot data, so the
    # ledger starts with a pull long before the deliveries these tests score.
    _delivery(c, "first-pull", session="boot", path="mcp_recall", u=None, at=T0 - timedelta(days=5))
    yield c
    c.close()


def _stamp(at: datetime) -> str:
    return at.isoformat(timespec="microseconds")


def _delivery(c, did, *, session="s1", path="session_init", at=T0, u=0.5, arm=None,
              channel=None, k=3, share=0.5, profile="default") -> None:
    randomised = path in rz.RANDOMISED_PATHS and u is not None
    c.execute(
        "INSERT INTO injection_deliveries (delivery_id, profile_id, session_id, client, path,"
        " design, holdout_state, epsilon, eligible_count, withhold_draw, pick_draw, key_id,"
        " candidates, delivered_at, arm, arm_draw, dropped_channel, memory_share)"
        " VALUES (?, ?, ?, 'c', ?, 'one-out-v2', ?, 0.1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (did, profile, session, path, "on" if randomised else "off", k,
         u if randomised else None, 0.3 if randomised else None, "k" if randomised else None,
         k, _stamp(at), arm, 0.2 if arm else None, channel, share if randomised else None))


def _used(c, did, signal="marker_edit", at=T0 + timedelta(hours=1)) -> None:
    c.execute("INSERT INTO injection_usage VALUES (?, 'f', 'default', ?, NULL, ?)",
              (did, signal, _stamp(at)))


def test_only_randomised_deliveries_with_a_closed_window_are_scored(conn) -> None:
    _delivery(conn, "closed")
    _delivery(conn, "open", at=LATER - timedelta(hours=2))
    _delivery(conn, "unrandomised", u=None)
    _delivery(conn, "pull", path="mcp_recall", u=None)
    _delivery(conn, "elsewhere", profile="other")
    units = fx.load_units(conn, "default", fx.USED, as_of=LATER)
    assert [u.delivery_id for u in units] == ["closed"]


def test_used_counts_acting_signals_inside_the_window_only(conn) -> None:
    for did in ("edit", "tool", "cite", "success", "failure", "overlap", "late", "none"):
        _delivery(conn, did)
    _used(conn, "edit")
    _used(conn, "tool", "marker_tool")
    _used(conn, "cite", "cite")
    _used(conn, "success", "explicit_success")
    _used(conn, "failure", "explicit_failure")
    _used(conn, "overlap", "overlap")
    _used(conn, "late", at=T0 + timedelta(hours=25))
    y = {u.delivery_id: u.y for u in fx.load_units(conn, "default", fx.USED, as_of=LATER)}
    assert y == {"edit": 1, "tool": 1, "cite": 1, "success": 1, "failure": 0, "overlap": 0,
                 "late": 0, "none": 0}


def test_reask_is_a_later_pull_in_the_same_conversation_within_the_window(conn) -> None:
    for did, session in (("asked", "a"), ("before", "b"), ("late", "c"), ("other", "d"),
                         ("none", "e"), ("no-session", None)):
        _delivery(conn, did, session=session)
    _delivery(conn, "p1", session="a", path="cli_recall", u=None, at=T0 + timedelta(hours=2))
    _delivery(conn, "p2", session="b", path="mcp_recall", u=None, at=T0 - timedelta(hours=1))
    _delivery(conn, "p3", session="c", path="hook_recall", u=None, at=T0 + timedelta(hours=30))
    _delivery(conn, "p4", session="zzz", path="mcp_recall", u=None, at=T0 + timedelta(hours=1))
    _delivery(conn, "p5", session="e", path="session_context", at=T0 + timedelta(hours=1))
    y = {u.delivery_id: u.y for u in fx.load_units(conn, "default", fx.REASK, as_of=LATER)}
    assert y == {"asked": 1, "before": 0, "late": 0, "other": 0, "none": 0, "p5": 0}
    assert "no-session" not in y  # no conversation, no re-ask to see


def test_each_delivery_gets_its_arm_and_exact_probabilities(conn) -> None:
    _delivery(conn, "ctrl", u=0.5)
    _delivery(conn, "mem", u=0.03, arm=rz.MEMORY_ARM)
    _delivery(conn, "chan", u=0.07, arm=rz.CHANNEL_ARM, channel="bm25")
    _delivery(conn, "late-draw", u=0.04, arm=rz.MEMORY_ARM, share=1.0)  # drawn after the fetch
    _delivery(conn, "no-session", session=None)
    units = {u.delivery_id: u for u in fx.load_units(conn, "default", fx.USED, as_of=LATER)}
    assert units["ctrl"].arm == fx.CONTROL and units["mem"].arm == fx.MEMORY
    assert units["chan"].arm == "channel:bm25"
    assert units["mem"].p_memory == pytest.approx(0.05)
    assert units["late-draw"].p_memory == pytest.approx(0.1)
    assert units["late-draw"].p_channel_each == 0
    assert units["chan"].p_channel_each == pytest.approx(0.05 / len(rz.DROPPABLE_CHANNELS))
    assert units["ctrl"].p_control == pytest.approx(0.9)
    assert units["no-session"].cluster == "delivery:no-session"
    assert units["ctrl"].cluster == "s1"


def test_a_store_without_the_ledger_gives_empty_estimates() -> None:
    c = sqlite3.connect(":memory:")
    out = fx.report(c, "default", as_of=LATER)
    assert out and all(e.aipw is None and e.claim is None and e.n == 0 for e in out)
    assert {(e.path, e.outcome) for e in out} == {(p, o) for p in rz.RANDOMISED_PATHS
                                                  for o in fx.OUTCOMES}
    assert all(e.planned > 0 for e in out)


def test_the_report_reads_a_ledger_end_to_end(conn) -> None:
    rng = random.Random(9)
    for i in range(3000):
        u = rng.random()
        arm = rz.MEMORY_ARM if u < 0.1 else None
        _delivery(conn, f"d{i}", session=f"s{i}", path="session_context", u=u, arm=arm,
                  share=1.0)
        if rng.random() < (0.2 if arm else 0.5):
            _used(conn, f"d{i}")
    out = {(e.path, e.contrast, e.outcome): e for e in fx.report(conn, "default", as_of=LATER)}
    est = out[("session_context", fx.MEMORY, fx.USED)]
    assert est.n == 3000 and est.n_treated + est.n_control == 3000
    assert est.aipw == pytest.approx(0.3, abs=0.1) and est.claim == "helps"
    assert est.interval[0] < est.aipw < est.interval[1]
    assert est.anytime[0] < est.interval[0] and est.anytime[1] > est.interval[1]
    assert out[("session_init", fx.MEMORY, fx.USED)].n == 0


def test_the_estimator_reads_what_the_real_receipt_writer_stores(tmp_path, monkeypatch) -> None:
    """No stand-in rows: deliveries written by ``receipts.plan`` / ``deliver``
    with the holdout on come back as units with their arm and exact probability."""
    from superlocalmemory.outcomes import receipts
    from superlocalmemory.outcomes.randomiser import Candidate
    from tests.test_core.test_coverage_backfill import _db
    from tests.test_outcomes.test_injection_receipts import _fact, _render

    monkeypatch.setenv("SLM_FEATURE_OUTCOMES_HOLDOUT", "on")
    store = _db(tmp_path / "memory.db")
    ids = [_fact(store, f"memory number {i}") for i in range(4)]
    withheld = {}
    pull = receipts.plan(slm_home=tmp_path, db_path=store.db_path, profile_id="default",
                         session_id="conv-pull", path="mcp_recall", query="q",
                         candidates=[Candidate(f, "recall", 0.5) for f in ids])
    receipts.deliver(pull, store.db_path, ids, fact_id=lambda f: f, render=_render,
                     shown=receipts.shown_in_text)  # W0-17b: evidence starts at the first pull
    for i in range(80):
        plan = receipts.plan(slm_home=tmp_path, db_path=store.db_path, profile_id="default",
                             session_id=f"conv-{i}", path="session_context", query="q",
                             candidates=[Candidate(f, "recent", 0.5) for f in ids])
        receipts.deliver(plan, store.db_path, ids, fact_id=lambda f: f, render=_render,
                         shown=receipts.shown_in_text)
        withheld[plan.delivery_id] = bool(plan.withheld)
    conn = sqlite3.connect(str(store.db_path))
    try:
        units = fx.load_units(conn, "default", fx.USED,
                              as_of=datetime.now(UTC) + timedelta(days=2))
        assert fx.load_units(conn, "default", fx.USED, as_of=datetime.now(UTC)) == []
    finally:
        conn.close()
    assert {u.delivery_id for u in units} == set(withheld)
    assert any(withheld.values())
    for u in units:
        assert (u.arm == fx.MEMORY) == withheld[u.delivery_id]
        assert u.p_memory == pytest.approx(rz.EPSILON) and u.p_control == pytest.approx(0.9)
        assert u.eligible_count == 4 and u.cluster.startswith("conv-")
