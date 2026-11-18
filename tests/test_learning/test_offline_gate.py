# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W3-6d (D-45, D-46): position attention from logged swaps, and the offline gate.

Checked by simulation against a known truth: the attention ratios are
recovered and covered, a candidate that orders by true relevance is
certified, the served order and a worse order are not, and nothing is
certified while attention is unmeasured.
"""

from __future__ import annotations

import json
import random
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from superlocalmemory.learning import learning_health as lh
from superlocalmemory.learning import offline_gate as og
from superlocalmemory.learning import position_bias as pb
from superlocalmemory.storage.migrations import M075_injection_receipts as M075

TRUE_P = [1.0, 0.7, 0.5, 0.35, 0.25]
AT = (datetime.now(UTC) - timedelta(days=3)).isoformat(timespec="microseconds")


# -- attention from swaps --------------------------------------------------------------------------

def _store(rng: random.Random, deliveries: int, swapped_share: float = 0.5,
           rel: float = 0.6) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    M075.apply(conn)
    for i in range(deliveries):
        swap_at = rng.randint(1, 4) if rng.random() < swapped_share else None
        did = f"d{i}"
        conn.execute("INSERT INTO injection_deliveries (delivery_id, profile_id, path, design,"
                     " holdout_state, epsilon, eligible_count, candidates, delivered_at,"
                     " swap_draw, swap_share, swap_at) VALUES (?, 'default', 'mcp_recall', 'x',"
                     " 'off', 0, 0, 5, ?, ?, 0.05, ?)",
                     (did, AT, 0.01 if swap_at else 0.5, swap_at))
        shown = list(range(1, 6))
        if swap_at:
            shown[swap_at - 1], shown[swap_at] = shown[swap_at], shown[swap_at - 1]
        for original, position in enumerate(shown, start=1):
            fact = f"{did}-o{original}"
            conn.execute("INSERT INTO injection_receipts VALUES (?, ?, 'default', 'recall', ?,"
                         " NULL, 0, 'path_not_randomised', 1.0, 1, 1)", (did, fact, position))
            if rng.random() < rel * TRUE_P[position - 1]:
                conn.execute("INSERT INTO injection_usage VALUES (?, ?, 'default',"
                             " 'marker_edit', NULL, ?)", (did, fact, AT))
    return conn


def test_the_attention_ratios_are_recovered_and_covered() -> None:
    model = pb.estimate(_store(random.Random(7), 8000), "default")
    assert model.measured
    for r in model.ratios:
        truth = TRUE_P[r.pair] / TRUE_P[r.pair - 1]
        assert r.low <= truth <= r.high, (r.pair, r.low, truth, r.high)
        assert abs(r.estimate - truth) < 0.1
    p = model.attention()
    assert p[0] == 1.0 and all(a >= b for a, b in zip(p, p[1:], strict=False))
    assert model.attention("low")[-1] <= p[-1] <= model.attention("high")[-1]


def test_without_swaps_attention_is_not_measured() -> None:
    model = pb.estimate(_store(random.Random(1), 300, swapped_share=0.0), "default")
    assert not model.measured and all(r.estimate is None for r in model.ratios)
    with pytest.raises(ValueError):
        model.attention()
    assert not pb.estimate(sqlite3.connect(":memory:"), "default").measured


def test_the_four_cells_come_from_the_right_items() -> None:
    rows = [("c", None, 1, 1, 5), ("c", None, 2, 0, 5),     # control: orig 1 at 1, orig 2 at 2
            ("s", 1, 2, 1, 5), ("s", 1, 1, 1, 5),           # swapped at 1: orig 1 at 2, orig 2 at 1
            ("short", None, 2, 1, 2)]                       # control too short for pair 2
    cells = pb._cells(rows)
    counts = lambda k: [(c.uses, c.shown) for c in cells[k]]  # noqa: E731
    assert counts(1) == [(1, 1), (1, 1), (1, 2), (1, 1)]    # a_s, a_u, b_u, b_s
    assert counts(2)[1] == (0, 1)                           # only the long control's position 2
    # a swap at pair 1 of a five-long list stands for pairs(5) = 4 deliveries (A-20)
    assert cells[1][0].w == 4.0 and cells[1][1].w == 1.0


def test_a_ratio_never_exceeds_one() -> None:
    cells = [pb.Cell.of(90, 100), pb.Cell.of(10, 100), pb.Cell.of(90, 100), pb.Cell.of(10, 100)]
    r = pb._ratio(1, cells, 2.0)
    assert r.estimate == r.high == 1.0


def test_the_store_window_must_have_closed() -> None:
    model = pb.estimate(_store(random.Random(3), 4000), "default",
                        as_of=datetime.now(UTC) - timedelta(days=10))
    assert not model.measured


# -- the gate ---------------------------------------------------------------------------------------

def _model(p=TRUE_P, spread=0.02) -> pb.PositionModel:
    ratios = []
    for k in range(1, 5):
        r = p[k] / p[k - 1]
        ratios.append(pb.Ratio(k, r, max(r - spread, 0.01), min(r + spread, 1.0), ()))
    return pb.PositionModel(tuple(ratios))


def _holdout(rng: random.Random, queries: int, per_session: int = 1) -> list[dict]:
    """Deliveries of 8 memories in a served order unrelated to relevance."""
    rows = []
    for q in range(queries):
        rels = [rng.random() for _ in range(8)]
        order = list(range(8))
        rng.shuffle(order)
        for shown, item in enumerate(order, start=1):
            p = TRUE_P[shown - 1] if shown <= 5 else 0.1
            rows.append({"query_id": f"q{q}", "session_id": f"s{q // per_session}",
                         "shown_position": shown, "label_source": "ledger_used",
                         "features": {"rel": rels[item]},
                         "outcome_reward": 1.0 if rng.random() < rels[item] * p else 0.0})
    return rows


def _by(key):
    return lambda rows: [key(r) for r in rows]


def test_a_candidate_that_orders_by_relevance_is_certified() -> None:
    v = og.evaluate(_by(lambda r: r["features"]["rel"]), _holdout(random.Random(5), 1500),
                    _model())
    assert v.certified and v.gain > 0 and all(b > 0 for b in v.lower.values())


def test_the_served_order_gains_nothing_and_is_not_certified() -> None:
    v = og.evaluate(_by(lambda r: -r["shown_position"]), _holdout(random.Random(5), 600),
                    _model())
    assert v.gain == pytest.approx(0.0) and not v.certified


def test_a_worse_order_is_not_certified() -> None:
    v = og.evaluate(_by(lambda r: -r["features"]["rel"]), _holdout(random.Random(6), 1500),
                    _model())
    assert not v.certified and v.gain < 0


def test_nothing_is_certified_while_attention_is_unmeasured() -> None:
    unmeasured = pb.PositionModel(tuple(pb.Ratio(k, None, None, None, ()) for k in range(1, 5)))
    v = og.evaluate(_by(lambda r: r["features"]["rel"]), _holdout(random.Random(5), 1500),
                    unmeasured)
    assert not v.certified and "not measured" in v.reason


def test_a_flat_end_of_the_interval_can_withhold_the_certificate() -> None:
    """If attention may be flat, reordering gains nothing: the gate must not certify."""
    v = og.evaluate(_by(lambda r: r["features"]["rel"]), _holdout(random.Random(5), 1500),
                    _model(spread=1.0))
    assert not v.certified and v.lower["high"] <= 0


def test_one_conversation_is_one_cluster() -> None:
    v = og.evaluate(_by(lambda r: r["features"]["rel"]),
                    _holdout(random.Random(5), 50, per_session=50), _model())
    assert not v.certified and v.clusters == 1 and "sessions" in v.reason


def test_a_lift_from_below_five_gets_a_conservative_credit() -> None:
    rows = [{"query_id": "q", "shown_position": 7, "label_source": "ledger_used",
             "outcome_reward": 1.0, "features": {}},
            {"query_id": "q", "shown_position": 1, "label_source": "ledger_used",
             "outcome_reward": 1.0, "features": {}}]
    scores = {id(rows[0]): 2.0, id(rows[1]): 1.0}         # lift the 7th to the top
    (delta,) = og._deltas({"q": rows}, scores, TRUE_P)
    # the lifted use counts p(1) / p(5) (true weight p(1) / p(7) is larger);
    # the displaced one loses p(2) / p(1) - 1
    assert delta == pytest.approx(TRUE_P[0] / TRUE_P[4] + TRUE_P[1] / TRUE_P[0] - 1.0)
    used = {"query_id": "q", "shown_position": 1, "label_source": "ledger_used",
            "outcome_reward": 1.0, "features": {}}
    others = [{"query_id": "q", "shown_position": k, "label_source": "ledger_unused",
               "outcome_reward": 0.0, "features": {}} for k in range(2, 7)]
    ranked = {id(used): 0.0, **{id(r): 1.0 for r in others}}   # five rows above it
    (lost,) = og._deltas({"q": [used, *others]}, ranked, TRUE_P)
    assert lost == pytest.approx(-1.0)                    # a use leaving the top five
    assert og._ledger([{"label_source": "outcome", "shown_position": 1}]) == []


# -- in the retrain cycle and in learning health ----------------------------------------------------

from tests.test_learning.test_online_retrain import (  # noqa: E402
    _seed_active_model,
    learning_db,  # noqa: F401
    memory_db,  # noqa: F401
)


def test_the_cycle_records_the_verdict_and_an_uncertified_candidate(
        learning_db, memory_db, monkeypatch) -> None:  # noqa: F811
    from superlocalmemory.learning import consolidation_worker as cw

    _seed_active_model(learning_db, profile_id="p", new_outcomes=100)

    class Booster:
        def model_to_string(self):
            return "tiny"

        def predict(self, x):
            return [0.0] * len(x)

    monkeypatch.setattr(cw, "_train_booster", lambda *a, **k: (Booster(), {}))
    monkeypatch.setattr(cw, "_measure_serialized_size", lambda b: 128)
    monkeypatch.setattr(cw, "_shadow_test_improved", lambda *a: True)
    monkeypatch.setattr(cw, "_fetch_training_rows", lambda db, pid: ([
        {"query_id": f"q{i // 6}", "fact_id": f"f{i}", "position": i % 6,
         "shown_position": i % 6 + 1, "label_source": "ledger_used",
         "features": {}, "outcome_reward": float(i % 2)} for i in range(120)], []))
    out = cw._run_shadow_cycle(memory_db_path=str(memory_db),
                               learning_db_path=str(learning_db), profile_id="p")
    assert out["candidate_persisted"] and not out["offline_gate"]["certified"]
    conn = sqlite3.connect(str(learning_db))
    (raw,) = conn.execute("SELECT shadow_results_json FROM learning_model_state"
                          " WHERE is_candidate = 1").fetchone()
    conn.close()
    stored = json.loads(raw)
    assert stored["offline_certificate"] is False
    assert "not measured" in stored["offline_gate"]["reason"]
    learners = {x.name: x for x in lh.measure(sqlite3.connect(str(learning_db)), None, "p",
                                              environ={})}
    assert learners["ranker_gate"].state == lh.WAITING
    assert "not measured" in learners["ranker_gate"].reason


def test_learning_health_reports_a_certified_candidate(learning_db) -> None:  # noqa: F811
    conn = sqlite3.connect(str(learning_db))
    assert {x.name: x for x in lh.measure(conn, None, "p", environ={})}[
        "ranker_gate"].state == lh.NO_INPUT
    conn.execute("INSERT INTO learning_model_state (profile_id, model_version, state_bytes,"
                 " bytes_sha256, trained_on_count, feature_names, metrics_json, is_active,"
                 " trained_at, updated_at, is_candidate, shadow_results_json) VALUES"
                 " ('p', 't', X'00', 'h', 1, '[]', '{}', 0, ?, ?, 1, ?)",
                 (AT, AT, json.dumps({"offline_certificate": True})))
    conn.commit()
    gate = {x.name: x for x in lh.measure(conn, None, "p", environ={})}["ranker_gate"]
    assert gate.state == lh.LEARNING and "live test" in gate.reason


def test_a_candidate_no_better_than_the_served_order_is_rarely_certified() -> None:
    """A true null: the candidate reshuffles the top five at random (nothing
    lifted from below, so its expected gain is exactly 0, with real variance).
    Certified at most at the gate's level; a fixed 1-SE bound would certify
    about one in six of these."""
    rng = random.Random(17)
    reps, certified = 150, 0
    for _ in range(reps):
        rows = _holdout(rng, 300)
        perm: dict[str, list[int]] = {}

        def score(r):
            order = perm.setdefault(r["query_id"], rng.sample(range(5), 5))
            k = r["shown_position"]
            return 100.0 - order[k - 1] if k <= 5 else -float(k)

        certified += og.evaluate(_by(score), rows, _model()).certified
    assert certified / reps <= og.GATE_ALPHA + 0.02


def test_the_interval_uses_the_joint_level_over_all_pairs() -> None:
    import math
    from statistics import NormalDist

    counts = [(30, 100), (50, 100), (40, 100), (45, 100)]
    z = NormalDist().inv_cdf(1.0 - pb.MODEL_ALPHA / (2.0 * 4))
    r = pb._ratio(1, [pb.Cell.of(u, n) for u, n in counts], z)
    se2 = 0.0
    for u, n in counts:                       # unit weights, one pseudo-observation (half a use)
        q = (u + 0.5) / (n + 1)
        var = (u * (1 - q) ** 2 + (n - u) * q ** 2 + 0.25 * ((1 - q) ** 2 + q ** 2)) / (n + 1) ** 2
        se2 += var / q ** 2
    se = 0.5 * math.sqrt(se2)
    assert math.log(r.estimate) - math.log(r.low) == pytest.approx(z * se)
    # the same multiplier inside estimate(): recompute one pair from the store's own rows
    conn = _store(random.Random(9), 3000)
    sql = pb._ROWS.format(signals=", ".join("?" * len(pb.effects.USED_SIGNALS)))
    rows = conn.execute(sql, (*pb.effects.USED_SIGNALS, pb.effects.OUTCOME_WINDOW_H, "default",
                              pb.SWAP_DEPTH, pb.effects.OUTCOME_WINDOW_H,
                              pb.usage.stamp(datetime.now(UTC)))).fetchall()
    model = pb.estimate(conn, "default")
    again = pb._ratio(2, pb._cells(rows)[2], NormalDist().inv_cdf(1.0 - pb.MODEL_ALPHA / 8.0))
    got = model.ratios[1]
    assert (got.estimate, got.low, got.high) == pytest.approx(
        (again.estimate, again.low, again.high))


def test_one_unmeasured_pair_makes_the_curve_unusable() -> None:
    ratios = list(_model().ratios)
    ratios[2] = pb.Ratio(3, None, None, None, ())
    model = pb.PositionModel(tuple(ratios))
    assert not model.measured
    v = og.evaluate(_by(lambda r: r["features"]["rel"]), _holdout(random.Random(5), 300), model)
    assert not v.certified and "not measured" in v.reason
