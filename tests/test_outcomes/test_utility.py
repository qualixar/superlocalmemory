# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W3-5 (M-13): per-memory use rate, hierarchical Beta pooled by kind, time-decayed.

Checked against simulated stores with a known truth (the prior is recovered,
intervals cover, pooling shrinks what little is known toward the kind and the
store), and the ledger reading against hand-built rows.
"""

from __future__ import annotations

import random
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from superlocalmemory.outcomes import memory_report as mr
from superlocalmemory.outcomes import utility as ut
from superlocalmemory.storage.migrations import M075_injection_receipts as M075

NOW = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)


def _sim(rng: random.Random, facts: int, mu: float, kappa: float,
         n_range=(3, 20), group="semantic") -> tuple[dict, dict]:
    counts, truth = {}, {}
    for i in range(facts):
        theta = rng.betavariate(kappa * mu, kappa * (1 - mu))
        n = rng.randint(*n_range)
        u = sum(rng.random() < theta for _ in range(n))
        fid = f"{group}-{i}"
        counts[fid] = [group, n, u, float(n), float(u)]
        truth[fid] = theta
    return counts, truth


# -- the model ------------------------------------------------------------------------------------

def test_the_store_prior_is_recovered_by_marginal_likelihood() -> None:
    counts, _ = _sim(random.Random(13), 600, mu=0.3, kappa=8.0)
    mu, kappa = ut.fit_prior([c[3] for c in counts.values()], [c[4] for c in counts.values()])
    assert mu == pytest.approx(0.3, abs=0.03)
    assert 4.0 < kappa < 16.0


def test_too_little_to_fit_gives_no_estimate() -> None:
    assert ut.fit_prior([3.0], [1.0]) is None
    assert ut.fit_prior([0.0, 0.0, 5.0], [0.0, 0.0, 1.0]) is None
    assert ut.estimate({"a": ["semantic", 2, 1, 2.0, 1.0]}) is ut.EMPTY


def test_credible_intervals_cover_the_true_rate() -> None:
    rng = random.Random(21)
    covered = total = 0
    for _ in range(4):
        counts, truth = _sim(rng, 300, mu=0.25, kappa=6.0)
        for f in ut.estimate(counts).facts:
            covered += f.low <= truth[f.fact_id] <= f.high
            total += 1
    assert 0.92 <= covered / total <= 0.98


def test_little_evidence_stays_near_the_kind_and_much_follows_the_data() -> None:
    rng = random.Random(2)
    counts, _ = _sim(rng, 300, mu=0.3, kappa=5.0)
    counts["once"] = ["semantic", 1, 1, 1.0, 1.0]
    counts["often"] = ["semantic", 200, 160, 200.0, 160.0]
    u = ut.estimate(counts)
    by = {f.fact_id: f for f in u.facts}
    assert abs(by["once"].mean - u.mean) < 0.2       # 1 of 1 is not 100%
    assert by["often"].mean == pytest.approx(0.8, abs=0.03)
    assert by["often"].low < 0.8 < by["often"].high
    assert by["once"].high - by["once"].low > 3 * (by["often"].high - by["often"].low)


def test_a_kind_seen_a_little_is_pooled_toward_the_store() -> None:
    rng = random.Random(8)
    big, _ = _sim(rng, 400, mu=0.2, kappa=10.0, group="semantic")
    rare = {"rule-0": ["rule", 2, 2, 2.0, 2.0]}
    lots, _ = _sim(rng, 400, mu=0.6, kappa=10.0, group="episodic")
    u = ut.estimate({**big, **rare, **lots})
    kinds = {g.group: g for g in u.groups}
    assert 0.2 < kinds["rule"].mean < 0.85   # 2 of 2 does not make rules 100% used
    assert kinds["episodic"].mean == pytest.approx(0.6, abs=0.05)
    assert kinds["semantic"].mean == pytest.approx(0.2, abs=0.05)
    assert kinds["rule"].facts == 1 and kinds["episodic"].facts == 400


def test_differences_between_kinds_are_not_mistaken_for_noise_within_one() -> None:
    """Two kinds at 0.2 and 0.6, each with kappa 10 inside: one store-wide fit
    would read the gap between kinds as spread within them (kappa near 3)."""
    rng = random.Random(31)
    a, _ = _sim(rng, 500, mu=0.2, kappa=10.0, group="semantic")
    b, _ = _sim(rng, 500, mu=0.6, kappa=10.0, group="episodic")
    u = ut.estimate({**a, **b})
    assert 6.0 < u.concentration < 16.0


def test_the_lists() -> None:
    rng = random.Random(4)
    counts, _ = _sim(rng, 200, mu=0.4, kappa=6.0)
    counts["ignored"] = ["semantic", 60, 0, 60.0, 0.0]
    counts["star"] = ["semantic", 60, 58, 60.0, 58.0]
    u = ut.estimate(counts)
    assert u.most_used(1)[0].fact_id == "star"
    rare = u.rarely_used(50)
    assert rare[0].fact_id == "ignored" and all(f.high < u.mean for f in rare)
    assert u.never_used() == sum(1 for c in counts.values() if c[2] == 0)
    assert ut.EMPTY.rarely_used() == [] and ut.EMPTY.never_used() == 0


# -- reading the ledger ---------------------------------------------------------------------------

@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY, memory_kind TEXT,"
              " fact_type TEXT)")
    c.executemany("INSERT INTO atomic_facts VALUES (?, ?, ?)",
                  [("a", "rule", "semantic"), ("b", None, "episodic"), ("c", None, None)])
    M075.apply(c)
    yield c
    c.close()


def _deliver(c, did, facts, *, at, profile="default", delivered=1) -> None:
    c.execute("INSERT INTO injection_deliveries (delivery_id, profile_id, session_id, path,"
              " design, holdout_state, epsilon, eligible_count, candidates, delivered_at)"
              " VALUES (?, ?, 's', 'mcp_recall', 'one-out-v2', 'off', 0, 0, ?, ?)",
              (did, profile, len(facts), at.isoformat(timespec="microseconds")))
    for i, f in enumerate(facts):
        c.execute("INSERT INTO injection_receipts VALUES (?, ?, ?, 'recall', ?, NULL, 0,"
                  " 'path_not_randomised', 1.0, 1, ?)", (did, f, profile, i + 1, delivered))


def _use(c, did, fact, *, at, signal="marker_edit") -> None:
    c.execute("INSERT INTO injection_usage VALUES (?, ?, 'default', ?, NULL, ?)",
              (did, fact, signal, at.isoformat(timespec="microseconds")))


def test_counts_read_delivered_closed_used_and_decayed(conn) -> None:
    t = NOW - timedelta(days=30)
    _deliver(conn, "d1", ["a", "b"], at=t)
    _use(conn, "d1", "a", at=t + timedelta(hours=2))
    _use(conn, "d1", "b", at=t + timedelta(hours=30))           # outside the window
    _deliver(conn, "d2", ["a"], at=NOW - timedelta(days=2))
    _use(conn, "d2", "a", at=NOW - timedelta(days=2), signal="overlap")  # not a use
    _deliver(conn, "d3", ["c"], at=NOW - timedelta(hours=3))    # window still open
    _deliver(conn, "d4", ["b"], at=t, delivered=0)              # cut by the budget
    _deliver(conn, "d5", ["a"], at=t, profile="other")
    counts = ut._counts(conn, "default", NOW)
    assert set(counts) == {"a", "b"}
    group, n, u, w, wu = counts["a"]
    assert (group, n, u) == ("rule", 2, 1)
    assert w == pytest.approx(0.5 + 2 ** (-2 / 30), rel=1e-6)
    assert wu == pytest.approx(0.5, rel=1e-6)                   # 30 days old: half
    assert counts["b"][:3] == ["episodic", 1, 0]                # kind falls back to type


def test_a_use_of_another_memory_in_the_same_delivery_does_not_count(conn) -> None:
    t = NOW - timedelta(days=5)
    _deliver(conn, "d1", ["a", "b"], at=t)
    _use(conn, "d1", "a", at=t + timedelta(hours=1))
    counts = ut._counts(conn, "default", NOW)
    assert counts["a"][2] == 1 and counts["b"][2] == 0


def test_a_store_without_the_ledger_has_no_use_estimate() -> None:
    assert ut.measure(sqlite3.connect(":memory:"), "default") is ut.EMPTY


# -- the report -----------------------------------------------------------------------------------

def test_the_report_has_every_section_and_fails_soft(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SLM_FEATURE_OUTCOMES_HOLDOUT", "on")
    report = mr.build(sqlite3.connect(":memory:"), "default", as_of=NOW, slm_home=tmp_path)
    assert set(report) >= {"ledger", "effects", "use", "certificates", "second_opinions"}
    assert report["ledger"]["withholding"] == "on" and report["ledger"]["deliveries"] == {}
    assert report["use"]["rate"] is None and report["certificates"] == []
    assert all(e["aipw"] is None and not e["enough_data"] for e in report["effects"])
    text = "\n".join(mr.describe(report))
    assert "withholding: on" in text and "no estimate yet" in text
    assert "Judge certificates: 0 active" in text


def test_the_report_counts_recent_deliveries_and_lists_memories(conn, tmp_path) -> None:
    rng = random.Random(3)
    for i in range(60):
        at = NOW - timedelta(days=rng.randint(2, 40))
        _deliver(conn, f"d{i}", ["a", "b"], at=at)
        if rng.random() < 0.8:
            _use(conn, f"d{i}", "a", at=at + timedelta(hours=1))
    report = mr.build(conn, "default", as_of=NOW, slm_home=tmp_path)
    recent = report["ledger"]["deliveries"]["mcp_recall"]
    assert 0 < recent < 60
    use = report["use"]
    assert use["facts_delivered"] == 2 and use["most_used"][0]["fact_id"] == "a"
    assert use["never_used"] == 1
    assert "most used a" in "\n".join(mr.describe(report))
    import json
    json.dumps(report)  # JSON-ready for every surface


def test_an_effect_line_says_what_it_can_and_cannot_claim() -> None:
    base = {"path": "session_init", "contrast": "memory", "outcome": "used", "n": 120,
            "planned": 7536, "aipw": 0.04, "anytime": (-0.2, 0.28), "claim": None}
    line = mr.describe_effect(base)
    assert "no claim yet" in line and "120 of the 7536" in line and "+4.0 points" in line
    assert "it helps" in mr.describe_effect({**base, "claim": "helps"})
    assert "bm25 channel" in mr.describe_effect({**base, "contrast": "channel:bm25"})
    assert "no estimate yet" in mr.describe_effect({**base, "aipw": None})
