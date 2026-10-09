"""Metrics, bootstrap, split, abstention, one-model rule, fusion."""
from __future__ import annotations

import hashlib
import math

import pytest

import stats
import ranking


QRELS = {"q1": {"a": 1, "b": 1}, "q2": {"c": 1}, "q3": {"z": 1}}
RUN = {
    "q1": {"x": 0.9, "a": 0.8, "b": 0.1},
    "q2": {"d": 0.9, "e": 0.8, "f": 0.7, "g": 0.6, "h": 0.5, "c": 0.4},
    "q3": {"y": 0.5},
}


def test_metrics_hand_computed():
    assert stats.per_query(QRELS, RUN, "recall@5") == {"q1": 1.0, "q2": 0.0, "q3": 0.0}
    mrr = stats.per_query(QRELS, RUN, "mrr@10")
    assert mrr["q1"] == pytest.approx(0.5)
    assert mrr["q2"] == pytest.approx(1 / 6)
    ndcg = stats.per_query(QRELS, RUN, "ndcg@10")
    # q1: ranks 2 and 3 relevant; ideal has ranks 1 and 2
    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg["q1"] == pytest.approx(dcg / idcg)
    assert ndcg["q3"] == 0.0


def test_metrics_agree_with_ranx():
    ranx = pytest.importorskip("ranx")
    got = ranx.evaluate(ranx.Qrels(QRELS), ranx.Run(RUN),
                        ["recall@5", "mrr@10", "ndcg@10"], return_mean=False)
    for metric in got:
        mine = stats.per_query(QRELS, RUN, metric)
        for i, qid in enumerate(sorted(QRELS)):
            assert mine[qid] == pytest.approx(float(got[metric][i]), abs=1e-6)


def test_ties_broken_by_doc_id():
    assert stats.ranked_docs({"b": 1.0, "a": 1.0, "c": 2.0}) == ["c", "a", "b"]


def test_bootstrap_deterministic_and_contains_mean():
    vals = [1, 0, 1, 1, 0, 0, 1, 1, 1, 0]
    a = stats.bootstrap_ci(vals, n_boot=2000)
    b = stats.bootstrap_ci(vals, n_boot=2000)
    assert a == b
    mean, lo, hi = a
    assert lo <= mean <= hi
    assert mean == pytest.approx(0.6)


def test_paired_delta_of_identical_runs_is_zero():
    vals = [1.0, 0.0, 1.0, 0.5]
    assert stats.paired_delta_ci(vals, list(vals), n_boot=500) == (0.0, 0.0, 0.0)


def test_split_stratified_and_stable():
    qs = [{"id": f"q{i}", "stratum": "a" if i % 2 else "b"} for i in range(100)]
    dev, test = stats.stratified_split(qs)
    for s in ("a", "b"):
        n = sum(1 for q in qs if q["stratum"] == s)
        n_dev = sum(1 for q in dev if q["stratum"] == s)
        assert abs(n_dev - 0.4 * n) <= 1
    dev2, test2 = stats.stratified_split(list(reversed(qs)))
    assert {q["id"] for q in dev} == {q["id"] for q in dev2}
    assert {q["id"] for q in test} == {q["id"] for q in test2}
    assert len(dev) + len(test) == 100


def test_split_assignment_uses_sha256_order():
    qs = [{"id": f"q{i}", "stratum": "s"} for i in range(10)]
    dev, _ = stats.stratified_split(qs)
    order = sorted(qs, key=lambda q: hashlib.sha256(q["id"].encode()).hexdigest())
    assert [q["id"] for q in dev] == [q["id"] for q in order[:4]]


def test_one_model_rule_cases():
    n = 40
    s1 = [1.0 if i % 2 else 0.0 for i in range(n)]
    same = stats.one_model_rule(s1, list(s1), n_boot=500)
    assert same["verdict"] == "PASS"
    worse = stats.one_model_rule(s1, [0.0] * n, n_boot=500)
    assert worse["verdict"] == "FAIL"
    small = stats.one_model_rule(s1[:10], s1[:10], n_boot=500)
    assert small["verdict"] == "SAMPLE TOO SMALL TO DECIDE"
    assert "delta" in small and "literal_form_pass" in small


def test_abstention_threshold_uses_dev_only():
    dev_scores = [0.9, 0.8, 0.2, 0.1]
    dev_unans = [False, False, True, True]
    tau = stats.tune_threshold(dev_scores, dev_unans)
    assert 0.2 < tau <= 0.8
    rep = stats.abstention_report([0.95, 0.15, 0.85], [False, True, True], tau)
    assert rep["false_answer_rate"] == pytest.approx(0.5)
    assert rep["abstention_precision"] == pytest.approx(1.0)
    assert rep["abstention_recall"] == pytest.approx(0.5)


def test_rrf_fusion_ordering():
    fused = ranking.rrf_fuse([["a", "b", "c"], ["b", "c", "d"]], k=60)
    ids = [d for d, _ in fused]
    assert ids == ["b", "c", "a", "d"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)
