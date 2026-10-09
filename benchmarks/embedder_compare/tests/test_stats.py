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


def test_one_model_rule_outcomes():
    n = 40
    s1 = [1.0 if i % 2 else 0.0 for i in range(n)]
    out = stats.one_model_rule(s1, [1.0 if i % 2 else 0.0 if i % 4 else 1.0 for i in range(n)], n_boot=500)
    assert out["verdict"] in ("PASS", "NOT SHOWN NON-INFERIOR")
    assert out["criterion"].startswith("criterion 1 of 4")
    assert out["ci_width"] >= 0
    # point delta within margin but CI wide: not shown
    s2 = list(s1)
    s2[0], s2[1] = 1.0, 0.0
    s2[3] = 0.0
    wide = stats.one_model_rule(s1, s2, n_boot=500)
    assert wide["verdict"] == "NOT SHOWN NON-INFERIOR" and wide["delta"] >= -0.03
    assert stats.one_model_rule(s1, [0.0] * n, n_boot=500)["verdict"] == "FAIL"
    assert stats.one_model_rule([0.0] * n, [0.0] * n, n_boot=500)["verdict"] == "INVALID"
    assert stats.one_model_rule(s1, s1, n_boot=500, bm25_point=0.9)["verdict"] == "INVALID"
    assert stats.one_model_rule(s1[:5], s1[:5], n_boot=500)["verdict"] == "SAMPLE TOO SMALL TO DECIDE"


def test_empty_paired_delta_is_nan_and_missing_query_raises():
    assert all(math.isnan(x) for x in stats.paired_delta_ci([], []))
    with pytest.raises(ValueError, match="q9"):
        stats.per_query({"q9": {"a": 1}}, {}, "recall@5")


def test_weighted_rrf_and_tie_break():
    fused = ranking.rrf_fuse([["a", "b"], ["b", "a"]], weights=[1.0, 0.0])
    assert [d for d, _ in fused] == ["a", "b"]
    tied = ranking.rrf_fuse([["a"], ["b"]], weights=[1.0, 1.0])
    assert [d for d, _ in tied] == ["a", "b"]  # equal score: text (first) channel first


def test_fusion_weight_tuned_on_dev_only_and_abstain_channel():
    import fusion

    junk = [(f"j{i}", 0.5 - i / 100) for i in range(6)]
    text = [[("t1", 0.9)] + junk + [("m1", 0.2)], [("t2", 0.8)] + junk + [("m2", 0.1)]]
    media = [[("m1", 0.7)], [("m2", 0.6)]]
    dev_qrels = {"q0": {"m1": 1}, "q1": {"m2": 1}}
    w, table = fusion.tune_weight(text, media, {"m1", "m2"}, ["q0", "q1"], dev_qrels)
    assert w > 0 and table[0.0] == 0.0 and table[1.0] == 1.0
    w0, _ = fusion.tune_weight(text, media, {"m1", "m2"}, ["q0", "q1"], {"q0": {"t1": 1}, "q1": {"t2": 1}})
    assert w0 == 0.0  # ties go to the text channel
    fused, abst = fusion.fuse_all(text, media, {"m1", "m2"}, 1.0)
    assert fused[0][0][0] == "m1" and abst[0] == 0.7  # media doc: media-channel cosine
    fused0, abst0 = fusion.fuse_all(text, media, {"m1", "m2"}, 0.0)
    assert fused0[0][0][0] == "t1" and abst0[0] == 0.9
