# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-02 §4.5 (ranker.rank)

"""Extra tests for AdaptiveRanker native inference + model_cache tombstoning."""

from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("lightgbm")

import lightgbm as lgb
import numpy as np

from superlocalmemory.learning import model_cache
from superlocalmemory.learning.consolidation_worker import _retrain_ranker_impl
from superlocalmemory.learning.features import FEATURE_NAMES
from superlocalmemory.learning.labeler import label_gain
from superlocalmemory.learning.model_cache import (
    ActiveModel,
    drift_mode,
    load_active,
)
from superlocalmemory.learning.ranker import AdaptiveRanker
from superlocalmemory.learning.signals import SignalCandidate, record_signal_batch
from tests.test_learning._signal_fixtures import (
    make_db_with_migrations,
    make_batch,
    open_conn,
)


def _trained_model(tmp_path):
    db = make_db_with_migrations(tmp_path)
    conn = open_conn(db)
    for q in range(20):
        record_signal_batch(
            conn,
            make_batch(query_id=f"q{q}", n_candidates=10, profile_id="p1"),
        )
    conn.close()
    assert _retrain_ranker_impl(db._db_path, "p1")
    model = load_active(db, "p1", use_cache=False)
    assert model is not None
    return db, model


# ---------------------------------------------------------------------------
# rank() — native booster path — accepts candidates with to_result_dict.
# ---------------------------------------------------------------------------


def test_rank_returns_reordered_candidates(tmp_path):
    _, model = _trained_model(tmp_path)
    ranker = AdaptiveRanker(signal_count=500, active_model=model)
    candidates = [
        SignalCandidate(fact_id=f"f-{i}",
                        channel_scores={"semantic": i / 10},
                        cross_encoder_score=float(i))
        for i in range(5)
    ]
    out = ranker.rank(candidates, {"query_type": "single_hop"})
    assert len(out) == 5
    assert {c.fact_id for c in out} == {c.fact_id for c in candidates}


def test_rank_empty_candidates_returns_empty(tmp_path):
    _, model = _trained_model(tmp_path)
    ranker = AdaptiveRanker(signal_count=500, active_model=model)
    assert ranker.rank([], {"query_type": "single_hop"}) == []


def test_rank_without_model_returns_input(tmp_path):
    ranker = AdaptiveRanker(signal_count=10)
    cands = [SignalCandidate(fact_id="x")]
    out = ranker.rank(cands, {})
    assert out == cands


def test_rank_with_unknown_drift_falls_back(tmp_path):
    _, good_model = _trained_model(tmp_path)
    # Wrap into an ActiveModel that claims to carry an unknown feature name.
    drifted = ActiveModel(
        profile_id=good_model.profile_id,
        booster=good_model.booster,
        feature_names=tuple(FEATURE_NAMES) + ("mystery",),
        trained_at=good_model.trained_at,
        sha256=good_model.sha256,
    )
    ranker = AdaptiveRanker(signal_count=500, active_model=drifted)
    cands = [SignalCandidate(fact_id=f"f-{i}") for i in range(3)]
    out = ranker.rank(cands, {})
    # Unknown drift → input order preserved.
    assert [c.fact_id for c in out] == ["f-0", "f-1", "f-2"]


def test_rank_accepts_dict_candidates(tmp_path):
    _, model = _trained_model(tmp_path)
    ranker = AdaptiveRanker(signal_count=500, active_model=model)
    dicts = [
        {"fact_id": "a", "channel_scores": {"semantic": 0.9}},
        {"fact_id": "b", "channel_scores": {"semantic": 0.2}},
    ]
    out = ranker.rank(dicts, {})
    assert len(out) == 2


def test_rank_unknown_candidate_type_returns_input():
    ranker = AdaptiveRanker(signal_count=500)
    # Force active_model to be non-None so we reach the loop.
    class _Sentinel:
        pass
    ranker._active = _Sentinel()
    # A non-dict, non-to_result_dict candidate should short-circuit.
    out = ranker.rank(["bad_candidate"], {})
    assert out == ["bad_candidate"]


# ---------------------------------------------------------------------------
# phase property — transitions
# ---------------------------------------------------------------------------


def test_phase_progression_without_model():
    assert AdaptiveRanker(signal_count=0).phase == 1
    assert AdaptiveRanker(signal_count=49).phase == 1
    assert AdaptiveRanker(signal_count=50).phase == 2
    assert AdaptiveRanker(signal_count=199).phase == 2
    # No model → phase 2 even at 200+.
    assert AdaptiveRanker(signal_count=500).phase == 2


def test_phase3_needs_active_model(tmp_path):
    _, model = _trained_model(tmp_path)
    r3 = AdaptiveRanker(signal_count=500, active_model=model)
    assert r3.phase == 3


def test_signal_count_setter_roundtrip():
    r = AdaptiveRanker(signal_count=0)
    r.signal_count = 150
    assert r.signal_count == 150


# ---------------------------------------------------------------------------
# model_cache extra branches
# ---------------------------------------------------------------------------


def test_load_active_returns_none_for_missing_profile(tmp_path):
    db = make_db_with_migrations(tmp_path)
    # Fresh DB — no active row.
    assert load_active(db, "nonexistent", use_cache=False) is None
    # Cached None too.
    assert load_active(db, "nonexistent", use_cache=True) is None
    # Second call should hit cache — still None.
    assert load_active(db, "nonexistent", use_cache=True) is None


def test_load_active_handles_bad_feature_names_json(tmp_path):
    # Use an isolated subdir so we can reuse make_db_with_migrations without
    # colliding with the other fixture.
    sub = tmp_path / "corrupt"
    sub.mkdir()
    db = make_db_with_migrations(sub)
    conn2 = open_conn(db)
    for q in range(20):
        record_signal_batch(
            conn2, make_batch(query_id=f"q{q}", n_candidates=10),
        )
    conn2.close()
    _retrain_ranker_impl(db._db_path, "p1")

    # Set feature_names to garbage JSON.
    direct = sqlite3.connect(db._db_path)
    direct.execute(
        "UPDATE learning_model_state SET feature_names = ? WHERE is_active=1",
        ("this is not JSON", ),
    )
    direct.commit()
    direct.close()
    model_cache.invalidate("p1")
    m = load_active(db, "p1", use_cache=False)
    # Should still succeed — feature_names defaults to current on JSON error.
    assert m is not None


def test_invalidate_all(tmp_path):
    db = make_db_with_migrations(tmp_path)
    # Populate cache with a None.
    load_active(db, "profile_x", use_cache=True)
    model_cache.invalidate()  # No arg → clears entire LRU.
    # Still works afterwards.
    assert load_active(db, "profile_x", use_cache=True) is None


def test_drift_mode_classifies_three_ways():
    from superlocalmemory.learning.model_cache import drift_mode

    class _M:
        def __init__(self, names):
            self.feature_names = names

    assert drift_mode(_M(tuple(FEATURE_NAMES))) == "aligned"
    assert drift_mode(_M(tuple(FEATURE_NAMES[:5]))) == "subset"
    assert drift_mode(_M(tuple(FEATURE_NAMES) + ("alien",))) == "unknown"


# ---------------------------------------------------------------------------
# Signals module — exercise legacy boost/decay/stats for coverage.
# ---------------------------------------------------------------------------


def test_legacy_learning_signals_class_unchanged(tmp_path):
    """``LearningSignals`` class is the 3.4.20 API — must still work."""
    from superlocalmemory.learning.signals import LearningSignals

    db_path = str(tmp_path / "sig.db")
    ls = LearningSignals(db_path)
    ls.credit_channel("p1", "single_hop", "semantic", hit=True)
    ls.credit_channel("p1", "single_hop", "semantic", hit=False)
    ls.record_co_retrieval("p1", ["a", "b", "c"])
    stats = ls.get_signal_stats("p1")
    assert "co_retrieval_edges" in stats
    # Entropy-gap static helper.
    gap = LearningSignals.compute_entropy_gap([1.0, 0.0], [[0.0, 1.0]])
    assert 0.0 <= gap <= 1.0
    assert LearningSignals.compute_entropy_gap([], []) == 0.5


def test_hash_query_deterministic():
    from superlocalmemory.learning.signals import _hash_query

    a = _hash_query("Hello World")
    b = _hash_query("hello world")
    c = _hash_query("  hello world  ")
    assert a == b == c  # case + whitespace fold.
    assert len(a) == 32


# ---------------------------------------------------------------------------
# rerank() — exercise phase 1/2/3 branches
# ---------------------------------------------------------------------------


def test_rerank_baseline_orders_by_cross_encoder():
    r = AdaptiveRanker(signal_count=0)
    results = [
        {"fact_id": "a", "cross_encoder_score": 0.1},
        {"fact_id": "b", "cross_encoder_score": 0.9},
        {"fact_id": "c", "cross_encoder_score": 0.5},
    ]
    out = r.rerank(results, {})
    assert [x["fact_id"] for x in out] == ["b", "c", "a"]


def test_rerank_baseline_empty_returns_empty():
    r = AdaptiveRanker(signal_count=0)
    assert r.rerank([], {}) == []


def test_rerank_heuristic_applies_boosts():
    r = AdaptiveRanker(signal_count=100)  # phase 2
    results = [
        {"fact_id": "a", "cross_encoder_score": 0.5, "trust_score": 0.9,
         "fact": {"age_days": 1, "access_count": 5}},
        {"fact_id": "b", "cross_encoder_score": 0.5, "trust_score": 0.1,
         "fact": {"age_days": 365, "access_count": 0}},
    ]
    out = r.rerank(results, {})
    # Higher trust + recency → first.
    assert out[0]["fact_id"] == "a"


def test_rerank_ml_falls_back_without_model():
    r = AdaptiveRanker(signal_count=500)
    # phase would be 3 only if active_model also set; without, stays 2.
    results = [{"fact_id": "x", "cross_encoder_score": 0.5,
                "trust_score": 0.5, "fact": {"age_days": 0, "access_count": 0}}]
    out = r.rerank(results, {})
    assert len(out) == 1


def test_rerank_ml_uses_booster(tmp_path):
    _, model = _trained_model(tmp_path)
    r = AdaptiveRanker(signal_count=500, active_model=model)
    results = [
        {"fact_id": f"f-{i}",
         "channel_scores": {"semantic": i * 0.1},
         "cross_encoder_score": 0.5,
         "trust_score": 0.5,
         "fact": {"age_days": 0, "access_count": 0}}
        for i in range(5)
    ]
    out = r.rerank(results, {"query_type": "single_hop"})
    assert len(out) == 5


def test_rerank_ml_unknown_drift_falls_back(tmp_path):
    _, model = _trained_model(tmp_path)
    # Wrap into a drifted model.
    drifted = ActiveModel(
        profile_id=model.profile_id,
        booster=model.booster,
        feature_names=tuple(FEATURE_NAMES) + ("alien",),
        trained_at=model.trained_at,
        sha256=model.sha256,
    )
    r = AdaptiveRanker(signal_count=500, active_model=drifted)
    results = [
        {"fact_id": "a", "cross_encoder_score": 0.5,
         "trust_score": 0.5, "fact": {"age_days": 0, "access_count": 0}},
        {"fact_id": "b", "cross_encoder_score": 0.9,
         "trust_score": 0.5, "fact": {"age_days": 0, "access_count": 0}},
    ]
    out = r.rerank(results, {})
    # Heuristic fallback preserves ordering sensitivity to CE score.
    assert out[0]["fact_id"] == "b"


def test_legacy_bytes_load_roundtrip(tmp_path):
    _, model = _trained_model(tmp_path)
    state = model.booster.model_to_string().encode("utf-8")
    r = AdaptiveRanker(signal_count=500, model_state=state)
    assert r.active_model is not None
    # get_model_state returns bytes.
    assert r.get_model_state() is not None


def test_legacy_bytes_load_failure_sets_none():
    r = AdaptiveRanker(signal_count=500, model_state=b"not a valid lgbm model")
    assert r.active_model is None


def test_get_model_state_returns_none_without_model():
    r = AdaptiveRanker(signal_count=0)
    assert r.get_model_state() is None


def test_legacy_train_below_threshold_returns_false():
    r = AdaptiveRanker(signal_count=0)
    # 10 rows is well below PHASE_3_THRESHOLD (=200).
    assert r.train([{"features": {}, "label": 1.0}] * 10) is False


def test_legacy_train_empty_returns_false():
    r = AdaptiveRanker(signal_count=0)
    assert r.train([]) is False


def test_legacy_train_above_threshold_fits_booster():
    # 200 rows with varying features/labels — enough to fit and promote.
    training = []
    for i in range(220):
        training.append({
            "features": {
                "semantic_score": (i % 10) / 10.0,
                "cross_encoder_score": (i % 5) / 5.0,
            },
            "label": 1.0 if i % 2 == 0 else 0.0,
        })
    r = AdaptiveRanker(signal_count=0)
    ok = r.train(training)
    assert ok is True
    assert r.active_model is not None
    # get_model_state returns bytes after legacy train.
    state = r.get_model_state()
    assert state is not None
    assert isinstance(state, bytes)


# ---------------------------------------------------------------------------
# signals edge: _apply_shown_flip missing keys.
# ---------------------------------------------------------------------------


def test_apply_shown_flip_noop_when_missing_fact(tmp_path):
    from superlocalmemory.learning.signals import (
        SignalBatch,
        record_signal_batch,
    )

    db = make_db_with_migrations(tmp_path)
    conn = open_conn(db)
    batch = SignalBatch(
        profile_id="",
        query_id="",  # empty → no-op per implementation
        query_text="",
        candidates=(),
        query_context={"_shown_flip": {"fact_id": "x", "shown": True}},
    )
    # Should not raise.
    record_signal_batch(conn, batch)
    conn.close()


def test_reset_counters_helper():
    from superlocalmemory.learning.signals import (
        get_counters, reset_counters, _bump,
    )
    _bump("signal_dropped_total", 5)
    assert get_counters()["signal_dropped_total"] >= 5
    reset_counters()
    assert get_counters()["signal_dropped_total"] == 0


# ---------------------------------------------------------------------------
# Legacy LearningSignals class — smoke coverage for unchanged 3.4.20 API
# ---------------------------------------------------------------------------


def test_legacy_get_channel_weights_empty(tmp_path):
    from superlocalmemory.learning.signals import LearningSignals

    ls = LearningSignals(str(tmp_path / "s.db"))
    assert ls.get_channel_weights("p1", "single_hop") == {}


def test_legacy_get_channel_weights_after_hits(tmp_path):
    from superlocalmemory.learning.signals import LearningSignals

    ls = LearningSignals(str(tmp_path / "s.db"))
    for _ in range(6):
        ls.credit_channel("p1", "single_hop", "semantic", hit=True)
    weights = ls.get_channel_weights("p1", "single_hop")
    assert "semantic" in weights
    assert weights["semantic"] > 0.7


def test_legacy_co_retrieval_boost(tmp_path):
    from superlocalmemory.learning.signals import LearningSignals

    ls = LearningSignals(str(tmp_path / "s.db"))
    ls.record_co_retrieval("p1", ["a", "b", "c"])
    boosts = ls.get_co_retrieval_boost("p1", "a", top_k=2)
    assert len(boosts) <= 2
    assert all("fact_id" in b for b in boosts)


def test_legacy_co_retrieval_ignores_single_fact(tmp_path):
    from superlocalmemory.learning.signals import LearningSignals

    ls = LearningSignals(str(tmp_path / "s.db"))
    assert ls.record_co_retrieval("p1", ["only_one"]) == 0


def test_legacy_decay_missing_db_returns_zero():
    from superlocalmemory.learning.signals import LearningSignals

    # Point at a non-existent file — sqlite3 will raise, caught → 0.
    assert LearningSignals.decay_confidence(
        "/nonexistent/dir/nope.db", "p1", rate=0.01,
    ) == 0


def test_legacy_boost_missing_db_swallows():
    from superlocalmemory.learning.signals import LearningSignals

    # Must NOT raise.
    LearningSignals.boost_confidence("/nonexistent/dir/nope.db", "fid")


def test_cosine_sim_helper_safety():
    from superlocalmemory.learning.signals import _cosine_sim

    assert _cosine_sim([], []) == 0.0
    assert _cosine_sim([0.0], [0.0]) == 0.0
    assert abs(_cosine_sim([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-9


def test_legacy_get_channel_weights_filters_low_count(tmp_path):
    from superlocalmemory.learning.signals import LearningSignals

    ls = LearningSignals(str(tmp_path / "s.db"))
    # Only 2 events — falls below total>=5 threshold.
    ls.credit_channel("p1", "q", "bm25", hit=True)
    ls.credit_channel("p1", "q", "bm25", hit=False)
    weights = ls.get_channel_weights("p1", "q")
    assert weights == {}


def test_legacy_entropy_gap_identical_cluster():
    from superlocalmemory.learning.signals import LearningSignals

    # Same vector → high similarity → low gap.
    gap = LearningSignals.compute_entropy_gap([1.0, 0.0], [[1.0, 0.0]])
    assert gap < 0.2


# ---------------------------------------------------------------------------
# Extra model_cache defensive branches.
# ---------------------------------------------------------------------------


def test_load_active_wraps_db_exception(tmp_path):
    class _Bang:
        def load_active_model(self, _pid):
            raise RuntimeError("db on fire")

    result = model_cache.load_active(_Bang(), "p1", use_cache=False)
    assert result is None


def test_parse_row_empty_state_bytes():
    from superlocalmemory.learning.model_cache import _parse_row

    assert _parse_row(
        "pid",
        {"state_bytes": b"",
         "bytes_sha256": "0" * 64,
         "feature_names": "[]",
         "trained_at": ""},
    ) is None


def test_parse_row_bytearray_coerced(tmp_path):
    _, model = _trained_model(tmp_path)
    from superlocalmemory.learning.model_cache import _parse_row

    state = bytearray(model.booster.model_to_string().encode("utf-8"))
    import hashlib as _h
    sha = _h.sha256(state).hexdigest()
    parsed = _parse_row("p1", {
        "state_bytes": state,
        "bytes_sha256": sha,
        "feature_names": "[\"semantic_score\"]",
        "trained_at": "",
    })
    # Success with bytearray input → not None.
    assert parsed is not None


def test_lru_eviction_and_refresh():
    from superlocalmemory.learning.model_cache import _LRU

    lru = _LRU(maxsize=2)
    lru.set("a", None)
    lru.set("b", None)
    lru.set("a", None)  # refresh-existing path (line 77 move_to_end).
    lru.set("c", None)  # eviction path (line 80 popitem).
    assert lru.get("a") == (True, None)
    assert lru.get("b") == (False, None)  # evicted.
    assert lru.get("c") == (True, None)


def test_parse_row_feature_drift_info_logged(tmp_path, caplog):
    _, model = _trained_model(tmp_path)
    from superlocalmemory.learning.model_cache import _parse_row

    state = model.booster.model_to_string().encode("utf-8")
    import hashlib as _h
    sha = _h.sha256(state).hexdigest()
    import json as _json
    with caplog.at_level("INFO"):
        parsed = _parse_row("p1", {
            "state_bytes": state,
            "bytes_sha256": sha,
            "feature_names": _json.dumps(["only_one_name"]),
            "trained_at": "",
        })
    assert parsed is not None
    assert any("feature-drift" in rec.message for rec in caplog.records)
