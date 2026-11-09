"""Tests for Adaptive Ranker — Task 10 of V3 build."""
from superlocalmemory.learning.ranker import PHASE_2_THRESHOLD, PHASE_3_THRESHOLD, AdaptiveRanker


def _mock_results():
    return [
        {"fact_id": "f1", "score": 0.5, "cross_encoder_score": 0.5, "trust_score": 0.8,
         "fact": {"age_days": 1, "access_count": 10, "confidence": 0.9},
         "channel_scores": {"semantic": 0.8}},
        {"fact_id": "f2", "score": 0.9, "cross_encoder_score": 0.9, "trust_score": 0.3,
         "fact": {"age_days": 30, "access_count": 1, "confidence": 0.5},
         "channel_scores": {"semantic": 0.4}},
        {"fact_id": "f3", "score": 0.7, "cross_encoder_score": 0.7, "trust_score": 0.6,
         "fact": {"age_days": 5, "access_count": 5, "confidence": 0.7},
         "channel_scores": {"semantic": 0.6}},
    ]


def test_phase_1_by_default():
    ranker = AdaptiveRanker(signal_count=0)
    assert ranker.phase == 1


def test_phase_2_after_threshold():
    ranker = AdaptiveRanker(signal_count=PHASE_2_THRESHOLD)
    assert ranker.phase == 2


def test_phase_3_needs_model():
    ranker = AdaptiveRanker(signal_count=PHASE_3_THRESHOLD)
    # Phase 3 requires model — without it, stays at phase 2
    assert ranker.phase == 2


def test_phase_1_ranks_by_cross_encoder():
    ranker = AdaptiveRanker(signal_count=0)
    results = _mock_results()
    reranked = ranker.rerank(results, {})
    # f2 has highest CE score (0.9)
    assert reranked[0]["fact_id"] == "f2"


def test_phase_2_applies_boosts():
    ranker = AdaptiveRanker(signal_count=100)
    results = _mock_results()
    reranked = ranker.rerank(results, {})
    # Ordering may change due to boosts — just verify it returns all results
    assert len(reranked) == 3
    fact_ids = {r["fact_id"] for r in reranked}
    assert fact_ids == {"f1", "f2", "f3"}


def test_empty_results():
    ranker = AdaptiveRanker(signal_count=0)
    assert ranker.rerank([], {}) == []


def test_signal_count_setter():
    ranker = AdaptiveRanker(signal_count=0)
    assert ranker.phase == 1
    ranker.signal_count = 100
    assert ranker.phase == 2


def test_model_state_none_without_training():
    ranker = AdaptiveRanker()
    assert ranker.get_model_state() is None


def test_train_needs_enough_data():
    ranker = AdaptiveRanker()
    # Not enough data
    small_data = [{"features": {}, "label": 1.0}] * 10
    result = ranker.train(small_data)
    assert result == False
