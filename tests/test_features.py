"""Tests for V3 Feature Extractor — Task 9 of V3 build."""
import pytest

from superlocalmemory.learning.features import (
    FEATURE_DIM,
    FEATURE_NAMES,
    FeatureExtractor,
    FeatureVector,
)


def _mock_result(**overrides):
    base = {
        "fact_id": "f1",
        "score": 0.8,
        "channel_scores": {"semantic": 0.9, "bm25": 0.3, "entity_graph": 0.5, "temporal": 0.1},
        "rrf_rank": 1,
        "rrf_score": 0.85,
        "cross_encoder_score": 0.7,
        "fisher_distance": 0.2,
        "fisher_confidence": 0.8,
        "sheaf_consistent": True,
        "trust_score": 0.9,
        "fact": {"age_days": 5, "access_count": 3, "confidence": 0.85},
    }
    base.update(overrides)
    return base


def _mock_query_context(**overrides):
    base = {
        "query_id": "q1",
        "query_type": "single_hop",
        "profile_recall_count": 50,
        "topic_affinity": 0.7,
    }
    base.update(overrides)
    return base


def test_feature_dim_matches_names():
    assert len(FEATURE_NAMES) == FEATURE_DIM


def test_extract_returns_feature_vector():
    result = _mock_result()
    ctx = _mock_query_context()
    fv = FeatureExtractor.extract(result, ctx)
    assert isinstance(fv, FeatureVector)
    assert fv.fact_id == "f1"


def test_extract_has_all_features():
    fv = FeatureExtractor.extract(_mock_result(), _mock_query_context())
    assert len(fv.features) == FEATURE_DIM


def test_to_list_has_fixed_length():
    fv = FeatureExtractor.extract(_mock_result(), _mock_query_context())
    vec = fv.to_list()
    assert len(vec) == FEATURE_DIM


def test_channel_scores_extracted():
    fv = FeatureExtractor.extract(_mock_result(), _mock_query_context())
    assert fv.features["semantic_score"] == 0.9
    assert fv.features["bm25_score"] == 0.3


def test_math_features_extracted():
    fv = FeatureExtractor.extract(_mock_result(), _mock_query_context())
    assert fv.features["fisher_distance"] == 0.2
    assert fv.features["sheaf_consistent"] == 1.0


def test_query_type_one_hot():
    ctx = _mock_query_context(query_type="multi_hop")
    fv = FeatureExtractor.extract(_mock_result(), ctx)
    assert fv.features["query_type_mh"] == 1.0
    assert fv.features["query_type_sh"] == 0.0


def test_missing_channel_defaults_to_zero():
    result = _mock_result(channel_scores={})
    fv = FeatureExtractor.extract(result, _mock_query_context())
    assert fv.features["semantic_score"] == 0.0
    assert fv.features["bm25_score"] == 0.0


def test_batch_extract():
    results = [_mock_result(fact_id="f1"), _mock_result(fact_id="f2")]
    batch = FeatureExtractor.extract_batch(results, _mock_query_context())
    assert len(batch) == 2
    assert batch[0].fact_id == "f1"
    assert batch[1].fact_id == "f2"


def test_feature_vector_immutable():
    fv = FeatureExtractor.extract(_mock_result(), _mock_query_context())
    with pytest.raises(AttributeError):
        fv.fact_id = "changed"


def test_safe_float_handles_none():
    result = _mock_result(trust_score=None)
    fv = FeatureExtractor.extract(result, _mock_query_context())
    assert fv.features["fact_trust_score"] == 0.0
