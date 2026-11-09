# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for Learning Database — Task 8 of V3 build."""


import pytest

from superlocalmemory.learning.database import LearningDatabase


@pytest.fixture
def db(tmp_path):
    return LearningDatabase(tmp_path / "learning.db")


def test_store_and_get_signal(db):
    db.store_signal("p1", "where is alice?", "f1", "recall_hit", 1.0)
    count = db.get_signal_count("p1")
    assert count >= 1


def test_store_and_get_features(db):
    features = {"semantic_score": 0.8, "bm25_score": 0.3, "fisher_distance": 0.2}
    db.store_features("p1", "q1", "f1", features, label=1.0)
    data = db.get_training_data("p1", limit=10)
    assert len(data) >= 1
    assert "semantic_score" in data[0]["features"]


def test_signal_count_increases(db):
    assert db.get_signal_count("p1") == 0
    db.store_signal("p1", "q", "f1", "recall_hit", 1.0)
    db.store_signal("p1", "q", "f2", "recall_miss", 0.0)
    assert db.get_signal_count("p1") == 2


def test_model_state_roundtrip(db):
    state = b"fake_model_bytes_here"
    db.store_model_state("p1", state)
    loaded = db.load_model_state("p1")
    assert loaded == state


def test_model_state_returns_none_when_empty(db):
    assert db.load_model_state("p1") is None


def test_engagement_tracking(db):
    db.record_engagement("p1", "recall_count", 1)
    db.record_engagement("p1", "recall_count", 1)
    db.record_engagement("p1", "store_count", 1)
    stats = db.get_engagement_stats("p1")
    assert stats.get("recall_count", 0) >= 2
    assert stats.get("store_count", 0) >= 1


def test_training_data_empty_profile(db):
    data = db.get_training_data("empty_profile", limit=10)
    assert data == []


def test_multiple_profiles_isolated(db):
    db.store_signal("p1", "q", "f1", "hit", 1.0)
    db.store_signal("p2", "q", "f2", "hit", 1.0)
    assert db.get_signal_count("p1") == 1
    assert db.get_signal_count("p2") == 1
