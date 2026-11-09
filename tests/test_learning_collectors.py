#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Qualixar / SuperLocalMemory (superlocalmemory.com)
# Part of Qualixar | Author: Varun Pratap Bhardwaj (qualixar.com | varunpratap.com)
"""Tests for Learning Data Collectors -- Task 11 of V3 build."""

import pytest

from superlocalmemory.learning.engagement import EngagementTracker
from superlocalmemory.learning.feedback import FeedbackCollector
from superlocalmemory.learning.source_quality import SourceQualityScorer


@pytest.fixture
def feedback(tmp_path):
    return FeedbackCollector(tmp_path / "feedback.db")

@pytest.fixture
def engagement(tmp_path):
    return EngagementTracker(tmp_path / "engagement.db")

@pytest.fixture
def quality(tmp_path):
    return SourceQualityScorer(tmp_path / "quality.db")


# -- Feedback --
def test_record_implicit_feedback(feedback):
    feedback.record_implicit("p1", "where is alice?", ["f1", "f2"], ["f1", "f2", "f3"])
    count = feedback.get_feedback_count("p1")
    assert count >= 2  # f1,f2 hits + f3 miss

def test_record_explicit_feedback(feedback):
    feedback.record_explicit("p1", "f1", "user_positive", 1.0)
    records = feedback.get_feedback("p1", limit=10)
    assert len(records) >= 1

def test_feedback_count_increases(feedback):
    assert feedback.get_feedback_count("p1") == 0
    feedback.record_explicit("p1", "f1", "user_positive", 1.0)
    assert feedback.get_feedback_count("p1") == 1

# -- Engagement --
def test_record_engagement_event(engagement):
    engagement.record_event("p1", "recall")
    engagement.record_event("p1", "recall")
    engagement.record_event("p1", "store")
    stats = engagement.get_stats("p1")
    assert stats["recall_count"] >= 2
    assert stats["store_count"] >= 1

def test_engagement_health_inactive(engagement):
    health = engagement.get_health("p1")
    assert health == "inactive"  # no events

def test_engagement_health_active(engagement):
    for _ in range(20):
        engagement.record_event("p1", "recall")
    health = engagement.get_health("p1")
    assert health in ("active", "warm")

# -- Source Quality --
def test_new_source_default_quality(quality):
    score = quality.get_quality("p1", "source-1")
    assert score == 0.5  # uniform prior

def test_positive_outcome_increases_quality(quality):
    quality.record_outcome("p1", "source-1", "positive")
    quality.record_outcome("p1", "source-1", "positive")
    score = quality.get_quality("p1", "source-1")
    assert score > 0.5

def test_negative_outcome_decreases_quality(quality):
    quality.record_outcome("p1", "source-1", "negative")
    quality.record_outcome("p1", "source-1", "negative")
    score = quality.get_quality("p1", "source-1")
    assert score < 0.5

def test_all_qualities(quality):
    quality.record_outcome("p1", "s1", "positive")
    quality.record_outcome("p1", "s2", "negative")
    all_q = quality.get_all_qualities("p1")
    assert "s1" in all_q
    assert "s2" in all_q
    assert all_q["s1"] > all_q["s2"]
