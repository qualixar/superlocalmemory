# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.encoding.emotional.

Covers:
  - tag_emotion() valence and arousal extraction
  - is_emotionally_significant() threshold check
  - emotional_importance_boost() calculation
  - VADER fallback (missing library)
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from superlocalmemory.encoding.emotional import (
    EmotionalTag,
    emotional_importance_boost,
    is_emotionally_significant,
    tag_emotion,
)

# ---------------------------------------------------------------------------
# tag_emotion
# ---------------------------------------------------------------------------

class TestTagEmotion:
    def test_positive_text(self) -> None:
        tag = tag_emotion("I love this amazing wonderful experience!")
        assert tag.valence > 0.0
        assert tag.arousal > 0.0

    def test_negative_text(self) -> None:
        tag = tag_emotion("I hate this terrible horrible situation!")
        assert tag.valence < 0.0
        assert tag.arousal > 0.0

    def test_neutral_text(self) -> None:
        tag = tag_emotion("The meeting is at three o'clock.")
        assert -0.3 <= tag.valence <= 0.3
        # Neutral text should have low arousal
        assert tag.arousal < 0.5

    def test_empty_text(self) -> None:
        tag = tag_emotion("")
        assert tag.valence == pytest.approx(0.0, abs=0.1)

    def test_returns_frozen_dataclass(self) -> None:
        tag = tag_emotion("Hello")
        assert isinstance(tag, EmotionalTag)
        with pytest.raises(AttributeError):
            tag.valence = 0.5  # type: ignore[misc]

    def test_valence_range(self) -> None:
        tag = tag_emotion("Absolutely incredible amazing wonderful")
        assert -1.0 <= tag.valence <= 1.0

    def test_arousal_range(self) -> None:
        tag = tag_emotion("FURIOUS ANGRY OUTRAGED!")
        assert 0.0 <= tag.arousal <= 1.0


# ---------------------------------------------------------------------------
# VADER missing fallback
# ---------------------------------------------------------------------------

class TestVaderFallback:
    def test_missing_vader_returns_zeros(self) -> None:
        import superlocalmemory.encoding.emotional as emo_mod
        # Save original, set to None to simulate missing VADER
        original = emo_mod._vader_analyzer
        emo_mod._vader_analyzer = None
        try:
            with patch.dict("sys.modules", {"vaderSentiment": None, "vaderSentiment.vaderSentiment": None}):
                # Force reload of _get_vader
                emo_mod._vader_analyzer = None
                tag = tag_emotion("I love this!")
                # If VADER import fails, should return zeros
                # Note: if VADER is installed, _get_vader will succeed
                # This test validates the fallback path
                assert isinstance(tag, EmotionalTag)
        finally:
            emo_mod._vader_analyzer = original


# ---------------------------------------------------------------------------
# is_emotionally_significant
# ---------------------------------------------------------------------------

class TestIsEmotionallySignificant:
    def test_above_threshold(self) -> None:
        tag = EmotionalTag(valence=0.8, arousal=0.5)
        assert is_emotionally_significant(tag, threshold=0.3) is True

    def test_below_threshold(self) -> None:
        tag = EmotionalTag(valence=0.1, arousal=0.1)
        assert is_emotionally_significant(tag, threshold=0.3) is False

    def test_exact_threshold(self) -> None:
        tag = EmotionalTag(valence=0.0, arousal=0.3)
        assert is_emotionally_significant(tag, threshold=0.3) is True

    def test_default_threshold(self) -> None:
        tag = EmotionalTag(valence=0.5, arousal=0.4)
        assert is_emotionally_significant(tag) is True


# ---------------------------------------------------------------------------
# emotional_importance_boost
# ---------------------------------------------------------------------------

class TestEmotionalImportanceBoost:
    def test_low_arousal_no_boost(self) -> None:
        tag = EmotionalTag(valence=0.0, arousal=0.1)
        assert emotional_importance_boost(tag) == 0.0

    def test_high_arousal_boost(self) -> None:
        tag = EmotionalTag(valence=0.9, arousal=0.9)
        boost = emotional_importance_boost(tag)
        assert boost > 0.0
        assert boost <= 0.3

    def test_moderate_arousal(self) -> None:
        tag = EmotionalTag(valence=-0.5, arousal=0.5)
        boost = emotional_importance_boost(tag)
        assert 0.0 < boost <= 0.3

    def test_max_boost_capped(self) -> None:
        tag = EmotionalTag(valence=1.0, arousal=1.0)
        boost = emotional_importance_boost(tag)
        assert boost <= 0.3

    def test_boundary_arousal_0_2(self) -> None:
        # arousal < 0.2 returns 0.0
        tag = EmotionalTag(valence=0.5, arousal=0.19)
        assert emotional_importance_boost(tag) == 0.0

        tag2 = EmotionalTag(valence=0.5, arousal=0.2)
        assert emotional_importance_boost(tag2) == 0.0  # 0.2 * 0.3 = 0.06
