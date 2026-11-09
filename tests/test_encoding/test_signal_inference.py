# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.encoding.signal_inference.

Covers:
  - infer_signal() for all 6 signal types
  - infer_signal_scores() multi-signal scoring
  - Priority ordering (REQUEST > TEMPORAL > EMOTIONAL > OPINION > SOCIAL)
  - Default FACTUAL fallback
"""

from __future__ import annotations

from superlocalmemory.encoding.signal_inference import (
    infer_signal,
    infer_signal_scores,
)
from superlocalmemory.storage.models import SignalType

# ---------------------------------------------------------------------------
# infer_signal — single type
# ---------------------------------------------------------------------------

class TestInferSignal:
    def test_factual_default(self) -> None:
        result = infer_signal("Paris is the capital of France")
        assert result == SignalType.FACTUAL

    def test_emotional(self) -> None:
        result = infer_signal("I am feeling so happy and excited today")
        assert result == SignalType.EMOTIONAL

    def test_temporal(self) -> None:
        result = infer_signal("The meeting is scheduled for next month")
        assert result == SignalType.TEMPORAL

    def test_opinion(self) -> None:
        result = infer_signal("I think Python is the best language")
        assert result == SignalType.OPINION

    def test_request(self) -> None:
        result = infer_signal("Can you help me find the document")
        assert result == SignalType.REQUEST

    def test_social(self) -> None:
        result = infer_signal("I met with my colleague and friend yesterday")
        assert result == SignalType.SOCIAL

    def test_empty_text(self) -> None:
        result = infer_signal("")
        assert result == SignalType.FACTUAL

    def test_priority_request_over_temporal(self) -> None:
        # REQUEST is higher priority than TEMPORAL
        result = infer_signal("Please remind me about the deadline tomorrow")
        assert result == SignalType.REQUEST

    def test_priority_temporal_over_emotional(self) -> None:
        # TEMPORAL is higher priority than EMOTIONAL
        result = infer_signal("I was happy when I heard the schedule yesterday")
        assert result == SignalType.TEMPORAL

    def test_priority_emotional_over_opinion(self) -> None:
        # EMOTIONAL is higher priority than OPINION
        result = infer_signal("I am angry and I think they are wrong")
        assert result == SignalType.EMOTIONAL


# ---------------------------------------------------------------------------
# infer_signal_scores — multi-signal
# ---------------------------------------------------------------------------

class TestInferSignalScores:
    def test_returns_all_types(self) -> None:
        scores = infer_signal_scores("Alice works at Google in Paris")
        assert SignalType.FACTUAL in scores
        assert SignalType.EMOTIONAL in scores
        assert SignalType.TEMPORAL in scores
        assert SignalType.OPINION in scores
        assert SignalType.REQUEST in scores
        assert SignalType.SOCIAL in scores

    def test_emotional_text_scores_high(self) -> None:
        scores = infer_signal_scores(
            "I am happy and excited and thrilled and grateful and proud"
        )
        assert scores[SignalType.EMOTIONAL] > 0.5

    def test_factual_inversely_proportional(self) -> None:
        # FACTUAL = 1 - max(other scores)
        scores = infer_signal_scores("neutral text about nothing special")
        assert scores[SignalType.FACTUAL] > 0.5

    def test_mixed_content(self) -> None:
        scores = infer_signal_scores(
            "I think the deadline is tomorrow and my friend is worried"
        )
        # Multiple types should have non-zero scores
        non_zero = [s for s in scores.values() if s > 0]
        assert len(non_zero) >= 3

    def test_capped_at_one(self) -> None:
        # Many temporal words
        scores = infer_signal_scores(
            "when date time schedule deadline tomorrow yesterday "
            "last week next month ago soon"
        )
        for score in scores.values():
            assert score <= 1.0

    def test_empty_text(self) -> None:
        scores = infer_signal_scores("")
        assert scores[SignalType.FACTUAL] == 1.0
        for stype in (SignalType.EMOTIONAL, SignalType.TEMPORAL,
                      SignalType.OPINION, SignalType.REQUEST, SignalType.SOCIAL):
            assert scores[stype] == 0.0
