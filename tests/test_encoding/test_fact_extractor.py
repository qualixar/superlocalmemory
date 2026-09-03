# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.encoding.fact_extractor.

Covers:
  - chunk_turns() partitioning and overlap
  - Mode A local extraction: sentences, entities, classification, scoring
  - Mode B/C LLM extraction: parsing, fallback to local
  - Deduplication (content normalization, importance tiebreaker)
  - Helper functions: _is_filler, _extract_entities, _classify_sentence, etc.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from superlocalmemory.core.config import EncodingConfig
from superlocalmemory.encoding.fact_extractor import (
    FactExtractor,
    _classify_sentence,
    _extract_date_string,
    _extract_entities,
    _extract_interval,
    _is_filler,
    _score_importance,
    _signal_from_fact_type,
    _split_sentences,
    _try_parse_date,
    chunk_turns,
)
from superlocalmemory.storage.models import AtomicFact, FactType, Mode, SignalType


# ---------------------------------------------------------------------------
# chunk_turns
# ---------------------------------------------------------------------------

class TestChunkTurns:
    def test_empty_input(self) -> None:
        assert chunk_turns([]) == []

    def test_single_turn(self) -> None:
        result = chunk_turns(["hello"])
        assert result == [["hello"]]

    def test_under_chunk_size(self) -> None:
        turns = [f"turn_{i}" for i in range(5)]
        result = chunk_turns(turns, chunk_size=10)
        assert len(result) == 1
        assert result[0] == turns

    def test_exact_chunk_size(self) -> None:
        turns = [f"turn_{i}" for i in range(10)]
        result = chunk_turns(turns, chunk_size=10)
        assert len(result) == 1

    def test_overlap_present(self) -> None:
        turns = [f"turn_{i}" for i in range(15)]
        chunks = chunk_turns(turns, chunk_size=10, overlap=2)
        assert len(chunks) >= 2
        # Overlapping turns should appear in both chunks
        first_end = set(chunks[0][-2:])
        second_start = set(chunks[1][:2])
        assert len(first_end & second_start) > 0

    def test_trailing_fragment_merged(self) -> None:
        # 12 turns, chunk_size=10, overlap=2 → step=8
        # After first chunk (0-10), remaining = 2 < overlap+1=3
        # So fragment merges into chunk[0]
        turns = [f"t{i}" for i in range(12)]
        chunks = chunk_turns(turns, chunk_size=10, overlap=2)
        assert len(chunks) == 1
        assert len(chunks[0]) == 12


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_split_sentences(self) -> None:
        text = "Alice is an engineer. Bob works at Google. Hi."
        parts = _split_sentences(text)
        assert len(parts) == 2  # "Hi." is too short (<8 chars)

    def test_extract_date_string_iso(self) -> None:
        assert _extract_date_string("Meeting on 2026-03-11 at noon") == "2026-03-11"

    def test_extract_date_string_us(self) -> None:
        result = _extract_date_string("Due date is 3/15/2026")
        assert result is not None

    def test_extract_date_string_relative(self) -> None:
        result = _extract_date_string("I went there yesterday")
        assert result == "yesterday"

    def test_extract_date_string_none(self) -> None:
        assert _extract_date_string("No dates here at all") is None

    def test_try_parse_date_iso(self) -> None:
        assert _try_parse_date("2026-03-11") == "2026-03-11"

    def test_try_parse_date_none(self) -> None:
        assert _try_parse_date("") is None
        assert _try_parse_date(None) is None

    def test_extract_interval(self) -> None:
        start, end = _extract_interval("From January 1 to March 1, 2026")
        assert start is not None
        assert end is not None

    def test_extract_interval_none(self) -> None:
        start, end = _extract_interval("No interval here")
        assert start is None and end is None

    def test_extract_entities(self) -> None:
        entities = _extract_entities("Alice Smith met Bob Jones at Google HQ")
        assert "Alice Smith" in entities
        assert "Bob Jones" in entities

    def test_extract_entities_filters_stopwords(self) -> None:
        entities = _extract_entities("The quick brown fox")
        # "The" should be filtered; others not capitalized
        assert "The" not in entities

    def test_extract_entities_quoted(self) -> None:
        entities = _extract_entities('She watched "Inception" last night')
        assert "Inception" in entities

    def test_is_filler_positive(self) -> None:
        assert _is_filler("hello there!") is True
        assert _is_filler("Thanks for the info") is True
        assert _is_filler("Okay") is True

    def test_is_filler_negative(self) -> None:
        assert _is_filler("Alice works at Google as an engineer") is False

    def test_classify_sentence_prospective(self) -> None:
        """A deadline is something planned for later, not a time-bounded event."""
        assert _classify_sentence("The deadline is next week") == FactType.PROSPECTIVE

    def test_classify_sentence_opinion(self) -> None:
        assert _classify_sentence("I think Python is great") == FactType.OPINION

    def test_classify_sentence_episodic(self) -> None:
        assert _classify_sentence("I went to Paris last summer") == FactType.EPISODIC

    def test_classify_sentence_semantic(self) -> None:
        assert _classify_sentence("Paris is the capital of France") == FactType.SEMANTIC


class TestScoreImportance:
    def test_base_score(self) -> None:
        score = _score_importance("neutral text", [], {}, False)
        assert score == pytest.approx(0.3, abs=0.01)

    def test_emotional_boost(self) -> None:
        score = _score_importance("I love this amazing thing", [], {}, False)
        assert score > 0.3

    def test_temporal_boost(self) -> None:
        score = _score_importance("some text", [], {}, True)
        assert score >= 0.5

    def test_entity_frequency_boost(self) -> None:
        score = _score_importance(
            "Alice is great", ["Alice"], {"Alice": 10, "Bob": 1}, False,
        )
        assert score > 0.3

    def test_capped_at_one(self) -> None:
        score = _score_importance(
            "I love this amazing incredible thing",
            ["Alice"], {"Alice": 100}, True,
        )
        assert score <= 1.0


class TestSignalFromFactType:
    def test_mapping(self) -> None:
        assert _signal_from_fact_type(FactType.EPISODIC) == SignalType.FACTUAL
        assert _signal_from_fact_type(FactType.OPINION) == SignalType.OPINION
        # SignalType.TEMPORAL is a different concept and keeps its name: the
        # signal is about a sentence carrying time information, not about the
        # memory being something the user intends to do later.
        assert _signal_from_fact_type(FactType.PROSPECTIVE) == SignalType.TEMPORAL
        assert _signal_from_fact_type(FactType.SEMANTIC) == SignalType.FACTUAL


# ---------------------------------------------------------------------------
# FactExtractor Mode A
# ---------------------------------------------------------------------------

class TestFactExtractorModeA:
    def _make_extractor(self, **overrides) -> FactExtractor:
        cfg = EncodingConfig(**overrides)
        return FactExtractor(config=cfg, mode=Mode.A)

    def test_empty_turns(self) -> None:
        ext = self._make_extractor()
        assert ext.extract_facts([], session_id="s1") == []

    def test_basic_extraction(self) -> None:
        ext = self._make_extractor(min_fact_confidence=0.0)
        turns = [
            "Alice Smith works at Google as a software engineer.",
            "She visited Paris last March and loved it.",
        ]
        facts = ext.extract_facts(turns, session_id="s1", session_date="2026-03-11")
        assert len(facts) > 0
        assert all(isinstance(f, AtomicFact) for f in facts)
        assert all(f.session_id == "s1" for f in facts)

    def test_filler_filtered(self) -> None:
        ext = self._make_extractor(min_fact_confidence=0.0)
        turns = ["Hello there!", "Thanks for the info.", "Alice works at Google as an engineer."]
        facts = ext.extract_facts(turns, session_id="s1")
        # The greetings should be filtered out
        contents = [f.content.lower() for f in facts]
        assert not any("hello" in c for c in contents)

    def test_deduplication(self) -> None:
        ext = self._make_extractor(min_fact_confidence=0.0)
        turns = [
            "Alice Smith works at Google.",
            "Alice Smith works at Google.",  # exact duplicate
        ]
        facts = ext.extract_facts(turns, session_id="s1")
        # Dedup should collapse duplicates
        contents = [f.content for f in facts]
        # All should be unique after dedup
        normalized = [c.lower().strip() for c in contents]
        assert len(normalized) == len(set(normalized))

    def test_max_facts_per_chunk(self) -> None:
        ext = self._make_extractor(max_facts_per_chunk=2, min_fact_confidence=0.0)
        turns = [
            "Alice works at Google. Bob works at Apple. Carol works at Meta.",
            "Dave works at Amazon. Eve works at Tesla.",
        ]
        facts = ext.extract_facts(turns, session_id="s1")
        assert len(facts) <= 2

    def test_speaker_inference(self) -> None:
        result = FactExtractor._infer_speaker(
            "Alice works at Google",
            ["Alice works at Google", "Nice"],
            "User", "Assistant",
        )
        assert result == "User"  # Index 0 = even = speaker_a

    def test_speaker_inference_no_speakers(self) -> None:
        result = FactExtractor._infer_speaker("test", ["test"], "", "")
        assert result == ""


# ---------------------------------------------------------------------------
# FactExtractor Mode B/C (LLM)
# ---------------------------------------------------------------------------

class TestFactExtractorModeLLM:
    def _mock_llm(self, response: str) -> MagicMock:
        llm = MagicMock()
        llm.is_available.return_value = True
        llm.generate.return_value = response
        return llm

    def test_llm_extraction(self) -> None:
        response = json.dumps([
            {"text": "Alice works at Google as a software engineer",
             "fact_type": "semantic", "entities": ["Alice", "Google"],
             "importance": 7, "confidence": 0.95},
        ])
        llm = self._mock_llm(response)
        ext = FactExtractor(config=EncodingConfig(), llm=llm, mode=Mode.C)

        facts = ext.extract_facts(
            ["Alice works at Google"], session_id="s1", session_date="2026-03-11",
        )
        assert len(facts) == 1
        assert facts[0].content == "Alice works at Google as a software engineer"
        assert facts[0].fact_type == FactType.SEMANTIC
        assert "Alice" in facts[0].entities

    def test_llm_fallback_to_local(self) -> None:
        llm = self._mock_llm("")  # Empty response triggers fallback
        ext = FactExtractor(
            config=EncodingConfig(min_fact_confidence=0.0),
            llm=llm, mode=Mode.B,
        )
        facts = ext.extract_facts(
            ["Alice Smith works at Google as an engineer."],
            session_id="s1",
        )
        assert len(facts) > 0  # Falls back to Mode A

    def test_llm_exception_fallback(self) -> None:
        llm = MagicMock()
        llm.is_available.return_value = True
        llm.generate.side_effect = RuntimeError("API timeout")
        ext = FactExtractor(
            config=EncodingConfig(min_fact_confidence=0.0),
            llm=llm, mode=Mode.C,
        )
        facts = ext.extract_facts(
            ["Alice Smith works at Google as an engineer."],
            session_id="s1",
        )
        assert len(facts) > 0  # Falls back to Mode A

    def test_llm_bad_json(self) -> None:
        llm = self._mock_llm("not valid json at all {{{")
        ext = FactExtractor(config=EncodingConfig(), llm=llm, mode=Mode.C)
        facts = ext.extract_facts(
            ["Alice works at Google"], session_id="s1",
        )
        # Should fall back to local extraction

    def test_llm_importance_normalization(self) -> None:
        response = json.dumps([
            {"text": "Alice loves hiking", "fact_type": "opinion",
             "entities": ["Alice"], "importance": 10, "confidence": 0.9},
        ])
        llm = self._mock_llm(response)
        ext = FactExtractor(config=EncodingConfig(), llm=llm, mode=Mode.C)
        facts = ext.extract_facts(["Alice loves hiking"], session_id="s1")
        assert len(facts) == 1
        assert facts[0].importance == 1.0  # 10/10 = 1.0

    def test_llm_item_with_string_entities(self) -> None:
        response = json.dumps([
            {"text": "Alice is an engineer", "entities": "Alice",
             "importance": 5, "confidence": 0.8},
        ])
        llm = self._mock_llm(response)
        ext = FactExtractor(config=EncodingConfig(), llm=llm, mode=Mode.C)
        facts = ext.extract_facts(["Alice is an engineer"], session_id="s1")
        assert len(facts) == 1
        assert facts[0].entities == ["Alice"]

    def test_llm_unavailable_uses_local(self) -> None:
        llm = MagicMock()
        llm.is_available.return_value = False
        ext = FactExtractor(
            config=EncodingConfig(min_fact_confidence=0.0),
            llm=llm, mode=Mode.C,
        )
        facts = ext.extract_facts(
            ["Alice Smith works at Google as an engineer."],
            session_id="s1",
        )
        assert len(facts) > 0

    def test_llm_thinking_trace_yields_real_facts(self) -> None:
        # Issue #128: a thinking model's answer arrives wrapped in its
        # reasoning trace (content empty, thinking populated). The JSON
        # array inside the trace must still extract — and the assertion
        # below must ONLY pass when the LLM path ran: the expected
        # content string ("...as a software engineer") never appears in
        # the input turns, so Mode-A fallback can never produce it.
        trace = (
            "Let me think step by step. Alice is the subject, Google the "
            "employer. Final answer: "
            + json.dumps([
                {"text": "Alice works at Google as a software engineer",
                 "fact_type": "semantic", "entities": ["Alice", "Google"],
                 "importance": 7, "confidence": 0.95},
            ])
        )
        llm = self._mock_llm(trace)
        ext = FactExtractor(config=EncodingConfig(), llm=llm, mode=Mode.B)
        facts = ext.extract_facts(
            ["Alice works at Google"], session_id="s1",
        )
        assert len(facts) == 1
        assert (
            facts[0].content
            == "Alice works at Google as a software engineer"
        )
        assert "Alice" in facts[0].entities
        assert "Google" in facts[0].entities
        assert facts[0].importance == 0.7

    def test_llm_thinking_trace_with_bracket_noise(self) -> None:
        # Real qwen traces interleave bracket noise ([Alice, Google],
        # [1]) before the answer. The legacy first-[-to-last-] span is
        # not valid JSON there; array recovery must skip the noise and
        # still extract the answer.
        trace = (
            "Candidates [Alice, Google] ranked [1] and [2]. "
            "Trial parse failed. Final answer: "
            + json.dumps([
                {"text": "Alice works at Google as a software engineer",
                 "fact_type": "semantic", "entities": ["Alice", "Google"],
                 "importance": 7, "confidence": 0.95},
            ])
            + " Done."
        )
        llm = self._mock_llm(trace)
        ext = FactExtractor(config=EncodingConfig(), llm=llm, mode=Mode.B)
        facts = ext.extract_facts(
            ["Alice works at Google"], session_id="s1",
        )
        assert len(facts) == 1
        assert (
            facts[0].content
            == "Alice works at Google as a software engineer"
        )

    def test_llm_trial_payload_does_not_shadow_final_answer(self) -> None:
        # A mid-trace trial array is itself viable — first-wins would
        # store the draft as a real memory. Last-viable must store the
        # final answer instead.
        trial = json.dumps([
            {"text": "Alice drinks coffee every morning",
             "fact_type": "habit", "entities": ["Alice"],
             "importance": 3, "confidence": 0.8},
        ])
        final = json.dumps([
            {"text": "Alice works at Google as a software engineer",
             "fact_type": "semantic", "entities": ["Alice", "Google"],
             "importance": 7, "confidence": 0.95},
        ])
        trace = f"Draft attempt: {trial} On reflection, better: Final: {final}"
        llm = self._mock_llm(trace)
        ext = FactExtractor(config=EncodingConfig(), llm=llm, mode=Mode.B)
        facts = ext.extract_facts(
            ["Alice works at Google"], session_id="s1",
        )
        assert len(facts) == 1
        assert (
            facts[0].content
            == "Alice works at Google as a software engineer"
        )

    def test_llm_trailing_decoy_falls_through_to_answer(self) -> None:
        # A trailing decoy yielding zero facts ([{"name": ...}] has no
        # text) must not shadow the real answer above it.
        final = json.dumps([
            {"text": "Alice works at Google as a software engineer",
             "fact_type": "semantic", "entities": ["Alice", "Google"],
             "importance": 7, "confidence": 0.95},
        ])
        trace = f"Final: {final} Metadata: [{{\"name\": \"Alice\"}}]"
        llm = self._mock_llm(trace)
        ext = FactExtractor(config=EncodingConfig(), llm=llm, mode=Mode.B)
        facts = ext.extract_facts(
            ["Alice works at Google"], session_id="s1",
        )
        assert len(facts) == 1
        assert (
            facts[0].content
            == "Alice works at Google as a software engineer"
        )

    def test_llm_empty_response_warns(self, caplog) -> None:
        llm = self._mock_llm("")
        ext = FactExtractor(
            config=EncodingConfig(min_fact_confidence=0.0),
            llm=llm, mode=Mode.B,
        )
        with caplog.at_level("WARNING"):
            facts = ext.extract_facts(
                ["Alice Smith works at Google as an engineer."],
                session_id="s1",
            )
        assert len(facts) > 0  # Falls back to Mode A
        assert "empty response text" in caplog.text


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_keeps_higher_importance(self) -> None:
        f1 = AtomicFact(fact_id="a", content="Alice works at Google", importance=0.3)
        f2 = AtomicFact(fact_id="b", content="alice works at google", importance=0.9)
        result = FactExtractor._deduplicate([f1, f2])
        assert len(result) == 1
        assert result[0].importance == 0.9

    def test_whitespace_normalization(self) -> None:
        f1 = AtomicFact(fact_id="a", content="Alice   works  at Google", importance=0.5)
        f2 = AtomicFact(fact_id="b", content="Alice works at Google", importance=0.5)
        result = FactExtractor._deduplicate([f1, f2])
        assert len(result) == 1

    def test_unique_facts_preserved(self) -> None:
        f1 = AtomicFact(fact_id="a", content="Alice works at Google", importance=0.5)
        f2 = AtomicFact(fact_id="b", content="Bob works at Apple", importance=0.5)
        result = FactExtractor._deduplicate([f1, f2])
        assert len(result) == 2
