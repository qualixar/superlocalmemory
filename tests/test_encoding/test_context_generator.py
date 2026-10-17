# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Tests for encoding/context_generator.py

"""Tests for ContextGenerator -- rules-based and LLM-based context generation."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from superlocalmemory.encoding.context_generator import ContextGenerator, ContextResult
from superlocalmemory.storage.models import AtomicFact, FactType, SignalType


def _make_fact(**overrides) -> AtomicFact:
    """Create a test AtomicFact with sensible defaults."""
    defaults = {
        "fact_id": "fact_test_001",
        "content": "Alice prefers Python for data science work",
        "fact_type": FactType.SEMANTIC,
        "signal_type": SignalType.FACTUAL,
        "canonical_entities": ["Alice", "Python"],
        "observation_date": "2026-03-15",
        "created_at": "2026-03-15T10:00:00",
    }
    defaults.update(overrides)
    return AtomicFact(**defaults)


class TestRulesBasedContext:
    """Rules-based (Mode A) context generation."""

    def test_template_includes_fact_type(self):
        fact = _make_fact(fact_type=FactType.EPISODIC)
        gen = ContextGenerator()
        result = gen.generate(fact, mode="a")
        assert "episodic event" in result.description

    def test_template_includes_entities(self):
        fact = _make_fact(canonical_entities=["Alice", "Python"])
        gen = ContextGenerator()
        result = gen.generate(fact, mode="a")
        assert "Alice" in result.description
        assert "Python" in result.description

    def test_template_includes_signal_type(self):
        fact = _make_fact(signal_type=SignalType.EMOTIONAL)
        gen = ContextGenerator()
        result = gen.generate(fact, mode="a")
        assert "emotional context" in result.description

    def test_template_includes_date(self):
        fact = _make_fact(observation_date="2026-03-15")
        gen = ContextGenerator()
        result = gen.generate(fact, mode="a")
        assert "2026-03-15" in result.description

    def test_keywords_extracted_from_entities(self):
        fact = _make_fact(canonical_entities=["Alice", "Python"])
        gen = ContextGenerator()
        result = gen.generate(fact, mode="a")
        assert "Alice" in result.keywords
        assert "Python" in result.keywords

    def test_keywords_extracted_from_content(self):
        fact = _make_fact(
            content="Alice strongly prefers Python for scientific computing",
            canonical_entities=[],
        )
        gen = ContextGenerator()
        result = gen.generate(fact, mode="a")
        # Words > 4 chars from content should appear
        assert any(w in result.keywords for w in ["alice", "strongly", "prefers", "python", "scientific"])

    def test_keywords_capped_at_10(self):
        fact = _make_fact(
            content="one two three four five six seven eight nine ten eleven twelve thirteen",
            canonical_entities=["A", "B", "C", "D", "E"],
        )
        gen = ContextGenerator()
        result = gen.generate(fact, mode="a")
        assert len(result.keywords) <= 10

    def test_empty_entities_handled(self):
        fact = _make_fact(canonical_entities=[])
        gen = ContextGenerator()
        result = gen.generate(fact, mode="a")
        assert "general knowledge" in result.description

    def test_generated_by_is_rules(self):
        gen = ContextGenerator()
        result = gen.generate(_make_fact(), mode="a")
        assert result.generated_by == "rules"

    def test_result_is_frozen_dataclass(self):
        gen = ContextGenerator()
        result = gen.generate(_make_fact(), mode="a")
        assert isinstance(result, ContextResult)
        with pytest.raises(AttributeError):
            result.description = "modified"


class TestLLMBasedContext:
    """LLM-based (Mode B/C) context generation."""

    def test_llm_prompt_contains_fact_content(self):
        mock_llm = MagicMock()
        mock_llm.is_available.return_value = True
        mock_llm.generate.return_value = json.dumps({
            "description": "Shows language preference",
            "keywords": ["Python", "preference"],
        })

        gen = ContextGenerator(llm=mock_llm)
        gen.generate(_make_fact(), mode="b")

        call_args = mock_llm.generate.call_args
        prompt = call_args[0][0]
        assert "Alice prefers Python" in prompt

    def test_llm_fenced_json_is_parsed(self):
        """4.2.9: gemma4:e2b wraps its answer in a ```json fence. A bare
        json.loads rejected it, so after the 12 Oct switch 2,147 of 2,149
        context calls on the live store fell back to rules - the model's
        work was thrown away."""
        mock_llm = MagicMock()
        mock_llm.is_available.return_value = True
        mock_llm.generate.return_value = (
            '```json\n{"description": "Shows a palette rule", '
            '"keywords": ["palette", "video"]}\n```'
        )
        result = ContextGenerator(llm=mock_llm).generate(_make_fact(), mode="b")
        assert result.generated_by == "ollama"
        assert result.description == "Shows a palette rule"
        assert result.keywords == ["palette", "video"]

    def test_llm_call_is_capped_and_not_thinking(self):
        """One sentence and five keywords need well under 200 tokens; the
        backbone default is 4,096. Thinking is off for this short answer."""
        mock_llm = MagicMock()
        mock_llm.is_available.return_value = True
        mock_llm.generate.return_value = '{"description": "d", "keywords": []}'
        ContextGenerator(llm=mock_llm).generate(_make_fact(), mode="b")
        kwargs = mock_llm.generate.call_args.kwargs
        assert 0 < kwargs["max_tokens"] <= 200
        assert kwargs["think"] is False

    def test_llm_response_parsed_as_json(self):
        mock_llm = MagicMock()
        mock_llm.is_available.return_value = True
        mock_llm.generate.return_value = json.dumps({
            "description": "Reveals programming language preference",
            "keywords": ["Python", "preference"],
        })

        gen = ContextGenerator(llm=mock_llm)
        result = gen.generate(_make_fact(), mode="b")
        assert result.description == "Reveals programming language preference"
        assert "Python" in result.keywords

    def test_llm_failure_falls_back_to_rules(self):
        mock_llm = MagicMock()
        mock_llm.is_available.return_value = True
        mock_llm.generate.side_effect = RuntimeError("LLM error")

        gen = ContextGenerator(llm=mock_llm)
        result = gen.generate(_make_fact(), mode="b")
        assert result.generated_by == "rules"

    def test_llm_unavailable_falls_back_to_rules(self):
        mock_llm = MagicMock()
        mock_llm.is_available.return_value = False

        gen = ContextGenerator(llm=mock_llm)
        result = gen.generate(_make_fact(), mode="b")
        assert result.generated_by == "rules"

    def test_mode_b_tags_as_ollama(self):
        mock_llm = MagicMock()
        mock_llm.is_available.return_value = True
        mock_llm.generate.return_value = json.dumps({
            "description": "test",
            "keywords": ["test"],
        })

        gen = ContextGenerator(llm=mock_llm)
        result = gen.generate(_make_fact(), mode="b")
        assert result.generated_by == "ollama"

    def test_mode_c_tags_as_cloud(self):
        mock_llm = MagicMock()
        mock_llm.is_available.return_value = True
        mock_llm.generate.return_value = json.dumps({
            "description": "test",
            "keywords": ["test"],
        })

        gen = ContextGenerator(llm=mock_llm)
        result = gen.generate(_make_fact(), mode="c")
        assert result.generated_by == "cloud"

    def test_llm_invalid_json_falls_back_to_rules(self):
        mock_llm = MagicMock()
        mock_llm.is_available.return_value = True
        mock_llm.generate.return_value = "not valid json"

        gen = ContextGenerator(llm=mock_llm)
        result = gen.generate(_make_fact(), mode="b")
        assert result.generated_by == "rules"

    def test_no_llm_uses_rules(self):
        gen = ContextGenerator(llm=None)
        result = gen.generate(_make_fact(), mode="b")
        assert result.generated_by == "rules"
