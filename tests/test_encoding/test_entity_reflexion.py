# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for encoding.entity_reflexion (Wave Q1).

Reflexion self-review over LLM-extracted entities: drop hallucinated,
add clearly-missed. Mode B/C only; fail-open on every failure path.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from superlocalmemory.core.config import EncodingConfig
from superlocalmemory.encoding.entity_reflexion import EntityReflexion
from superlocalmemory.encoding.fact_extractor import FactExtractor
from superlocalmemory.storage.models import AtomicFact, FactType, Mode


def _fact(content: str, entities: list[str]) -> AtomicFact:
    return AtomicFact(
        fact_id="f", memory_id="m", profile_id="default",
        content=content, fact_type=FactType.SEMANTIC, entities=list(entities),
    )


class _LLM:
    """Stub LLM that returns a scripted response and records calls."""

    def __init__(self, response: str, available: bool = True) -> None:
        self._response = response
        self._available = available
        self.calls = 0

    def is_available(self) -> bool:
        return self._available

    def generate(self, prompt: str, system: str = "", **kw) -> str:
        self.calls += 1
        return self._response


class TestEntityReflexionDrop:
    def test_drops_hallucinated_entity(self) -> None:
        llm = _LLM('[{"index": 0, "drop": ["Zeus"], "add": []}]')
        facts = [_fact("Varun met his manager at Accenture", ["Accenture", "Zeus"])]
        out = EntityReflexion(llm).refine("Varun met his manager at Accenture", facts)
        assert out[0].entities == ["Accenture"]
        # Immutability: a new object, original untouched.
        assert facts[0].entities == ["Accenture", "Zeus"]


class TestEntityReflexionAdd:
    def test_adds_missed_entity_when_grounded(self) -> None:
        llm = _LLM('[{"index": 0, "drop": [], "add": ["Paris"]}]')
        src = "Varun booked a trip to Paris next month"
        facts = [_fact("Varun booked a trip", [])]
        out = EntityReflexion(llm).refine(src, facts)
        assert "Paris" in out[0].entities

    def test_ungrounded_add_is_rejected(self) -> None:
        # LLM tries to inject an entity NOT present in the source text.
        llm = _LLM('[{"index": 0, "drop": [], "add": ["Tokyo"]}]')
        src = "Varun booked a trip to Paris"
        facts = [_fact("Varun booked a trip", [])]
        out = EntityReflexion(llm).refine(src, facts)
        assert "Tokyo" not in out[0].entities


class TestEntityReflexionFailOpen:
    def test_no_llm_returns_unchanged(self) -> None:
        facts = [_fact("a fact", ["X"])]
        assert EntityReflexion(None).refine("a fact", facts) is facts

    def test_unavailable_llm_returns_unchanged(self) -> None:
        llm = _LLM("[]", available=False)
        facts = [_fact("a fact", ["X"])]
        out = EntityReflexion(llm).refine("a fact", facts)
        assert out is facts
        assert llm.calls == 0

    def test_llm_error_is_fail_open(self) -> None:
        class _Boom:
            def is_available(self) -> bool:
                return True

            def generate(self, *a, **k) -> str:
                raise RuntimeError("model down")

        facts = [_fact("a fact", ["X"])]
        out = EntityReflexion(_Boom()).refine("a fact", facts)
        assert out is facts

    def test_unparseable_response_returns_unchanged(self) -> None:
        llm = _LLM("not json at all")
        facts = [_fact("a fact", ["X"])]
        out = EntityReflexion(llm).refine("a fact", facts)
        assert out[0].entities == ["X"]

    def test_empty_facts_short_circuits(self) -> None:
        llm = _LLM("[]")
        assert EntityReflexion(llm).refine("src", []) == []
        assert llm.calls == 0


class TestEntityReflexionBounds:
    def test_only_reviews_max_facts(self) -> None:
        # max_facts=1 -> only fact index 0 is sent/corrected; index 1 untouched
        # even if a (spurious) correction references it.
        llm = _LLM('[{"index": 1, "drop": ["Y"], "add": []}]')
        facts = [_fact("first", ["A"]), _fact("second", ["Y"])]
        out = EntityReflexion(llm, max_facts=1).refine("first second Y", facts)
        assert out[1].entities == ["Y"]  # beyond review window, unchanged

    def test_entities_capped(self) -> None:
        adds = ", ".join(f'"e{i}"' for i in range(50))
        src = " ".join(f"e{i}" for i in range(50))
        llm = _LLM(f'[{{"index": 0, "drop": [], "add": [{adds}]}}]')
        facts = [_fact("many", [])]
        out = EntityReflexion(llm).refine(src, facts)
        assert len(out[0].entities) <= 12


class TestFactExtractorReflexionIntegration:
    """Reflexion is wired into the Mode B/C extraction path only."""

    def _extraction_json(self) -> str:
        return json.dumps([
            {"text": "Alice works at Google as an engineer",
             "fact_type": "semantic",
             "entities": ["Alice", "Google", "Zeus"],
             "importance": 7, "confidence": 0.95},
        ])

    def test_reflexion_runs_in_mode_c(self) -> None:
        llm = MagicMock()
        llm.is_available.return_value = True
        # Call 1 = extraction, call 2 = reflexion (drops hallucinated "Zeus").
        llm.generate.side_effect = [
            self._extraction_json(),
            '[{"index": 0, "drop": ["Zeus"], "add": []}]',
        ]
        ext = FactExtractor(config=EncodingConfig(), llm=llm, mode=Mode.C)
        facts = ext.extract_facts(["Alice works at Google"], session_id="s1")
        assert len(facts) == 1
        assert "Zeus" not in facts[0].entities
        assert "Alice" in facts[0].entities
        assert llm.generate.call_count == 2

    def test_reflexion_disabled_by_config(self) -> None:
        llm = MagicMock()
        llm.is_available.return_value = True
        llm.generate.side_effect = [self._extraction_json()]
        ext = FactExtractor(
            config=EncodingConfig(enable_entity_reflexion=False),
            llm=llm, mode=Mode.C,
        )
        facts = ext.extract_facts(["Alice works at Google"], session_id="s1")
        # Reflexion skipped -> only the extraction call, "Zeus" survives.
        assert llm.generate.call_count == 1
        assert "Zeus" in facts[0].entities

    def test_mode_a_never_calls_llm(self) -> None:
        llm = MagicMock()
        llm.is_available.return_value = True
        ext = FactExtractor(config=EncodingConfig(), llm=llm, mode=Mode.A)
        ext.extract_facts(
            ["Alice Smith works at Google as an engineer."], session_id="s1",
        )
        llm.generate.assert_not_called()
