# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.encoding.type_router.

Covers:
  - _cosine() helper
  - TypeRouter.classify() keyword, embedding, and LLM modes
  - TypeRouter.route_facts() batch classification
  - Fallback chains: LLM -> embedding -> keywords
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from superlocalmemory.encoding.type_router import TypeRouter, _cosine
from superlocalmemory.storage.models import AtomicFact, FactType, Mode

# ---------------------------------------------------------------------------
# _cosine
# ---------------------------------------------------------------------------

class TestCosine:
    def test_identical(self) -> None:
        a = [1.0, 0.0, 0.0]
        assert _cosine(a, a) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite(self) -> None:
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert _cosine(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_zero_vector(self) -> None:
        a = [0.0, 0.0]
        b = [1.0, 1.0]
        assert _cosine(a, b) == 0.0

    def test_both_zero(self) -> None:
        assert _cosine([0.0], [0.0]) == 0.0


# ---------------------------------------------------------------------------
# Keyword classification
# ---------------------------------------------------------------------------

class TestKeywordClassification:
    def _make_fact(self, content: str) -> AtomicFact:
        return AtomicFact(fact_id="f1", content=content)

    def test_opinion(self) -> None:
        router = TypeRouter(mode=Mode.A)
        fact = self._make_fact("I think Python is better than Java")
        assert router.classify(fact) == FactType.OPINION

    def test_prospective(self) -> None:
        """"scheduled" and "deadline" are what this type was always matching."""
        router = TypeRouter(mode=Mode.A)
        fact = self._make_fact("The deadline is scheduled for next week")
        assert router.classify(fact) == FactType.PROSPECTIVE

    def test_episodic(self) -> None:
        router = TypeRouter(mode=Mode.A)
        fact = self._make_fact("We visited the museum and attended a show")
        assert router.classify(fact) == FactType.EPISODIC

    def test_semantic_default(self) -> None:
        router = TypeRouter(mode=Mode.A)
        fact = self._make_fact("Paris is the capital of France")
        assert router.classify(fact) == FactType.SEMANTIC

    def test_no_embedder_falls_to_keywords(self) -> None:
        router = TypeRouter(mode=Mode.A, embedder=None)
        fact = self._make_fact("I believe this is correct")
        assert router.classify(fact) == FactType.OPINION


# ---------------------------------------------------------------------------
# Embedding classification
# ---------------------------------------------------------------------------

class TestEmbeddingClassification:
    def _mock_embedder(self) -> MagicMock:
        embedder = MagicMock()
        # Return different embeddings for different content
        def embed(text: str) -> list[float]:
            if "visited" in text or "went" in text:
                return [1.0, 0.0, 0.0, 0.0]
            elif "fact" in text or "capital" in text:
                return [0.0, 1.0, 0.0, 0.0]
            elif "think" in text or "prefer" in text:
                return [0.0, 0.0, 1.0, 0.0]
            elif "deadline" in text or "scheduled" in text:
                return [0.0, 0.0, 0.0, 1.0]
            return [0.25, 0.25, 0.25, 0.25]

        def embed_batch(texts: list[str]) -> list[list[float]]:
            return [embed(t) for t in texts]

        embedder.embed = embed
        embedder.embed_batch = embed_batch
        return embedder

    def test_embedding_classification(self) -> None:
        embedder = self._mock_embedder()
        router = TypeRouter(mode=Mode.A, embedder=embedder)
        fact = AtomicFact(fact_id="f1", content="The capital of France is Paris")
        # With mock embedder the exact type depends on template embeddings
        result = router.classify(fact)
        assert isinstance(result, FactType)

    def test_template_embeddings_built_once(self) -> None:
        embedder = self._mock_embedder()
        router = TypeRouter(mode=Mode.A, embedder=embedder)
        f1 = AtomicFact(fact_id="f1", content="fact about world")
        f2 = AtomicFact(fact_id="f2", content="another fact")
        router.classify(f1)
        # Template embeddings should be cached
        assert router._template_embeddings is not None
        router.classify(f2)


# ---------------------------------------------------------------------------
# LLM classification
# ---------------------------------------------------------------------------

class TestLLMClassification:
    def test_llm_classifies(self) -> None:
        llm = MagicMock()
        llm.generate.return_value = "episodic"
        router = TypeRouter(mode=Mode.C, llm=llm)
        fact = AtomicFact(fact_id="f1", content="Alice visited Paris")
        assert router.classify(fact) == FactType.EPISODIC

    def test_llm_unknown_type_defaults_semantic(self) -> None:
        llm = MagicMock()
        llm.generate.return_value = "unknown_type"
        router = TypeRouter(mode=Mode.C, llm=llm)
        fact = AtomicFact(fact_id="f1", content="Something")
        assert router.classify(fact) == FactType.SEMANTIC

    def test_llm_exception_falls_to_keywords(self) -> None:
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("API fail")
        router = TypeRouter(mode=Mode.C, llm=llm)
        fact = AtomicFact(fact_id="f1", content="I think this is true")
        # Should fall back to keyword classification
        result = router.classify(fact)
        assert result == FactType.OPINION

    def test_llm_none_falls_to_embedding(self) -> None:
        embedder = MagicMock()
        embedder.embed.return_value = [0.5, 0.5]
        embedder.embed_batch.return_value = [[0.5, 0.5]] * 5
        router = TypeRouter(mode=Mode.C, llm=None, embedder=embedder)
        fact = AtomicFact(fact_id="f1", content="Something")
        result = router.classify(fact)
        assert isinstance(result, FactType)


# ---------------------------------------------------------------------------
# route_facts (batch)
# ---------------------------------------------------------------------------

class TestRouteFacts:
    def test_batch_classification(self) -> None:
        router = TypeRouter(mode=Mode.A)
        facts = [
            AtomicFact(fact_id="f1", content="I think cats are great"),
            AtomicFact(fact_id="f2", content="Paris is the capital of France"),
            AtomicFact(fact_id="f3", content="We visited the Louvre and saw art"),
        ]
        result = router.route_facts(facts)
        assert len(result) == 3
        assert result[0].fact_type == FactType.OPINION
        assert result[1].fact_type == FactType.SEMANTIC
        assert result[2].fact_type == FactType.EPISODIC

    def test_immutability(self) -> None:
        router = TypeRouter(mode=Mode.A)
        original = AtomicFact(fact_id="f1", content="I think this is good")
        result = router.route_facts([original])
        # Original should be unchanged (immutability pattern)
        assert result[0] is not original
        assert result[0].fact_id == original.fact_id

    def test_empty_batch(self) -> None:
        router = TypeRouter(mode=Mode.A)
        assert router.route_facts([]) == []

    def test_preserves_other_fields(self) -> None:
        router = TypeRouter(mode=Mode.A)
        original = AtomicFact(
            fact_id="f1", content="I believe this is correct",
            profile_id="work", importance=0.8,
            entities=["Alice"], session_id="s1",
        )
        result = router.route_facts([original])
        assert result[0].profile_id == "work"
        assert result[0].importance == 0.8
        assert result[0].entities == ["Alice"]
        assert result[0].session_id == "s1"


class TestRoutingChangesOnlyTheType:
    """Routing decides a fact's type. It must not quietly reset anything else
    about the fact — sharing, pinning, lifecycle — on the way through."""

    def test_every_other_field_survives_routing(self) -> None:
        import dataclasses

        from superlocalmemory.encoding.type_router import TypeRouter
        from superlocalmemory.storage.models import AtomicFact, FactType, Mode

        fact = AtomicFact(
            fact_id="f1", memory_id="m1", profile_id="p1",
            content="I think we should ship the release on Friday.",
            fact_type=FactType.SEMANTIC, scope="shared", shared_with=["team-a"],
            pinned=True, access_count=7, langevin_position=[0.1] * 8,
        )
        routed = TypeRouter(mode=Mode.A).route_facts([fact])[0]
        for f in dataclasses.fields(AtomicFact):
            if f.name == "fact_type":
                continue
            assert getattr(routed, f.name) == getattr(fact, f.name), f.name
