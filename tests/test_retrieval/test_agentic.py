# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.retrieval.agentic — 2-Round Sufficiency Verification.

Covers:
  - Skip temporal queries (S15)
  - Round 1 sufficient → return early
  - Round 1 insufficient → round 2 refinement
  - Mode A heuristic expansion (no LLM)
  - Sufficiency check (heuristic + LLM)
  - _parse_json_strings and _parse_sufficiency helpers
  - _avg helper
  - Empty results handling
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from superlocalmemory.retrieval.agentic import (
    AgenticRetriever,
    _avg,
    _parse_json_strings,
    _parse_sufficiency,
)
from superlocalmemory.storage.models import AtomicFact

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fact(fact_id: str, content: str = "") -> AtomicFact:
    return AtomicFact(
        fact_id=fact_id, memory_id="m0",
        content=content or f"fact {fact_id}",
    )


def _make_results(n: int, score: float = 0.5) -> list[tuple[AtomicFact, float]]:
    return [(_make_fact(f"f{i}", f"content {i}"), score) for i in range(n)]


def _mock_llm(available: bool = True, response: str = "[]") -> MagicMock:
    llm = MagicMock()
    llm.is_available = available
    llm.generate.return_value = MagicMock(text=response)
    return llm


def _mock_engine(results: list[tuple[AtomicFact, float]]) -> MagicMock:
    engine = MagicMock()
    engine.recall_facts.return_value = results
    return engine


# ---------------------------------------------------------------------------
# _parse_json_strings
# ---------------------------------------------------------------------------

class TestParseJsonStrings:
    def test_valid_json_array(self) -> None:
        assert _parse_json_strings('["q1", "q2"]') == ["q1", "q2"]

    def test_json_with_surrounding_text(self) -> None:
        assert _parse_json_strings('Here: ["query1"]') == ["query1"]

    def test_empty_array(self) -> None:
        assert _parse_json_strings("[]") == []

    def test_no_array(self) -> None:
        assert _parse_json_strings("no json here") == []

    def test_invalid_json(self) -> None:
        assert _parse_json_strings("[invalid}") == []

    def test_max_3_items(self) -> None:
        result = _parse_json_strings('["a", "b", "c", "d", "e"]')
        assert len(result) <= 3

    def test_non_list_json(self) -> None:
        assert _parse_json_strings('{"key": "val"}') == []


# ---------------------------------------------------------------------------
# _parse_sufficiency
# ---------------------------------------------------------------------------

class TestParseSufficiency:
    def test_sufficient(self) -> None:
        assert _parse_sufficiency('{"is_sufficient": true}') is True

    def test_insufficient(self) -> None:
        assert _parse_sufficiency('{"is_sufficient": false, "missing_information": "date"}') is False

    def test_with_surrounding_text(self) -> None:
        assert _parse_sufficiency('Result: {"is_sufficient": true}') is True

    def test_invalid_json(self) -> None:
        assert _parse_sufficiency("not json") is None

    def test_missing_key(self) -> None:
        assert _parse_sufficiency('{"other": true}') is None


# ---------------------------------------------------------------------------
# _avg
# ---------------------------------------------------------------------------

class TestAvg:
    def test_normal(self) -> None:
        results = [(_make_fact("f1"), 0.8), (_make_fact("f2"), 0.4)]
        assert _avg(results) == pytest.approx(0.6)

    def test_empty(self) -> None:
        assert _avg([]) == 0.0

    def test_single(self) -> None:
        assert _avg([(_make_fact("f1"), 0.7)]) == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# AgenticRetriever.retrieve
# ---------------------------------------------------------------------------

class TestAgenticRetrieverRetrieve:
    def test_skip_temporal(self) -> None:
        engine = _mock_engine(_make_results(5, 0.6))
        retriever = AgenticRetriever()
        results = retriever.retrieve(
            "query", "default", engine, query_type="temporal",
        )
        assert len(results) == 5
        engine.recall_facts.assert_called_once_with(
            "query", "default", top_k=20, skip_agentic=True, facets=None,
        )

    def test_multi_hop_not_skipped(self) -> None:
        """Multi-hop is handled by bridge discovery, NOT skipped."""
        engine = _mock_engine(_make_results(10, 0.8))
        retriever = AgenticRetriever()
        results = retriever.retrieve(
            "query", "default", engine, query_type="multi_hop",
        )
        assert len(results) > 0
        assert len(retriever.rounds) >= 1

    def test_round1_sufficient_high_score(self) -> None:
        results_data = _make_results(10, 0.9)
        engine = _mock_engine(results_data)
        retriever = AgenticRetriever()
        results = retriever.retrieve("query", "default", engine, top_k=20)
        assert len(results) > 0
        assert len(retriever.rounds) == 1
        assert retriever.rounds[0].is_sufficient is True

    def test_round1_insufficient_triggers_round2(self) -> None:
        # Score 0.4 avoids the fast-path (< 0.3) but fails sufficiency (< 0.6)
        r1_data = _make_results(3, 0.4)
        r2_data = _make_results(5, 0.7)
        engine = MagicMock()
        engine.recall_facts.side_effect = [r1_data, r2_data]

        llm = _mock_llm(True, "")
        llm.generate.side_effect = [
            MagicMock(text='{"is_sufficient": false}'),
            MagicMock(text='["better query"]'),
        ]

        retriever = AgenticRetriever()
        results = retriever.retrieve(
            "query", "default", engine, llm=llm, top_k=20,
        )
        assert len(retriever.rounds) >= 2

    def test_no_llm_returns_round1(self) -> None:
        results_data = _make_results(3, 0.1)
        engine = _mock_engine(results_data)
        retriever = AgenticRetriever()
        results = retriever.retrieve("query", "default", engine, llm=None)
        assert len(results) == 3

    def test_empty_round1(self) -> None:
        engine = _mock_engine([])
        retriever = AgenticRetriever()
        results = retriever.retrieve("query", "default", engine)
        assert results == []

    def test_rounds_metadata_tracked(self) -> None:
        engine = _mock_engine(_make_results(15, 0.9))
        retriever = AgenticRetriever()
        retriever.retrieve("test query", "default", engine)
        assert len(retriever.rounds) == 1
        assert retriever.rounds[0].query == "test query"
        assert retriever.rounds[0].round_num == 1
        assert retriever.rounds[0].result_count == 15

    # -- L2-08: facets must survive every round, not just round 1 -----------

    def test_facets_forwarded_on_round1(self) -> None:
        from superlocalmemory.retrieval.facets import Facets

        engine = _mock_engine(_make_results(10, 0.9))
        facets = Facets.of(project="zephyr")
        retriever = AgenticRetriever()
        retriever.retrieve("query", "default", engine, facets=facets)
        engine.recall_facts.assert_called_once_with(
            "query", "default", top_k=20, skip_agentic=True, facets=facets,
        )

    def test_facets_forwarded_on_round2_mode_a_heuristic(self) -> None:
        """Round 2's heuristic-expansion re-retrieval (Mode A, no LLM) must
        carry the SAME facets object round 1 used — before this fix, round 2
        re-ran with none and its results REPLACED the filtered round-1
        response."""
        from superlocalmemory.retrieval.facets import Facets

        r1_data = _make_results(1, 0.2)  # forces round 2 (score < 0.3, len < 3)
        r2_data = _make_results(2, 0.7)
        engine = MagicMock()
        # Mode A's heuristic expansion can generate up to 3 sub-queries
        # (entity+action, action-only, entity-only); round 1 + headroom.
        engine.recall_facts.side_effect = [r1_data] + [r2_data] * 4
        facets = Facets.of(project="zephyr")

        retriever = AgenticRetriever()
        retriever.retrieve("Atlas Postgres deploy", "default", engine, facets=facets)

        assert engine.recall_facts.call_count >= 2
        for call in engine.recall_facts.call_args_list:
            assert call.kwargs.get("facets") is facets

    def test_facets_forwarded_on_round2_llm_refine(self) -> None:
        """Mode C (LLM refine) round 2 must carry facets too."""
        from superlocalmemory.retrieval.facets import Facets

        r1_data = _make_results(3, 0.4)
        r2_data = _make_results(5, 0.7)
        engine = MagicMock()
        engine.recall_facts.side_effect = [r1_data, r2_data]
        llm = _mock_llm(True, "")
        llm.generate.side_effect = [
            MagicMock(text='{"is_sufficient": false}'),
            MagicMock(text='["better query"]'),
        ]
        facets = Facets.of(kind="status")

        retriever = AgenticRetriever()
        retriever.retrieve("query", "default", engine, llm=llm, top_k=20, facets=facets)

        for call in engine.recall_facts.call_args_list:
            assert call.kwargs.get("facets") is facets

    def test_facets_forwarded_on_skip_branch(self, monkeypatch) -> None:
        """``_SKIP_TYPES`` is empty in production (S15 removed "temporal"),
        so this early-return branch is currently unreachable — but it must
        stay correct if a skip type is ever reinstated. Force it via the
        module constant rather than asserting on dead behaviour."""
        from superlocalmemory.retrieval import agentic as agentic_mod
        from superlocalmemory.retrieval.facets import Facets

        monkeypatch.setattr(agentic_mod, "_SKIP_TYPES", frozenset({"temporal"}))
        engine = _mock_engine(_make_results(5, 0.6))
        facets = Facets.of(project="zephyr")
        retriever = AgenticRetriever()
        retriever.retrieve(
            "query", "default", engine, query_type="temporal", facets=facets,
        )
        engine.recall_facts.assert_called_once_with(
            "query", "default", top_k=20, skip_agentic=True, facets=facets,
        )


# ---------------------------------------------------------------------------
# Mode A heuristic expansion
# ---------------------------------------------------------------------------

class TestHeuristicExpansion:
    def test_no_db_generates_sub_queries(self) -> None:
        """V3.3.19: Without DB, still generates entity+action sub-queries."""
        retriever = AgenticRetriever(db=None)
        expanded = retriever._heuristic_expand("What about Alice?", "default")
        # "Alice" is extracted as entity, generates entity-only sub-query
        assert any("Alice" in q for q in expanded)

    def test_with_db_expands_aliases(self) -> None:
        db = MagicMock()
        entity = MagicMock()
        entity.entity_id = "e1"
        db.get_entity_by_name.return_value = entity
        alias = MagicMock()
        alias.alias = "Al"
        db.get_aliases_for_entity.return_value = [alias]

        retriever = AgenticRetriever(db=db)
        expanded = retriever._heuristic_expand("What about Alice?", "default")
        # V3.3.19: Returns entity sub-queries + alias expansion
        assert len(expanded) >= 1
        assert any("Al" in q for q in expanded)

    def test_decomposition_multi_hop(self) -> None:
        """V3.3.19: Multi-hop decomposition generates entity+action sub-queries."""
        retriever = AgenticRetriever(db=None)
        expanded = retriever._heuristic_expand(
            "When did Caroline go to the LGBTQ support group?", "default",
        )
        assert len(expanded) >= 2
        # Should generate entity+action and action-only sub-queries
        assert any("support" in q.lower() for q in expanded)
        assert any("Caroline" in q for q in expanded)
