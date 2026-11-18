# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""R-43 (4.2.30, F-37): a word in the query never widens the answer past the caller's limit.

On the 16 Nov golden run, 12 of 200 queries came back with 36-86 results for
``limit=10``. Each held a bare quantifier ("all passed", "in all", "different") or a
plural noun the aggregation list names, which sent the query down the aggregation path;
that path replaced the caller's limit with 100. MCP ``recall`` and the CLI printed all of
them, past the caller's token budget.

Two guarantees, both tested here:
* the engine returns at most ``limit`` results whatever the query says;
* a bare quantifier (a word like "all" with no request around it) is not an aggregation
  intent, while a real request for a collection still is.
"""

from __future__ import annotations

import pytest

from superlocalmemory.retrieval.strategy import QueryStrategyClassifier
from tests.test_retrieval.test_engine import _build_engine, _make_fact, _mock_db

#: The shapes the 12 golden queries took (generic text, same words).
BARE_QUANTIFIER_QUERIES = [
    "The full test suite consists of 3,066 tests that have all passed",
    "PEER DISPATCH SMOKE TESTS - ALL PASSED: | Test | Orchestrator |",
    "should be so strong, and in all",
    "There are different paradoxes and different theories which we are getting to",
    "A quote naming a different company than the claim's subject",
    "If you cannot find the claim in the evidence at all",
    "The plugins (Claude Code, Codex, Hermes, all",
    "Many agents were running at the time of the release",
]

REAL_REQUESTS = [
    "What are all the hobbies Alice has?",
    "list all the cities Bob visited",
    "How many different cities did she visit?",
    "Which places has Melanie visited in all her trips?",
    "summarize everything about the release",
    "give me an overview of the project",
    "What activities does Melanie partake in?",
]


@pytest.mark.parametrize("query", BARE_QUANTIFIER_QUERIES)
def test_a_bare_quantifier_is_not_an_aggregation_intent(query: str) -> None:
    assert QueryStrategyClassifier()._detect_type(query) != "aggregation"


@pytest.mark.parametrize("query", REAL_REQUESTS)
def test_a_real_request_for_a_collection_is_still_an_aggregation(query: str) -> None:
    assert QueryStrategyClassifier()._detect_type(query) == "aggregation"


@pytest.mark.parametrize("query", [
    "What did they do at all?", "Did it work at all", "after all, what changed?",
])
def test_at_all_and_in_all_are_not_a_quantifier(query: str) -> None:
    assert QueryStrategyClassifier()._detect_type(query) != "aggregation"


def _engine(n: int = 40):
    facts = [_make_fact(f"f{i}", f"fact about topic {i}") for i in range(n)]
    results = [(f"f{i}", 0.9 - i * 0.01) for i in range(n)]
    return _build_engine(db=_mock_db(facts), semantic_results=results, bm25_results=results)


@pytest.mark.parametrize("query", BARE_QUANTIFIER_QUERIES + REAL_REQUESTS)
@pytest.mark.parametrize("limit", [1, 5, 10])
def test_the_engine_never_returns_more_than_the_limit(query: str, limit: int) -> None:
    response = _engine().recall(query, "default", limit=limit)
    assert len(response.results) <= limit
    assert len(response.results) == limit  # 40 candidates qualify: the cap, not a shortage


def test_a_larger_limit_still_returns_more() -> None:
    """The cap is the caller's number, not a new constant."""
    response = _engine().recall("list all the hobbies", "default", limit=30)
    assert len(response.results) == 30


def test_the_aggregation_weights_still_apply_to_a_real_request() -> None:
    """Respecting the limit does not turn the intent off: the channel weights follow it."""
    response = _engine().recall("What are all the hobbies Alice has?", "default", limit=5)
    assert response.query_type == "aggregation"
    assert len(response.results) == 5
