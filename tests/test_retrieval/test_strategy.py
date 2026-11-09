# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.retrieval.strategy — Query-Adaptive Strategy.

Covers:
  - Query type detection (temporal, multi_hop, opinion, entity, factual, general)
  - Weight adaptation from base weights + presets
  - QueryStrategy dataclass
  - QueryStrategyClassifier.classify() full flow
  - Edge cases: empty query, all-lowercase, mixed signals
"""

from __future__ import annotations

import pytest

from superlocalmemory.retrieval.strategy import (
    STRATEGY_PRESETS,
    QueryStrategy,
    QueryStrategyClassifier,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def classifier() -> QueryStrategyClassifier:
    return QueryStrategyClassifier()


@pytest.fixture()
def base_weights() -> dict[str, float]:
    return {"semantic": 1.2, "bm25": 1.0, "entity_graph": 1.0, "temporal": 0.8}


# ---------------------------------------------------------------------------
# Query type detection
# ---------------------------------------------------------------------------

class TestQueryTypeDetection:
    def test_temporal_when(self, classifier: QueryStrategyClassifier) -> None:
        strat = classifier.classify("When did Alice start her job?", {})
        assert strat.query_type == "temporal"

    def test_temporal_date_word(self, classifier: QueryStrategyClassifier) -> None:
        strat = classifier.classify("what happened in january?", {})
        assert strat.query_type == "temporal"

    def test_temporal_ago(self, classifier: QueryStrategyClassifier) -> None:
        strat = classifier.classify("what did Bob say recently?", {})
        assert strat.query_type == "temporal"

    def test_multi_hop_and_then(self, classifier: QueryStrategyClassifier) -> None:
        strat = classifier.classify("what happened and then what did she do?", {})
        assert strat.query_type == "multi_hop"

    def test_multi_hop_because(self, classifier: QueryStrategyClassifier) -> None:
        strat = classifier.classify("why did that happen because of the event?", {})
        assert strat.query_type == "multi_hop"

    def test_multi_hop_connection(self, classifier: QueryStrategyClassifier) -> None:
        strat = classifier.classify("connection between Alice and Bob?", {})
        assert strat.query_type == "multi_hop"

    def test_opinion_think(self, classifier: QueryStrategyClassifier) -> None:
        strat = classifier.classify("does Alice think that is right?", {})
        assert strat.query_type == "opinion"

    def test_opinion_favorite(self, classifier: QueryStrategyClassifier) -> None:
        strat = classifier.classify("what is her favorite color?", {})
        assert strat.query_type == "opinion"

    def test_entity_two_proper_nouns(
        self, classifier: QueryStrategyClassifier
    ) -> None:
        # V3.3.19: 2 entities WITHOUT causal/temporal word → entity
        strat = classifier.classify("Tell me about Alice and Bob", {})
        assert strat.query_type == "entity"

    def test_multi_hop_two_entities_with_causal_word(
        self, classifier: QueryStrategyClassifier
    ) -> None:
        # V3.3.19: 2 entities + causal verb → multi_hop
        strat = classifier.classify("Did Alice meet Bob?", {})
        assert strat.query_type == "multi_hop"

    def test_multi_hop_entity_before(
        self, classifier: QueryStrategyClassifier
    ) -> None:
        # V3.3.19: Classic LoCoMo multi-hop pattern
        strat = classifier.classify("What did Alice study before moving to New York?", {})
        assert strat.query_type == "multi_hop"

    def test_multi_hop_entity_after(
        self, classifier: QueryStrategyClassifier
    ) -> None:
        strat = classifier.classify("What happened after Alice told Bob?", {})
        assert strat.query_type == "multi_hop"

    def test_factual_what(self, classifier: QueryStrategyClassifier) -> None:
        strat = classifier.classify("what is the capital of france?", {})
        assert strat.query_type == "factual"

    def test_factual_who(self, classifier: QueryStrategyClassifier) -> None:
        strat = classifier.classify("who wrote this?", {})
        assert strat.query_type == "factual"

    def test_general_no_signal(self, classifier: QueryStrategyClassifier) -> None:
        strat = classifier.classify("tell me more", {})
        assert strat.query_type == "general"

    def test_empty_query(self, classifier: QueryStrategyClassifier) -> None:
        strat = classifier.classify("", {})
        assert strat.query_type == "general"


# ---------------------------------------------------------------------------
# Weight adaptation
# ---------------------------------------------------------------------------

class TestWeightAdaptation:
    def test_temporal_boosts_temporal_weight(
        self,
        classifier: QueryStrategyClassifier,
        base_weights: dict[str, float],
    ) -> None:
        strat = classifier.classify("When did Alice start?", base_weights)
        # MULTIPLY: base 0.8 * preset 2.0 = 1.6
        assert strat.weights["temporal"] == pytest.approx(1.6)

    def test_multi_hop_boosts_entity_graph(
        self,
        classifier: QueryStrategyClassifier,
        base_weights: dict[str, float],
    ) -> None:
        strat = classifier.classify(
            "connection between Alice and Bob?", base_weights
        )
        # MULTIPLY: base 1.0 * preset 2.5 = 2.5 (v3.4.11: boosted for multi_hop)
        assert strat.weights["entity_graph"] == pytest.approx(2.5)

    def test_general_preserves_base_weights(
        self,
        classifier: QueryStrategyClassifier,
        base_weights: dict[str, float],
    ) -> None:
        strat = classifier.classify("tell me more", base_weights)
        assert strat.weights == base_weights

    def test_preset_multiplies_base(
        self,
        classifier: QueryStrategyClassifier,
    ) -> None:
        base = {"semantic": 5.0, "bm25": 5.0, "entity_graph": 5.0, "temporal": 5.0}
        strat = classifier.classify("When did that happen?", base)
        # MULTIPLY: base 5.0 * preset 2.0 = 10.0
        assert strat.weights["temporal"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# QueryStrategy dataclass
# ---------------------------------------------------------------------------

class TestQueryStrategy:
    def test_default_values(self) -> None:
        qs = QueryStrategy()
        assert qs.query_type == "general"
        assert qs.weights == {}
        assert qs.confidence == 0.5

    def test_custom_values(self) -> None:
        qs = QueryStrategy(
            query_type="temporal",
            weights={"temporal": 2.0},
            confidence=0.7,
        )
        assert qs.query_type == "temporal"
        assert qs.confidence == 0.7


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_general_has_lower_confidence(
        self, classifier: QueryStrategyClassifier
    ) -> None:
        strat = classifier.classify("tell me more", {})
        assert strat.confidence == 0.5

    def test_typed_has_higher_confidence(
        self, classifier: QueryStrategyClassifier
    ) -> None:
        strat = classifier.classify("When did Alice start?", {})
        assert strat.confidence == 0.7


# ---------------------------------------------------------------------------
# Strategy presets exist
# ---------------------------------------------------------------------------

class TestStrategyPresets:
    def test_all_presets_have_keys(self) -> None:
        for name, preset in STRATEGY_PRESETS.items():
            if name == "general":
                assert preset == {}
            else:
                assert "semantic" in preset
                assert "bm25" in preset

    def test_preset_names(self) -> None:
        expected = {"temporal", "multi_hop", "aggregation", "opinion", "factual", "entity", "general", "vague", "recency"}
        assert set(STRATEGY_PRESETS.keys()) == expected
