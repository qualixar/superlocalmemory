# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Query-Adaptive Strategy.

Classifies query type and returns per-type channel weights.
V1 had this code (strategy_learner.py) but never wired it in.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: MIT
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

STRATEGY_PRESETS: dict[str, dict[str, float]] = {
    "temporal": {"semantic": 0.8, "bm25": 1.5, "entity_graph": 0.8, "temporal": 2.0, "spreading_activation": 0.5},
    "multi_hop": {"semantic": 1.0, "bm25": 0.8, "entity_graph": 2.0, "temporal": 0.5, "spreading_activation": 2.0},
    "aggregation": {"semantic": 1.2, "bm25": 1.5, "entity_graph": 1.0, "temporal": 0.5, "spreading_activation": 0.8},
    "opinion": {"semantic": 1.8, "bm25": 0.6, "entity_graph": 0.8, "temporal": 0.3, "spreading_activation": 0.5},
    "factual": {"semantic": 1.2, "bm25": 1.4, "entity_graph": 1.0, "temporal": 0.6, "spreading_activation": 0.8},
    "entity": {"semantic": 1.0, "bm25": 1.5, "entity_graph": 1.2, "temporal": 0.5, "spreading_activation": 1.0},
    "general": {},
}

_TEMPORAL_WORDS: frozenset[str] = frozenset({
    "when", "date", "time", "year", "month", "ago", "before", "after",
    "during", "last", "next", "recently", "earlier", "later", "since",
    "until", "while", "between", "january", "february", "march",
    "april", "may", "june", "july", "august", "september", "october",
    "november", "december",
})

_MULTI_HOP_PHRASES: tuple[str, ...] = (
    "and then", "after that", "because", "how did",
    "as a result", "led to", "connection between", "relationship between",
)

_AGGREGATION_WORDS: frozenset[str] = frozenset({
    "all", "list", "every", "everything", "various", "different",
    "many", "several", "multiple", "summarize", "overview",
})

_OPINION_WORDS: tuple[str, ...] = (
    "think", "feel", "opinion", "prefer", "favorite", "best", "worst",
    "believe", "like about", "dislike", "enjoy", "hate", "love",
)


@dataclass
class QueryStrategy:
    """Classified query type + adapted weights."""
    query_type: str = "general"
    weights: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.5


class QueryStrategyClassifier:
    """Classifies queries and produces adaptive channel weights."""

    def classify(self, query: str, base_weights: dict[str, float]) -> QueryStrategy:
        """Classify query and return adapted weights."""
        qtype = self._detect_type(query)
        adapted = dict(base_weights)
        for ch, w in STRATEGY_PRESETS.get(qtype, {}).items():
            adapted[ch] = base_weights.get(ch, 1.0) * w
        return QueryStrategy(qtype, adapted, 0.7 if qtype != "general" else 0.5)

    def _detect_type(self, query: str) -> str:
        q = query.lower()
        # Strip punctuation from words so "january?" matches "january"
        words = set(re.sub(r"[^\w\s'-]", "", q).split())

        # Check multi_hop BEFORE temporal — phrases like "connection between"
        # must not be short-circuited by the word "between" in _TEMPORAL_WORDS.
        if any(p in q for p in _MULTI_HOP_PHRASES):
            return "multi_hop"
        if words & _TEMPORAL_WORDS:
            return "temporal"
        if words & _AGGREGATION_WORDS:
            return "aggregation"
        if any(w in q for w in _OPINION_WORDS):
            return "opinion"
        # Proper nouns — exclude common sentence-initial words
        _SENTENCE_STARTERS = {"What", "Where", "Who", "Which", "How", "When",
                              "Does", "Did", "Can", "Could", "Would", "Should",
                              "Are", "Is", "Was", "Were", "Has", "Have", "The", "Tell"}
        proper_nouns = [m for m in re.findall(r"\b[A-Z][a-z]{1,}\b", query)
                        if m not in _SENTENCE_STARTERS]
        if len(proper_nouns) >= 2:
            return "entity"
        if q.startswith(("what ", "where ", "who ", "which ", "how ")):
            return "factual"
        return "general"
