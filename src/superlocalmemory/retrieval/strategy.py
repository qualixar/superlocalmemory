# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Query-Adaptive Strategy.

Classifies query type and returns per-type channel weights.
V1 had this code (strategy_learner.py) but never wired it in.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

STRATEGY_PRESETS: dict[str, dict[str, float]] = {
    "temporal": {"semantic": 0.8, "bm25": 1.5, "entity_graph": 0.8, "temporal": 2.0, "spreading_activation": 0.5, "hopfield": 0.5},
    "multi_hop": {"semantic": 1.0, "bm25": 0.8, "entity_graph": 2.5, "temporal": 0.5, "spreading_activation": 2.0, "hopfield": 0.7},
    "aggregation": {"semantic": 1.2, "bm25": 1.5, "entity_graph": 1.0, "temporal": 0.5, "spreading_activation": 0.8, "hopfield": 0.6},
    "opinion": {"semantic": 1.8, "bm25": 0.6, "entity_graph": 0.8, "temporal": 0.3, "spreading_activation": 0.5, "hopfield": 0.5},
    "factual": {"semantic": 1.2, "bm25": 1.4, "entity_graph": 1.0, "temporal": 0.6, "spreading_activation": 0.8, "hopfield": 0.8},
    "entity": {"semantic": 1.0, "bm25": 1.2, "entity_graph": 3.0, "temporal": 0.5, "spreading_activation": 1.5, "hopfield": 0.9},
    "general": {},
    "vague": {"semantic": 0.8, "bm25": 0.5, "entity_graph": 0.6, "temporal": 0.3, "spreading_activation": 1.5, "hopfield": 1.1},
}

_TEMPORAL_WORDS: frozenset[str] = frozenset({
    "when", "date", "time", "year", "month", "ago", "before", "after",
    "during", "last", "next", "recently", "earlier", "later", "since",
    "until", "while", "between", "january", "february", "march",
    "april", "may", "june", "july", "august", "september", "october",
    "november", "december",
})

_MULTI_HOP_PHRASES: tuple[str, ...] = (
    # Original 8 phrases
    "and then", "after that", "because", "how did",
    "as a result", "led to", "connection between", "relationship between",
    # V3.3.19: LoCoMo-style multi-hop patterns (causal/temporal chains)
    "what happened when", "what was happening",
    "during the time", "at the same time",
    "how did it affect", "what changed after",
    "what did they do after", "what did they do before",
    "what was the result", "what was the outcome",
    "what was the reason", "why did they",
    "in response to", "as a consequence",
    "prior to", "following that", "subsequent to",
    "in the meantime", "at that point",
    "which led to", "which caused", "which resulted in",
)

# Words that signal causal/temporal chain when combined with 2+ entities.
# Excludes common instruction verbs (tell, help) to avoid false positives
# on queries like "Tell me about Alice and Bob".
_CAUSAL_TEMPORAL_WORDS: frozenset[str] = frozenset({
    "before", "after", "when", "while", "because", "then",
    "during", "since", "until", "once",
    "affect", "cause", "change", "happen", "result",
    "influence", "impact", "lead", "meet",
    "start", "stop", "begin", "end", "move", "leave",
    "join", "visit", "return",
})

_AGGREGATION_WORDS: frozenset[str] = frozenset({
    "all", "list", "every", "everything", "various", "different",
    "many", "several", "multiple", "summarize", "overview",
    # V3.3.21 R5: LoCoMo cat 1 patterns — "What X does/did Y Z?" needs aggregation.
    # "What activities does Melanie partake in?" = aggregation, not factual.
    "activities", "events", "hobbies", "instruments", "types",
    "things", "places", "jobs", "skills", "interests", "pets",
})

# V3.3.21 R5: Plural noun patterns that signal aggregation queries.
# "What [noun]s has/does [entity] [verb]?" = needs cross-session aggregation.
_AGGREGATION_PATTERNS: tuple[str, ...] = (
    r"what (?:\w+ )?(?:activities|events|hobbies|types|things|places|jobs)",
    r"what (?:\w+ )?has .+ (?:done|visited|attended|participated|played|practiced)",
    r"how many (?:\w+ )?(?:times|events|things|places)",
    r"what are .+(?:'s|s') (?:\w+ )?(?:hobbies|interests|activities|skills)",
)

_OPINION_WORDS: tuple[str, ...] = (
    "think", "feel", "opinion", "prefer", "favorite", "best", "worst",
    "believe", "like about", "dislike", "enjoy", "hate", "love",
)

_VAGUE_PHRASES: tuple[str, ...] = (
    "something about", "i think", "maybe", "not sure",
    "vaguely remember", "partially recall", "sort of",
    "kind of", "i forgot", "what was that",
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

        # Check multi_hop phrases FIRST (exact phrase match)
        if any(p in q for p in _MULTI_HOP_PHRASES):
            return "multi_hop"

        # Extract proper nouns EARLY for the multi-entity heuristic
        _SENTENCE_STARTERS = {"What", "Where", "Who", "Which", "How", "When",
                              "Does", "Did", "Can", "Could", "Would", "Should",
                              "Are", "Is", "Was", "Were", "Has", "Have", "The", "Tell"}
        proper_nouns = [m for m in re.findall(r"\b[A-Z][a-z]{1,}\b", query)
                        if m not in _SENTENCE_STARTERS]

        # V3.3.19: 2+ entities + causal/temporal word → multi_hop
        # This MUST fire BEFORE the temporal check, otherwise "What did
        # Alice study before moving to New York?" would classify as
        # "temporal" instead of "multi_hop".
        if len(proper_nouns) >= 2 and words & _CAUSAL_TEMPORAL_WORDS:
            return "multi_hop"

        if words & _TEMPORAL_WORDS:
            return "temporal"
        if words & _AGGREGATION_WORDS:
            return "aggregation"
        # V3.3.21 R5: Regex patterns for aggregation questions
        if any(re.search(p, q) for p in _AGGREGATION_PATTERNS):
            return "aggregation"
        if any(w in q for w in _OPINION_WORDS):
            return "opinion"
        if len(proper_nouns) >= 2:
            return "entity"
        if q.startswith(("what ", "where ", "who ", "which ", "how ")):
            return "factual"
        # Vague/fuzzy recall — Hopfield pattern completion excels here
        if any(p in q for p in _VAGUE_PHRASES):
            return "vague"
        return "general"
