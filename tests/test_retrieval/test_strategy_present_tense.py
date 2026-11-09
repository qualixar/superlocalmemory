# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for present-tense and recency query classification in strategy.py.

Covers:
  - Present-tense words (now, today, yesterday, currently) classify
    as temporal or recency — not factual.
  - Present-activity phrases ("working on", "at the moment", etc.) classify
    as recency, not factual.
  - Plainly factual queries without temporal/recency signals stay factual.
  - No factual query without a temporal or recency signal becomes recency.
  - classify_query() module-level convenience function is importable.
  - STRATEGY_PRESETS includes a "recency" key with the expected channel weights.
"""

from __future__ import annotations

import pytest

from superlocalmemory.retrieval.strategy import (
    STRATEGY_PRESETS,
    QueryStrategyClassifier,
    classify_query,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def clf() -> QueryStrategyClassifier:
    return QueryStrategyClassifier()


def _type(clf: QueryStrategyClassifier, query: str) -> str:
    return clf.classify(query, {}).query_type


# ---------------------------------------------------------------------------
# Table-driven: present-tense temporal words → temporal or recency (not factual)
# These FAIL before the _TEMPORAL_WORDS extension (task 2.1).
# ---------------------------------------------------------------------------

PRESENT_TEMPORAL_WORDS = [
    ("now",       "what did I do now"),
    ("today",     "what did I do today"),
    ("yesterday", "what did I do yesterday"),
    # "currently" was removed from _TEMPORAL_WORDS to prevent "what is the
    # currently supported format" from triggering the temporal/recency path
    # (which dumps newest facts with no topic filter). The present-activity
    # patterns "currently working", "currently doing", and "currently focused"
    # are handled by _RECENCY_PHRASES; standalone "currently" with a concrete
    # topic ("currently supported format") should be factual.
    ("currently", "what am I currently focused on"),
    ("tonight",   "what did I do tonight"),
    ("tomorrow",  "what is happening tomorrow"),
    # "latest" was removed from _TEMPORAL_WORDS. In "what is the latest update"
    # it means "most recent version of", not "at what time did an event happen".
    # Routing it to temporal causes the recency fallback to return 50 newest
    # facts with no topic filter — worse than factual's balanced BM25+semantic.
    # "what is the latest update" now classifies as factual, which is correct.
]


@pytest.mark.parametrize("word,query", PRESENT_TEMPORAL_WORDS, ids=[w for w, _ in PRESENT_TEMPORAL_WORDS])
def test_present_temporal_word_is_not_factual(
    clf: QueryStrategyClassifier, word: str, query: str
) -> None:
    """Each entry must classify as temporal, recency, or (for 'currently') recency via phrase.

    "currently" in "what am I currently focused on" → recency via the
    "currently focused" recency phrase. "latest" was removed from this table
    because "what is the latest update" should be factual — see table docstring.
    """
    result = _type(clf, query)
    assert result in ("temporal", "recency"), (
        f"word {word!r} in {query!r} -> {result!r}; expected temporal or recency"
    )


# ---------------------------------------------------------------------------
# Table-driven: recency phrases → recency (not factual)
# These FAIL before the _RECENCY_PHRASES + recency strategy (task 2.2).
# ---------------------------------------------------------------------------

RECENCY_QUERIES = [
    # These must NOT contain temporal words so the recency phrase check fires.
    "what am I working on",
    "what are we working on at the moment",
    "what's happening these days",
    "what am i doing these days",       # "these days" phrase, no temporal word
    "what have i been working on",
    "what are we working on",           # "working on" phrase, no temporal word
]


@pytest.mark.parametrize("query", RECENCY_QUERIES)
def test_recency_phrase_classifies_as_recency(
    clf: QueryStrategyClassifier, query: str
) -> None:
    """Present-activity phrase queries must classify as recency."""
    result = _type(clf, query)
    assert result == "recency", (
        f"{query!r} -> {result!r}; expected recency"
    )


# ---------------------------------------------------------------------------
# Regression: plainly factual queries must NOT become recency or temporal.
# Over-triggering is the critical risk — these guard against it.
# ---------------------------------------------------------------------------

FACTUAL_QUERIES = [
    ("what is the architecture",       "factual"),
    ("what is the database schema",    "factual"),
    ("who maintains this codebase",    "factual"),
    ("who wrote the retrieval module", "factual"),
    ("what happened in march",         "temporal"),  # retrospective, not recency
    ("when did we deploy",             "temporal"),  # retrospective, not recency
    ("who was at the meeting",         "factual"),
    ("where is the config file",       "factual"),
    ("how does authentication work",   "factual"),
]


# A word that hints at time does not make a question about a topic.
# "current" was originally in the present-tense set, so "what is the current
# database schema" was routed to the time-aware path — which weights
# word-matching MORE heavily than the topical path, amplifying the exact signal
# that made this release necessary. These are the forms that were missing
# from the regression table: time-flavoured wording, topical intent.
TIME_FLAVOURED_BUT_TOPICAL = [
    "what is the current database schema",
    "what is the current state of the migration",
    "what is the current architecture",
]


@pytest.mark.parametrize("query", TIME_FLAVOURED_BUT_TOPICAL)
def test_time_flavoured_topical_question_stays_factual(query: str) -> None:
    got = classify_query(query).query_type
    assert got == "factual", (
        f"{query!r} classified as {got!r}. It asks about a topic; routing it to a "
        f"time-aware strategy boosts word-matching above the topical preset and "
        f"answers a question about the schema with whatever is newest"
    )


@pytest.mark.parametrize("query,expected", FACTUAL_QUERIES, ids=[q for q, _ in FACTUAL_QUERIES])
def test_factual_query_not_over_triggered(
    clf: QueryStrategyClassifier, query: str, expected: str
) -> None:
    """Plainly factual or retrospective queries must not become recency."""
    result = _type(clf, query)
    assert result == expected, (
        f"{query!r} -> {result!r}; expected {expected!r}"
    )


# ---------------------------------------------------------------------------
# Gate-compatibility: classify_query module-level function
# The gate calls: from superlocalmemory.retrieval.strategy import classify_query
# ---------------------------------------------------------------------------

class TestClassifyQueryFunction:
    def test_importable(self) -> None:
        """classify_query must be importable at module level."""
        # Already imported at module level; this checks that import path works.
        assert callable(classify_query)

    def test_returns_query_strategy(self) -> None:
        from superlocalmemory.retrieval.strategy import QueryStrategy
        result = classify_query("what am I working on")
        assert isinstance(result, QueryStrategy)

    def test_mandatory_words_via_classify_query(self) -> None:
        """Gate condition 2: core time words must classify as temporal or recency.

        "currently" was removed from this list because standalone "currently"
        in a topical query ("what is the currently supported format") was
        routing those queries to the temporal path, which returned newest facts
        with no topic filter instead of matching the stated subject. Present-
        activity uses of "currently" are covered by _RECENCY_PHRASES phrases
        such as "currently working", "currently doing", "currently focused".
        """
        mandatory = ("now", "today", "yesterday")
        for w in mandatory:
            qt = classify_query(f"what did I do {w}").query_type
            assert qt in ("temporal", "recency"), (
                f"word {w!r} -> {qt!r}; expected temporal or recency"
            )

    def test_present_activity_via_classify_query(self) -> None:
        """Gate condition 3: present-activity query classifies as recency."""
        qt = classify_query("what am I working on").query_type
        assert qt == "recency", f"expected recency, got {qt!r}"


# ---------------------------------------------------------------------------
# STRATEGY_PRESETS: recency key and channel weights
# ---------------------------------------------------------------------------

class TestRecencyPreset:
    def test_recency_key_exists(self) -> None:
        assert "recency" in STRATEGY_PRESETS, "STRATEGY_PRESETS missing 'recency' key"

    def test_recency_temporal_weight_highest(self) -> None:
        preset = STRATEGY_PRESETS["recency"]
        assert preset.get("temporal", 0.0) >= 2.0, (
            "recency preset: temporal weight must be >= 2.0 (time proximity should dominate)"
        )

    def test_recency_bm25_weight_present(self) -> None:
        preset = STRATEGY_PRESETS["recency"]
        assert "bm25" in preset

    def test_recency_hopfield_damped(self) -> None:
        """Hopfield should be damped for recency — pattern completion is noise here."""
        preset = STRATEGY_PRESETS["recency"]
        assert preset.get("hopfield", 1.0) <= 0.6, (
            "recency preset: hopfield weight should be <= 0.6 (damped for recency)"
        )

    def test_recency_weights_are_floats(self) -> None:
        for k, v in STRATEGY_PRESETS["recency"].items():
            assert isinstance(v, float), f"recency preset key {k!r} value must be float"


# ---------------------------------------------------------------------------
# The switch that disables present-tense ranking must actually disable it
#
# The config field RetrievalConfig.enable_recency_strategy is documented as a
# one-line rollback. Before the fix, _recency_enabled() only read the env var;
# the config field was declared and never read. An inert implementation that
# always returns True would pass the entire test suite, so these tests close
# that gap by exercising both the env var path and the config-field path with
# both on and off.
# ---------------------------------------------------------------------------

class TestRecencyRollbackSwitch:
    """Both rollback mechanisms must actually suppress the recency path."""

    def test_env_var_off_present_phrase_is_not_recency(self, monkeypatch) -> None:
        """With SLM_DISABLE_RECENCY_STRATEGY=1, a present-activity phrase
        must NOT classify as recency.
        """
        monkeypatch.setenv("SLM_DISABLE_RECENCY_STRATEGY", "1")
        clf = QueryStrategyClassifier()
        result = clf.classify("what am I working on", {}).query_type
        assert result != "recency", (
            f"SLM_DISABLE_RECENCY_STRATEGY=1 must suppress recency classification, "
            f"but got {result!r}. "
            "An inert _recency_enabled() that always returns True fails here."
        )

    def test_env_var_on_present_phrase_is_recency(self, monkeypatch) -> None:
        """Without the disable flag, the same query must classify as recency."""
        monkeypatch.delenv("SLM_DISABLE_RECENCY_STRATEGY", raising=False)
        clf = QueryStrategyClassifier()
        result = clf.classify("what am I working on", {}).query_type
        assert result == "recency", (
            f"With rollback off, 'what am I working on' must classify as recency, "
            f"got {result!r}."
        )

    def test_config_field_off_present_phrase_is_not_recency(self) -> None:
        """RetrievalConfig(enable_recency_strategy=False) must suppress recency.

        This fails before the fix because QueryStrategyClassifier does not
        accept a config argument and the config field is never read.
        """
        from superlocalmemory.core.config import RetrievalConfig
        config = RetrievalConfig(enable_recency_strategy=False)
        clf = QueryStrategyClassifier(config=config)
        result = clf.classify("what am I working on", {}).query_type
        assert result != "recency", (
            f"RetrievalConfig(enable_recency_strategy=False) must suppress recency, "
            f"got {result!r}. "
            "The config field enable_recency_strategy must be wired to _recency_enabled()."
        )

    def test_config_field_on_present_phrase_is_recency(self) -> None:
        """RetrievalConfig(enable_recency_strategy=True) must keep recency on."""
        from superlocalmemory.core.config import RetrievalConfig
        config = RetrievalConfig(enable_recency_strategy=True)
        clf = QueryStrategyClassifier(config=config)
        result = clf.classify("what am I working on", {}).query_type
        assert result == "recency", (
            f"With enable_recency_strategy=True, 'what am I working on' must classify "
            f"as recency, got {result!r}."
        )

    def test_env_var_wins_over_config_when_env_disables(self, monkeypatch) -> None:
        """Env var must override config: env=off, config=on → disabled."""
        from superlocalmemory.core.config import RetrievalConfig
        monkeypatch.setenv("SLM_DISABLE_RECENCY_STRATEGY", "1")
        config = RetrievalConfig(enable_recency_strategy=True)
        clf = QueryStrategyClassifier(config=config)
        result = clf.classify("what am I working on", {}).query_type
        assert result != "recency", (
            "Env var SLM_DISABLE_RECENCY_STRATEGY=1 must override "
            "RetrievalConfig(enable_recency_strategy=True)."
        )


# ---------------------------------------------------------------------------
# 'latest' and 'currently' must not route topical questions to
# the temporal/recency path, which dumps newest facts with no topic filter
# ---------------------------------------------------------------------------

class TestTopicalQueriesNotOverTriggered:
    """Time-flavoured words used to mean 'most recent version' must not
    trigger the temporal or recency path.

    When 'latest' or 'currently' describes a topic ('the latest authentication
    design', 'the currently supported format'), the temporal path is wrong:
    it finds no entities and runs the recency fallback, which returns the 50
    newest facts with no topic filter and at temporal weight 2.0, burying the
    topical subject the user named.
    """

    def test_latest_authentication_design_is_factual(self) -> None:
        """'what is the latest authentication design' must classify as factual.

        This fails before the fix because 'latest' is in _TEMPORAL_WORDS,
        routing a topical query to the temporal path where the recency fallback
        returns newest facts unrelated to authentication.
        """
        result = classify_query("what is the latest authentication design").query_type
        assert result == "factual", (
            f"'what is the latest authentication design' -> {result!r}. "
            "The word 'latest' here means 'most recent version', not a temporal event. "
            "Routing this to temporal causes the fallback to dump 50 newest facts "
            "at weight 2.0 with no regard for the authentication topic."
        )

    def test_latest_update_is_factual(self) -> None:
        """'what is the latest update' must classify as factual.

        After removing 'latest' from _TEMPORAL_WORDS, this topical question
        receives factual weights (balanced BM25 + semantic) which are better
        for finding the actual update content than a recency dump.
        Note: the previous PRESENT_TEMPORAL_WORDS test expected 'temporal or recency'
        for this query. That expectation was wrong — 'latest' in this context
        means 'most recent version of', not 'what time did an event happen'.
        """
        result = classify_query("what is the latest update").query_type
        assert result == "factual", (
            f"'what is the latest update' -> {result!r}. "
            "After removing 'latest' from _TEMPORAL_WORDS, this should be factual."
        )

    def test_currently_supported_format_is_factual(self) -> None:
        """'what is the currently supported format' must classify as factual.

        This is the same pattern as 'what is the current database schema' (already
        fixed by removing 'current'). 'Currently' in this context means 'as of
        now', not a temporal event query.
        """
        result = classify_query("what is the currently supported format").query_type
        assert result == "factual", (
            f"'what is the currently supported format' -> {result!r}. "
            "Should be factual — 'currently' here means 'as of now', not a time query."
        )
