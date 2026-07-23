# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for 4-channel retrieval pipeline — end-to-end integration.

Phase 0 Safety Net: exercises the full RetrievalEngine pipeline with
all 4 channels mocked independently. Captures current behavior of
RRF fusion, channel disabling, trust weighting, agentic adapter,
and content quality penalty before Phase 1 restructuring.

Covers:
  - All 4 channels contributing to fusion
  - Single-channel sufficiency (semantic, bm25, entity_graph, temporal)
  - RRF fusion ordering
  - Channel disabling via disabled_channels config
  - Trust weighting (boost, demote, disabled)
  - Agentic adapter (recall_facts returns tuples, respects top_k)
  - Content quality penalty for short content
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from superlocalmemory.core.config import RetrievalConfig
from superlocalmemory.retrieval.bm25_channel import BM25Channel
from superlocalmemory.retrieval.engine import RetrievalEngine
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, MemoryRecord


# ---------------------------------------------------------------------------
# Helpers — match existing test_engine.py conventions
# ---------------------------------------------------------------------------

def _make_fact(
    fact_id: str,
    content: str = "",
    confidence: float = 0.9,
    trust: float = 0.5,
) -> AtomicFact:
    """Create a minimal AtomicFact for testing."""
    return AtomicFact(
        fact_id=fact_id,
        memory_id="m0",
        content=content or f"Detailed factual content about {fact_id} and related information",
        confidence=confidence,
    )


def _mock_db(facts: list[AtomicFact] | None = None) -> MagicMock:
    """Return a MagicMock DB that returns the given facts from get_all_facts."""
    db = MagicMock()
    _facts = facts or []
    db.get_all_facts.return_value = _facts
    # V3.3.13: _load_facts uses get_facts_by_ids instead of get_all_facts
    db.get_facts_by_ids.side_effect = (
        lambda ids, pid, **kwargs: [f for f in _facts if f.fact_id in ids]
    )
    db.get_scenes_for_fact.return_value = []
    return db


def _mock_embedder(embedding: list[float] | None = None) -> MagicMock:
    """Return a MagicMock embedder producing a fixed vector."""
    emb = MagicMock()
    emb.embed.return_value = embedding or [0.1, 0.2, 0.3]
    return emb


def _mock_channel(results: list[tuple[str, float]]) -> MagicMock:
    """Return a MagicMock channel whose search() returns given results."""
    ch = MagicMock()
    ch.search.return_value = results
    return ch


def _build_engine(
    db: MagicMock | None = None,
    semantic_results: list[tuple[str, float]] | None = None,
    bm25_results: list[tuple[str, float]] | None = None,
    entity_results: list[tuple[str, float]] | None = None,
    temporal_results: list[tuple[str, float]] | None = None,
    reranker: MagicMock | None = None,
    embedder: MagicMock | None = None,
    config: RetrievalConfig | None = None,
    trust_scorer: MagicMock | None = None,
) -> RetrievalEngine:
    """Build a RetrievalEngine with mocked channels, matching test_engine.py pattern."""
    channels: dict = {}
    if semantic_results is not None:
        channels["semantic"] = _mock_channel(semantic_results)
    if bm25_results is not None:
        channels["bm25"] = _mock_channel(bm25_results)
    if entity_results is not None:
        channels["entity_graph"] = _mock_channel(entity_results)
    if temporal_results is not None:
        channels["temporal"] = _mock_channel(temporal_results)

    return RetrievalEngine(
        db=db or _mock_db(),
        config=config or RetrievalConfig(),
        channels=channels,
        embedder=embedder or (_mock_embedder() if semantic_results is not None else None),
        reranker=reranker,
        trust_scorer=trust_scorer,
    )


# ---------------------------------------------------------------------------
# 4-channel pipeline
# ---------------------------------------------------------------------------

class TestFourChannelPipeline:
    """Verify all 4 channels contribute to fused results."""

    def test_all_channels_contribute_to_fusion(self) -> None:
        """When all 4 channels return results, fusion includes candidates from each."""
        facts = [
            _make_fact("f_sem", "Alice works at NovaTech as a senior architect with deep expertise"),
            _make_fact("f_bm25", "Bob joined the ML team and leads the data science projects"),
            _make_fact("f_entity", "Charlie mentioned the Qualixar product suite during the meeting"),
            _make_fact("f_temp", "Last Tuesday the deployment pipeline was refactored completely"),
        ]
        db = _mock_db(facts)
        engine = _build_engine(
            db=db,
            semantic_results=[("f_sem", 0.9)],
            bm25_results=[("f_bm25", 0.8)],
            entity_results=[("f_entity", 0.7)],
            temporal_results=[("f_temp", 0.6)],
        )
        response = engine.recall("What happened?", "default")
        result_ids = {r.fact.fact_id for r in response.results}
        assert "f_sem" in result_ids
        assert "f_bm25" in result_ids
        # V3.4.12: entity_graph is now a signal enhancer (post-RRF boost),
        # not an independent channel. It scores candidates from other channels
        # rather than producing its own results. f_entity won't appear unless
        # it's also found by semantic/BM25/temporal.
        assert "f_temp" in result_ids

    def test_semantic_only_works(self) -> None:
        """A single semantic channel is sufficient to produce results."""
        facts = [_make_fact("f1", "Alice is a senior architect building enterprise systems")]
        db = _mock_db(facts)
        engine = _build_engine(db=db, semantic_results=[("f1", 0.9)])
        response = engine.recall("q", "default")
        assert len(response.results) == 1
        assert response.results[0].fact.fact_id == "f1"

    def test_bm25_only_works(self) -> None:
        """A single BM25 channel is sufficient to produce results."""
        facts = [_make_fact("f1", "Bob manages the infrastructure deployment pipeline")]
        db = _mock_db(facts)
        engine = _build_engine(db=db, bm25_results=[("f1", 0.8)])
        response = engine.recall("q", "default")
        assert len(response.results) == 1

    def test_exact_bm25_hit_survives_semantic_top_k_pressure(self) -> None:
        """A newly queryable exact-token fact must not hide below semantic hits."""
        semantic_ids = [f"semantic-{index}" for index in range(5)]
        facts = [
            *[
                _make_fact(
                    fact_id,
                    f"Semantically similar older memory number {index}",
                )
                for index, fact_id in enumerate(semantic_ids)
            ],
            _make_fact(
                "exact-new",
                "SLM381_UNIQUE_QUERYABLE_MARKER is immediately recallable",
            ),
        ]
        engine = _build_engine(
            db=_mock_db(facts),
            semantic_results=[
                (fact_id, 0.95 - index * 0.01)
                for index, fact_id in enumerate(semantic_ids)
            ],
            bm25_results=[("exact-new", 12.0)],
        )

        response = engine.recall(
            "SLM381_UNIQUE_QUERYABLE_MARKER",
            "default",
            limit=3,
        )

        assert len(response.results) == 3
        assert "exact-new" in {result.fact.fact_id for result in response.results}

    def test_real_fts5_exact_hit_keeps_bounded_slot(
        self,
        tmp_path,
    ) -> None:
        """A real sub-1.0 FTS5 hit remains visible under semantic pressure."""
        db = DatabaseManager(tmp_path / "fts5-exact-slot.db")
        db.initialize(real_schema)
        semantic_ids = [f"semantic-{index}" for index in range(3)]
        for index, fact_id in enumerate([*semantic_ids, "exact-new"]):
            memory_id = f"memory-{fact_id}"
            content = (
                "SLM381_UNIQUE_QUERYABLE_MARKER is immediately recallable"
                if fact_id == "exact-new"
                else f"Semantically similar older memory number {index}"
            )
            db.store_memory(MemoryRecord(memory_id=memory_id, content=content))
            db.store_fact(AtomicFact(
                fact_id=fact_id,
                memory_id=memory_id,
                content=content,
            ))

        bm25 = BM25Channel(db)
        lexical = bm25.search("SLM381_UNIQUE_QUERYABLE_MARKER", "default")
        assert len(lexical) == 1
        assert lexical[0][0] == "exact-new"
        assert 0.0 < lexical[0][1] < 1.0

        engine = RetrievalEngine(
            db=db,
            config=RetrievalConfig(),
            channels={
                "semantic": _mock_channel([
                    (fact_id, 0.95 - index * 0.01)
                    for index, fact_id in enumerate(semantic_ids)
                ]),
                "bm25": bm25,
            },
            embedder=_mock_embedder(),
        )
        try:
            response = engine.recall(
                "SLM381_UNIQUE_QUERYABLE_MARKER",
                "default",
                limit=2,
            )
        finally:
            engine.close()

        assert len(response.results) == 2
        assert "exact-new" in {result.fact.fact_id for result in response.results}

    def test_entity_graph_enhances_bm25(self) -> None:
        """V3.4.12: entity_graph is a signal enhancer, not independent channel.
        It boosts BM25/semantic candidates by graph proximity."""
        facts = [_make_fact("f1", "Charlie mentioned the product roadmap in the planning session")]
        db = _mock_db(facts)
        # entity_graph alone produces 0 results (it's a post-RRF enhancer now)
        # but combined with BM25, it boosts the result
        engine = _build_engine(db=db, bm25_results=[("f1", 0.8)], entity_results=[("f1", 0.7)])
        response = engine.recall("q", "default")
        assert len(response.results) == 1

    def test_temporal_only_works(self) -> None:
        """A single temporal channel is sufficient to produce results."""
        facts = [_make_fact("f1", "Last week the team completed the migration to the new database")]
        db = _mock_db(facts)
        engine = _build_engine(db=db, temporal_results=[("f1", 0.6)])
        response = engine.recall("q", "default")
        assert len(response.results) == 1

    def test_fusion_ranks_by_rrf_score(self) -> None:
        """Facts appearing in more channels should rank higher via RRF.

        Both facts carry above-floor semantic evidence (v3.6.6 evidence floor
        requires semantic >= 0.60 or other channel signal); this test isolates
        RRF *ranking*, not the floor (which is covered in test_evidence_floor).
        """
        facts = [
            _make_fact("f_multi", "Alice is an engineer working on multiple critical projects"),
            _make_fact("f_single", "Bob mentioned he likes coffee during the morning standup"),
        ]
        db = _mock_db(facts)
        engine = _build_engine(
            db=db,
            semantic_results=[("f_multi", 0.9), ("f_single", 0.65)],
            bm25_results=[("f_multi", 0.8)],
        )
        response = engine.recall("q", "default")
        assert len(response.results) == 2
        # f_multi appears in both channels -> higher RRF score
        assert response.results[0].fact.fact_id == "f_multi"


# ---------------------------------------------------------------------------
# Channel disabling
# ---------------------------------------------------------------------------

class TestChannelDisabling:
    """Verify disabled_channels config suppresses channels."""

    def test_disabled_channels_skipped(self) -> None:
        """Channels in disabled_channels list are not called."""
        facts = [
            _make_fact("f_sem", "Semantic channel fact with detailed architecture content"),
            _make_fact("f_bm25", "BM25 channel fact about keyword matching and relevance"),
        ]
        db = _mock_db(facts)
        config = RetrievalConfig(disabled_channels=["bm25"])
        sem_ch = _mock_channel([("f_sem", 0.9)])
        bm25_ch = _mock_channel([("f_bm25", 0.8)])

        engine = RetrievalEngine(
            db=db, config=config,
            channels={"semantic": sem_ch, "bm25": bm25_ch},
            embedder=_mock_embedder(),
        )
        engine.recall("q", "default")
        bm25_ch.search.assert_not_called()
        sem_ch.search.assert_called_once()

    def test_empty_disabled_all_channels_run(self) -> None:
        """An empty disabled_channels list means all channels are active."""
        facts = [
            _make_fact("f1", "Semantic result about the enterprise architecture discussion"),
            _make_fact("f2", "BM25 result about the code review process and findings"),
        ]
        db = _mock_db(facts)
        config = RetrievalConfig(disabled_channels=[])
        sem_ch = _mock_channel([("f1", 0.9)])
        bm25_ch = _mock_channel([("f2", 0.8)])

        engine = RetrievalEngine(
            db=db, config=config,
            channels={"semantic": sem_ch, "bm25": bm25_ch},
            embedder=_mock_embedder(),
        )
        engine.recall("q", "default")
        sem_ch.search.assert_called_once()
        bm25_ch.search.assert_called_once()


# ---------------------------------------------------------------------------
# Trust weighting
# ---------------------------------------------------------------------------

class TestTrustWeighting:
    """Verify Bayesian trust weight modulates final ranking."""

    def _build_trust_engine(
        self,
        facts: list[AtomicFact],
        trust_map: dict[str, float],
        use_trust: bool = True,
    ) -> RetrievalEngine:
        """Helper: engine with a mock trust scorer returning preset trust values."""
        db = _mock_db(facts)
        config = RetrievalConfig(use_trust_weighting=use_trust)
        scorer = MagicMock()
        scorer.get_fact_trust.side_effect = lambda fid, pid: trust_map.get(fid, 0.5)
        return _build_engine(
            db=db,
            semantic_results=[(f.fact_id, 0.9) for f in facts],
            config=config,
            trust_scorer=scorer,
        )

    def test_trust_weight_boosts_high_trust_facts(self) -> None:
        """trust=1.0 maps to weight=1.5, boosting the fact's score."""
        f_high = _make_fact("f_high", "High-trust fact with comprehensive verified evidence")
        f_low = _make_fact("f_low", "Low-trust fact with minimal unverified source information")
        engine = self._build_trust_engine(
            [f_high, f_low],
            trust_map={"f_high": 1.0, "f_low": 0.0},
        )
        response = engine.recall("q", "default")
        scores = {r.fact.fact_id: r.score for r in response.results}
        assert scores["f_high"] > scores["f_low"]

    def test_trust_weight_demotes_low_trust_facts(self) -> None:
        """trust=0.0 maps to weight=0.5, demoting the fact's score."""
        f_untrusted = _make_fact("f_untrusted", "Untrusted fact about dubious claim with no evidence")
        engine = self._build_trust_engine(
            [f_untrusted],
            trust_map={"f_untrusted": 0.0},
        )
        response = engine.recall("q", "default")
        # The trust_score field should reflect low trust
        assert response.results[0].trust_score == pytest.approx(0.0, abs=0.1)

    def test_trust_disabled_returns_neutral(self) -> None:
        """When use_trust_weighting=False, trust_score defaults to 0.5 (neutral)."""
        f1 = _make_fact("f1", "Fact that should not have trust applied to its retrieval score")
        engine = self._build_trust_engine(
            [f1],
            trust_map={"f1": 0.0},  # Would demote if enabled
            use_trust=False,
        )
        response = engine.recall("q", "default")
        # Default trust when disabled is 0.5 (neutral)
        assert response.results[0].trust_score == pytest.approx(0.5, abs=0.1)


# ---------------------------------------------------------------------------
# Agentic adapter (recall_facts)
# ---------------------------------------------------------------------------

class TestAgenticAdapter:
    """Verify the simplified recall_facts() used by AgenticRetriever."""

    def test_recall_facts_returns_tuples(self) -> None:
        """recall_facts must return list of (AtomicFact, float) tuples."""
        facts = [_make_fact("f1", "Alice is a senior engineer building production systems at scale")]
        db = _mock_db(facts)
        engine = _build_engine(db=db, semantic_results=[("f1", 0.9)])
        pairs = engine.recall_facts("q", "default", top_k=10)
        assert len(pairs) == 1
        fact_obj, score = pairs[0]
        assert isinstance(fact_obj, AtomicFact)
        assert isinstance(score, float)
        assert fact_obj.fact_id == "f1"

    def test_recall_facts_respects_top_k(self) -> None:
        """recall_facts limits results to top_k."""
        facts = [_make_fact(f"f{i}", f"Fact number {i} with enough content to pass quality check") for i in range(10)]
        db = _mock_db(facts)
        sem_results = [(f"f{i}", 0.9 - i * 0.01) for i in range(10)]
        engine = _build_engine(db=db, semantic_results=sem_results)
        pairs = engine.recall_facts("q", "default", top_k=3)
        assert len(pairs) <= 3


# ---------------------------------------------------------------------------
# Content quality penalty
# ---------------------------------------------------------------------------

class TestContentQualityPenalty:
    """Verify short/low-info facts are penalized in final scoring."""

    def test_short_content_penalized(self) -> None:
        """Facts with content < 25 chars get quality=0.1, reducing their score."""
        f_short = _make_fact("f_short", "Hi!")  # 3 chars -> quality=0.1
        f_long = _make_fact(
            "f_long",
            "Alice is a senior architect at NovaTech with 15 years of experience building enterprise systems",
        )  # 94 chars -> quality=1.0
        db = _mock_db([f_short, f_long])
        engine = _build_engine(
            db=db,
            semantic_results=[("f_short", 0.9), ("f_long", 0.9)],
        )
        response = engine.recall("q", "default")
        scores = {r.fact.fact_id: r.score for r in response.results}
        # f_long should have a significantly higher final score due to quality multiplier
        assert scores.get("f_long", 0) > scores.get("f_short", 0), (
            f"Long content ({scores.get('f_long')}) should outscore short content ({scores.get('f_short')})"
        )


# ---------------------------------------------------------------------------
# T1: bi-temporal validity filter wired into the recall pipeline
# ---------------------------------------------------------------------------

class TestTemporalValidityWiring:
    """Superseded (system-invalidated) facts must not surface in recall."""

    @staticmethod
    def _facts() -> list[AtomicFact]:
        return [
            _make_fact("f_valid", "Alice currently lives in Mumbai and works there daily"),
            _make_fact("f_superseded", "Alice used to live in Delhi before relocating recently"),
        ]

    def test_superseded_fact_absent_after_filter(self) -> None:
        from superlocalmemory.core.config import TemporalValidatorConfig
        from superlocalmemory.retrieval.temporal_validity_filter import (
            register_temporal_validity_filter,
        )
        facts = self._facts()
        db = _mock_db(facts)
        db.get_invalidated_fact_ids.side_effect = (
            lambda ids, pid: {"f_superseded"} & set(ids)
        )
        engine = _build_engine(
            db=db,
            semantic_results=[("f_valid", 0.9), ("f_superseded", 0.85)],
        )
        register_temporal_validity_filter(
            engine._registry, db, TemporalValidatorConfig(),
        )

        response = engine.recall("where does Alice live?", "default")
        ids = {r.fact.fact_id for r in response.results}
        assert "f_valid" in ids
        assert "f_superseded" not in ids

    def test_without_filter_superseded_fact_present(self) -> None:
        # Control: without the filter, both facts surface — proving the filter
        # (not some other pipeline stage) is what removes the superseded one.
        facts = self._facts()
        db = _mock_db(facts)
        engine = _build_engine(
            db=db,
            semantic_results=[("f_valid", 0.9), ("f_superseded", 0.85)],
        )
        response = engine.recall("where does Alice live?", "default")
        ids = {r.fact.fact_id for r in response.results}
        assert "f_valid" in ids
        assert "f_superseded" in ids


# ---------------------------------------------------------------------------
# T-window: event-time range pruning in recall()
# ---------------------------------------------------------------------------

class TestTimeWindowRecall:
    """recall(window=...) keeps only candidates whose event time is in range."""

    @staticmethod
    def _facts() -> list[AtomicFact]:
        return [
            _make_fact("f_recent", "The deployment pipeline was refactored this month in detail"),
            _make_fact("f_old", "The original prototype was written a long time ago back then"),
        ]

    def _engine(self) -> tuple[RetrievalEngine, MagicMock]:
        facts = self._facts()
        db = _mock_db(facts)
        db.get_fact_event_times.side_effect = lambda ids, pid: {
            k: v for k, v in
            {"f_recent": "2026-07-20 09:00:00", "f_old": "2020-01-01 09:00:00"}.items()
            if k in ids
        }
        engine = _build_engine(
            db=db, semantic_results=[("f_recent", 0.9), ("f_old", 0.88)],
        )
        return engine, db

    def test_window_keeps_only_in_range(self) -> None:
        engine, _ = self._engine()
        response = engine.recall(
            "q", "default", window=("2026-07-01", "2026-07-31"),
        )
        ids = {r.fact.fact_id for r in response.results}
        assert "f_recent" in ids
        assert "f_old" not in ids

    def test_no_window_returns_both(self) -> None:
        engine, db = self._engine()
        response = engine.recall("q", "default")  # window=None
        ids = {r.fact.fact_id for r in response.results}
        assert {"f_recent", "f_old"} <= ids
        db.get_fact_event_times.assert_not_called()  # no window -> no lookup

    def test_unparseable_window_applies_no_filter(self) -> None:
        engine, _ = self._engine()
        response = engine.recall("q", "default", window="someday")
        ids = {r.fact.fact_id for r in response.results}
        assert {"f_recent", "f_old"} <= ids  # bad spec => additive no-op

    def test_query_infers_window(self) -> None:
        # T3: no explicit window, but "last week" in the query auto-windows to 7d.
        from datetime import datetime, timedelta, timezone
        recent = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        facts = self._facts()
        db = _mock_db(facts)
        db.get_fact_event_times.side_effect = lambda ids, pid: {
            k: v for k, v in
            {"f_recent": recent, "f_old": "2019-01-01 00:00:00"}.items()
            if k in ids
        }
        engine = _build_engine(
            db=db, semantic_results=[("f_recent", 0.9), ("f_old", 0.88)],
        )
        response = engine.recall("what did I work on last week", "default")
        ids = {r.fact.fact_id for r in response.results}
        assert "f_recent" in ids
        assert "f_old" not in ids

    def test_inferred_window_falls_back_when_empty(self) -> None:
        # An inferred (query-derived) window must never zero-out results: if it
        # would exclude everything, recall falls back to the unwindowed set.
        facts = self._facts()
        db = _mock_db(facts)
        db.get_fact_event_times.side_effect = lambda ids, pid: {
            k: "2019-01-01 00:00:00" for k in ids  # all far outside "last week"
        }
        engine = _build_engine(
            db=db, semantic_results=[("f_recent", 0.9), ("f_old", 0.88)],
        )
        response = engine.recall("what did I do last week", "default")  # inferred 7d
        ids = {r.fact.fact_id for r in response.results}
        assert {"f_recent", "f_old"} <= ids  # fallback keeps all

    def test_explicit_empty_window_is_honored(self) -> None:
        # An explicit user window is authoritative even when it empties results.
        engine, _ = self._engine()   # f_recent=2026-07, f_old=2020
        response = engine.recall(
            "q", "default", window=("1900-01-01", "1900-12-31"),
        )
        assert response.results == []
