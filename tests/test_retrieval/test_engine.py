# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.retrieval.engine — five-producer Retrieval Engine.

Covers:
  - RetrievalEngine.recall() full pipeline
  - Channel execution with mocked channels
  - RRF fusion integration
  - Cross-encoder reranking (with and without)
  - Fact loading from DB
  - Response building (RecallResponse structure)
  - Error handling (channel failures logged, not raised)
  - Empty results handling
  - Strategy classification integration
  - Profile forwarding
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from superlocalmemory.core.config import RetrievalConfig
from superlocalmemory.retrieval.engine import RetrievalEngine

# Imported at collection time on purpose: tests/conftest.py replaces the
# module attribute with a MagicMock for the whole session so no test loads
# the real model. Importing inside a test would get that mock instead.
from superlocalmemory.retrieval.reranker import (
    CrossEncoderReranker as _RealCrossEncoderReranker,
)
from superlocalmemory.storage.models import (
    AtomicFact,
    Mode,
    RecallResponse,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fact(fact_id: str, content: str = "") -> AtomicFact:
    return AtomicFact(
        fact_id=fact_id, memory_id="m0",
        content=content or f"fact {fact_id}",
        confidence=0.9,
    )


def _mock_db(facts: list[AtomicFact] | None = None) -> MagicMock:
    db = MagicMock()
    _facts = facts or []
    db.get_all_facts.return_value = _facts
    # **kwargs absorbs the multi-scope include_global/include_shared params that
    # the real DatabaseManager.get_facts_by_ids now accepts (defaulted, additive).
    db.get_facts_by_ids.side_effect = (
        lambda ids, pid, **kwargs: [f for f in _facts if f.fact_id in ids]
    )
    db.get_scenes_for_fact.return_value = []
    # Correction-admission contract (4.0.2 Brain Core / M042) — see the matching
    # note in test_evidence_floor.py. temporal_validity_filter fails CLOSED when
    # the correction-lifecycle read is unprovable, so a bare MagicMock made every
    # recall abstain and these assertions silently exercised nothing. Real empty
    # sets mean "admission proved, no corrections in play".
    db.get_invalidated_fact_ids.return_value = set()
    db.get_nonapplied_correction_successor_ids.return_value = set()
    db.get_strict_temporal_excluded_fact_ids.return_value = set()
    return db


def _mock_embedder(embedding: list[float] | None = None) -> MagicMock:
    emb = MagicMock()
    emb.embed.return_value = embedding or [0.1, 0.2, 0.3]
    return emb


def _mock_channel(results: list[tuple[str, float]]) -> MagicMock:
    ch = MagicMock()
    ch.search.return_value = results
    return ch


def _mock_reranker(scored: list[tuple[str, float]] | None = None) -> MagicMock:
    rr = MagicMock()
    rr.rerank.return_value = scored or []
    return rr


def _build_engine(
    db: MagicMock | None = None,
    semantic_results: list[tuple[str, float]] | None = None,
    bm25_results: list[tuple[str, float]] | None = None,
    entity_results: list[tuple[str, float]] | None = None,
    temporal_results: list[tuple[str, float]] | None = None,
    reranker: MagicMock | None = None,
    embedder: MagicMock | None = None,
    config: RetrievalConfig | None = None,
) -> RetrievalEngine:
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
    )


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class TestRecallPipeline:
    def test_basic_recall_with_semantic_channel(self) -> None:
        facts = [_make_fact("f1", "Alice is an engineer")]
        db = _mock_db(facts)
        engine = _build_engine(
            db=db,
            semantic_results=[("f1", 0.9)],
        )
        response = engine.recall("What does Alice do?", "default")
        assert isinstance(response, RecallResponse)
        assert response.query == "What does Alice do?"
        assert len(response.results) == 1
        assert response.results[0].fact.fact_id == "f1"

    def test_recall_with_multiple_channels(self) -> None:
        facts = [
            _make_fact("f1", "Alice is an engineer"),
            _make_fact("f2", "Bob is a doctor"),
        ]
        db = _mock_db(facts)
        engine = _build_engine(
            db=db,
            semantic_results=[("f1", 0.9), ("f2", 0.5)],
            bm25_results=[("f2", 0.8), ("f1", 0.3)],
        )
        response = engine.recall("What do they do?", "default")
        assert len(response.results) == 2
        assert response.total_candidates > 0

    def test_recall_mode_propagated(self) -> None:
        db = _mock_db([_make_fact("f1")])
        engine = _build_engine(db=db, semantic_results=[("f1", 0.9)])
        response = engine.recall("q", "default", mode=Mode.C)
        assert response.mode == Mode.C

    def test_recall_timing_recorded(self) -> None:
        db = _mock_db([_make_fact("f1")])
        engine = _build_engine(db=db, semantic_results=[("f1", 0.9)])
        response = engine.recall("q", "default")
        assert response.retrieval_time_ms > 0

    def test_recall_limit_respected(self) -> None:
        facts = [_make_fact(f"f{i}") for i in range(20)]
        db = _mock_db(facts)
        sem_results = [(f"f{i}", 0.9 - i * 0.01) for i in range(20)]
        engine = _build_engine(db=db, semantic_results=sem_results)
        response = engine.recall("q", "default", limit=5)
        assert len(response.results) <= 5


# ---------------------------------------------------------------------------
# Channel error handling
# ---------------------------------------------------------------------------

class TestChannelErrors:
    def test_semantic_channel_error_logged(self) -> None:
        ch = MagicMock()
        ch.search.side_effect = RuntimeError("Embedding fail")
        engine = RetrievalEngine(
            db=_mock_db(), config=RetrievalConfig(),
            channels={"semantic": ch},
            embedder=_mock_embedder(),
        )
        # Should not raise
        response = engine.recall("q", "default")
        assert isinstance(response, RecallResponse)

    def test_bm25_channel_error_logged(self) -> None:
        ch = MagicMock()
        ch.search.side_effect = RuntimeError("BM25 fail")
        engine = RetrievalEngine(
            db=_mock_db(), config=RetrievalConfig(),
            channels={"bm25": ch},
        )
        response = engine.recall("q", "default")
        assert isinstance(response, RecallResponse)

    def test_entity_channel_error_logged(self) -> None:
        ch = MagicMock()
        ch.search.side_effect = RuntimeError("Entity fail")
        engine = RetrievalEngine(
            db=_mock_db(), config=RetrievalConfig(),
            channels={"entity_graph": ch},
        )
        response = engine.recall("q", "default")
        assert isinstance(response, RecallResponse)

    def test_temporal_channel_error_logged(self) -> None:
        ch = MagicMock()
        ch.search.side_effect = RuntimeError("Temporal fail")
        engine = RetrievalEngine(
            db=_mock_db(), config=RetrievalConfig(),
            channels={"temporal": ch},
        )
        response = engine.recall("q", "default")
        assert isinstance(response, RecallResponse)


# ---------------------------------------------------------------------------
# Empty results
# ---------------------------------------------------------------------------

class TestEmptyResults:
    def test_no_channels_returns_empty(self) -> None:
        engine = RetrievalEngine(
            db=_mock_db(), config=RetrievalConfig(), channels={},
        )
        response = engine.recall("q", "default")
        assert response.results == []
        assert response.total_candidates == 0

    def test_channels_return_empty(self) -> None:
        engine = _build_engine(
            semantic_results=[],
            bm25_results=[],
        )
        response = engine.recall("q", "default")
        assert response.results == []

    def test_facts_not_found_in_db(self) -> None:
        # Channel returns IDs but DB has no matching facts
        db = _mock_db([])
        engine = _build_engine(
            db=db, semantic_results=[("f_missing", 0.9)],
        )
        response = engine.recall("q", "default")
        assert response.results == []


# ---------------------------------------------------------------------------
# Cross-encoder reranking
# ---------------------------------------------------------------------------

class TestCrossEncoderIntegration:
    def test_reranker_called_when_provided(self) -> None:
        f1 = _make_fact("f1")
        f2 = _make_fact("f2")
        facts = [f1, f2]
        db = _mock_db(facts)
        reranker = MagicMock()
        # Reranker receives (AtomicFact, score) tuples, returns same format
        reranker.rerank.return_value = [(f2, 0.95), (f1, 0.4)]

        engine = _build_engine(
            db=db,
            semantic_results=[("f1", 0.9), ("f2", 0.5)],
            reranker=reranker,
        )
        response = engine.recall("q", "default")
        reranker.rerank.assert_called_once()
        assert response.reranker_applied is True
        assert response.reranker_status == "applied"

    def test_no_reranker_skips_rerank(self) -> None:
        facts = [_make_fact("f1")]
        db = _mock_db(facts)
        engine = _build_engine(
            db=db,
            semantic_results=[("f1", 0.9)],
            reranker=None,
        )
        response = engine.recall("q", "default")
        assert len(response.results) == 1
        assert response.reranker_applied is False
        assert response.reranker_status == "not_configured"

    def test_reranker_failure_graceful(self) -> None:
        facts = [_make_fact("f1")]
        db = _mock_db(facts)
        reranker = MagicMock()
        reranker.rerank.side_effect = RuntimeError("Model crash")

        engine = _build_engine(
            db=db,
            semantic_results=[("f1", 0.9)],
            reranker=reranker,
        )
        # Should not raise, fallback to un-reranked results
        response = engine.recall("q", "default")
        assert len(response.results) == 1
        assert response.reranker_applied is False
        assert response.reranker_status == "error"

    def test_cold_reranker_fallback_is_explicit_and_self_heals(self) -> None:
        """A cold reranker reports an explicit fallback — and is still asked.

        This test used to assert the engine must NOT consult a reranker whose
        worker was down. That assertion was the bug: the reranker's own
        re-warm lives inside the call the engine refused to make, so once the
        worker died — the routine 500-request recycle, the 30-minute idle kill,
        a crash — reranking stayed off until the daemon was restarted. The live
        daemon ran unranked for seven days that way. The fallback must stay
        explicit; the engine must still let the reranker heal itself.
        """
        facts = [_make_fact("f1"), _make_fact("f2")]
        with patch.object(_RealCrossEncoderReranker, "_start_background_warmup"):
            reranker = _RealCrossEncoderReranker(model_name="fake-model")
        assert type(reranker) is _RealCrossEncoderReranker  # not the session mock
        reranker._worker_ready = False  # the state _kill_worker leaves behind
        reranker._model_loaded = False
        engine = _build_engine(
            db=_mock_db(facts),
            semantic_results=[("f1", 0.9), ("f2", 0.5)],
            reranker=reranker,
        )

        with patch.object(reranker, "_start_background_warmup") as rewarm:
            response = engine.recall("q", "default")

        rewarm.assert_called_once()
        assert response.reranker_applied is False
        assert response.reranker_status == "fallback_not_ready"
        assert response.results and response.results[0].fact.fact_id == "f1"

    def test_a_reranker_type_without_worker_ready_is_used(self) -> None:
        """RemoteReranker has no ``_worker_ready`` attribute at all.

        The engine gated on ``getattr(reranker, "_worker_ready", False)``, so
        a configured remote reranker was never called: every recall reported
        ``fallback_not_ready`` and returned the fused order, silently.
        """
        f1, f2 = _make_fact("f1"), _make_fact("f2")

        class NoWorkerFlagReranker:
            def rerank_with_status(self, query, candidates, top_k=10):
                return [(f2, 0.95), (f1, 0.4)][:top_k], True, "applied"

        assert not hasattr(NoWorkerFlagReranker(), "_worker_ready")
        engine = _build_engine(
            db=_mock_db([f1, f2]),
            semantic_results=[("f1", 0.9), ("f2", 0.5)],
            reranker=NoWorkerFlagReranker(),
        )

        response = engine.recall("q", "default")

        assert response.reranker_applied is True
        assert response.reranker_status == "applied"


# ---------------------------------------------------------------------------
# Response structure
# ---------------------------------------------------------------------------

class TestResponseStructure:
    def test_query_type_from_strategy(self) -> None:
        facts = [_make_fact("f1")]
        db = _mock_db(facts)
        engine = _build_engine(db=db, semantic_results=[("f1", 0.9)])
        response = engine.recall("When did Alice start?", "default")
        # Strategy should detect temporal
        assert response.query_type == "temporal"

    def test_channel_weights_in_response(self) -> None:
        facts = [_make_fact("f1")]
        db = _mock_db(facts)
        engine = _build_engine(db=db, semantic_results=[("f1", 0.9)])
        response = engine.recall("q", "default")
        assert isinstance(response.channel_weights, dict)
        assert len(response.channel_weights) > 0

    def test_result_has_evidence_chain(self) -> None:
        facts = [_make_fact("f1")]
        db = _mock_db(facts)
        engine = _build_engine(
            db=db,
            semantic_results=[("f1", 0.9)],
            bm25_results=[("f1", 0.7)],
        )
        response = engine.recall("q", "default")
        assert len(response.results) > 0
        result = response.results[0]
        assert isinstance(result.evidence_chain, list)
        # Should have evidence from both channels
        assert len(result.evidence_chain) >= 1

    def test_result_confidence_bounded(self) -> None:
        facts = [_make_fact("f1")]
        db = _mock_db(facts)
        engine = _build_engine(db=db, semantic_results=[("f1", 0.9)])
        response = engine.recall("q", "default")
        for result in response.results:
            assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# Embedder requirement
# ---------------------------------------------------------------------------

class TestEmbedderRequirement:
    def test_semantic_skipped_without_embedder(self) -> None:
        facts = [_make_fact("f1")]
        db = _mock_db(facts)
        sem_ch = _mock_channel([("f1", 0.9)])
        engine = RetrievalEngine(
            db=db, config=RetrievalConfig(),
            channels={"semantic": sem_ch},
            embedder=None,  # No embedder
        )
        engine.recall("q", "default")
        # Semantic channel should be skipped
        sem_ch.search.assert_not_called()
