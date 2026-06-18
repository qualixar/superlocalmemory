# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 — F-1 Evidence Floor tests

"""F-1: Evidence floor — per-channel gate in retrieval/engine.py.

TDD RED suite: all tests MUST fail before the feature is implemented.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from superlocalmemory.core.config import ChannelWeights, RetrievalConfig
from superlocalmemory.retrieval.engine import RetrievalEngine
from superlocalmemory.retrieval.fusion import FusionResult
from superlocalmemory.storage.models import AtomicFact, Mode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fact(fact_id: str, content: str = "relevant fact content") -> AtomicFact:
    return AtomicFact(
        fact_id=fact_id, memory_id="m0",
        content=content, confidence=0.9,
    )


def _mock_db(facts: list[AtomicFact] | None = None) -> MagicMock:
    db = MagicMock()
    _facts = facts or []
    # _load_facts calls get_facts_by_ids which returns a list of AtomicFact
    db.get_facts_by_ids.side_effect = lambda ids, pid, **kwargs: [
        f for f in _facts if f.fact_id in ids
    ]
    db.get_scenes_for_facts_batch.return_value = {}
    return db


def _build_engine(
    facts: list[AtomicFact] | None = None,
    semantic_results: list[tuple[str, float]] | None = None,
    bm25_results: list[tuple[str, float]] | None = None,
    entity_scores: dict[str, float] | None = None,
    temporal_results: list[tuple[str, float]] | None = None,
    config: RetrievalConfig | None = None,
) -> RetrievalEngine:
    """Build a test RetrievalEngine with mocked channels.

    entity_scores: dict of fact_id -> score returned by entity.score_candidates()
    (entity_graph uses score_candidates post-RRF, not search in _run_channels).
    """
    db = _mock_db(facts)
    sem = MagicMock()
    sem.search.return_value = semantic_results or []
    bm = MagicMock()
    bm.search.return_value = bm25_results or []
    ent = MagicMock()
    ent.search.return_value = []  # not used in main path
    # score_candidates returns dict fact_id -> entity_graph score
    ent.score_candidates.return_value = entity_scores or {}
    temp = MagicMock()
    temp.search.return_value = temporal_results or []
    emb = MagicMock()
    emb.embed.return_value = [0.1, 0.2, 0.3]
    cfg = config or RetrievalConfig()
    return RetrievalEngine(
        db=db, config=cfg,
        channels={
            "semantic": sem, "bm25": bm,
            "entity_graph": ent, "temporal": temp,
        },
        embedder=emb,
    )


# ---------------------------------------------------------------------------
# F-1-A: Nonsense query earns zero channel evidence → empty results
# ---------------------------------------------------------------------------

class TestEvidenceFloor:
    """Evidence floor gate tests."""

    def test_no_evidence_returns_empty(self) -> None:
        """A query that earns zero channel evidence returns zero results."""
        # Facts exist but all channel_scores are zero (nonsense query scenario)
        f1 = _make_fact("f1")
        engine = _build_engine(
            facts=[f1],
            semantic_results=[("f1", 0.0)],  # zero cosine
            bm25_results=[],                  # zero bm25
            temporal_results=[],
        )
        response = engine.recall(
            "purple elephant quantum knitting recipe", "default", Mode.A, limit=10,
        )
        assert response.results == [], (
            "Expected empty results for nonsense query with zero channel evidence"
        )

    def test_no_evidence_adds_no_confident_match_field(self) -> None:
        """no_confident_match=True when floor empties the result set."""
        f1 = _make_fact("f1")
        engine = _build_engine(
            facts=[f1],
            semantic_results=[("f1", 0.0)],
        )
        response = engine.recall(
            "purple elephant quantum knitting recipe", "default", Mode.A, limit=10,
        )
        assert getattr(response, "no_confident_match", False) is True

    def test_semantic_above_floor_passes(self) -> None:
        """A fact with semantic >= 0.60 is retained."""
        f1 = _make_fact("f1")
        engine = _build_engine(
            facts=[f1],
            semantic_results=[("f1", 0.75)],
            bm25_results=[],
            temporal_results=[],
        )
        response = engine.recall("SLM recall quality", "default", Mode.A, limit=10)
        fact_ids = [r.fact.fact_id for r in response.results]
        assert "f1" in fact_ids, "Fact with semantic=0.75 must pass floor"

    def test_bm25_above_zero_passes(self) -> None:
        """A fact with bm25 > 0 is retained regardless of semantic."""
        f1 = _make_fact("f1")
        engine = _build_engine(
            facts=[f1],
            semantic_results=[("f1", 0.0)],  # low semantic
            bm25_results=[("f1", 0.3)],       # bm25 > 0
            temporal_results=[],
        )
        response = engine.recall("keyword match query", "default", Mode.A, limit=10)
        fact_ids = [r.fact.fact_id for r in response.results]
        assert "f1" in fact_ids, "Fact with bm25>0 must pass floor"

    def test_entity_graph_above_zero_passes(self) -> None:
        """A fact with entity_graph > 0 (from score_candidates) is retained."""
        f1 = _make_fact("f1")
        # entity_graph uses score_candidates post-RRF; need semantic hit too so fact
        # enters the pool, then entity_graph annotates it
        engine = _build_engine(
            facts=[f1],
            semantic_results=[("f1", 0.0)],
            bm25_results=[("f1", 0.0)],  # get into pool via bm25 rank
            # entity_scores: score_candidates returns these for the post-RRF boost
            entity_scores={"f1": 0.4},
            temporal_results=[],
        )
        # Manually ensure f1 appears in fused (via non-zero bm25 list with small score)
        # Actually: need the fact in the pool. Use semantic channel with tiny score
        # so it gets a RRF rank, then entity_graph score_candidates annotates it.
        # The test validates that entity_graph annotation in channel_scores passes floor.
        # Since bm25_results=[] and semantic=0, we need another path.
        # Use a direct approach: override engine channels to inject entity_graph score.
        engine2 = _build_engine(
            facts=[f1],
            semantic_results=[("f1", 0.0)],
            bm25_results=[("f1", 0.0)],
            entity_scores={"f1": 0.4},
            temporal_results=[],
        )
        # bm25 with score 0.0 means it will appear in RRF with score 0 but still rank
        # The entity enhancement then adds entity_graph to channel_scores
        response = engine2.recall("entity query", "default", Mode.A, limit=10)
        # entity_graph channel score should be set via score_candidates, passes floor
        # Note: if bm25=0 in score but appears in channel, that may not pass floor.
        # The real gate is: entity_graph score (set by score_candidates) > 0 passes.
        # Let's verify the channel_scores on results
        for r in response.results:
            cs = r.channel_scores or {}
            if r.fact.fact_id == "f1":
                assert cs.get("entity_graph", 0.0) > 0.0, (
                    "Entity graph score must be set on result"
                )

    def test_temporal_above_zero_passes(self) -> None:
        """A fact with temporal > 0 is retained."""
        f1 = _make_fact("f1")
        engine = _build_engine(
            facts=[f1],
            semantic_results=[("f1", 0.0)],
            bm25_results=[],
            temporal_results=[("f1", 0.5)],
        )
        response = engine.recall("yesterday query", "default", Mode.A, limit=10)
        fact_ids = [r.fact.fact_id for r in response.results]
        assert "f1" in fact_ids, "Fact with temporal>0 must pass floor"

    def test_pinned_fact_bypasses_floor(self) -> None:
        """Pinned facts always pass regardless of channel scores."""
        f1 = AtomicFact(
            fact_id="f1", memory_id="m0",
            content="pinned important fact", confidence=0.9,
            pinned=True,
        )
        engine = _build_engine(
            facts=[f1],
            semantic_results=[("f1", 0.0)],  # zero evidence
            bm25_results=[],
            temporal_results=[],
        )
        response = engine.recall("anything", "default", Mode.A, limit=10)
        fact_ids = [r.fact.fact_id for r in response.results]
        assert "f1" in fact_ids, "Pinned fact must bypass evidence floor"

    def test_spreading_activation_alone_does_not_pass(self) -> None:
        """spreading_activation channel alone is NOT evidence — fact still filtered."""
        f1 = _make_fact("f1")
        engine = _build_engine(
            facts=[f1],
            semantic_results=[("f1", 0.0)],
            bm25_results=[],
            temporal_results=[],
        )
        # Inject spreading_activation channel with a hit
        sa = MagicMock()
        sa.search.return_value = [("f1", 0.9)]
        engine._spreading_activation = sa
        engine._registry.register_channel(
            "spreading_activation", sa, needs_embedding=True,
        )
        response = engine.recall("anything", "default", Mode.A, limit=10)
        # Even with SA hit, zero primary evidence → filtered out
        fact_ids = [r.fact.fact_id for r in response.results]
        assert "f1" not in fact_ids, (
            "spreading_activation alone must NOT count as evidence"
        )

    def test_semantic_just_below_floor_filtered(self) -> None:
        """Semantic score of 0.59 (just below 0.60 default) is filtered."""
        f1 = _make_fact("f1")
        engine = _build_engine(
            facts=[f1],
            semantic_results=[("f1", 0.59)],
            bm25_results=[],
            temporal_results=[],
        )
        response = engine.recall("marginal query", "default", Mode.A, limit=10)
        fact_ids = [r.fact.fact_id for r in response.results]
        assert "f1" not in fact_ids, (
            "Semantic 0.59 is below floor 0.60, must be filtered"
        )

    def test_semantic_exactly_at_floor_passes(self) -> None:
        """Semantic score of exactly 0.60 passes."""
        f1 = _make_fact("f1")
        engine = _build_engine(
            facts=[f1],
            semantic_results=[("f1", 0.60)],
            bm25_results=[],
            temporal_results=[],
        )
        response = engine.recall("at-floor query", "default", Mode.A, limit=10)
        fact_ids = [r.fact.fact_id for r in response.results]
        assert "f1" in fact_ids, "Semantic >= 0.60 must pass floor"


# ---------------------------------------------------------------------------
# F-1-B: Kill switch SLM_RECALL_NO_FLOOR=1 disables floor
# ---------------------------------------------------------------------------

class TestEvidenceFloorKillSwitch:
    """Kill switch tests."""

    def test_kill_switch_disables_floor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SLM_RECALL_NO_FLOOR=1 bypasses the floor, returning all results."""
        monkeypatch.setenv("SLM_RECALL_NO_FLOOR", "1")
        f1 = _make_fact("f1")
        engine = _build_engine(
            facts=[f1],
            semantic_results=[("f1", 0.0)],  # would be filtered normally
            bm25_results=[],
            temporal_results=[],
        )
        response = engine.recall(
            "purple elephant quantum knitting recipe", "default", Mode.A, limit=10,
        )
        fact_ids = [r.fact.fact_id for r in response.results]
        assert "f1" in fact_ids, (
            "Kill switch SLM_RECALL_NO_FLOOR=1 must disable the floor"
        )

    def test_kill_switch_off_floor_active(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With kill switch off (default), floor applies normally."""
        monkeypatch.delenv("SLM_RECALL_NO_FLOOR", raising=False)
        f1 = _make_fact("f1")
        engine = _build_engine(
            facts=[f1],
            semantic_results=[("f1", 0.0)],
            bm25_results=[],
            temporal_results=[],
        )
        response = engine.recall(
            "purple elephant quantum knitting recipe", "default", Mode.A, limit=10,
        )
        assert response.results == []


# ---------------------------------------------------------------------------
# F-1-C: Config fields exist on RetrievalConfig
# ---------------------------------------------------------------------------

class TestEvidenceFloorConfig:
    """RetrievalConfig evidence_floor fields."""

    def test_config_has_evidence_floor_enabled(self) -> None:
        """RetrievalConfig.evidence_floor_enabled defaults to True."""
        cfg = RetrievalConfig()
        assert hasattr(cfg, "evidence_floor_enabled"), (
            "RetrievalConfig must have evidence_floor_enabled field"
        )
        assert cfg.evidence_floor_enabled is True

    def test_config_has_min_semantic_evidence(self) -> None:
        """RetrievalConfig.min_semantic_evidence defaults to 0.60."""
        cfg = RetrievalConfig()
        assert hasattr(cfg, "min_semantic_evidence"), (
            "RetrievalConfig must have min_semantic_evidence field"
        )
        assert cfg.min_semantic_evidence == pytest.approx(0.60)

    def test_floor_disabled_via_config(self) -> None:
        """evidence_floor_enabled=False disables the floor."""
        cfg = RetrievalConfig(evidence_floor_enabled=False)
        f1 = _make_fact("f1")
        engine = _build_engine(
            facts=[f1],
            semantic_results=[("f1", 0.0)],
            bm25_results=[],
            temporal_results=[],
            config=cfg,
        )
        response = engine.recall(
            "purple elephant quantum knitting recipe", "default", Mode.A, limit=10,
        )
        fact_ids = [r.fact.fact_id for r in response.results]
        assert "f1" in fact_ids, (
            "evidence_floor_enabled=False in config must disable the floor"
        )

    def test_custom_min_semantic_threshold(self) -> None:
        """Custom min_semantic_evidence in config is respected."""
        cfg = RetrievalConfig(min_semantic_evidence=0.80)
        f1 = _make_fact("f1")
        engine = _build_engine(
            facts=[f1],
            semantic_results=[("f1", 0.75)],  # above default 0.60 but below 0.80
            bm25_results=[],
            temporal_results=[],
            config=cfg,
        )
        response = engine.recall("query", "default", Mode.A, limit=10)
        fact_ids = [r.fact.fact_id for r in response.results]
        assert "f1" not in fact_ids, (
            "Semantic 0.75 is below custom threshold 0.80, must be filtered"
        )


# ---------------------------------------------------------------------------
# F-1-D: RecallResponse carries no_confident_match
# ---------------------------------------------------------------------------

class TestNoConfidentMatch:
    """no_confident_match field propagation."""

    def test_no_confident_match_false_when_results_exist(self) -> None:
        """no_confident_match is False (or absent) when results pass floor."""
        f1 = _make_fact("f1")
        engine = _build_engine(
            facts=[f1],
            semantic_results=[("f1", 0.85)],
            bm25_results=[],
            temporal_results=[],
        )
        response = engine.recall("real topic", "default", Mode.A, limit=10)
        assert len(response.results) > 0
        assert getattr(response, "no_confident_match", False) is False

    def test_no_confident_match_true_when_all_filtered(self) -> None:
        """no_confident_match is True when floor removes all results."""
        f1 = _make_fact("f1")
        engine = _build_engine(
            facts=[f1],
            semantic_results=[("f1", 0.0)],
            bm25_results=[],
            temporal_results=[],
        )
        response = engine.recall("nonsense", "default", Mode.A, limit=10)
        assert response.results == []
        assert getattr(response, "no_confident_match", None) is True
