# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.storage.models — data models and enums.

Covers:
  - All enum values and string representations
  - Dataclass creation with default values
  - _new_id() uniqueness (UUID-based)
  - Frozen vs mutable dataclass behavior
  - Field defaults and factory functions
"""

from __future__ import annotations

import pytest

from superlocalmemory.storage.models import (
    ActionOutcome,
    AtomicFact,
    BehavioralPattern,
    CanonicalEntity,
    ComplianceAuditEntry,
    ConsolidationAction,
    ConsolidationActionType,
    EdgeType,
    EntityAlias,
    EntityProfile,
    FactType,
    FeedbackRecord,
    GraphEdge,
    MemoryLifecycle,
    MemoryRecord,
    MemoryScene,
    Mode,
    Profile,
    ProvenanceRecord,
    RecallResponse,
    RetrievalResult,
    SignalType,
    TemporalEvent,
    TrustScore,
    _new_id,
)

# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestFactType:
    def test_values_complete(self) -> None:
        expected = {"episodic", "semantic", "opinion", "prospective"}
        assert {ft.value for ft in FactType} == expected

    def test_is_str_enum(self) -> None:
        assert isinstance(FactType.SEMANTIC, str)
        assert FactType.SEMANTIC == "semantic"


class TestEdgeType:
    def test_values_complete(self) -> None:
        expected = {
            "entity", "temporal", "semantic",
            "causal", "contradiction", "supersedes",
        }
        assert {et.value for et in EdgeType} == expected


class TestConsolidationActionType:
    def test_values_complete(self) -> None:
        expected = {"add", "update", "supersede", "noop"}
        assert {ca.value for ca in ConsolidationActionType} == expected


class TestMemoryLifecycle:
    def test_values_complete(self) -> None:
        expected = {"active", "warm", "cold", "archived"}
        assert {ml.value for ml in MemoryLifecycle} == expected


class TestSignalType:
    def test_values_complete(self) -> None:
        expected = {
            "factual", "emotional", "temporal",
            "opinion", "request", "social",
        }
        assert {st.value for st in SignalType} == expected


class TestMode:
    def test_values_complete(self) -> None:
        expected = {"a", "b", "c"}
        assert {m.value for m in Mode} == expected

    def test_mode_a_is_default_guardian(self) -> None:
        assert Mode.A == "a"

    def test_mode_c_is_full_power(self) -> None:
        assert Mode.C == "c"


# ---------------------------------------------------------------------------
# _new_id() uniqueness
# ---------------------------------------------------------------------------

class TestNewId:
    def test_returns_hex_string_of_length_16(self) -> None:
        result = _new_id()
        assert isinstance(result, str)
        assert len(result) == 16
        # Hex characters only
        int(result, 16)

    def test_uniqueness_across_1000_calls(self) -> None:
        ids = {_new_id() for _ in range(1_000)}
        assert len(ids) == 1_000, "Collision detected in 1000 IDs"


# ---------------------------------------------------------------------------
# Profile (frozen dataclass)
# ---------------------------------------------------------------------------

class TestProfile:
    def test_create_with_defaults(self) -> None:
        p = Profile(profile_id="abc", name="test")
        assert p.profile_id == "abc"
        assert p.name == "test"
        assert p.description == ""
        assert p.personality == ""
        assert p.mode == Mode.A
        assert p.last_used is None
        assert p.config == {}

    def test_frozen_immutability(self) -> None:
        p = Profile(profile_id="abc", name="test")
        with pytest.raises(AttributeError):
            p.name = "changed"  # type: ignore[misc]

    def test_created_at_auto_populated(self) -> None:
        p = Profile(profile_id="x", name="y")
        assert p.created_at  # Non-empty ISO string


# ---------------------------------------------------------------------------
# MemoryRecord (mutable dataclass)
# ---------------------------------------------------------------------------

class TestMemoryRecord:
    def test_create_with_defaults(self) -> None:
        m = MemoryRecord()
        assert len(m.memory_id) == 16
        assert m.profile_id == "default"
        assert m.content == ""
        assert m.metadata == {}

    def test_auto_id_unique(self) -> None:
        ids = {MemoryRecord().memory_id for _ in range(100)}
        assert len(ids) == 100

    def test_mutable_content(self) -> None:
        m = MemoryRecord()
        m.content = "updated"
        assert m.content == "updated"


# ---------------------------------------------------------------------------
# AtomicFact
# ---------------------------------------------------------------------------

class TestAtomicFact:
    def test_create_with_defaults(self) -> None:
        f = AtomicFact()
        assert len(f.fact_id) == 16
        assert f.profile_id == "default"
        assert f.fact_type == FactType.SEMANTIC
        assert f.entities == []
        assert f.canonical_entities == []
        assert f.confidence == 1.0
        assert f.importance == 0.5
        assert f.evidence_count == 1
        assert f.access_count == 0
        assert f.embedding is None
        assert f.fisher_mean is None
        assert f.fisher_variance is None
        assert f.lifecycle == MemoryLifecycle.ACTIVE
        assert f.langevin_position is None
        assert f.emotional_valence == 0.0
        assert f.emotional_arousal == 0.0
        assert f.signal_type == SignalType.FACTUAL

    def test_create_with_custom_fields(self) -> None:
        f = AtomicFact(
            fact_id="custom",
            content="Alice works at Acme",
            fact_type=FactType.EPISODIC,
            entities=["Alice", "Acme"],
            confidence=0.9,
            importance=0.8,
        )
        assert f.fact_id == "custom"
        assert f.fact_type == FactType.EPISODIC
        assert "Alice" in f.entities
        assert f.confidence == 0.9

    def test_source_turn_ids_independent(self) -> None:
        """Factory-created lists must NOT share state between instances."""
        f1 = AtomicFact()
        f2 = AtomicFact()
        f1.source_turn_ids.append("turn_1")
        assert f2.source_turn_ids == [], "List factory should create independent lists"


# ---------------------------------------------------------------------------
# CanonicalEntity
# ---------------------------------------------------------------------------

class TestCanonicalEntity:
    def test_create_with_defaults(self) -> None:
        e = CanonicalEntity()
        assert len(e.entity_id) == 16
        assert e.profile_id == "default"
        assert e.canonical_name == ""
        assert e.fact_count == 0


# ---------------------------------------------------------------------------
# EntityAlias
# ---------------------------------------------------------------------------

class TestEntityAlias:
    def test_create_with_defaults(self) -> None:
        a = EntityAlias()
        assert len(a.alias_id) == 16
        assert a.entity_id == ""
        assert a.alias == ""
        assert a.confidence == 1.0


# ---------------------------------------------------------------------------
# EntityProfile
# ---------------------------------------------------------------------------

class TestEntityProfile:
    def test_create_with_defaults(self) -> None:
        ep = EntityProfile()
        assert ep.knowledge_summary == ""
        assert ep.fact_ids == []


# ---------------------------------------------------------------------------
# MemoryScene
# ---------------------------------------------------------------------------

class TestMemoryScene:
    def test_create_with_defaults(self) -> None:
        s = MemoryScene()
        assert s.theme == ""
        assert s.fact_ids == []
        assert s.entity_ids == []

    def test_list_independence(self) -> None:
        s1 = MemoryScene()
        s2 = MemoryScene()
        s1.fact_ids.append("f1")
        assert s2.fact_ids == []


# ---------------------------------------------------------------------------
# TemporalEvent
# ---------------------------------------------------------------------------

class TestTemporalEvent:
    def test_create_with_defaults(self) -> None:
        t = TemporalEvent()
        assert t.observation_date is None
        assert t.referenced_date is None
        assert t.interval_start is None
        assert t.interval_end is None
        assert t.description == ""


# ---------------------------------------------------------------------------
# GraphEdge
# ---------------------------------------------------------------------------

class TestGraphEdge:
    def test_create_with_defaults(self) -> None:
        g = GraphEdge()
        assert g.edge_type == EdgeType.ENTITY
        assert g.weight == 1.0
        assert g.source_id == ""
        assert g.target_id == ""


# ---------------------------------------------------------------------------
# ConsolidationAction
# ---------------------------------------------------------------------------

class TestConsolidationAction:
    def test_create_with_defaults(self) -> None:
        ca = ConsolidationAction()
        assert ca.action_type == ConsolidationActionType.ADD
        assert ca.reason == ""


# ---------------------------------------------------------------------------
# TrustScore
# ---------------------------------------------------------------------------

class TestTrustScore:
    def test_create_with_defaults(self) -> None:
        ts = TrustScore()
        assert ts.trust_score == 0.5
        assert ts.evidence_count == 0


# ---------------------------------------------------------------------------
# ProvenanceRecord
# ---------------------------------------------------------------------------

class TestProvenanceRecord:
    def test_create_with_defaults(self) -> None:
        pr = ProvenanceRecord()
        assert pr.source_type == ""
        assert pr.created_by == ""


# ---------------------------------------------------------------------------
# FeedbackRecord
# ---------------------------------------------------------------------------

class TestFeedbackRecord:
    def test_create_with_defaults(self) -> None:
        fr = FeedbackRecord()
        assert fr.dwell_time_ms == 0
        assert fr.feedback_type == ""


# ---------------------------------------------------------------------------
# BehavioralPattern
# ---------------------------------------------------------------------------

class TestBehavioralPattern:
    def test_create_with_defaults(self) -> None:
        bp = BehavioralPattern()
        assert bp.observation_count == 0
        assert bp.confidence == 0.0


# ---------------------------------------------------------------------------
# ActionOutcome
# ---------------------------------------------------------------------------

class TestActionOutcome:
    def test_create_with_defaults(self) -> None:
        ao = ActionOutcome()
        assert ao.outcome == ""
        assert ao.fact_ids == []
        assert ao.context == {}


# ---------------------------------------------------------------------------
# ComplianceAuditEntry
# ---------------------------------------------------------------------------

class TestComplianceAuditEntry:
    def test_create_with_defaults(self) -> None:
        c = ComplianceAuditEntry()
        assert c.action == ""
        assert c.target_type == ""


# ---------------------------------------------------------------------------
# RetrievalResult / RecallResponse (runtime-only, no persistence)
# ---------------------------------------------------------------------------

class TestRetrievalResult:
    def test_create_with_fact(self) -> None:
        fact = AtomicFact(content="test fact")
        rr = RetrievalResult(fact=fact, score=0.95)
        assert rr.score == 0.95
        assert rr.fact.content == "test fact"
        assert rr.channel_scores == {}
        assert rr.trust_score == 0.5


class TestRecallResponse:
    def test_create_with_defaults(self) -> None:
        resp = RecallResponse(query="who is Alice?")
        assert resp.mode == Mode.A
        assert resp.results == []
        assert resp.total_candidates == 0
        assert resp.retrieval_time_ms == 0.0
