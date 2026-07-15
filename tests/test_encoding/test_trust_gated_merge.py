# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for the trust-gated merge / quarantine / corroboration pipeline
added to align with reference_diag/full_framework_detailed.puml:

  - Noisy-OR confidence merge (encoding/consolidator.py)
  - Quarantine on untrusted merge / intent-flagged content
  - Independent corroboration on first mention (multi-source deployment)
  - Single-source ungated bootstrap
  - MergePolicy action selection (encoding/policy.py)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from superlocalmemory.core.config import EncodingConfig
from superlocalmemory.encoding.consolidator import MemoryConsolidator
from superlocalmemory.encoding.policy import MergeAction, MergePolicy
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import (
    AtomicFact,
    ConsolidationActionType,
    MemoryRecord,
)


class _StubTrustScorer:
    """Deterministic tau(s) for tests — no Beta-distribution machinery."""

    def __init__(self, trust: float) -> None:
        self._trust = trust

    def get_agent_trust(self, agent_id: str, profile_id: str) -> float:
        return self._trust


@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    db_path = tmp_path / "test.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize(real_schema)
    return mgr


def _store_fact(
    db: DatabaseManager, fact_id: str, content: str, **kwargs,
) -> AtomicFact:
    mem_id = f"m_{fact_id}"
    db.store_memory(MemoryRecord(memory_id=mem_id, content="parent"))
    fact = AtomicFact(fact_id=fact_id, memory_id=mem_id, content=content, **kwargs)
    db.store_fact(fact)
    return fact


# A pair of embeddings + shared entity that reliably lands the combined
# candidate score in (_MATCH_THRESHOLD, _NOOP_THRESHOLD] = (0.85, 0.95]:
# combined = 0.4 * jaccard(entities) + 0.6 * cosine(embeddings)
#          = 0.4 * 1.0            + 0.6 * 0.8169   ~= 0.890
_EXISTING_EMBEDDING = [1.0, 0.0, 0.0]
_MATCHING_NEW_EMBEDDING = [0.85, 0.6, 0.0]


# ---------------------------------------------------------------------------
# Noisy-OR merge
# ---------------------------------------------------------------------------

class TestNoisyOrMerge:
    def test_update_commits_noisy_or_formula(self, db: DatabaseManager) -> None:
        existing = _store_fact(
            db, "f_old", "Alice works at Google",
            canonical_entities=["ent_alice"], embedding=_EXISTING_EMBEDDING,
        )
        db.update_fact("f_old", {"confidence": 0.6})
        consolidator = MemoryConsolidator(db=db, trust_scorer=_StubTrustScorer(0.9))
        new_fact = AtomicFact(
            fact_id="f_upd", memory_id="m_upd",
            content="Alice works at Google in the cloud team",
            canonical_entities=["ent_alice"], embedding=_MATCHING_NEW_EMBEDDING,
            confidence=0.8,
        )
        db.store_memory(MemoryRecord(memory_id="m_upd", content="parent"))

        action = consolidator.consolidate(new_fact, "default", agent_id="agent_a")

        assert action.action_type == ConsolidationActionType.UPDATE
        updated = db.get_fact("f_old")
        expected_confidence = min(1 - (1 - 0.6) * (1 - 0.8), 0.99)
        assert updated.confidence == pytest.approx(expected_confidence)
        assert "agent_a" in updated.corroboration_agents
        assert updated.pending_corroboration is False

    def test_update_quarantines_when_untrusted(self, db: DatabaseManager) -> None:
        _store_fact(
            db, "f_old", "Alice works at Google",
            canonical_entities=["ent_alice"], embedding=_EXISTING_EMBEDDING,
        )
        db.update_fact("f_old", {"confidence": 0.6})
        consolidator = MemoryConsolidator(db=db, trust_scorer=_StubTrustScorer(0.05))
        new_fact = AtomicFact(
            fact_id="f_upd2", memory_id="m_upd2",
            content="Alice works at Google in the cloud team",
            canonical_entities=["ent_alice"], embedding=_MATCHING_NEW_EMBEDDING,
            confidence=0.8,
        )
        db.store_memory(MemoryRecord(memory_id="m_upd2", content="parent"))

        action = consolidator.consolidate(new_fact, "default", agent_id="agent_untrusted")

        # Quarantine is logged as NOOP (no CHECK-constraint widening — see plan).
        assert action.action_type == ConsolidationActionType.NOOP
        assert "QUARANTINE" in action.reason
        existing = db.get_fact("f_old")
        assert existing.confidence == pytest.approx(0.6)  # untouched
        assert existing.pending_corroboration is True

    def test_intent_flagged_content_always_quarantines_merge(
        self, db: DatabaseManager,
    ) -> None:
        _store_fact(
            db, "f_old", "Alice works at Google",
            canonical_entities=["ent_alice"], embedding=_EXISTING_EMBEDDING,
        )
        db.update_fact("f_old", {"confidence": 0.6})
        # High trust — would normally commit — but intent_flagged forces quarantine.
        consolidator = MemoryConsolidator(db=db, trust_scorer=_StubTrustScorer(0.99))
        new_fact = AtomicFact(
            fact_id="f_upd3", memory_id="m_upd3",
            content="Alice works at Google in the cloud team",
            canonical_entities=["ent_alice"], embedding=_MATCHING_NEW_EMBEDDING,
            confidence=0.9, intent_flagged=True,
        )
        db.store_memory(MemoryRecord(memory_id="m_upd3", content="parent"))

        action = consolidator.consolidate(new_fact, "default", agent_id="agent_a")

        assert action.action_type == ConsolidationActionType.NOOP
        assert "intent_flagged" in action.reason
        assert db.get_fact("f_old").confidence == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Add path: single-source bootstrap vs multi-source corroboration
# ---------------------------------------------------------------------------

class TestAddBootstrapAndCorroboration:
    def test_single_source_commits_immediately(self, db: DatabaseManager) -> None:
        cfg = EncodingConfig(deployment_mode="single_source")
        consolidator = MemoryConsolidator(db=db, config=cfg)
        new_fact = AtomicFact(
            fact_id="f_new", memory_id="m_new", content="Bob likes chess",
        )
        db.store_memory(MemoryRecord(memory_id="m_new", content="parent"))

        action = consolidator.consolidate(new_fact, "default", agent_id="agent_a")

        assert action.action_type == ConsolidationActionType.ADD
        assert "ungated-bootstrap" in action.reason
        stored = db.get_fact("f_new")
        assert stored.pending_corroboration is False
        assert 0.7 <= stored.confidence <= 0.9

    def test_multi_source_quarantines_first_mention(self, db: DatabaseManager) -> None:
        cfg = EncodingConfig(deployment_mode="multi_source")
        consolidator = MemoryConsolidator(db=db, config=cfg)
        new_fact = AtomicFact(
            fact_id="f_new2", memory_id="m_new2", content="Bob likes chess",
        )
        db.store_memory(MemoryRecord(memory_id="m_new2", content="parent"))

        action = consolidator.consolidate(new_fact, "default", agent_id="agent_a")

        assert action.action_type == ConsolidationActionType.NOOP
        stored = db.get_fact("f_new2")
        assert stored.pending_corroboration is True
        assert stored.corroboration_agents == ["agent_a"]

    def test_independent_corroboration_commits_both(self, db: DatabaseManager) -> None:
        cfg = EncodingConfig(deployment_mode="multi_source")
        consolidator = MemoryConsolidator(db=db, config=cfg)
        db.store_memory(MemoryRecord(memory_id="m_p1", content="parent"))
        first = AtomicFact(
            fact_id="f_p1", memory_id="m_p1",
            content="Bob likes chess", canonical_entities=["ent_bob"],
            embedding=_EXISTING_EMBEDDING,
        )
        action1 = consolidator.consolidate(first, "default", agent_id="agent_a")
        assert action1.action_type == ConsolidationActionType.NOOP  # quarantined

        db.store_memory(MemoryRecord(memory_id="m_p2", content="parent"))
        second = AtomicFact(
            fact_id="f_p2", memory_id="m_p2",
            content="Bob likes playing chess", canonical_entities=["ent_bob"],
            embedding=_MATCHING_NEW_EMBEDDING, confidence=0.8,
        )
        action2 = consolidator.consolidate(second, "default", agent_id="agent_b")

        assert action2.action_type == ConsolidationActionType.UPDATE
        assert "corroboration" in action2.reason
        promoted = db.get_fact("f_p1")
        assert promoted.pending_corroboration is False
        assert set(promoted.corroboration_agents) == {"agent_a", "agent_b"}
        assert 0.7 <= promoted.confidence <= 0.9

    def test_same_source_repeat_stays_pending(self, db: DatabaseManager) -> None:
        cfg = EncodingConfig(deployment_mode="multi_source")
        consolidator = MemoryConsolidator(db=db, config=cfg)
        db.store_memory(MemoryRecord(memory_id="m_q1", content="parent"))
        first = AtomicFact(
            fact_id="f_q1", memory_id="m_q1",
            content="Bob likes chess", canonical_entities=["ent_bob"],
            embedding=_EXISTING_EMBEDDING,
        )
        consolidator.consolidate(first, "default", agent_id="agent_a")

        db.store_memory(MemoryRecord(memory_id="m_q2", content="parent"))
        second = AtomicFact(
            fact_id="f_q2", memory_id="m_q2",
            content="Bob likes playing chess", canonical_entities=["ent_bob"],
            embedding=_MATCHING_NEW_EMBEDDING, confidence=0.8,
        )
        action2 = consolidator.consolidate(second, "default", agent_id="agent_a")

        assert action2.action_type == ConsolidationActionType.NOOP
        assert "still awaiting" in action2.reason
        still = db.get_fact("f_q1")
        assert still.pending_corroboration is True

    def test_intent_flagged_add_always_quarantines(self, db: DatabaseManager) -> None:
        cfg = EncodingConfig(deployment_mode="single_source")
        consolidator = MemoryConsolidator(db=db, config=cfg)
        new_fact = AtomicFact(
            fact_id="f_iflag", memory_id="m_iflag",
            content="What is Bob's favorite game?", intent_flagged=True,
        )
        db.store_memory(MemoryRecord(memory_id="m_iflag", content="parent"))

        action = consolidator.consolidate(new_fact, "default", agent_id="agent_a")

        assert action.action_type == ConsolidationActionType.NOOP
        stored = db.get_fact("f_iflag")
        assert stored.pending_corroboration is True


# ---------------------------------------------------------------------------
# MergePolicy — direct unit tests
# ---------------------------------------------------------------------------

class TestMergePolicy:
    def test_no_candidates_single_source_returns_add(self) -> None:
        policy = MergePolicy(lambda a, b: False, EncodingConfig(deployment_mode="single_source"))
        decision = policy.select(AtomicFact(content="x"), [], tau=0.5)
        assert decision.action == MergeAction.ADD

    def test_no_candidates_multi_source_returns_add_quarantine(self) -> None:
        policy = MergePolicy(lambda a, b: False, EncodingConfig(deployment_mode="multi_source"))
        decision = policy.select(AtomicFact(content="x"), [], tau=0.5)
        assert decision.action == MergeAction.ADD_QUARANTINE

    def test_high_score_contradiction_returns_supersede(self) -> None:
        policy = MergePolicy(lambda a, b: True, EncodingConfig())
        existing = AtomicFact(fact_id="e", content="old")
        decision = policy.select(AtomicFact(content="new"), [(existing, 0.99)], tau=0.5)
        assert decision.action == MergeAction.SUPERSEDE

    def test_high_score_no_contradiction_returns_noop(self) -> None:
        policy = MergePolicy(lambda a, b: False, EncodingConfig())
        existing = AtomicFact(fact_id="e", content="old")
        decision = policy.select(AtomicFact(content="new"), [(existing, 0.99)], tau=0.5)
        assert decision.action == MergeAction.NOOP
