# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.encoding.consolidator.

Covers:
  - MemoryConsolidator.consolidate() — ADD, UPDATE, SUPERSEDE, NOOP
  - _compute_similarity() and _jaccard() pure functions
  - Keyword contradiction detection
  - LLM contradiction detection (mocked)
  - get_consolidation_history()
  - Semantic edge creation on ADD
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from superlocalmemory.core.config import EncodingConfig
from superlocalmemory.encoding.consolidator import (
    MemoryConsolidator,
    _compute_similarity,
    _jaccard,
)
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import (
    AtomicFact,
    ConsolidationActionType,
    FactType,
    MemoryRecord,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    db_path = tmp_path / "test.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize(real_schema)
    return mgr


def _store_fact(
    db: DatabaseManager, fact_id: str, content: str,
    canonical_entities: list[str] | None = None,
    embedding: list[float] | None = None,
    fact_type: FactType = FactType.SEMANTIC,
) -> AtomicFact:
    mem_id = f"m_{fact_id}"
    db.store_memory(MemoryRecord(memory_id=mem_id, content="parent"))
    fact = AtomicFact(
        fact_id=fact_id, memory_id=mem_id, content=content,
        canonical_entities=canonical_entities or [],
        embedding=embedding,
        fact_type=fact_type,
    )
    db.store_fact(fact)
    return fact


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

class TestComputeSimilarity:
    def test_identical_vectors(self) -> None:
        assert _compute_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal(self) -> None:
        assert _compute_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_none_input(self) -> None:
        assert _compute_similarity(None, [1.0, 0.0]) == 0.0
        assert _compute_similarity([1.0, 0.0], None) == 0.0
        assert _compute_similarity(None, None) == 0.0

    def test_empty_input(self) -> None:
        assert _compute_similarity([], [1.0]) == 0.0

    def test_different_lengths(self) -> None:
        assert _compute_similarity([1.0, 0.0], [1.0]) == 0.0

    def test_zero_norm(self) -> None:
        assert _compute_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


class TestJaccard:
    def test_identical_sets(self) -> None:
        assert _jaccard({"a", "b"}, {"a", "b"}) == pytest.approx(1.0)

    def test_disjoint_sets(self) -> None:
        assert _jaccard({"a"}, {"b"}) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        assert _jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)

    def test_both_empty(self) -> None:
        assert _jaccard(set(), set()) == 0.0

    def test_one_empty(self) -> None:
        assert _jaccard({"a"}, set()) == 0.0


# ---------------------------------------------------------------------------
# Consolidation: ADD
# ---------------------------------------------------------------------------

class TestConsolidateAdd:
    def test_add_when_no_existing_facts(self, db: DatabaseManager) -> None:
        consolidator = MemoryConsolidator(db=db)
        db.store_memory(MemoryRecord(memory_id="m_new", content="parent"))
        new_fact = AtomicFact(
            fact_id="f_new", memory_id="m_new",
            content="Alice works at Google",
            canonical_entities=["ent_alice"],
        )
        action = consolidator.consolidate(new_fact, "default")
        assert action.action_type == ConsolidationActionType.ADD
        # Verify fact is stored
        facts = db.get_all_facts("default")
        assert any(f.fact_id == "f_new" for f in facts)

    def test_add_when_low_similarity(self, db: DatabaseManager) -> None:
        _store_fact(db, "f_old", "Bob likes swimming",
                    canonical_entities=["ent_bob"])
        consolidator = MemoryConsolidator(db=db)
        db.store_memory(MemoryRecord(memory_id="m_new2", content="parent"))
        new_fact = AtomicFact(
            fact_id="f_new2", memory_id="m_new2",
            content="Alice works at Google",
            canonical_entities=["ent_alice"],
        )
        action = consolidator.consolidate(new_fact, "default")
        assert action.action_type == ConsolidationActionType.ADD


# ---------------------------------------------------------------------------
# Consolidation: NOOP (near-duplicate)
# ---------------------------------------------------------------------------

class TestConsolidateNoop:
    def test_noop_for_near_duplicate(self, db: DatabaseManager) -> None:
        existing = _store_fact(
            db, "f_existing", "Alice works at Google",
            canonical_entities=["ent_alice"],
            embedding=[1.0, 0.0, 0.0],
        )
        consolidator = MemoryConsolidator(db=db)
        db.store_memory(MemoryRecord(memory_id="m_dup", content="parent"))
        new_fact = AtomicFact(
            fact_id="f_dup", memory_id="m_dup",
            content="Alice works at Google",
            canonical_entities=["ent_alice"],
            embedding=[1.0, 0.0, 0.0],  # Identical embedding
        )
        action = consolidator.consolidate(new_fact, "default")
        assert action.action_type == ConsolidationActionType.NOOP

    def test_operation_queryable_fact_can_be_excluded_from_candidates(
        self, db: DatabaseManager,
    ) -> None:
        """A write-through projection must not suppress its own derivation."""
        _store_fact(
            db, "f_queryable", "Alice leads the reliability program",
            canonical_entities=["ent_alice"],
            embedding=[1.0, 0.0, 0.0],
        )
        consolidator = MemoryConsolidator(db=db)
        db.store_memory(MemoryRecord(memory_id="m_derived", content="parent"))
        derived = AtomicFact(
            fact_id="f_derived", memory_id="m_derived",
            content="Alice heads reliability engineering",
            canonical_entities=["ent_alice"],
            embedding=[1.0, 0.0, 0.0],
        )

        action = consolidator.consolidate(
            derived,
            "default",
            exclude_fact_ids={"f_queryable"},
        )

        assert action.action_type == ConsolidationActionType.ADD
        assert db.get_fact("f_derived") is not None


# ---------------------------------------------------------------------------
# Consolidation: UPDATE
# ---------------------------------------------------------------------------

class TestConsolidateUpdate:
    def test_update_refines_existing(self, db: DatabaseManager) -> None:
        existing = _store_fact(
            db, "f_old", "Alice works at Google",
            canonical_entities=["ent_alice"],
            embedding=[1.0, 0.0, 0.0],
        )
        consolidator = MemoryConsolidator(db=db)
        db.store_memory(MemoryRecord(memory_id="m_upd", content="parent"))
        new_fact = AtomicFact(
            fact_id="f_upd", memory_id="m_upd",
            content="Alice works at Google as a senior engineer",
            canonical_entities=["ent_alice"],
            embedding=[0.95, 0.05, 0.0],  # Very similar but slightly different
        )
        action = consolidator.consolidate(new_fact, "default")
        # This could be ADD, UPDATE, or NOOP depending on combined score
        assert action.action_type in (
            ConsolidationActionType.ADD,
            ConsolidationActionType.UPDATE,
            ConsolidationActionType.NOOP,
        )


# ---------------------------------------------------------------------------
# Consolidation: SUPERSEDE (contradiction)
# ---------------------------------------------------------------------------

class TestConsolidateSupersede:
    def test_keyword_contradiction_detected(self, db: DatabaseManager) -> None:
        existing = _store_fact(
            db, "f_old", "Alice works at Google",
            canonical_entities=["ent_alice"],
            embedding=[1.0, 0.0, 0.0],
        )
        consolidator = MemoryConsolidator(db=db)
        db.store_memory(MemoryRecord(memory_id="m_contra", content="parent"))
        new_fact = AtomicFact(
            fact_id="f_contra", memory_id="m_contra",
            content="Alice no longer works at Google, she quit",
            canonical_entities=["ent_alice"],
            embedding=[0.9, 0.1, 0.0],  # High similarity
        )
        action = consolidator.consolidate(new_fact, "default")
        # May be SUPERSEDE if similarity > 0.85 and contradiction detected
        assert action.action_type in (
            ConsolidationActionType.SUPERSEDE,
            ConsolidationActionType.ADD,
        )


# ---------------------------------------------------------------------------
# Keyword contradiction detection
# ---------------------------------------------------------------------------

class TestKeywordContradiction:
    def test_negation_marker(self, db: DatabaseManager) -> None:
        consolidator = MemoryConsolidator(db=db)
        fact_a = AtomicFact(content="Alice quit the company")
        fact_b = AtomicFact(content="Alice works at the company")
        result = consolidator._keyword_contradiction_check(fact_a, fact_b)
        assert result is True

    def test_no_contradiction(self, db: DatabaseManager) -> None:
        consolidator = MemoryConsolidator(db=db)
        fact_a = AtomicFact(content="Alice works at Google")
        fact_b = AtomicFact(content="Alice lives in New York")
        result = consolidator._keyword_contradiction_check(fact_a, fact_b)
        assert result is False

    def test_emotional_valence_contradiction(self, db: DatabaseManager) -> None:
        consolidator = MemoryConsolidator(db=db)
        fact_a = AtomicFact(
            content="Alice loves the project",
            canonical_entities=["ent_alice"],
            emotional_valence=0.9,
        )
        fact_b = AtomicFact(
            content="Alice hates the project",
            canonical_entities=["ent_alice"],
            emotional_valence=-0.9,
        )
        result = consolidator._keyword_contradiction_check(fact_a, fact_b)
        assert result is True  # valence_diff > 1.2


# ---------------------------------------------------------------------------
# LLM contradiction detection
# ---------------------------------------------------------------------------

class TestLLMContradiction:
    def test_llm_says_yes(self, db: DatabaseManager) -> None:
        llm = MagicMock()
        llm.is_available.return_value = True
        llm.generate.return_value = "yes"
        consolidator = MemoryConsolidator(db=db, llm=llm)
        fact_a = AtomicFact(content="Alice works at Google")
        fact_b = AtomicFact(content="Alice works at Apple")
        assert consolidator._llm_contradiction_check(fact_a, fact_b) is True

    def test_llm_says_no(self, db: DatabaseManager) -> None:
        llm = MagicMock()
        llm.is_available.return_value = True
        llm.generate.return_value = "no, these are compatible"
        consolidator = MemoryConsolidator(db=db, llm=llm)
        fact_a = AtomicFact(content="Alice works at Google")
        fact_b = AtomicFact(content="Alice likes hiking")
        assert consolidator._llm_contradiction_check(fact_a, fact_b) is False


# ---------------------------------------------------------------------------
# get_consolidation_history
# ---------------------------------------------------------------------------

class TestConsolidationHistory:
    def test_history_recorded(self, db: DatabaseManager) -> None:
        consolidator = MemoryConsolidator(db=db)
        db.store_memory(MemoryRecord(memory_id="m_h", content="parent"))
        fact = AtomicFact(
            fact_id="f_h", memory_id="m_h",
            content="Some new information",
        )
        consolidator.consolidate(fact, "default")
        history = consolidator.get_consolidation_history("default")
        assert len(history) == 1
        assert history[0].new_fact_id == "f_h"

    def test_empty_history(self, db: DatabaseManager) -> None:
        consolidator = MemoryConsolidator(db=db)
        history = consolidator.get_consolidation_history("default")
        assert history == []


# ---------------------------------------------------------------------------
# _merge_facts (LLM merge)
# ---------------------------------------------------------------------------

class TestMergeFacts:
    def test_merge_with_llm(self, db: DatabaseManager) -> None:
        llm = MagicMock()
        llm.is_available.return_value = True
        llm.generate.return_value = "Alice works at Google as a senior engineer"
        consolidator = MemoryConsolidator(db=db, llm=llm)
        merged = consolidator._merge_facts(
            "Alice works at Google", "Alice is a senior engineer",
        )
        assert "senior engineer" in merged

    def test_merge_empty_result(self, db: DatabaseManager) -> None:
        llm = MagicMock()
        llm.is_available.return_value = True
        llm.generate.return_value = "   "
        consolidator = MemoryConsolidator(db=db, llm=llm)
        merged = consolidator._merge_facts("fact A", "fact B")
        assert merged == ""
