# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for Phase 5: ConsolidationEngine.

Covers: full cycle, lightweight cycle, Mode A blocks, idempotency,
        no-fact-deletion, error handling, step-count trigger, temporal check.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from superlocalmemory.core.config import ConsolidationConfig, SLMConfig
from superlocalmemory.core.consolidation_engine import ConsolidationEngine
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage import schema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_db(tmp_path: Path) -> DatabaseManager:
    """Create an in-memory-ish test database with full schema."""
    db_path = tmp_path / "test.db"
    db = DatabaseManager(db_path)
    db.initialize(schema)
    return db


@pytest.fixture()
def config() -> ConsolidationConfig:
    return ConsolidationConfig(
        enabled=True,
        step_count_trigger=5,
        block_char_limit=500,
        core_memory_char_limit=2000,
        promotion_min_access=2,
        promotion_min_trust=0.3,
    )


@pytest.fixture()
def slm_config() -> SLMConfig:
    return SLMConfig.default()


@pytest.fixture()
def engine(
    tmp_db: DatabaseManager, config: ConsolidationConfig, slm_config: SLMConfig,
) -> ConsolidationEngine:
    return ConsolidationEngine(
        db=tmp_db,
        config=config,
        summarizer=None,
        behavioral_store=None,
        auto_linker=None,
        graph_analyzer=None,
        temporal_validator=None,
        slm_config=slm_config,
    )


def _seed_facts(
    db: DatabaseManager, profile_id: str, count: int = 5,
) -> list[str]:
    """Insert sample facts and return their fact_ids."""
    from superlocalmemory.storage.models import (
        AtomicFact, FactType, MemoryRecord, SignalType,
    )
    record = MemoryRecord(
        profile_id=profile_id, content="test memory",
        session_id="s1",
    )
    db.store_memory(record)
    ids = []
    for i in range(count):
        fact = AtomicFact(
            memory_id=record.memory_id,
            profile_id=profile_id,
            content=f"Fact number {i}: test content about topic {i}",
            fact_type=FactType.SEMANTIC if i % 2 == 0 else FactType.OPINION,
            confidence=0.8,
            importance=0.5,
            signal_type=SignalType.FACTUAL,
        )
        db.store_fact(fact)
        ids.append(fact.fact_id)
    return ids


# ---------------------------------------------------------------------------
# Core Tests
# ---------------------------------------------------------------------------

class TestConsolidationEngine:
    """Tests for ConsolidationEngine core functionality."""

    def test_full_consolidation_runs_all_6_steps(
        self, engine: ConsolidationEngine, tmp_db: DatabaseManager,
    ) -> None:
        """Full consolidation returns results for all 6 steps."""
        _seed_facts(tmp_db, "default", 3)
        result = engine.consolidate("default", lightweight=False)

        assert result["success"] is True
        assert "compressed" in result
        assert "blocks" in result
        assert "promoted" in result
        assert "decayed" in result
        assert "graph_stats" in result
        assert "new_associations" in result

    def test_lightweight_consolidation_runs_2_steps(
        self, engine: ConsolidationEngine, tmp_db: DatabaseManager,
    ) -> None:
        """Lightweight consolidation only runs steps 2 + 4."""
        _seed_facts(tmp_db, "default", 3)
        result = engine.consolidate("default", lightweight=True)

        assert result["success"] is True
        assert "blocks" in result
        assert "decayed" in result
        # Full-cycle steps should NOT be present
        assert "compressed" not in result
        assert "promoted" not in result
        assert "graph_stats" not in result

    def test_consolidation_returns_results_dict(
        self, engine: ConsolidationEngine,
    ) -> None:
        """Result dict always has profile_id, lightweight, and success keys."""
        result = engine.consolidate("default")
        assert result["profile_id"] == "default"
        assert result["lightweight"] is False
        assert "success" in result

    def test_consolidation_handles_empty_profile(
        self, engine: ConsolidationEngine,
    ) -> None:
        """Profile with no facts produces no errors, all blocks compiled."""
        # 'default' profile exists (created by schema) but has no facts
        result = engine.consolidate("default")
        assert result["success"] is True
        assert result["blocks"]["blocks_compiled"] == 5  # 5 block types

    def test_consolidation_does_not_delete_facts(
        self, engine: ConsolidationEngine, tmp_db: DatabaseManager,
    ) -> None:
        """After consolidation, all original facts still exist (Rule 17)."""
        ids = _seed_facts(tmp_db, "default", 5)
        before_count = tmp_db.get_fact_count("default")

        engine.consolidate("default", lightweight=False)

        after_count = tmp_db.get_fact_count("default")
        assert after_count >= before_count
        for fid in ids:
            assert tmp_db.get_fact(fid) is not None

    def test_consolidation_errors_are_non_fatal(
        self, tmp_db: DatabaseManager, config: ConsolidationConfig,
        slm_config: SLMConfig,
    ) -> None:
        """Exception during consolidation returns success=False (Rule 19)."""
        broken_engine = ConsolidationEngine(
            db=tmp_db, config=config, slm_config=slm_config,
        )
        # Patch _step2 to raise
        with patch.object(
            broken_engine, "_step2_compile_blocks", side_effect=RuntimeError("boom"),
        ):
            result = broken_engine.consolidate("default")
            assert result["success"] is False
            assert "boom" in result["error"]


class TestConsolidationModeA:
    """Tests for Mode A rules-based block compilation."""

    def test_compile_mode_a_produces_5_blocks(
        self, engine: ConsolidationEngine, tmp_db: DatabaseManager,
    ) -> None:
        """Mode A compilation creates all 5 block types."""
        _seed_facts(tmp_db, "default", 5)
        result = engine.compile_core_blocks_mode_a("default")

        assert result["blocks_compiled"] == 5
        assert result["mode"] == "rules"

        blocks = tmp_db.get_core_blocks("default")
        types = {b["block_type"] for b in blocks}
        assert types == {
            "user_profile", "project_context", "behavioral_patterns",
            "active_decisions", "learned_preferences",
        }

    def test_user_profile_uses_semantic_opinion(
        self, engine: ConsolidationEngine, tmp_db: DatabaseManager,
    ) -> None:
        """user_profile block sourced from semantic/opinion facts."""
        _seed_facts(tmp_db, "default", 5)
        engine.compile_core_blocks_mode_a("default")
        block = tmp_db.get_core_block("default", "user_profile")
        assert block is not None
        assert block["compiled_by"] == "rules"
        # Should contain fact content (not default placeholder) when facts exist
        assert len(block["content"]) > 0

    def test_block_char_limit_enforced(
        self, tmp_db: DatabaseManager, slm_config: SLMConfig,
    ) -> None:
        """Blocks are truncated at block_char_limit."""
        small_config = ConsolidationConfig(
            enabled=True, block_char_limit=50,
        )
        eng = ConsolidationEngine(
            db=tmp_db, config=small_config, slm_config=slm_config,
        )
        _seed_facts(tmp_db, "default", 10)
        eng.compile_core_blocks_mode_a("default")

        for block in tmp_db.get_core_blocks("default"):
            assert len(block["content"]) <= 50

    def test_fallback_content_when_empty(
        self, engine: ConsolidationEngine, tmp_db: DatabaseManager,
    ) -> None:
        """Blocks with no matching facts show fallback text."""
        # No facts seeded — all blocks should have fallback
        engine.compile_core_blocks_mode_a("default")
        block = tmp_db.get_core_block("default", "active_decisions")
        assert block is not None
        assert block["content"] == "No data available."

    def test_no_llm_called_in_mode_a(
        self, engine: ConsolidationEngine, tmp_db: DatabaseManager,
    ) -> None:
        """Mode A compilation does NOT call any LLM."""
        _seed_facts(tmp_db, "default", 3)
        # If summarizer were called, it would fail (it's None)
        result = engine.compile_core_blocks_mode_a("default")
        assert result["mode"] == "rules"

    def test_active_decisions_uses_signal_type(
        self, engine: ConsolidationEngine, tmp_db: DatabaseManager,
    ) -> None:
        """active_decisions block queries signal_type='decision', not fact_type."""
        from superlocalmemory.storage.models import (
            AtomicFact, FactType, MemoryRecord, SignalType, _new_id,
        )
        # Create a fact and set signal_type='decision' via SQL
        # (signal_type column is free-text; 'decision' is not in SignalType enum)
        record = MemoryRecord(
            profile_id="default", content="decision memory", session_id="s1",
        )
        tmp_db.store_memory(record)
        fact = AtomicFact(
            memory_id=record.memory_id, profile_id="default",
            content="We decided to use Python for the backend",
            fact_type=FactType.SEMANTIC,
            signal_type=SignalType.FACTUAL,
            confidence=0.9, importance=0.8,
        )
        tmp_db.store_fact(fact)
        # Set signal_type to 'decision' directly
        tmp_db.execute(
            "UPDATE atomic_facts SET signal_type = 'decision' WHERE fact_id = ?",
            (fact.fact_id,),
        )

        # Add access log entries to meet min_access threshold (config=2)
        for _ in range(3):
            tmp_db.execute(
                "INSERT INTO fact_access_log (log_id, fact_id, profile_id, access_type) "
                "VALUES (?, ?, ?, 'recall')",
                (_new_id(), fact.fact_id, "default"),
            )

        engine.compile_core_blocks_mode_a("default")
        block = tmp_db.get_core_block("default", "active_decisions")
        assert block is not None
        assert "Python" in block["content"]


class TestConsolidationPromote:
    """Tests for Step 3: Auto-promotion with temporal check."""

    def test_promote_filters_expired_facts(
        self, engine: ConsolidationEngine, tmp_db: DatabaseManager,
    ) -> None:
        """Facts with valid_until in the past are NOT promoted (L12)."""
        ids = _seed_facts(tmp_db, "default", 1)
        fid = ids[0]
        # Add access entries to meet threshold
        from superlocalmemory.storage.models import _new_id
        for _ in range(5):
            tmp_db.execute(
                "INSERT INTO fact_access_log (log_id, fact_id, profile_id) "
                "VALUES (?, ?, ?)",
                (_new_id(), fid, "default"),
            )
        # Mark as expired
        tmp_db.store_temporal_validity(fid, "default", valid_until="2020-01-01T00:00:00")

        result = engine._step3_promote("default")
        # The expired fact should NOT have been promoted
        fact = tmp_db.get_fact(fid)
        assert fact.lifecycle.value == "active"  # Not promoted to 'warm'

    def test_promote_accepts_valid_facts(
        self, engine: ConsolidationEngine, tmp_db: DatabaseManager,
    ) -> None:
        """Facts with valid_until=NULL are promoted normally."""
        ids = _seed_facts(tmp_db, "default", 1)
        fid = ids[0]
        from superlocalmemory.storage.models import _new_id
        for _ in range(5):
            tmp_db.execute(
                "INSERT INTO fact_access_log (log_id, fact_id, profile_id) "
                "VALUES (?, ?, ?)",
                (_new_id(), fid, "default"),
            )
        # No temporal validity record = valid
        result = engine._step3_promote("default")
        fact = tmp_db.get_fact(fid)
        assert fact.lifecycle.value == "warm"
        retention = tmp_db.execute(
            "SELECT lifecycle_zone FROM fact_retention WHERE fact_id = ?",
            (fid,),
        )
        assert retention and dict(retention[0])["lifecycle_zone"] == "warm"

    def test_promote_requires_min_access(
        self, engine: ConsolidationEngine, tmp_db: DatabaseManager,
    ) -> None:
        """Facts below access threshold are NOT promoted."""
        ids = _seed_facts(tmp_db, "default", 1)
        fid = ids[0]
        # Only 1 access (threshold=2 in fixture config)
        from superlocalmemory.storage.models import _new_id
        tmp_db.execute(
            "INSERT INTO fact_access_log (log_id, fact_id, profile_id) "
            "VALUES (?, ?, ?)",
            (_new_id(), fid, "default"),
        )
        result = engine._step3_promote("default")
        fact = tmp_db.get_fact(fid)
        assert fact.lifecycle.value == "active"

    def test_promote_requires_min_trust(
        self, tmp_db: DatabaseManager, slm_config: SLMConfig,
    ) -> None:
        """Facts below trust threshold are NOT promoted."""
        strict_config = ConsolidationConfig(
            enabled=True, promotion_min_access=1, promotion_min_trust=0.99,
        )
        eng = ConsolidationEngine(
            db=tmp_db, config=strict_config, slm_config=slm_config,
        )
        ids = _seed_facts(tmp_db, "default", 1)
        fid = ids[0]
        from superlocalmemory.storage.models import _new_id
        for _ in range(5):
            tmp_db.execute(
                "INSERT INTO fact_access_log (log_id, fact_id, profile_id) "
                "VALUES (?, ?, ?)",
                (_new_id(), fid, "default"),
            )
        result = eng._step3_promote("default")
        fact = tmp_db.get_fact(fid)
        assert fact.lifecycle.value == "active"  # confidence 0.8 < 0.99


class TestConsolidationDelegation:
    """Tests for step delegation to AutoLinker and GraphAnalyzer."""

    def test_step4_decay_delegates_to_autolinker(
        self, tmp_db: DatabaseManager, slm_config: SLMConfig,
    ) -> None:
        """Step 4 delegates to AutoLinker.decay_unused()."""
        mock_linker = MagicMock()
        mock_linker.decay_unused.return_value = 7
        config = ConsolidationConfig(enabled=True)
        eng = ConsolidationEngine(
            db=tmp_db, config=config, auto_linker=mock_linker,
            slm_config=slm_config,
        )
        result = eng._step4_decay_edges("default")
        mock_linker.decay_unused.assert_called_once_with(
            "default", days_threshold=config.decay_days_threshold,
        )
        assert result["decayed"] == 7

    def test_step5_recompute_delegates_to_analyzer(
        self, tmp_db: DatabaseManager, slm_config: SLMConfig,
    ) -> None:
        """Step 5 delegates to GraphAnalyzer.compute_and_store()."""
        mock_analyzer = MagicMock()
        mock_analyzer.compute_and_store.return_value = {
            "node_count": 10, "community_count": 2,
        }
        config = ConsolidationConfig(enabled=True)
        eng = ConsolidationEngine(
            db=tmp_db, config=config, graph_analyzer=mock_analyzer,
            slm_config=slm_config,
        )
        result = eng._step5_recompute_graph("default")
        mock_analyzer.compute_and_store.assert_called_once_with("default")
        assert result["node_count"] == 10

    def test_step4_no_autolinker_returns_zero(
        self, engine: ConsolidationEngine,
    ) -> None:
        """Step 4 without AutoLinker returns zero decayed."""
        result = engine._step4_decay_edges("default")
        assert result["decayed"] == 0

    def test_step5_no_analyzer_returns_zero(
        self, engine: ConsolidationEngine,
    ) -> None:
        """Step 5 without GraphAnalyzer returns zero."""
        result = engine._step5_recompute_graph("default")
        assert result["node_count"] == 0


class TestStepCountTrigger:
    """Tests for step-count trigger (L7)."""

    def test_step_count_trigger_at_threshold(
        self, engine: ConsolidationEngine, tmp_db: DatabaseManager,
    ) -> None:
        """After N stores, lightweight consolidation fires."""
        _seed_facts(tmp_db, "default", 3)
        # Config has step_count_trigger=5
        for i in range(4):
            assert engine.increment_store_count("default") is False
        # 5th call triggers
        assert engine.increment_store_count("default") is True

    def test_step_count_resets_after_trigger(
        self, engine: ConsolidationEngine, tmp_db: DatabaseManager,
    ) -> None:
        """Counter resets to 0 after trigger fires."""
        _seed_facts(tmp_db, "default", 3)
        # Trigger once
        for _ in range(5):
            engine.increment_store_count("default")
        # Counter should be reset — need 5 more
        for i in range(4):
            assert engine.increment_store_count("default") is False
        assert engine.increment_store_count("default") is True

    def test_step_count_disabled_skips(
        self, tmp_db: DatabaseManager, slm_config: SLMConfig,
    ) -> None:
        """When disabled, increment_store_count always returns False."""
        disabled_config = ConsolidationConfig(enabled=False, step_count_trigger=1)
        eng = ConsolidationEngine(
            db=tmp_db, config=disabled_config, slm_config=slm_config,
        )
        assert eng.increment_store_count("default") is False


class TestConsolidationIdempotent:
    """Tests for idempotency guarantee (L18)."""

    def test_double_consolidation_identical_state(
        self, engine: ConsolidationEngine, tmp_db: DatabaseManager,
    ) -> None:
        """Running consolidate() twice produces identical Core Memory state."""
        _seed_facts(tmp_db, "default", 5)

        engine.consolidate("default", lightweight=False)
        blocks_first = tmp_db.get_core_blocks("default")
        contents_first = {
            b["block_type"]: b["content"] for b in blocks_first
        }

        engine.consolidate("default", lightweight=False)
        blocks_second = tmp_db.get_core_blocks("default")
        contents_second = {
            b["block_type"]: b["content"] for b in blocks_second
        }

        # Content should be identical
        assert contents_first == contents_second
        # Should still have exactly 5 blocks
        assert len(blocks_second) == 5

    def test_double_block_compilation_identical(
        self, engine: ConsolidationEngine, tmp_db: DatabaseManager,
    ) -> None:
        """compile_core_blocks_mode_a() twice gives same blocks."""
        _seed_facts(tmp_db, "default", 3)

        engine.compile_core_blocks_mode_a("default")
        first = {
            b["block_type"]: b["content"]
            for b in tmp_db.get_core_blocks("default")
        }

        engine.compile_core_blocks_mode_a("default")
        second = {
            b["block_type"]: b["content"]
            for b in tmp_db.get_core_blocks("default")
        }

        assert first == second

    def test_version_incremented_on_recompile(
        self, engine: ConsolidationEngine, tmp_db: DatabaseManager,
    ) -> None:
        """Block version increments on each recompilation."""
        _seed_facts(tmp_db, "default", 3)

        engine.compile_core_blocks_mode_a("default")
        v1 = tmp_db.get_core_block("default", "user_profile")["version"]

        engine.compile_core_blocks_mode_a("default")
        v2 = tmp_db.get_core_block("default", "user_profile")["version"]

        assert v2 == v1 + 1
