# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for CCQ Worker — Phase E.

TDD: 5 tests covering block creation, audit trail, consolidation
engine integration, scheduling logic, and disabled mode.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from superlocalmemory.core.config import CCQConfig
from superlocalmemory.encoding.cognitive_consolidator import (
    CCQPipelineResult,
    CognitiveConsolidator,
)
from superlocalmemory.learning.consolidation_quantization_worker import CCQWorker
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import _new_id

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path) -> DatabaseManager:
    """DatabaseManager with full schema."""
    from superlocalmemory.storage import schema

    db_path = tmp_path / "test_ccq_worker.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize(schema)
    return mgr


@pytest.fixture
def ccq_config() -> CCQConfig:
    return CCQConfig(use_llm_gist=False)


@pytest.fixture
def profile_id() -> str:
    return "test-worker-profile"


def _seed_profile(db: DatabaseManager, profile_id: str) -> None:
    db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
        (profile_id, "Worker Test"),
    )


def _seed_warm_cluster(
    db: DatabaseManager,
    profile_id: str,
    *,
    count: int = 3,
    shared_entities: list[str] | None = None,
) -> list[str]:
    """Seed warm facts for a valid cluster."""
    entities = shared_entities or ["Python", "FastAPI"]
    ents_json = json.dumps(entities)
    fact_ids = []
    for i in range(count):
        fid = _new_id()
        mid = _new_id()
        dt = datetime(2026, 1, 15, 12, 0) + timedelta(hours=i)
        content = f"Fact about {', '.join(entities)} #{i}"

        # Seed parent memory record (FK requirement)
        db.execute(
            "INSERT OR IGNORE INTO memories "
            "(memory_id, profile_id, content, session_id, speaker, role, "
            " session_date, created_at, metadata_json) "
            "VALUES (?, ?, ?, 'test', 'user', 'user', "
            " datetime('now'), datetime('now'), '{}')",
            (mid, profile_id, content),
        )

        db.execute(
            "INSERT INTO atomic_facts "
            "(fact_id, memory_id, profile_id, content, fact_type, "
            " entities_json, canonical_entities_json, observation_date, "
            " confidence, importance, evidence_count, access_count, "
            " source_turn_ids_json, session_id, lifecycle, "
            " emotional_valence, emotional_arousal, signal_type, created_at) "
            "VALUES (?, ?, ?, ?, 'semantic', ?, ?, ?, 0.8, ?, 1, 0, '[]', "
            " 'test', 'active', 0.0, 0.0, 'factual', datetime('now'))",
            (fid, mid, profile_id, content,
             ents_json, ents_json, dt.isoformat(), 0.5 + i * 0.1),
        )
        db.execute(
            "INSERT INTO fact_retention "
            "(fact_id, profile_id, retention_score, memory_strength, "
            " access_count, last_accessed_at, lifecycle_zone) "
            "VALUES (?, ?, 0.3, 1.0, 1, datetime('now'), 'warm')",
            (fid, profile_id),
        )
        fact_ids.append(fid)
    return fact_ids


# ---------------------------------------------------------------------------
# Test 13: CCQ block created
# ---------------------------------------------------------------------------


def test_ccq_block_created(db, ccq_config, profile_id):
    """Pipeline creates a row in ccq_consolidated_blocks with correct source_fact_ids."""
    _seed_profile(db, profile_id)
    fact_ids = _seed_warm_cluster(db, profile_id, count=3)

    cons = CognitiveConsolidator(db=db, config=ccq_config)
    result = cons.run_pipeline(profile_id)

    blocks = db.get_ccq_blocks(profile_id)
    assert len(blocks) >= 1

    block = blocks[0]
    stored_ids = json.loads(block["source_fact_ids"])
    assert set(stored_ids) == set(fact_ids)
    assert block["compiled_by"] == "ccq"
    assert result.blocks_created >= 1


# ---------------------------------------------------------------------------
# Test 14: Audit trail complete
# ---------------------------------------------------------------------------


def test_audit_trail_complete(db, ccq_config, profile_id):
    """Pipeline creates audit entry with all fields populated."""
    _seed_profile(db, profile_id)
    _seed_warm_cluster(db, profile_id, count=3)

    cons = CognitiveConsolidator(db=db, config=ccq_config)
    cons.run_pipeline(profile_id)

    audits = db.get_ccq_audit(profile_id)
    assert len(audits) >= 1

    audit = audits[0]
    assert audit["cluster_id"]
    assert audit["block_id"]
    assert audit["fact_count"] >= 3
    assert audit["gist_text"]
    assert audit["extraction_mode"] in ("rules", "llm")
    assert audit["bytes_before"] >= 0
    assert audit["bytes_after"] >= 0


# ---------------------------------------------------------------------------
# Test 15: CCQ integration with consolidation engine
# ---------------------------------------------------------------------------


def test_ccq_integration_with_consolidation(db, profile_id, tmp_path):
    """consolidation_engine.consolidate() includes 'ccq' in results when wired."""
    from superlocalmemory.core.config import ConsolidationConfig, SLMConfig
    from superlocalmemory.core.consolidation_engine import ConsolidationEngine
    from superlocalmemory.storage.models import Mode

    _seed_profile(db, profile_id)
    _seed_warm_cluster(db, profile_id, count=3)

    consolidation_config = ConsolidationConfig(enabled=True)
    slm_config = SLMConfig.for_mode(Mode.A, base_dir=tmp_path)

    ccq_config = CCQConfig(use_llm_gist=False, store_count_trigger=1)
    cons = CognitiveConsolidator(db=db, config=ccq_config)
    worker = CCQWorker(consolidator=cons, config=ccq_config)

    engine = ConsolidationEngine(
        db=db,
        config=consolidation_config,
        slm_config=slm_config,
        ccq_worker=worker,
    )

    results = engine.consolidate(profile_id, lightweight=False)

    assert "ccq" in results
    assert results["success"] is True


# ---------------------------------------------------------------------------
# Test 16: Worker should_run on session end
# ---------------------------------------------------------------------------


def test_worker_should_run_on_session_end(ccq_config):
    """should_run returns True on session end when configured."""
    mock_cons = MagicMock(spec=CognitiveConsolidator)
    worker = CCQWorker(consolidator=mock_cons, config=ccq_config)

    assert worker.should_run(store_count=0, is_session_end=True) is True
    assert worker.should_run(store_count=0, is_session_end=False) is False


# ---------------------------------------------------------------------------
# Test 17: Worker disabled
# ---------------------------------------------------------------------------


def test_worker_disabled():
    """Disabled config produces empty result with no DB operations."""
    config = CCQConfig(enabled=False)
    mock_cons = MagicMock(spec=CognitiveConsolidator)
    worker = CCQWorker(consolidator=mock_cons, config=config)

    result = worker.run("some-profile")

    assert isinstance(result, CCQPipelineResult)
    assert result.clusters_processed == 0
    assert result.blocks_created == 0
    mock_cons.run_pipeline.assert_not_called()

    assert worker.should_run(store_count=100, is_session_end=True) is False


# ---------------------------------------------------------------------------
# Coverage: Worker run() with actual consolidator
# ---------------------------------------------------------------------------


def test_worker_run_executes_pipeline(db, ccq_config, profile_id):
    """Worker.run() delegates to consolidator and increments run_count."""
    _seed_profile(db, profile_id)
    _seed_warm_cluster(db, profile_id, count=3)

    cons = CognitiveConsolidator(db=db, config=ccq_config)
    worker = CCQWorker(consolidator=cons, config=ccq_config)

    result = worker.run(profile_id)

    assert isinstance(result, CCQPipelineResult)
    assert result.blocks_created >= 1

    stats = worker.get_stats()
    assert stats["total_runs"] == 1
    assert stats["enabled"] is True


# ---------------------------------------------------------------------------
# Coverage: Worker should_run on store count trigger
# ---------------------------------------------------------------------------


def test_worker_should_run_on_store_count():
    """should_run returns True when store_count hits the trigger."""
    config = CCQConfig(store_count_trigger=50)
    mock_cons = MagicMock(spec=CognitiveConsolidator)
    worker = CCQWorker(consolidator=mock_cons, config=config)

    assert worker.should_run(store_count=50, is_session_end=False) is True
    assert worker.should_run(store_count=100, is_session_end=False) is True
    assert worker.should_run(store_count=25, is_session_end=False) is False
    assert worker.should_run(store_count=0, is_session_end=False) is False
