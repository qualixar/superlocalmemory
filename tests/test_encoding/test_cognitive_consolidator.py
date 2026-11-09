# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for Cognitive Consolidation Quantization — Phase E.

TDD: 12 tests covering identification, clustering, gist extraction,
embedding compression, storage, and audit trail.
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
    ConsolidationCluster,
    GistResult,
)
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import _new_id

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path) -> DatabaseManager:
    """DatabaseManager with full schema (on-disk for WAL support)."""
    from superlocalmemory.storage import schema

    db_path = tmp_path / "test_ccq.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize(schema)
    return mgr


@pytest.fixture
def ccq_config() -> CCQConfig:
    """Default CCQ config with LLM gist disabled for deterministic tests."""
    return CCQConfig(use_llm_gist=False)


@pytest.fixture
def consolidator(db, ccq_config) -> CognitiveConsolidator:
    """CognitiveConsolidator with no embedder/LLM (rules-only)."""
    return CognitiveConsolidator(db=db, config=ccq_config)


@pytest.fixture
def profile_id() -> str:
    return "test-profile-ccq"


def _seed_profile(db: DatabaseManager, profile_id: str) -> None:
    """Insert a profile row."""
    db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
        (profile_id, "CCQ Test Profile"),
    )


def _seed_fact(
    db: DatabaseManager,
    profile_id: str,
    *,
    fact_id: str | None = None,
    content: str = "Test fact content",
    entities: list[str] | None = None,
    importance: float = 0.5,
    confidence: float = 0.8,
    observation_date: str | None = None,
    lifecycle: str = "active",
) -> str:
    """Insert an atomic_fact (with parent memory record) and return fact_id."""
    fid = fact_id or _new_id()
    mid = _new_id()
    ents = json.dumps(entities or [])
    obs_date = observation_date or datetime(2026, 1, 15, 12, 0).isoformat()

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
        "VALUES (?, ?, ?, ?, 'semantic', ?, ?, ?, ?, ?, 1, 0, '[]', "
        " 'test', ?, 0.0, 0.0, 'factual', datetime('now'))",
        (fid, mid, profile_id, content, ents, ents, obs_date,
         confidence, importance, lifecycle),
    )
    return fid


def _seed_retention(
    db: DatabaseManager,
    fact_id: str,
    profile_id: str,
    *,
    retention_score: float = 0.3,
    lifecycle_zone: str = "warm",
    memory_strength: float = 1.0,
) -> None:
    """Insert a fact_retention row."""
    db.execute(
        "INSERT INTO fact_retention "
        "(fact_id, profile_id, retention_score, memory_strength, "
        " access_count, last_accessed_at, lifecycle_zone) "
        "VALUES (?, ?, ?, ?, 1, datetime('now'), ?)",
        (fact_id, profile_id, retention_score, memory_strength, lifecycle_zone),
    )


def _seed_warm_cluster(
    db: DatabaseManager,
    profile_id: str,
    *,
    count: int = 3,
    shared_entities: list[str] | None = None,
    base_date: str | None = None,
    retention: float = 0.3,
) -> list[str]:
    """Seed a cluster of warm facts sharing entities. Returns fact_ids."""
    entities = shared_entities or ["Python", "FastAPI"]
    base = base_date or "2026-01-15T12:00:00"
    fact_ids = []
    for i in range(count):
        # Stagger dates slightly within the temporal window
        dt = datetime.fromisoformat(base) + timedelta(hours=i * 2)
        fid = _seed_fact(
            db, profile_id,
            content=f"Fact {i} about {', '.join(entities)}",
            entities=entities,
            importance=0.5 + i * 0.1,
            confidence=0.8,
            observation_date=dt.isoformat(),
        )
        _seed_retention(
            db, fid, profile_id,
            retention_score=retention,
            lifecycle_zone="warm",
        )
        fact_ids.append(fid)
    return fact_ids


# ---------------------------------------------------------------------------
# Test 1: Identify warm/cold candidates
# ---------------------------------------------------------------------------


def test_identify_warm_cold_candidates(db, consolidator, profile_id):
    """Step 1 returns only warm/cold facts below retention threshold."""
    _seed_profile(db, profile_id)

    # 3 warm facts (retention 0.3)
    warm_ids = []
    for i in range(3):
        fid = _seed_fact(db, profile_id, content=f"warm-{i}", entities=["A", "B"])
        _seed_retention(db, fid, profile_id, retention_score=0.3, lifecycle_zone="warm")
        warm_ids.append(fid)

    # 3 cold facts (retention 0.1)
    cold_ids = []
    for i in range(3):
        fid = _seed_fact(db, profile_id, content=f"cold-{i}", entities=["A", "B"])
        _seed_retention(db, fid, profile_id, retention_score=0.1, lifecycle_zone="cold")
        cold_ids.append(fid)

    # 4 active facts (retention 0.9) — should NOT be returned
    for i in range(4):
        fid = _seed_fact(db, profile_id, content=f"active-{i}", entities=["X", "Y"])
        _seed_retention(db, fid, profile_id, retention_score=0.9, lifecycle_zone="active")

    candidates = consolidator._step1_identify(profile_id)
    candidate_ids = {c["fact_id"] for c in candidates}

    assert len(candidates) == 6
    assert set(warm_ids + cold_ids) == candidate_ids


# ---------------------------------------------------------------------------
# Test 2: Skip already-consolidated facts
# ---------------------------------------------------------------------------


def test_skip_already_consolidated(db, consolidator, profile_id):
    """Facts already in ccq_consolidated_blocks are excluded."""
    _seed_profile(db, profile_id)

    # Seed 5 warm facts
    fact_ids = []
    for i in range(5):
        fid = _seed_fact(db, profile_id, content=f"fact-{i}", entities=["A", "B"])
        _seed_retention(db, fid, profile_id, retention_score=0.3, lifecycle_zone="warm")
        fact_ids.append(fid)

    # Consolidate first 3 into a CCQ block
    db.store_ccq_block(
        block_id=_new_id(),
        profile_id=profile_id,
        content="Gist of first 3 facts",
        source_fact_ids=json.dumps(fact_ids[:3]),
        gist_embedding_rowid=None,
        char_count=21,
        cluster_id=_new_id(),
    )

    candidates = consolidator._step1_identify(profile_id)
    candidate_ids = {c["fact_id"] for c in candidates}

    assert len(candidates) == 2
    assert candidate_ids == set(fact_ids[3:])


# ---------------------------------------------------------------------------
# Test 3: Cluster by entity overlap
# ---------------------------------------------------------------------------


def test_cluster_by_entity_overlap(db, consolidator, profile_id):
    """Union-Find groups facts by shared entities (min_overlap=2)."""
    _seed_profile(db, profile_id)

    # Group 1: A,B,C share entities {x, y}
    group1 = []
    for i in range(3):
        fid = _seed_fact(
            db, profile_id, content=f"g1-{i}",
            entities=["x", "y"], observation_date="2026-01-15T12:00:00",
        )
        _seed_retention(db, fid, profile_id, retention_score=0.3, lifecycle_zone="warm")
        group1.append(fid)

    # Group 2: D,E,F share entities {z, w}
    group2 = []
    for i in range(3):
        fid = _seed_fact(
            db, profile_id, content=f"g2-{i}",
            entities=["z", "w"], observation_date="2026-01-15T12:00:00",
        )
        _seed_retention(db, fid, profile_id, retention_score=0.3, lifecycle_zone="warm")
        group2.append(fid)

    candidates = consolidator._step1_identify(profile_id)
    clusters = consolidator._step2_cluster(candidates, profile_id)

    assert len(clusters) == 2
    cluster_fact_sets = [set(c.fact_ids) for c in clusters]
    assert set(group1) in cluster_fact_sets
    assert set(group2) in cluster_fact_sets


# ---------------------------------------------------------------------------
# Test 4: Temporal sub-clustering
# ---------------------------------------------------------------------------


def test_cluster_temporal_proximity(db, consolidator, profile_id):
    """Facts separated by >7 days form separate temporal sub-clusters."""
    _seed_profile(db, profile_id)

    # 3 facts from Jan 2026 + 3 facts from Mar 2026 (>7 day gap)
    # All share entities {x, y}
    jan_ids = []
    for i in range(3):
        dt = datetime(2026, 1, 10, 12, 0) + timedelta(hours=i)
        fid = _seed_fact(
            db, profile_id, content=f"jan-{i}",
            entities=["x", "y"], observation_date=dt.isoformat(),
        )
        _seed_retention(db, fid, profile_id, retention_score=0.3, lifecycle_zone="warm")
        jan_ids.append(fid)

    mar_ids = []
    for i in range(3):
        dt = datetime(2026, 3, 10, 12, 0) + timedelta(hours=i)
        fid = _seed_fact(
            db, profile_id, content=f"mar-{i}",
            entities=["x", "y"], observation_date=dt.isoformat(),
        )
        _seed_retention(db, fid, profile_id, retention_score=0.3, lifecycle_zone="warm")
        mar_ids.append(fid)

    candidates = consolidator._step1_identify(profile_id)
    clusters = consolidator._step2_cluster(candidates, profile_id)

    assert len(clusters) == 2
    cluster_fact_sets = [set(c.fact_ids) for c in clusters]
    assert set(jan_ids) in cluster_fact_sets
    assert set(mar_ids) in cluster_fact_sets


# ---------------------------------------------------------------------------
# Test 5: Minimum cluster size
# ---------------------------------------------------------------------------


def test_minimum_cluster_size(db, profile_id):
    """Clusters below min_cluster_size are discarded."""
    _seed_profile(db, profile_id)
    config = CCQConfig(use_llm_gist=False, min_cluster_size=3)
    cons = CognitiveConsolidator(db=db, config=config)

    # 3 facts sharing entities (valid cluster)
    for i in range(3):
        fid = _seed_fact(
            db, profile_id, content=f"good-{i}",
            entities=["x", "y"], observation_date="2026-01-15T12:00:00",
        )
        _seed_retention(db, fid, profile_id, retention_score=0.3, lifecycle_zone="warm")

    # 1 isolated fact (different entities, should NOT form cluster)
    fid = _seed_fact(
        db, profile_id, content="isolated",
        entities=["z", "w"], observation_date="2026-01-15T12:00:00",
    )
    _seed_retention(db, fid, profile_id, retention_score=0.3, lifecycle_zone="warm")

    candidates = cons._step1_identify(profile_id)
    clusters = cons._step2_cluster(candidates, profile_id)

    assert len(clusters) == 1
    assert clusters[0].fact_count == 3


# ---------------------------------------------------------------------------
# Test 6: Gist extraction Mode A (rules-based)
# ---------------------------------------------------------------------------


def test_gist_extraction_mode_a(db, consolidator, profile_id):
    """Mode A gist uses representative fact content + entity summary."""
    _seed_profile(db, profile_id)

    fact_ids = _seed_warm_cluster(
        db, profile_id,
        count=3,
        shared_entities=["Python", "FastAPI"],
    )

    cluster = ConsolidationCluster(
        cluster_id=_new_id(),
        fact_ids=tuple(fact_ids),
        shared_entities=("Python", "FastAPI"),
        temporal_centroid="2026-01-15T12:00:00",
        avg_retention=0.3,
        fact_count=3,
    )

    gist = consolidator._step3_extract_gist(cluster, profile_id)

    assert isinstance(gist, GistResult)
    assert gist.extraction_mode == "rules"
    assert "Python" in gist.gist_text or "FastAPI" in gist.gist_text
    assert len(gist.key_entities) > 0
    assert gist.representative_fact_id in fact_ids


# ---------------------------------------------------------------------------
# Test 7: Gist extraction Mode B (LLM)
# ---------------------------------------------------------------------------


def test_gist_extraction_mode_b(db, profile_id):
    """Mode B gist calls LLM and validates entity coverage."""
    _seed_profile(db, profile_id)

    mock_llm = MagicMock()
    mock_llm.is_available.return_value = True
    mock_llm.generate.return_value = (
        "Python is a programming language used with FastAPI for APIs."
    )

    config = CCQConfig(use_llm_gist=True)
    cons = CognitiveConsolidator(db=db, llm=mock_llm, config=config)

    fact_ids = _seed_warm_cluster(
        db, profile_id,
        count=3,
        shared_entities=["Python", "FastAPI"],
    )

    cluster = ConsolidationCluster(
        cluster_id=_new_id(),
        fact_ids=tuple(fact_ids),
        shared_entities=("Python", "FastAPI"),
        temporal_centroid="2026-01-15T12:00:00",
        avg_retention=0.3,
        fact_count=3,
    )

    gist = cons._step3_extract_gist(cluster, profile_id)

    assert gist.extraction_mode == "llm"
    assert "Python" in gist.gist_text
    mock_llm.generate.assert_called_once()


# ---------------------------------------------------------------------------
# Test 8: Gist fallback to Mode A when LLM fails
# ---------------------------------------------------------------------------


def test_gist_fallback_to_mode_a(db, profile_id):
    """When LLM raises exception, gist falls back to Mode A rules."""
    _seed_profile(db, profile_id)

    mock_llm = MagicMock()
    mock_llm.is_available.return_value = True
    mock_llm.generate.side_effect = RuntimeError("LLM timeout")

    config = CCQConfig(use_llm_gist=True)
    cons = CognitiveConsolidator(db=db, llm=mock_llm, config=config)

    fact_ids = _seed_warm_cluster(
        db, profile_id,
        count=3,
        shared_entities=["Python", "FastAPI"],
    )

    cluster = ConsolidationCluster(
        cluster_id=_new_id(),
        fact_ids=tuple(fact_ids),
        shared_entities=("Python", "FastAPI"),
        temporal_centroid="2026-01-15T12:00:00",
        avg_retention=0.3,
        fact_count=3,
    )

    gist = cons._step3_extract_gist(cluster, profile_id)

    assert gist.extraction_mode == "rules"
    assert len(gist.gist_text) > 0


# ---------------------------------------------------------------------------
# Test 9: Embedding compression (mocked PolarQuant)
# ---------------------------------------------------------------------------


def test_embedding_compression_with_polar(db, consolidator, profile_id):
    """Step 4 reports bytes_before > bytes_after when embeddings exist."""
    _seed_profile(db, profile_id)

    fact_ids = _seed_warm_cluster(db, profile_id, count=3)

    # Seed embedding_metadata for each fact
    for i, fid in enumerate(fact_ids):
        db.execute(
            "INSERT INTO embedding_metadata "
            "(vec_rowid, fact_id, profile_id, model_name, dimension) "
            "VALUES (?, ?, ?, 'nomic', 768)",
            (i + 1, fid, profile_id),
        )

    cluster = ConsolidationCluster(
        cluster_id=_new_id(),
        fact_ids=tuple(fact_ids),
        shared_entities=("Python", "FastAPI"),
        temporal_centroid="2026-01-15T12:00:00",
        avg_retention=0.3,
        fact_count=3,
    )

    bytes_before, bytes_after = consolidator._step4_compress_embeddings(
        cluster, profile_id,
    )

    # Each fact: 768 dims * 4 bytes = 3072 bytes before
    assert bytes_before == 3 * 768 * 4
    # After should be <= before (PolarQuant may not be available,
    # in which case bytes_after == bytes_before with pending_polar2)
    assert bytes_after <= bytes_before
    assert bytes_after > 0


# ---------------------------------------------------------------------------
# Test 10: Gist embedding at full precision
# ---------------------------------------------------------------------------


def test_gist_embedding_full_precision(db, profile_id):
    """Pipeline stores gist block with full-precision embedding when embedder available."""
    _seed_profile(db, profile_id)

    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = [0.1] * 768

    config = CCQConfig(use_llm_gist=False, min_cluster_size=3)
    cons = CognitiveConsolidator(db=db, embedder=mock_embedder, config=config)

    fact_ids = _seed_warm_cluster(db, profile_id, count=3)

    cluster = ConsolidationCluster(
        cluster_id=_new_id(),
        fact_ids=tuple(fact_ids),
        shared_entities=("Python", "FastAPI"),
        temporal_centroid="2026-01-15T12:00:00",
        avg_retention=0.3,
        fact_count=3,
    )
    gist = GistResult(
        gist_text="Gist about Python and FastAPI",
        key_entities=("Python", "FastAPI"),
        extraction_mode="rules",
        representative_fact_id=fact_ids[0],
    )

    block_id = cons._step5_store_block(cluster, gist, profile_id)

    # Verify block was stored
    blocks = db.get_ccq_blocks(profile_id)
    assert len(blocks) == 1
    assert blocks[0]["block_id"] == block_id

    # Verify embedder was called with gist text
    mock_embedder.encode.assert_called_once_with(gist.gist_text)


# ---------------------------------------------------------------------------
# Test 11: Compression ratio
# ---------------------------------------------------------------------------


def test_compression_ratio(db, profile_id):
    """Full pipeline produces correct metrics even without PolarQuant."""
    _seed_profile(db, profile_id)

    # Seed 5 warm facts with shared entities and embeddings
    fact_ids = _seed_warm_cluster(
        db, profile_id, count=5, shared_entities=["Python", "FastAPI"],
    )
    for i, fid in enumerate(fact_ids):
        db.execute(
            "INSERT INTO embedding_metadata "
            "(vec_rowid, fact_id, profile_id, model_name, dimension) "
            "VALUES (?, ?, ?, 'nomic', 768)",
            (i + 100, fid, profile_id),
        )

    config = CCQConfig(use_llm_gist=False, min_cluster_size=3)
    cons = CognitiveConsolidator(db=db, config=config)

    result = cons.run_pipeline(profile_id)

    assert isinstance(result, CCQPipelineResult)
    assert result.clusters_processed >= 1
    assert result.blocks_created >= 1
    # Without PolarQuant: bytes_before == bytes_after (no compression)
    # But bytes should still be tracked
    assert result.total_bytes_before > 0
    # bytes_before >= bytes_after (either equal or compressed)
    assert result.total_bytes_before >= result.total_bytes_after


# ---------------------------------------------------------------------------
# Test 12: Source facts archived
# ---------------------------------------------------------------------------


def test_source_facts_archived(db, profile_id):
    """After pipeline, source facts have lifecycle='archived' and zone='archive'."""
    _seed_profile(db, profile_id)

    fact_ids = _seed_warm_cluster(
        db, profile_id, count=3, shared_entities=["Python", "FastAPI"],
    )

    config = CCQConfig(use_llm_gist=False, min_cluster_size=3)
    cons = CognitiveConsolidator(db=db, config=config)

    result = cons.run_pipeline(profile_id)
    assert result.facts_archived == 3

    # Verify lifecycle in atomic_facts
    for fid in fact_ids:
        rows = db.execute(
            "SELECT lifecycle FROM atomic_facts WHERE fact_id = ?",
            (fid,),
        )
        assert dict(rows[0])["lifecycle"] == "archived"

    # Verify lifecycle_zone in fact_retention
    for fid in fact_ids:
        rows = db.execute(
            "SELECT lifecycle_zone FROM fact_retention WHERE fact_id = ?",
            (fid,),
        )
        assert dict(rows[0])["lifecycle_zone"] == "archive"


# ---------------------------------------------------------------------------
# Coverage: date helper edge cases
# ---------------------------------------------------------------------------


def test_parse_date_none_and_empty():
    """_parse_date returns None for None/empty input."""
    from superlocalmemory.encoding.cognitive_consolidator import _parse_date

    assert _parse_date(None) is None
    assert _parse_date("") is None


def test_parse_date_multiple_formats():
    """_parse_date handles all supported ISO-8601 formats."""
    from superlocalmemory.encoding.cognitive_consolidator import _parse_date

    assert _parse_date("2026-01-15T12:00:00.123456") is not None
    assert _parse_date("2026-01-15T12:00:00") is not None
    assert _parse_date("2026-01-15 12:00:00") is not None
    assert _parse_date("2026-01-15") is not None
    assert _parse_date("not-a-date") is None


def test_temporal_midpoint_empty():
    """No known dates -> no centroid. It used to substitute 'now' (BRN-32)."""
    from superlocalmemory.encoding.cognitive_consolidator import _temporal_midpoint

    assert _temporal_midpoint([]) == ""


def test_parse_date_accepts_every_iso_spelling_slm_writes():
    """BRN-32: offsets and Z returned None, which disabled the window."""
    from datetime import datetime, timezone

    from superlocalmemory.encoding.cognitive_consolidator import _parse_date

    want = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for raw in ("2026-01-01", "2026-01-01T00:00:00", "2026-01-01 00:00:00",
                "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:00.000000+00:00", "2026-01-01T05:30:00+05:30",
                "2025-12-31T19:00:00-05:00"):
        assert _parse_date(raw) == want, raw
    assert _parse_date("not a date") is None


# ---------------------------------------------------------------------------
# Coverage: Union-Find rank swap
# ---------------------------------------------------------------------------


def test_union_find_rank_swap():
    """Union-Find handles rank-based merging (both paths)."""
    from superlocalmemory.encoding.cognitive_consolidator import _UnionFind

    uf = _UnionFind(["a", "b", "c", "d"])
    # Build tree: a-b, c-d
    uf.union("a", "b")
    uf.union("c", "d")
    # Now merge two trees of equal rank (triggers rank increment)
    uf.union("a", "c")
    comps = uf.components()
    # All should be in one group
    assert len(comps) == 1
    assert sorted(list(comps.values())[0]) == ["a", "b", "c", "d"]


def test_union_find_noop_same_root():
    """Union of elements already in same set is a no-op."""
    from superlocalmemory.encoding.cognitive_consolidator import _UnionFind

    uf = _UnionFind(["a", "b"])
    uf.union("a", "b")
    uf.union("a", "b")  # Should be no-op
    comps = uf.components()
    assert len(comps) == 1


# ---------------------------------------------------------------------------
# Coverage: empty pipeline (no candidates)
# ---------------------------------------------------------------------------


def test_pipeline_no_candidates(db, consolidator, profile_id):
    """Pipeline returns empty result when no candidates exist."""
    _seed_profile(db, profile_id)

    result = consolidator.run_pipeline(profile_id)

    assert result.clusters_processed == 0
    assert result.blocks_created == 0
    assert result.facts_archived == 0
    assert result.errors == ()


# ---------------------------------------------------------------------------
# Coverage: pipeline with cluster error isolation
# ---------------------------------------------------------------------------


def test_pipeline_cluster_error_isolation(db, profile_id):
    """Pipeline continues when one cluster fails (HR-07)."""
    from unittest.mock import patch

    _seed_profile(db, profile_id)

    # Seed two valid clusters with different entities
    _seed_warm_cluster(
        db, profile_id, count=3, shared_entities=["A", "B"],
    )
    _seed_warm_cluster(
        db, profile_id, count=3, shared_entities=["C", "D"],
    )

    config = CCQConfig(use_llm_gist=False, min_cluster_size=3)
    cons = CognitiveConsolidator(db=db, config=config)

    # Use patch on the class method (not instance) to fail on first call
    original = CognitiveConsolidator._step3_extract_gist
    call_count = [0]

    def failing_step3(self, cluster, pid):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("Simulated cluster failure")
        return original(self, cluster, pid)

    with patch.object(
        CognitiveConsolidator, "_step3_extract_gist", failing_step3,
    ):
        result = cons.run_pipeline(profile_id)

    # One cluster should have succeeded, one failed
    assert result.blocks_created >= 1
    assert len(result.errors) >= 1


# ---------------------------------------------------------------------------
# Coverage: compress_embeddings disabled
# ---------------------------------------------------------------------------


def test_compress_embeddings_disabled(db, profile_id):
    """Step 4 returns (0, 0) when compress_embeddings=False."""
    _seed_profile(db, profile_id)

    config = CCQConfig(use_llm_gist=False, compress_embeddings=False)
    cons = CognitiveConsolidator(db=db, config=config)

    fact_ids = _seed_warm_cluster(db, profile_id, count=3)
    cluster = ConsolidationCluster(
        cluster_id=_new_id(),
        fact_ids=tuple(fact_ids),
        shared_entities=("Python", "FastAPI"),
        temporal_centroid="2026-01-15T12:00:00",
        avg_retention=0.3,
        fact_count=3,
    )

    bb, ba = cons._step4_compress_embeddings(cluster, profile_id)
    assert bb == 0
    assert ba == 0


# ---------------------------------------------------------------------------
# Coverage: compress single embedding with no metadata
# ---------------------------------------------------------------------------


def test_compress_no_embedding_metadata(db, consolidator, profile_id):
    """Step 4 returns (0, 0) for facts without embedding metadata."""
    _seed_profile(db, profile_id)
    fact_ids = _seed_warm_cluster(db, profile_id, count=3)

    # No embedding_metadata rows seeded
    cluster = ConsolidationCluster(
        cluster_id=_new_id(),
        fact_ids=tuple(fact_ids),
        shared_entities=("Python", "FastAPI"),
        temporal_centroid="2026-01-15T12:00:00",
        avg_retention=0.3,
        fact_count=3,
    )

    bb, ba = consolidator._step4_compress_embeddings(cluster, profile_id)
    assert bb == 0
    assert ba == 0


# ---------------------------------------------------------------------------
# Coverage: compress with already-quantized embedding
# ---------------------------------------------------------------------------


def test_compress_already_quantized(db, consolidator, profile_id):
    """Step 4 skips embeddings already at polar2/deleted."""
    _seed_profile(db, profile_id)
    fact_ids = _seed_warm_cluster(db, profile_id, count=3)

    for i, fid in enumerate(fact_ids):
        db.execute(
            "INSERT INTO embedding_metadata "
            "(vec_rowid, fact_id, profile_id, model_name, dimension) "
            "VALUES (?, ?, ?, 'nomic', 768)",
            (i + 200, fid, profile_id),
        )
        db.execute(
            "INSERT INTO embedding_quantization_metadata "
            "(fact_id, profile_id, quantization_level, bit_width) "
            "VALUES (?, ?, 'polar2', 2)",
            (fid, profile_id),
        )

    cluster = ConsolidationCluster(
        cluster_id=_new_id(),
        fact_ids=tuple(fact_ids),
        shared_entities=("Python", "FastAPI"),
        temporal_centroid="2026-01-15T12:00:00",
        avg_retention=0.3,
        fact_count=3,
    )

    bb, ba = consolidator._step4_compress_embeddings(cluster, profile_id)
    assert bb == 0
    assert ba == 0


# ---------------------------------------------------------------------------
# Coverage: LLM gist fails entity validation
# ---------------------------------------------------------------------------


def test_gist_llm_fails_entity_validation(db, profile_id):
    """LLM gist that doesn't mention enough entities falls back to rules."""
    _seed_profile(db, profile_id)

    mock_llm = MagicMock()
    mock_llm.is_available.return_value = True
    # Response that doesn't mention shared entities
    mock_llm.generate.return_value = "Some completely unrelated summary."

    config = CCQConfig(use_llm_gist=True, min_entity_coverage=0.5)
    cons = CognitiveConsolidator(db=db, llm=mock_llm, config=config)

    fact_ids = _seed_warm_cluster(
        db, profile_id, count=3, shared_entities=["Python", "FastAPI"],
    )

    cluster = ConsolidationCluster(
        cluster_id=_new_id(),
        fact_ids=tuple(fact_ids),
        shared_entities=("Python", "FastAPI"),
        temporal_centroid="2026-01-15T12:00:00",
        avg_retention=0.3,
        fact_count=3,
    )

    gist = cons._step3_extract_gist(cluster, profile_id)
    # Should fall back to rules since LLM gist didn't cover entities
    assert gist.extraction_mode == "rules"


# ---------------------------------------------------------------------------
# Coverage: LLM gist with truncation
# ---------------------------------------------------------------------------


def test_gist_llm_truncation(db, profile_id):
    """LLM gist longer than max_gist_chars is truncated."""
    _seed_profile(db, profile_id)

    mock_llm = MagicMock()
    mock_llm.is_available.return_value = True
    mock_llm.generate.return_value = "Python FastAPI " + "x" * 600

    config = CCQConfig(use_llm_gist=True, max_gist_chars=100)
    cons = CognitiveConsolidator(db=db, llm=mock_llm, config=config)

    fact_ids = _seed_warm_cluster(
        db, profile_id, count=3, shared_entities=["Python", "FastAPI"],
    )

    cluster = ConsolidationCluster(
        cluster_id=_new_id(),
        fact_ids=tuple(fact_ids),
        shared_entities=("Python", "FastAPI"),
        temporal_centroid="2026-01-15T12:00:00",
        avg_retention=0.3,
        fact_count=3,
    )

    gist = cons._step3_extract_gist(cluster, profile_id)
    assert gist.extraction_mode == "llm"
    assert len(gist.gist_text) <= 100


# ---------------------------------------------------------------------------
# Coverage: Mode A gist truncation
# ---------------------------------------------------------------------------


def test_gist_mode_a_truncation(db, profile_id):
    """Mode A gist exceeding max_gist_chars is truncated with ellipsis."""
    _seed_profile(db, profile_id)

    config = CCQConfig(use_llm_gist=False, max_gist_chars=50)
    cons = CognitiveConsolidator(db=db, config=config)

    # Seed facts with long content
    fact_ids = []
    for i in range(3):
        fid = _seed_fact(
            db, profile_id,
            content="A" * 100 + f" item {i}",
            entities=["LongEntity1", "LongEntity2"],
        )
        _seed_retention(db, fid, profile_id)
        fact_ids.append(fid)

    cluster = ConsolidationCluster(
        cluster_id=_new_id(),
        fact_ids=tuple(fact_ids),
        shared_entities=("LongEntity1", "LongEntity2"),
        temporal_centroid="2026-01-15T12:00:00",
        avg_retention=0.3,
        fact_count=3,
    )

    gist = cons._step3_extract_gist(cluster, profile_id)
    assert len(gist.gist_text) <= 50
    assert gist.gist_text.endswith("...")


# ---------------------------------------------------------------------------
# Coverage: gist embedding generation fails
# ---------------------------------------------------------------------------


def test_gist_embedding_generation_fails(db, profile_id):
    """Step 5 continues when embedder.encode() raises an exception."""
    _seed_profile(db, profile_id)

    mock_embedder = MagicMock()
    mock_embedder.encode.side_effect = RuntimeError("Embedding service down")

    config = CCQConfig(use_llm_gist=False)
    cons = CognitiveConsolidator(db=db, embedder=mock_embedder, config=config)

    fact_ids = _seed_warm_cluster(db, profile_id, count=3)

    cluster = ConsolidationCluster(
        cluster_id=_new_id(),
        fact_ids=tuple(fact_ids),
        shared_entities=("Python", "FastAPI"),
        temporal_centroid="2026-01-15T12:00:00",
        avg_retention=0.3,
        fact_count=3,
    )
    gist = GistResult(
        gist_text="Test gist",
        key_entities=("Python",),
        extraction_mode="rules",
        representative_fact_id=fact_ids[0],
    )

    # Should not raise — continues without embedding
    block_id = cons._step5_store_block(cluster, gist, profile_id)
    assert block_id


# ---------------------------------------------------------------------------
# Coverage: audit with zero bytes_after
# ---------------------------------------------------------------------------


def test_audit_zero_bytes_after(db, consolidator, profile_id):
    """Audit handles bytes_after=0 (compression_ratio=0.0)."""
    _seed_profile(db, profile_id)

    block_id = _new_id()
    cluster_id = _new_id()

    # Create the referenced block first (FK requirement)
    db.store_ccq_block(
        block_id=block_id,
        profile_id=profile_id,
        content="Test block",
        source_fact_ids='["f1", "f2"]',
        gist_embedding_rowid=None,
        char_count=10,
        cluster_id=cluster_id,
    )

    cluster = ConsolidationCluster(
        cluster_id=cluster_id,
        fact_ids=("f1", "f2"),
        shared_entities=("A",),
        temporal_centroid="2026-01-15T12:00:00",
        avg_retention=0.3,
        fact_count=2,
    )
    gist = GistResult(
        gist_text="Test", key_entities=("A",),
        extraction_mode="rules", representative_fact_id="f1",
    )

    audit_id = consolidator._step6_audit(
        cluster, gist, bytes_before=100, bytes_after=0,
        block_id=block_id, profile_id=profile_id,
    )
    assert audit_id

    audits = db.get_ccq_audit(profile_id)
    assert len(audits) == 1
    assert audits[0]["compression_ratio"] == 0.0


# ---------------------------------------------------------------------------
# Coverage: LLM gist with empty shared_entities
# ---------------------------------------------------------------------------


def test_gist_llm_empty_shared_entities(db, profile_id):
    """LLM gist with no shared entities skips entity validation."""
    _seed_profile(db, profile_id)

    mock_llm = MagicMock()
    mock_llm.is_available.return_value = True
    mock_llm.generate.return_value = "A summary of related facts."

    config = CCQConfig(use_llm_gist=True)
    cons = CognitiveConsolidator(db=db, llm=mock_llm, config=config)

    fact_ids = _seed_warm_cluster(
        db, profile_id, count=3, shared_entities=["Python", "FastAPI"],
    )

    cluster = ConsolidationCluster(
        cluster_id=_new_id(),
        fact_ids=tuple(fact_ids),
        shared_entities=(),  # Empty shared entities
        temporal_centroid="2026-01-15T12:00:00",
        avg_retention=0.3,
        fact_count=3,
    )

    gist = cons._step3_extract_gist(cluster, profile_id)
    assert gist.extraction_mode == "llm"


# ---------------------------------------------------------------------------
# Coverage: step2_cluster with too few candidates
# ---------------------------------------------------------------------------


def test_cluster_too_few_candidates(db, consolidator, profile_id):
    """Step 2 returns empty list when candidates < min_cluster_size."""
    _seed_profile(db, profile_id)

    # Only 2 candidates (below min_cluster_size=3)
    candidates = [
        {"fact_id": "f1", "canonical_entities": ["A", "B"],
         "observation_date": "2026-01-15T12:00:00", "retention_score": 0.3},
        {"fact_id": "f2", "canonical_entities": ["A", "B"],
         "observation_date": "2026-01-15T13:00:00", "retention_score": 0.3},
    ]

    clusters = consolidator._step2_cluster(candidates, profile_id)
    assert clusters == []


# ---------------------------------------------------------------------------
# Coverage: Union-Find with unequal ranks (line 142)
# ---------------------------------------------------------------------------


def test_union_find_unequal_ranks():
    """Union-Find swaps when rank[a] < rank[b]."""
    from superlocalmemory.encoding.cognitive_consolidator import _UnionFind

    uf = _UnionFind(["a", "b", "c", "d", "e"])
    # Build a tall tree on 'a' side: a-b, a-c (rank of a's root = 1)
    uf.union("a", "b")
    uf.union("a", "c")
    # 'd' is a leaf (rank 0). Union d with a's root triggers rank swap
    uf.union("d", "a")
    # 'e' is also a leaf. Union e with d (which merged into a's tree)
    uf.union("e", "b")

    comps = uf.components()
    assert len(comps) == 1


# ---------------------------------------------------------------------------
# Coverage: candidates exist but no valid clusters (line 231)
# ---------------------------------------------------------------------------


def test_pipeline_candidates_no_valid_clusters(db, profile_id):
    """Pipeline returns empty when candidates have no shared entities."""
    _seed_profile(db, profile_id)

    # 4 facts with all different entities — no overlap >= 2
    for i in range(4):
        fid = _seed_fact(
            db, profile_id, content=f"unique-{i}",
            entities=[f"unique_entity_{i}_a", f"unique_entity_{i}_b"],
        )
        _seed_retention(db, fid, profile_id, retention_score=0.3, lifecycle_zone="warm")

    config = CCQConfig(use_llm_gist=False, min_cluster_size=3)
    cons = CognitiveConsolidator(db=db, config=config)

    result = cons.run_pipeline(profile_id)
    assert result.clusters_processed == 0
    assert result.blocks_created == 0


# ---------------------------------------------------------------------------
# Coverage: temporal subcluster with None dates (lines 451-452)
# ---------------------------------------------------------------------------


def test_temporal_subcluster_with_none_dates(db, profile_id):
    """_temporal_subcluster handles facts with None dates (lines 451-452)."""
    _seed_profile(db, profile_id)

    config = CCQConfig(use_llm_gist=False, min_cluster_size=2)
    cons = CognitiveConsolidator(db=db, config=config)

    # Direct test of _temporal_subcluster with None dates in fact_map
    fact_map = {
        "f1": {"observation_date": "2026-01-15T12:00:00", "created_at": None},
        "f2": {"observation_date": None, "created_at": None},  # Both None
        "f3": {"observation_date": "2026-01-15T13:00:00", "created_at": None},
    }

    sub_clusters = cons._temporal_subcluster(["f1", "f2", "f3"], fact_map)

    # A fact with no known date cannot be shown to be inside the window, so
    # it joins no cluster (BRN-32: it used to join whichever was open).
    assert sub_clusters == [["f1", "f3"]]


# ---------------------------------------------------------------------------
# Coverage: gist with empty facts from DB query (lines 499-504)
# ---------------------------------------------------------------------------


def test_gist_empty_facts_query(db, consolidator, profile_id):
    """Step 3 returns empty gist when cluster fact_ids don't exist in DB."""
    _seed_profile(db, profile_id)

    # Cluster references non-existent fact_ids
    cluster = ConsolidationCluster(
        cluster_id=_new_id(),
        fact_ids=("nonexistent1", "nonexistent2", "nonexistent3"),
        shared_entities=("A",),
        temporal_centroid="2026-01-15T12:00:00",
        avg_retention=0.3,
        fact_count=3,
    )

    gist = consolidator._step3_extract_gist(cluster, profile_id)
    assert gist.gist_text == "[empty cluster]"
    assert gist.extraction_mode == "rules"


# ---------------------------------------------------------------------------
# Coverage: PolarQuant not importable (lines 686-691)
# ---------------------------------------------------------------------------


def test_compress_without_polar_quant(db, profile_id):
    """Compression fallback when PolarQuant is not importable."""
    from unittest.mock import patch

    _seed_profile(db, profile_id)
    config = CCQConfig(use_llm_gist=False)
    cons = CognitiveConsolidator(db=db, config=config)

    fact_ids = _seed_warm_cluster(db, profile_id, count=3)
    for i, fid in enumerate(fact_ids):
        db.execute(
            "INSERT INTO embedding_metadata "
            "(vec_rowid, fact_id, profile_id, model_name, dimension) "
            "VALUES (?, ?, ?, 'nomic', 768)",
            (i + 300, fid, profile_id),
        )

    cluster = ConsolidationCluster(
        cluster_id=_new_id(),
        fact_ids=tuple(fact_ids),
        shared_entities=("Python", "FastAPI"),
        temporal_centroid="2026-01-15T12:00:00",
        avg_retention=0.3,
        fact_count=3,
    )

    # Mock the import to fail
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if "polar_quant" in name:
            raise ImportError("Mocked: PolarQuant not available")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        bb, ba = cons._step4_compress_embeddings(cluster, profile_id)

    # Without PolarQuant: bytes_before == bytes_after
    assert bb == 3 * 768 * 4
    assert ba == bb


# ---------------------------------------------------------------------------
# Coverage: _extract_gist_llm called with None LLM (line 531)
# ---------------------------------------------------------------------------


def test_extract_gist_llm_none():
    """_extract_gist_llm returns None when llm is None."""
    from unittest.mock import MagicMock

    # Access the private method via an instance with no LLM
    from superlocalmemory.core.config import CCQConfig as _CCQConfig
    from superlocalmemory.encoding.cognitive_consolidator import (
        CognitiveConsolidator,
    )

    db_mock = MagicMock()
    cons = CognitiveConsolidator(db=db_mock, llm=None, config=_CCQConfig())

    result = cons._extract_gist_llm(
        [{"content": "test", "canonical_entities": []}],
        ("entity1",),
    )
    assert result is None


# ---------------------------------------------------------------------------
# Coverage: malformed canonical_entities_json in step1 (lines 337-340)
# ---------------------------------------------------------------------------


def test_identify_malformed_entities_json(db, profile_id):
    """Step 1 handles malformed canonical_entities_json gracefully."""
    _seed_profile(db, profile_id)

    fid = _seed_fact(db, profile_id, content="bad-json", entities=["A", "B"])
    _seed_retention(db, fid, profile_id, retention_score=0.3, lifecycle_zone="warm")

    # Corrupt the canonical_entities_json column
    db.execute(
        "UPDATE atomic_facts SET canonical_entities_json = 'not-valid-json' "
        "WHERE fact_id = ?",
        (fid,),
    )

    config = CCQConfig(use_llm_gist=False)
    cons = CognitiveConsolidator(db=db, config=config)

    candidates = cons._step1_identify(profile_id)
    assert len(candidates) == 1
    # Should default to empty list
    assert candidates[0]["canonical_entities"] == []


# ---------------------------------------------------------------------------
# Coverage: malformed entities in step3 gist extraction (lines 493-496)
# ---------------------------------------------------------------------------


def test_temporal_split_discards_small_subclusters(db, profile_id):
    """Temporal sub-clusters below min_cluster_size are discarded (line 386)."""
    _seed_profile(db, profile_id)

    config = CCQConfig(use_llm_gist=False, min_cluster_size=3, temporal_window_days=7)
    cons = CognitiveConsolidator(db=db, config=config)

    # 5 facts sharing entities {A, B}: 3 from Jan, 2 from June (>7 day gap)
    # The Jan group (3) is valid, the June group (2) is too small
    for i in range(3):
        dt = datetime(2026, 1, 10, 12, 0) + timedelta(hours=i)
        fid = _seed_fact(
            db, profile_id, content=f"jan-{i}",
            entities=["A", "B"], observation_date=dt.isoformat(),
        )
        _seed_retention(db, fid, profile_id, retention_score=0.3, lifecycle_zone="warm")

    for i in range(2):
        dt = datetime(2026, 6, 10, 12, 0) + timedelta(hours=i)
        fid = _seed_fact(
            db, profile_id, content=f"jun-{i}",
            entities=["A", "B"], observation_date=dt.isoformat(),
        )
        _seed_retention(db, fid, profile_id, retention_score=0.3, lifecycle_zone="warm")

    candidates = cons._step1_identify(profile_id)
    clusters = cons._step2_cluster(candidates, profile_id)

    # Only the Jan cluster (3 facts) should survive; June (2) discarded
    assert len(clusters) == 1
    assert clusters[0].fact_count == 3


def test_gist_with_malformed_entities(db, profile_id):
    """Step 3 handles malformed canonical_entities_json in fact rows."""
    _seed_profile(db, profile_id)

    fact_ids = _seed_warm_cluster(
        db, profile_id, count=3, shared_entities=["A", "B"],
    )

    # Corrupt entities on all facts
    for fid in fact_ids:
        db.execute(
            "UPDATE atomic_facts SET canonical_entities_json = '{invalid}' "
            "WHERE fact_id = ?",
            (fid,),
        )

    config = CCQConfig(use_llm_gist=False)
    cons = CognitiveConsolidator(db=db, config=config)

    cluster = ConsolidationCluster(
        cluster_id=_new_id(),
        fact_ids=tuple(fact_ids),
        shared_entities=("A", "B"),
        temporal_centroid="2026-01-15T12:00:00",
        avg_retention=0.3,
        fact_count=3,
    )

    gist = cons._step3_extract_gist(cluster, profile_id)
    assert gist.extraction_mode == "rules"
    assert len(gist.gist_text) > 0
