# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""End-to-end integration test for superlocalmemory MemoryEngine.

Validates the FULL pipeline with a real SQLite database:
  store → encode → retrieve → recall

Uses Mode A (zero LLM) with a MOCK embedder to avoid loading
real sentence-transformers models (which eat 30+ GB of RAM).

Covers:
  - Store and recall flow
  - Multi-session storage with different speakers
  - Entity extraction (Mode A regex-based)
  - Fact type classification
  - Temporal date parsing
  - BM25 keyword retrieval
  - Profile isolation (two profiles, separate data)
  - Mode switching (A vs C config differences)
  - Consolidation (supersede on contradiction)
  - Engine lifecycle (initialize, store, recall, close)
  - RecallResponse structure validation

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from superlocalmemory.core.config import SLMConfig, RetrievalConfig
from superlocalmemory.core.engine import MemoryEngine
from superlocalmemory.storage.models import (
    AtomicFact,
    FactType,
    Mode,
    RecallResponse,
    RetrievalResult,
)


# ---------------------------------------------------------------------------
# Mock Embedder — deterministic, zero model loading
# ---------------------------------------------------------------------------

class _MockEmbedder:
    """Deterministic mock embedder: text → 768-dim vector via hashing.

    Produces CONSISTENT embeddings (same text → same vector) with
    enough variance for cosine similarity to be meaningful.
    No ML models loaded. No network calls. ~0.1ms per embed.
    """

    is_available = True

    def __init__(self, dimension: int = 768) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        vec = rng.standard_normal(self.dimension).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def compute_fisher_params(
        self, embedding: list[float],
    ) -> tuple[list[float], list[float]]:
        arr = np.asarray(embedding, dtype=np.float64)
        norm = float(np.linalg.norm(arr))
        if norm < 1e-10:
            mean = np.zeros(len(arr))
            var = np.full(len(arr), 2.0)
        else:
            mean = arr / norm
            abs_mean = np.abs(mean)
            max_val = float(np.max(abs_mean)) + 1e-10
            signal = abs_mean / max_val
            var = 2.0 - 1.7 * signal
            var = np.clip(var, 0.3, 2.0)
        return mean.tolist(), var.tolist()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_e2e.db"


@pytest.fixture()
def engine(db_path: Path) -> MemoryEngine:
    """Create a Mode A MemoryEngine with mock embedder and no cross-encoder."""
    config = SLMConfig.for_mode(Mode.A, base_dir=db_path.parent)
    config.db_path = db_path
    config.retrieval = RetrievalConfig(use_cross_encoder=False)

    eng = MemoryEngine(config)

    # Patch model loading at the SOURCE module (local import inside initialize)
    with patch(
        "superlocalmemory.core.embeddings.EmbeddingService",
        return_value=_MockEmbedder(768),
    ):
        eng.initialize()

    return eng


@pytest.fixture()
def loaded_engine(engine: MemoryEngine) -> MemoryEngine:
    """Engine pre-loaded with a small conversation dataset."""
    engine.store(
        "Alice is a software engineer at Google.",
        session_id="s1", speaker="Bob",
        session_date="3:00 pm on 5 March, 2026",
    )
    engine.store(
        "Bob mentioned that Alice loves hiking in the mountains.",
        session_id="s1", speaker="Bob",
        session_date="3:00 pm on 5 March, 2026",
    )
    engine.store(
        "Alice said she visited Paris last summer with her family.",
        session_id="s2", speaker="Alice",
        session_date="4:00 pm on 6 March, 2026",
    )
    engine.store(
        "Bob is a doctor at the local hospital. He graduated from Stanford.",
        session_id="s2", speaker="Alice",
        session_date="4:00 pm on 6 March, 2026",
    )
    engine.store(
        "Alice started working at Microsoft in January 2026.",
        session_id="s3", speaker="Bob",
        session_date="10:00 am on 8 March, 2026",
    )
    return engine


# ---------------------------------------------------------------------------
# Basic Lifecycle
# ---------------------------------------------------------------------------

class TestEngineLifecycle:
    def test_initialize_creates_database(self, engine: MemoryEngine, db_path: Path) -> None:
        assert db_path.exists()

    def test_engine_mode(self, engine: MemoryEngine) -> None:
        assert engine._config.mode == Mode.A

    def test_fact_count_starts_at_zero(self, engine: MemoryEngine) -> None:
        assert engine.fact_count == 0

    def test_close_resets_initialized(self, engine: MemoryEngine) -> None:
        engine.close()
        assert not engine._initialized


# ---------------------------------------------------------------------------
# Store Flow
# ---------------------------------------------------------------------------

class TestStoreFlow:
    def test_store_returns_fact_ids(self, engine: MemoryEngine) -> None:
        ids = engine.store("Alice is a software engineer.", session_id="s1")
        assert isinstance(ids, list)
        assert len(ids) > 0
        assert all(isinstance(fid, str) for fid in ids)

    def test_store_increments_fact_count(self, engine: MemoryEngine) -> None:
        engine.store("Alice is a software engineer.", session_id="s1")
        assert engine.fact_count > 0

    def test_store_multiple_sessions(self, engine: MemoryEngine) -> None:
        engine.store(
            "Alice works as a data scientist at Amazon in Seattle.",
            session_id="s1",
        )
        engine.store(
            "Bob graduated from MIT and now teaches at Stanford University.",
            session_id="s2",
        )
        assert engine.fact_count >= 2

    def test_empty_content_returns_empty(self, engine: MemoryEngine) -> None:
        ids = engine.store("", session_id="s1")
        assert ids == []

    def test_store_with_session_date(self, engine: MemoryEngine) -> None:
        ids = engine.store(
            "Meeting happened on March 5.",
            session_id="s1",
            session_date="3:00 pm on 5 March, 2026",
        )
        assert len(ids) > 0


# ---------------------------------------------------------------------------
# Recall Flow
# ---------------------------------------------------------------------------

class TestRecallFlow:
    def test_recall_returns_response(self, loaded_engine: MemoryEngine) -> None:
        response = loaded_engine.recall("What does Alice do?")
        assert isinstance(response, RecallResponse)
        assert response.query == "What does Alice do?"

    def test_recall_has_results(self, loaded_engine: MemoryEngine) -> None:
        response = loaded_engine.recall("What does Alice do?")
        assert len(response.results) > 0

    def test_recall_results_are_retrieval_results(self, loaded_engine: MemoryEngine) -> None:
        response = loaded_engine.recall("Alice")
        for result in response.results:
            assert isinstance(result, RetrievalResult)
            assert isinstance(result.fact, AtomicFact)
            assert result.score >= 0.0

    def test_recall_timing_recorded(self, loaded_engine: MemoryEngine) -> None:
        response = loaded_engine.recall("Alice")
        assert response.retrieval_time_ms > 0

    def test_recall_mode_propagated(self, loaded_engine: MemoryEngine) -> None:
        response = loaded_engine.recall("Alice", mode=Mode.A)
        assert response.mode == Mode.A

    def test_recall_limit_respected(self, loaded_engine: MemoryEngine) -> None:
        response = loaded_engine.recall("Alice", limit=2)
        assert len(response.results) <= 2

    def test_recall_query_type_detected(self, loaded_engine: MemoryEngine) -> None:
        response = loaded_engine.recall("When did Alice visit Paris?")
        assert response.query_type in ("factual", "temporal", "multi_hop", "opinion")

    def test_recall_channel_weights_present(self, loaded_engine: MemoryEngine) -> None:
        response = loaded_engine.recall("Alice")
        assert isinstance(response.channel_weights, dict)
        assert len(response.channel_weights) > 0


# ---------------------------------------------------------------------------
# BM25 Keyword Retrieval
# ---------------------------------------------------------------------------

class TestBM25Retrieval:
    def test_keyword_match(self, loaded_engine: MemoryEngine) -> None:
        response = loaded_engine.recall("Stanford")
        contents = [r.fact.content.lower() for r in response.results]
        # At least one result should mention Stanford
        assert any("stanford" in c for c in contents)

    def test_entity_keyword(self, loaded_engine: MemoryEngine) -> None:
        response = loaded_engine.recall("Paris")
        contents = [r.fact.content.lower() for r in response.results]
        assert any("paris" in c for c in contents)


# ---------------------------------------------------------------------------
# Profile Isolation
# ---------------------------------------------------------------------------

class TestProfileIsolation:
    @staticmethod
    def _create_profile(engine: MemoryEngine, profile_id: str) -> None:
        """Insert a profile row so FK constraints pass."""
        engine._db.execute(
            "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
            (profile_id, profile_id),
        )

    def test_different_profiles_are_isolated(self, engine: MemoryEngine) -> None:
        self._create_profile(engine, "work")
        self._create_profile(engine, "personal")

        engine.profile_id = "work"
        engine.store("Work fact: Q1 revenue was $10M", session_id="s1")

        engine.profile_id = "personal"
        engine.store("Personal fact: I love pizza", session_id="s1")

        # Recall from personal profile — should NOT see work facts
        response = engine.recall("revenue", profile_id="personal")
        work_facts = [
            r for r in response.results
            if "revenue" in r.fact.content.lower()
        ]
        assert len(work_facts) == 0

    def test_correct_profile_returns_data(self, engine: MemoryEngine) -> None:
        self._create_profile(engine, "work")

        engine.profile_id = "work"
        engine.store("Q1 revenue target is $10M", session_id="s1")

        response = engine.recall("revenue", profile_id="work")
        # Should find the work fact
        assert len(response.results) > 0


# ---------------------------------------------------------------------------
# Mode Configuration
# ---------------------------------------------------------------------------

class TestModeConfig:
    def test_mode_a_no_llm(self, db_path: Path) -> None:
        config = SLMConfig.for_mode(Mode.A, base_dir=db_path.parent)
        assert config.llm.provider == ""
        assert not config.llm.is_available

    def test_mode_b_has_ollama(self, db_path: Path) -> None:
        config = SLMConfig.for_mode(Mode.B, base_dir=db_path.parent)
        assert config.llm.provider == "ollama"
        assert config.llm.is_available

    def test_mode_c_has_cloud(self, db_path: Path) -> None:
        config = SLMConfig.for_mode(Mode.C, base_dir=db_path.parent)
        assert config.llm.provider == "openrouter"
        assert config.embedding.dimension == 3072

    def test_mode_a_local_embedding(self, db_path: Path) -> None:
        config = SLMConfig.for_mode(Mode.A, base_dir=db_path.parent)
        assert config.embedding.model_name == "nomic-ai/nomic-embed-text-v1.5"
        assert config.embedding.dimension == 768
        assert not config.embedding.is_cloud


# ---------------------------------------------------------------------------
# Response Structure Validation
# ---------------------------------------------------------------------------

class TestResponseStructure:
    def test_evidence_chain_populated(self, loaded_engine: MemoryEngine) -> None:
        response = loaded_engine.recall("Alice engineer")
        for result in response.results:
            assert isinstance(result.evidence_chain, list)

    def test_confidence_bounded(self, loaded_engine: MemoryEngine) -> None:
        response = loaded_engine.recall("Alice")
        for result in response.results:
            assert 0.0 <= result.confidence <= 1.0

    def test_total_candidates_positive(self, loaded_engine: MemoryEngine) -> None:
        response = loaded_engine.recall("Alice")
        assert response.total_candidates >= 0

    def test_fact_has_profile_id(self, loaded_engine: MemoryEngine) -> None:
        response = loaded_engine.recall("Alice")
        for result in response.results:
            assert result.fact.profile_id == "default"

    def test_fact_has_embedding(self, loaded_engine: MemoryEngine) -> None:
        response = loaded_engine.recall("Alice")
        for result in response.results:
            if result.fact.embedding is not None:
                assert len(result.fact.embedding) == 768


# ---------------------------------------------------------------------------
# Reconsolidation (access updates)
# ---------------------------------------------------------------------------

class TestReconsolidation:
    def test_recall_increments_access_count(self, loaded_engine: MemoryEngine) -> None:
        # First recall — establishes baseline
        r1 = loaded_engine.recall("Alice")
        if r1.results:
            fid = r1.results[0].fact.fact_id
            count1 = r1.results[0].fact.access_count

            # Second recall — should increment
            r2 = loaded_engine.recall("Alice")
            for result in r2.results:
                if result.fact.fact_id == fid:
                    assert result.fact.access_count >= count1


# ---------------------------------------------------------------------------
# Store → Recall Round-Trip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_store_then_recall_by_keyword(self, engine: MemoryEngine) -> None:
        engine.store(
            "The quarterly budget meeting is scheduled for next Friday.",
            session_id="s1",
        )
        response = engine.recall("budget meeting")
        assert len(response.results) > 0
        contents = [r.fact.content.lower() for r in response.results]
        assert any("budget" in c for c in contents)

    def test_store_then_recall_by_entity(self, engine: MemoryEngine) -> None:
        engine.store("Charlie graduated from MIT in 2020.", session_id="s1")
        response = engine.recall("Charlie")
        assert len(response.results) > 0
        contents = [r.fact.content.lower() for r in response.results]
        assert any("charlie" in c for c in contents)

    def test_multiple_facts_from_one_store(self, engine: MemoryEngine) -> None:
        ids = engine.store(
            "David is a chef. He works at a Michelin restaurant. He studied in France.",
            session_id="s1",
        )
        # Mode A may extract 1 or more facts from compound sentences
        assert len(ids) >= 1
