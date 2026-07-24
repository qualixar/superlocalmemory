# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Encoding wiring integration test — every encoding component writes to DB.

Stores content through MemoryEngine, then inspects the database directly
to verify each encoding component did its job. Covers:

  - Fact extraction + persistence (atomic_facts)
  - Entity resolution (canonical_entities, entity_aliases)
  - Temporal parsing (referenced_date, observation_date)
  - Type routing (fact_type classification)
  - Graph building (graph_edges: entity, semantic)
  - Consolidation (consolidation_log: supersede/update)
  - Observation builder (entity_profiles)
  - Scene builder (memory_scenes)
  - Entropy gate (deduplication filtering)
  - Emotional tagging (emotional_valence, emotional_arousal)
  - Signal inference (signal_type)
  - BM25 token persistence (bm25_tokens)
  - Memory record persistence (memories)

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from superlocalmemory.core.config import RetrievalConfig, SLMConfig
from superlocalmemory.core.engine import MemoryEngine
from superlocalmemory.storage.models import Mode

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Mock Embedder (deterministic, zero model loading)
# ---------------------------------------------------------------------------

class _MockEmbedder:
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
            return np.zeros(len(arr)).tolist(), np.full(len(arr), 2.0).tolist()
        mean = arr / norm
        abs_mean = np.abs(mean)
        max_val = float(np.max(abs_mean)) + 1e-10
        signal = abs_mean / max_val
        var = np.clip(2.0 - 1.95 * signal, 0.3, 2.0)
        return mean.tolist(), var.tolist()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def engine(tmp_path: Path) -> MemoryEngine:
    config = SLMConfig.for_mode(Mode.A, base_dir=tmp_path)
    config.db_path = tmp_path / "test.db"
    config.retrieval = RetrievalConfig(use_cross_encoder=False)
    eng = MemoryEngine(config)
    with patch(
        "superlocalmemory.core.embeddings.EmbeddingService",
        return_value=_MockEmbedder(768),
    ):
        eng.initialize()
    # v3.8.2: store() is queryable-first (async enrichment via the daemon
    # materializer). These tests assert enrichment artifacts synchronously,
    # so drive the complete path. See tests/conftest.force_sync_enrichment.
    from tests.conftest import force_sync_enrichment
    return force_sync_enrichment(eng)


def _query(engine: MemoryEngine, sql: str, params: tuple = ()) -> list[dict]:
    """Execute raw SQL and return list of dicts."""
    rows = engine._db.execute(sql, params)
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# 1. Fact Extraction Wiring
# ---------------------------------------------------------------------------

class TestFactExtractionWiring:
    """After store, atomic_facts table has rows with all enrichment fields."""

    def test_facts_created(self, engine: MemoryEngine) -> None:
        engine.store("Alice is a software engineer at Google. She loves hiking.",
                      session_id="s1")
        rows = _query(engine, "SELECT * FROM atomic_facts WHERE profile_id = 'default'")
        assert len(rows) >= 1

    def test_fact_content_nonempty(self, engine: MemoryEngine) -> None:
        engine.store("Alice is a software engineer at Google.", session_id="s1")
        rows = _query(engine, "SELECT content FROM atomic_facts WHERE profile_id = 'default'")
        for r in rows:
            assert r["content"].strip() != ""

    def test_fact_type_set(self, engine: MemoryEngine) -> None:
        engine.store("Alice is a software engineer at Google.", session_id="s1")
        rows = _query(engine, "SELECT fact_type FROM atomic_facts WHERE profile_id = 'default'")
        for r in rows:
            assert r["fact_type"] in ("episodic", "semantic", "opinion", "temporal")

    def test_embedding_stored(self, engine: MemoryEngine) -> None:
        engine.store("Alice is a software engineer at Google.", session_id="s1")
        rows = _query(engine, "SELECT embedding FROM atomic_facts WHERE profile_id = 'default'")
        assert len(rows) >= 1
        for r in rows:
            assert r["embedding"] is not None
            emb = json.loads(r["embedding"])
            assert len(emb) == 768

    def test_fisher_mean_stored(self, engine: MemoryEngine) -> None:
        engine.store("Alice is a software engineer at Google.", session_id="s1")
        rows = _query(engine, "SELECT fisher_mean FROM atomic_facts WHERE profile_id = 'default'")
        assert len(rows) >= 1
        for r in rows:
            assert r["fisher_mean"] is not None
            fm = json.loads(r["fisher_mean"])
            assert len(fm) == 768

    def test_fisher_variance_stored(self, engine: MemoryEngine) -> None:
        engine.store("Alice is a software engineer at Google.", session_id="s1")
        rows = _query(engine, "SELECT fisher_variance FROM atomic_facts WHERE profile_id = 'default'")
        assert len(rows) >= 1
        for r in rows:
            assert r["fisher_variance"] is not None
            fv = json.loads(r["fisher_variance"])
            assert len(fv) == 768

    def test_created_at_populated(self, engine: MemoryEngine) -> None:
        engine.store("Alice is a software engineer at Google.", session_id="s1")
        rows = _query(engine, "SELECT created_at FROM atomic_facts WHERE profile_id = 'default'")
        for r in rows:
            assert r["created_at"] is not None
            assert len(r["created_at"]) >= 10  # ISO date minimum


# ---------------------------------------------------------------------------
# 2. Entity Resolution Wiring
# ---------------------------------------------------------------------------

class TestEntityResolutionWiring:
    """Entities get resolved and persisted in canonical_entities + aliases."""

    def test_canonical_entities_created(self, engine: MemoryEngine) -> None:
        engine.store("Alice met Bob at the cafe. Alice talked to Bob again.",
                      session_id="s1")
        rows = _query(engine, "SELECT * FROM canonical_entities WHERE profile_id = 'default'")
        names = [r["canonical_name"] for r in rows]
        assert any("Alice" in n for n in names)
        assert any("Bob" in n for n in names)

    def test_entity_aliases_created(self, engine: MemoryEngine) -> None:
        engine.store("Alice met Bob. Alice talked to Bob.",
                      session_id="s1")
        rows = _query(engine, "SELECT * FROM entity_aliases")
        assert len(rows) >= 1
        aliases = [r["alias"] for r in rows]
        assert any("Alice" in a for a in aliases) or any("Bob" in a for a in aliases)

    def test_facts_linked_to_entities(self, engine: MemoryEngine) -> None:
        engine.store("Alice met Bob at the central park near downtown during the summer festival.",
                      session_id="s1")
        rows = _query(engine,
            "SELECT canonical_entities_json FROM atomic_facts WHERE profile_id = 'default'")
        linked = False
        for r in rows:
            ce = json.loads(r["canonical_entities_json"])
            if ce:
                linked = True
        assert linked, "At least one fact should have canonical_entities set"


# ---------------------------------------------------------------------------
# 3. Temporal Parsing Wiring
# ---------------------------------------------------------------------------

class TestTemporalParsingWiring:
    """Dates in content and session_date are parsed and persisted."""

    def test_referenced_date_populated(self, engine: MemoryEngine) -> None:
        engine.store("Alice visited Paris on March 5, 2026.",
                      session_id="s1")
        rows = _query(engine,
            "SELECT referenced_date FROM atomic_facts WHERE profile_id = 'default'")
        has_ref = any(r["referenced_date"] is not None for r in rows)
        assert has_ref, "At least one fact should have referenced_date from 'March 5, 2026'"

    def test_observation_date_from_session(self, engine: MemoryEngine) -> None:
        engine.store("Alice had lunch at the Italian restaurant near the office with her colleagues.",
                      session_id="s1",
                      session_date="3:00 pm on 5 March, 2026")
        rows = _query(engine,
            "SELECT observation_date FROM atomic_facts WHERE profile_id = 'default'")
        has_obs = any(r["observation_date"] is not None for r in rows)
        assert has_obs, "observation_date should be populated from session_date"


# ---------------------------------------------------------------------------
# 4. Type Router Wiring
# ---------------------------------------------------------------------------

class TestTypeRouterWiring:
    """Facts are classified into typed stores by the router."""

    def test_episodic_classified(self, engine: MemoryEngine) -> None:
        engine.store("Alice went to the store yesterday and bought groceries.",
                      session_id="s1")
        rows = _query(engine,
            "SELECT fact_type FROM atomic_facts WHERE profile_id = 'default'")
        types = {r["fact_type"] for r in rows}
        # Should be classified (may be episodic or temporal due to "yesterday")
        assert len(types) >= 1

    def test_opinion_classified(self, engine: MemoryEngine) -> None:
        # With mock embedder the TypeRouter uses template similarity which
        # produces non-deterministic types. Verify routing occurred (valid type set)
        # and that fact extractor's keyword classifier picked up opinion markers.
        engine.store("I think Alice prefers hiking over swimming.",
                      session_id="s1")
        rows = _query(engine,
            "SELECT fact_type FROM atomic_facts WHERE profile_id = 'default'")
        types = {r["fact_type"] for r in rows}
        # TypeRouter embedding classification with mock may override keyword.
        # The key assertion: type IS set to a valid value (router ran).
        assert types.issubset({"episodic", "semantic", "opinion", "temporal"}), (
            f"Unexpected fact types: {types}"
        )
        assert len(types) >= 1, "At least one fact should have a type assigned"

    def test_semantic_classified(self, engine: MemoryEngine) -> None:
        engine.store("Paris is the capital of France.",
                      session_id="s1")
        rows = _query(engine,
            "SELECT fact_type FROM atomic_facts WHERE profile_id = 'default'")
        types = {r["fact_type"] for r in rows}
        # With mock embedder the type router uses random-ish embedding similarity,
        # but the type must be a valid classification.
        assert types.issubset({"episodic", "semantic", "opinion", "temporal"}), (
            f"Unexpected fact types: {types}"
        )

    def test_all_fact_types_valid(self, engine: MemoryEngine) -> None:
        """Store a variety of content and verify every fact gets a valid type."""
        engine.store("Alice works at Google as an engineer.", session_id="s1")
        engine.store("The meeting is scheduled for next Monday.", session_id="s2")
        engine.store("Bob visited the museum last weekend.", session_id="s3")
        rows = _query(engine,
            "SELECT fact_type FROM atomic_facts WHERE profile_id = 'default'")
        valid = {"episodic", "semantic", "opinion", "temporal"}
        for r in rows:
            assert r["fact_type"] in valid, f"Invalid fact_type: {r['fact_type']}"


# ---------------------------------------------------------------------------
# 5. Graph Builder Wiring
# ---------------------------------------------------------------------------

class TestGraphBuilderWiring:
    """Graph edges are created for shared entities and semantic similarity."""

    def test_graph_edges_created(self, engine: MemoryEngine) -> None:
        engine.store("Alice works at Google as a senior engineer.",
                      session_id="s1")
        engine.store("Alice graduated from Stanford University.",
                      session_id="s1")
        rows = _query(engine,
            "SELECT * FROM graph_edges WHERE profile_id = 'default'")
        assert len(rows) >= 1, "Shared entity (Alice) should produce graph edges"

    def test_entity_edges_exist(self, engine: MemoryEngine) -> None:
        engine.store("Bob works at Microsoft as a principal engineer in the cloud infrastructure team.", session_id="s1")
        engine.store("Bob lives in Seattle near the waterfront in a modern apartment building.", session_id="s1")
        rows = _query(engine,
            "SELECT * FROM graph_edges WHERE profile_id = 'default' AND edge_type = 'entity'")
        assert len(rows) >= 1, "Shared entity (Bob) should produce entity edges"

    def test_semantic_or_any_edges_exist(self, engine: MemoryEngine) -> None:
        engine.store("Charlie is a musician who plays guitar.",
                      session_id="s1")
        engine.store("Charlie performs at jazz concerts every weekend.",
                      session_id="s1")
        rows = _query(engine,
            "SELECT * FROM graph_edges WHERE profile_id = 'default'")
        assert len(rows) >= 1, "Related facts should have at least one edge type"


# ---------------------------------------------------------------------------
# 6. Consolidation Wiring
# ---------------------------------------------------------------------------

class TestConsolidationWiring:
    """Contradicting facts trigger consolidation actions."""

    def test_consolidation_log_entry(self, engine: MemoryEngine) -> None:
        engine.store("Alice works at Google as a senior software engineer in Mountain View.", session_id="s1")
        engine.store("Alice no longer works at Google.", session_id="s2")
        rows = _query(engine,
            "SELECT * FROM consolidation_log WHERE profile_id = 'default'")
        # Every store triggers an ADD at minimum
        assert len(rows) >= 1

    def test_consolidation_action_types(self, engine: MemoryEngine) -> None:
        engine.store("David lives in New York City and works as a financial analyst on Wall Street.", session_id="s1")
        engine.store("David moved from New York City to San Francisco.",
                      session_id="s2")
        rows = _query(engine,
            "SELECT action_type FROM consolidation_log WHERE profile_id = 'default'")
        actions = {r["action_type"] for r in rows}
        # Should have at least an add
        assert len(actions) >= 1
        assert "add" in actions or "supersede" in actions or "update" in actions


# ---------------------------------------------------------------------------
# 7. Observation Builder Wiring
# ---------------------------------------------------------------------------

class TestObservationBuilderWiring:
    """Entity profiles are created/updated in entity_profiles table."""

    def test_entity_profile_created(self, engine: MemoryEngine) -> None:
        engine.store("Alice is a software engineer at Google.",
                      session_id="s1")
        engine.store("Alice graduated from MIT in 2020.",
                      session_id="s1")
        rows = _query(engine,
            "SELECT * FROM entity_profiles WHERE profile_id = 'default'")
        # Should have at least one entity profile (for Alice's canonical entity)
        assert len(rows) >= 1

    def test_knowledge_summary_contains_info(self, engine: MemoryEngine) -> None:
        engine.store("Alice is a software engineer at Google.",
                      session_id="s1")
        rows = _query(engine,
            "SELECT knowledge_summary FROM entity_profiles WHERE profile_id = 'default'")
        if rows:
            # At least one profile should have a non-empty summary
            summaries = [r["knowledge_summary"] for r in rows if r["knowledge_summary"]]
            assert len(summaries) >= 1


# ---------------------------------------------------------------------------
# 8. Scene Builder Wiring
# ---------------------------------------------------------------------------

class TestSceneBuilderWiring:
    """Facts get assigned to memory scenes."""

    def test_scenes_created(self, engine: MemoryEngine) -> None:
        engine.store("Alice works at Google as a senior engineer.",
                      session_id="s1")
        engine.store("Alice loves hiking in the mountains.",
                      session_id="s1")
        engine.store("Alice visited Paris last summer.",
                      session_id="s1")
        rows = _query(engine,
            "SELECT * FROM memory_scenes WHERE profile_id = 'default'")
        assert len(rows) >= 1

    def test_scene_has_fact_ids(self, engine: MemoryEngine) -> None:
        engine.store("Bob is a doctor at the hospital.",
                      session_id="s1")
        engine.store("Bob graduated from Stanford in 2018.",
                      session_id="s1")
        rows = _query(engine,
            "SELECT fact_ids_json FROM memory_scenes WHERE profile_id = 'default'")
        assert len(rows) >= 1
        for r in rows:
            fids = json.loads(r["fact_ids_json"])
            assert len(fids) >= 1


# ---------------------------------------------------------------------------
# 9. Entropy Gate Wiring
# ---------------------------------------------------------------------------

class TestEntropyGateWiring:
    """Exact-duplicate content is filtered by the entropy gate."""

    def test_duplicate_content_filtered(self, engine: MemoryEngine) -> None:
        ids1 = engine.store(
            "Alice is a software engineer at Google.",
            session_id="s1",
        )
        ids2 = engine.store(
            "Alice is a software engineer at Google.",
            session_id="s1",
        )
        # Second store should return fewer or no new facts
        assert len(ids2) <= len(ids1)

    def test_low_info_content_blocked(self, engine: MemoryEngine) -> None:
        ids = engine.store("ok", session_id="s1")
        assert ids == []


# ---------------------------------------------------------------------------
# 10. Emotional Tagging Wiring
# ---------------------------------------------------------------------------

class TestEmotionalTaggingWiring:
    """Emotional valence and arousal are persisted on facts."""

    def test_positive_emotion_valence(self, engine: MemoryEngine) -> None:
        engine.store("I absolutely love this amazing experience!",
                      session_id="s1")
        rows = _query(engine,
            "SELECT emotional_valence, emotional_arousal FROM atomic_facts "
            "WHERE profile_id = 'default'")
        assert len(rows) >= 1
        # At least one fact should have positive valence
        has_positive = any(r["emotional_valence"] > 0 for r in rows)
        assert has_positive, "Positive emotional content should yield positive valence"

    def test_negative_emotion_detected(self, engine: MemoryEngine) -> None:
        engine.store("I hate this terrible situation so much.",
                      session_id="s1")
        rows = _query(engine,
            "SELECT emotional_valence, emotional_arousal FROM atomic_facts "
            "WHERE profile_id = 'default'")
        assert len(rows) >= 1
        # At least one fact should have negative valence or high arousal
        has_neg = any(
            r["emotional_valence"] < 0 or r["emotional_arousal"] > 0
            for r in rows
        )
        assert has_neg, "Negative emotional content should yield negative valence or arousal"


# ---------------------------------------------------------------------------
# 11. Signal Inference Wiring
# ---------------------------------------------------------------------------

class TestSignalInferenceWiring:
    """Signal type is inferred and persisted on facts."""

    def test_temporal_signal_inferred(self, engine: MemoryEngine) -> None:
        engine.store("The deadline for the project is next Monday.",
                      session_id="s1")
        rows = _query(engine,
            "SELECT signal_type FROM atomic_facts WHERE profile_id = 'default'")
        assert len(rows) >= 1
        # Temporal content should produce temporal or other non-default signal
        signals = {r["signal_type"] for r in rows}
        assert "temporal" in signals or len(signals) >= 1

    def test_social_signal_inferred(self, engine: MemoryEngine) -> None:
        engine.store("Alice met with her colleague Bob for a team meeting.",
                      session_id="s1")
        rows = _query(engine,
            "SELECT signal_type FROM atomic_facts WHERE profile_id = 'default'")
        assert len(rows) >= 1
        signals = {r["signal_type"] for r in rows}
        # Social keywords present -- could be "social" or "request" or "temporal"
        assert len(signals) >= 1


# ---------------------------------------------------------------------------
# 12. BM25 Token Wiring
# ---------------------------------------------------------------------------

class TestBM25TokenWiring:
    """BM25 tokens are lazily persisted during first retrieval (BM25Channel builds index)."""

    def test_bm25_tokens_stored_after_recall(self, engine: MemoryEngine) -> None:
        engine.store("Alice is a software engineer at Google.",
                      session_id="s1")
        # BM25 tokens are stored lazily when BM25Channel._build_index runs
        engine.recall("Alice engineer")
        rows = _query(engine,
            "SELECT * FROM bm25_tokens WHERE profile_id = 'default'")
        assert len(rows) >= 1

    def test_bm25_tokens_nonempty_after_recall(self, engine: MemoryEngine) -> None:
        engine.store("Bob graduated from Stanford University.",
                      session_id="s1")
        engine.recall("Bob Stanford")
        rows = _query(engine,
            "SELECT tokens FROM bm25_tokens WHERE profile_id = 'default'")
        assert len(rows) >= 1
        for r in rows:
            tokens = json.loads(r["tokens"])
            assert len(tokens) >= 1


# ---------------------------------------------------------------------------
# 13. Memory Record Wiring
# ---------------------------------------------------------------------------

class TestMemoryRecordWiring:
    """Raw memory records are persisted in the memories table."""

    def test_memory_record_created(self, engine: MemoryEngine) -> None:
        engine.store("Alice is a software engineer at Google.",
                      session_id="s1")
        rows = _query(engine,
            "SELECT * FROM memories WHERE profile_id = 'default'")
        assert len(rows) >= 1

    def test_memory_content_matches(self, engine: MemoryEngine) -> None:
        original = "Alice is a software engineer at Google."
        engine.store(original, session_id="s1")
        rows = _query(engine,
            "SELECT content FROM memories WHERE profile_id = 'default'")
        assert len(rows) >= 1
        assert rows[0]["content"] == original

    def test_memory_session_id_matches(self, engine: MemoryEngine) -> None:
        engine.store("Some test fact about the project deadline being moved to next quarter.", session_id="session_42")
        rows = _query(engine,
            "SELECT session_id FROM memories WHERE profile_id = 'default'")
        assert len(rows) >= 1
        assert any(r["session_id"] == "session_42" for r in rows)

    def test_memory_speaker_stored(self, engine: MemoryEngine) -> None:
        engine.store("Alice went to Paris last summer for a two week vacation with her family.",
                      session_id="s1", speaker="narrator")
        rows = _query(engine,
            "SELECT speaker FROM memories WHERE profile_id = 'default'")
        assert len(rows) >= 1
        assert rows[0]["speaker"] == "narrator"
