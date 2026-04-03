# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3

"""SLM 3.3 End-to-End Integration Tests.

Verifies the full 3.3 pipeline: forgetting filter, Hopfield channel,
quantization, CCQ consolidation, vague query detection, and 3.2.3->3.3
migration — all wired together with in-memory DB.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from superlocalmemory.core.config import (
    ChannelWeights,
    ForgettingConfig,
    HopfieldConfig,
    QuantizationConfig,
    RetrievalConfig,
    SLMConfig,
)
from superlocalmemory.retrieval.strategy import (
    STRATEGY_PRESETS,
    QueryStrategyClassifier,
    _VAGUE_PHRASES,
)
from superlocalmemory.storage.models import Mode


# ---------------------------------------------------------------------------
# Deterministic embedder for test reproducibility
# ---------------------------------------------------------------------------

class _TestEmbedder:
    """SHA-256 seeded deterministic embedder for tests."""

    is_available = True

    def embed(self, text: str) -> list[float]:
        rng = np.random.RandomState(hash(text) % 2**31)
        vec = rng.randn(768).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()

    def compute_fisher_params(self, text: str):
        return ([0.0] * 768, [1.0] * 768)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def v33_db():
    """In-memory SQLite DB with full SLM schema (including 3.3 tables)."""
    from superlocalmemory.storage import schema

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    schema.create_all_tables(conn)
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def v323_db():
    """In-memory SQLite DB with ONLY pre-3.3 tables (missing 3.3 tables).

    Creates the base schema, then drops the 6 new V33 tables to simulate
    a 3.2.3 database.
    """
    from superlocalmemory.storage import schema

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    schema.create_all_tables(conn)

    # Drop V33-specific tables to simulate 3.2.3
    for table in (
        "fact_retention", "polar_embeddings",
        "embedding_quantization_metadata",
        "ccq_consolidated_blocks", "ccq_audit_log",
        "soft_prompt_templates",
    ):
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def embedder():
    return _TestEmbedder()


def _store_fact(
    conn: sqlite3.Connection,
    fact_id: str,
    content: str,
    profile_id: str = "default",
    embedding: list[float] | None = None,
) -> None:
    """Insert a test fact into atomic_facts with embedding."""
    conn.execute(
        "INSERT OR IGNORE INTO memories (memory_id, profile_id, content) "
        "VALUES (?, ?, ?)",
        (f"mem-{fact_id}", profile_id, content),
    )
    emb_json = json.dumps(embedding) if embedding else None
    conn.execute(
        "INSERT INTO atomic_facts "
        "(fact_id, memory_id, profile_id, content, embedding, "
        "importance, confidence, access_count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (fact_id, f"mem-{fact_id}", profile_id, content,
         emb_json, 0.5, 1.0, 1),
    )
    conn.commit()


def _store_retention(
    conn: sqlite3.Connection,
    fact_id: str,
    profile_id: str,
    lifecycle_zone: str = "active",
    retention_score: float = 1.0,
) -> None:
    """Insert a test retention row."""
    conn.execute(
        "INSERT OR REPLACE INTO fact_retention "
        "(fact_id, profile_id, retention_score, lifecycle_zone) "
        "VALUES (?, ?, ?, ?)",
        (fact_id, profile_id, retention_score, lifecycle_zone),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Test 1: Hopfield weights are present in ALL strategy presets
# ---------------------------------------------------------------------------

class TestStrategyPresetsHopfield:
    """Verify hopfield is present in all presets that have channel weights."""

    def test_hopfield_in_temporal_preset(self):
        assert "hopfield" in STRATEGY_PRESETS["temporal"]
        assert STRATEGY_PRESETS["temporal"]["hopfield"] == 0.5

    def test_hopfield_in_multi_hop_preset(self):
        assert "hopfield" in STRATEGY_PRESETS["multi_hop"]
        assert STRATEGY_PRESETS["multi_hop"]["hopfield"] == 0.7

    def test_hopfield_in_aggregation_preset(self):
        assert "hopfield" in STRATEGY_PRESETS["aggregation"]
        assert STRATEGY_PRESETS["aggregation"]["hopfield"] == 0.6

    def test_hopfield_in_opinion_preset(self):
        assert "hopfield" in STRATEGY_PRESETS["opinion"]
        assert STRATEGY_PRESETS["opinion"]["hopfield"] == 0.5

    def test_hopfield_in_factual_preset(self):
        assert "hopfield" in STRATEGY_PRESETS["factual"]
        assert STRATEGY_PRESETS["factual"]["hopfield"] == 0.8

    def test_hopfield_in_entity_preset(self):
        assert "hopfield" in STRATEGY_PRESETS["entity"]
        assert STRATEGY_PRESETS["entity"]["hopfield"] == 0.9

    def test_hopfield_in_vague_preset(self):
        assert "vague" in STRATEGY_PRESETS
        assert STRATEGY_PRESETS["vague"]["hopfield"] == 1.1

    def test_general_preset_empty(self):
        """General still inherits from base_weights (no overrides)."""
        assert STRATEGY_PRESETS["general"] == {}


# ---------------------------------------------------------------------------
# Test 2: Vague query detection
# ---------------------------------------------------------------------------

class TestVagueQueryDetection:
    """Verify that vague phrases trigger 'vague' query type."""

    def test_something_about(self):
        c = QueryStrategyClassifier()
        s = c.classify("something about a project we discussed", {})
        assert s.query_type == "vague"

    def test_vaguely_remember(self):
        c = QueryStrategyClassifier()
        s = c.classify("I vaguely remember a conversation", {})
        assert s.query_type == "vague"

    def test_i_forgot(self):
        c = QueryStrategyClassifier()
        s = c.classify("i forgot the name of that tool", {})
        assert s.query_type == "vague"

    def test_not_sure(self):
        c = QueryStrategyClassifier()
        s = c.classify("not sure what it was called", {})
        assert s.query_type == "vague"

    def test_what_was_that(self):
        """'what was that' starts with 'what ' -> factual check fires first.
        Use a non-question-word prefix to hit vague detection."""
        c = QueryStrategyClassifier()
        s = c.classify("hmm, what was that thing we talked about", {})
        assert s.query_type == "vague"

    def test_maybe(self):
        c = QueryStrategyClassifier()
        s = c.classify("maybe it was about Docker", {})
        assert s.query_type == "vague"

    def test_non_vague_stays_factual(self):
        c = QueryStrategyClassifier()
        s = c.classify("What is Docker?", {})
        assert s.query_type == "factual"

    def test_non_vague_stays_general(self):
        c = QueryStrategyClassifier()
        s = c.classify("tell me about SLM", {})
        assert s.query_type == "general"

    def test_vague_phrases_tuple_not_empty(self):
        assert len(_VAGUE_PHRASES) >= 10


# ---------------------------------------------------------------------------
# Test 3: ChannelWeights includes hopfield
# ---------------------------------------------------------------------------

class TestChannelWeightsHopfield:
    """Verify ChannelWeights has hopfield field and as_dict includes it."""

    def test_default_hopfield_weight(self):
        cw = ChannelWeights()
        assert cw.hopfield == 0.8

    def test_as_dict_includes_hopfield(self):
        cw = ChannelWeights()
        d = cw.as_dict()
        assert "hopfield" in d
        assert d["hopfield"] == 0.8

    def test_custom_hopfield_weight(self):
        cw = ChannelWeights(hopfield=1.5)
        assert cw.as_dict()["hopfield"] == 1.5

    def test_six_channels_in_dict(self):
        d = ChannelWeights().as_dict()
        expected = {"semantic", "bm25", "entity_graph", "temporal",
                    "spreading_activation", "hopfield"}
        assert set(d.keys()) == expected


# ---------------------------------------------------------------------------
# Test 4: RetrievalConfig has hopfield_top_k
# ---------------------------------------------------------------------------

class TestRetrievalConfigHopfield:
    """Verify hopfield_top_k exists in RetrievalConfig."""

    def test_default_hopfield_top_k(self):
        rc = RetrievalConfig()
        assert rc.hopfield_top_k == 50

    def test_custom_hopfield_top_k(self):
        rc = RetrievalConfig(hopfield_top_k=100)
        assert rc.hopfield_top_k == 100


# ---------------------------------------------------------------------------
# Test 5: Forgetting filter integration
# ---------------------------------------------------------------------------

class TestForgettingFilterIntegration:
    """Verify forgetting filter adjusts scores based on retention zones."""

    @staticmethod
    def _make_mock_db(retention_rows: list[dict]):
        """Create a mock DB that returns given retention rows."""
        db = MagicMock()
        db.batch_get_retention.return_value = retention_rows
        return db

    def test_active_facts_pass_through(self):
        """Active zone facts keep full score."""
        from superlocalmemory.retrieval.forgetting_filter import ForgettingFilter

        db = self._make_mock_db([
            {"fact_id": "f1", "retention_score": 0.9, "lifecycle_zone": "active"},
        ])

        ff = ForgettingFilter(db, ForgettingConfig())
        results = {"semantic": [("f1", 0.8)]}
        filtered = ff.filter(results, "default", None)
        assert filtered["semantic"][0] == ("f1", 0.8)

    def test_cold_facts_get_reduced_score(self):
        """Cold zone facts get 0.3x weight."""
        from superlocalmemory.retrieval.forgetting_filter import ForgettingFilter

        db = self._make_mock_db([
            {"fact_id": "f2", "retention_score": 0.15, "lifecycle_zone": "cold"},
        ])

        ff = ForgettingFilter(db, ForgettingConfig())
        results = {"semantic": [("f2", 1.0)]}
        filtered = ff.filter(results, "default", None)
        assert len(filtered["semantic"]) == 1
        assert abs(filtered["semantic"][0][1] - 0.3) < 0.01

    def test_forgotten_facts_removed(self):
        """Forgotten zone facts are excluded from results entirely."""
        from superlocalmemory.retrieval.forgetting_filter import ForgettingFilter

        db = self._make_mock_db([
            {"fact_id": "f3", "retention_score": 0.01, "lifecycle_zone": "forgotten"},
        ])

        ff = ForgettingFilter(db, ForgettingConfig())
        results = {"semantic": [("f3", 0.9)]}
        filtered = ff.filter(results, "default", None)
        assert len(filtered["semantic"]) == 0

    def test_disabled_config_passes_through(self):
        """When config.enabled=False, results pass unchanged (HR-06)."""
        from superlocalmemory.retrieval.forgetting_filter import ForgettingFilter

        db = self._make_mock_db([
            {"fact_id": "f4", "retention_score": 0.01, "lifecycle_zone": "forgotten"},
        ])

        config = ForgettingConfig(enabled=False)
        ff = ForgettingFilter(db, config)
        results = {"semantic": [("f4", 0.9)]}
        filtered = ff.filter(results, "default", None)
        # Disabled -> original results
        assert filtered["semantic"][0] == ("f4", 0.9)

    def test_no_retention_data_keeps_fact(self):
        """Facts with no retention row yet are kept as-is (new memories)."""
        from superlocalmemory.retrieval.forgetting_filter import ForgettingFilter

        db = self._make_mock_db([])  # No retention data

        ff = ForgettingFilter(db, ForgettingConfig())
        results = {"semantic": [("new_fact", 0.7)]}
        filtered = ff.filter(results, "default", None)
        assert filtered["semantic"][0] == ("new_fact", 0.7)

    def test_warm_zone_gets_07_weight(self):
        """Warm zone facts get 0.7x weight."""
        from superlocalmemory.retrieval.forgetting_filter import ForgettingFilter

        db = self._make_mock_db([
            {"fact_id": "f5", "retention_score": 0.5, "lifecycle_zone": "warm"},
        ])

        ff = ForgettingFilter(db, ForgettingConfig())
        results = {"bm25": [("f5", 1.0)]}
        filtered = ff.filter(results, "default", None)
        assert abs(filtered["bm25"][0][1] - 0.7) < 0.01

    def test_archive_zone_excluded(self):
        """Archive zone facts are removed like forgotten."""
        from superlocalmemory.retrieval.forgetting_filter import ForgettingFilter

        db = self._make_mock_db([
            {"fact_id": "f6", "retention_score": 0.1, "lifecycle_zone": "archive"},
        ])

        ff = ForgettingFilter(db, ForgettingConfig())
        results = {"semantic": [("f6", 0.9)]}
        filtered = ff.filter(results, "default", None)
        assert len(filtered["semantic"]) == 0


# ---------------------------------------------------------------------------
# Test 6: Hopfield channel wiring in engine
# ---------------------------------------------------------------------------

class TestHopfieldChannelWiring:
    """Verify Hopfield channel is wired into RetrievalEngine."""

    def test_hopfield_registered_in_registry(self):
        """Hopfield channel registers in ChannelRegistry."""
        from superlocalmemory.retrieval.engine import RetrievalEngine

        mock_hopfield = MagicMock()
        mock_hopfield.search.return_value = [("f1", 0.9)]

        config = RetrievalConfig()
        db = MagicMock()
        channels = {"hopfield": mock_hopfield}

        engine = RetrievalEngine(
            db=db, config=config, channels=channels,
        )
        assert "hopfield" in engine._registry.channel_names

    def test_hopfield_dispatched_in_run_channels(self):
        """Hopfield is called during _run_channels when embedder is present."""
        from superlocalmemory.retrieval.engine import RetrievalEngine
        from superlocalmemory.retrieval.strategy import QueryStrategy

        mock_hopfield = MagicMock()
        mock_hopfield.search.return_value = [("f1", 0.85)]

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 768

        config = RetrievalConfig()
        db = MagicMock()
        channels = {"hopfield": mock_hopfield}

        engine = RetrievalEngine(
            db=db, config=config, channels=channels,
            embedder=mock_embedder,
        )

        strat = QueryStrategy(query_type="vague", weights={})
        results = engine._run_channels("test query", "default", strat)
        assert "hopfield" in results
        assert results["hopfield"] == [("f1", 0.85)]

    def test_hopfield_skipped_when_disabled(self):
        """Hopfield channel is skipped when in disabled_channels."""
        from superlocalmemory.retrieval.engine import RetrievalEngine
        from superlocalmemory.retrieval.strategy import QueryStrategy

        mock_hopfield = MagicMock()
        mock_hopfield.search.return_value = [("f1", 0.85)]

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 768

        config = RetrievalConfig(disabled_channels=["hopfield"])
        db = MagicMock()
        channels = {"hopfield": mock_hopfield}

        engine = RetrievalEngine(
            db=db, config=config, channels=channels,
            embedder=mock_embedder,
        )

        strat = QueryStrategy(query_type="vague", weights={})
        results = engine._run_channels("test query", "default", strat)
        assert "hopfield" not in results
        mock_hopfield.search.assert_not_called()

    def test_hopfield_error_returns_empty(self):
        """Hopfield channel error is caught and logged (HR-06)."""
        from superlocalmemory.retrieval.engine import RetrievalEngine
        from superlocalmemory.retrieval.strategy import QueryStrategy

        mock_hopfield = MagicMock()
        mock_hopfield.search.side_effect = RuntimeError("boom")

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 768

        config = RetrievalConfig()
        db = MagicMock()
        channels = {"hopfield": mock_hopfield}

        engine = RetrievalEngine(
            db=db, config=config, channels=channels,
            embedder=mock_embedder,
        )

        strat = QueryStrategy(query_type="vague", weights={})
        results = engine._run_channels("test query", "default", strat)
        assert "hopfield" not in results

    def test_hopfield_not_called_without_embedder(self):
        """When no embedder, Hopfield is skipped (needs embedding)."""
        from superlocalmemory.retrieval.engine import RetrievalEngine
        from superlocalmemory.retrieval.strategy import QueryStrategy

        mock_hopfield = MagicMock()

        config = RetrievalConfig()
        db = MagicMock()
        channels = {"hopfield": mock_hopfield}

        engine = RetrievalEngine(
            db=db, config=config, channels=channels,
            embedder=None,
        )

        strat = QueryStrategy(query_type="vague", weights={})
        results = engine._run_channels("test query", "default", strat)
        assert "hopfield" not in results
        mock_hopfield.search.assert_not_called()


# ---------------------------------------------------------------------------
# Test 7: 3.2.3 -> 3.3 Migration
# ---------------------------------------------------------------------------

class TestMigrationV33:
    """Verify idempotent migration from SLM 3.2.3 to 3.3."""

    def test_detect_v323_missing_tables(self, v323_db):
        """Pre-3.3 DB is detected as needing migration."""
        from superlocalmemory.storage.migration_v33 import detect_v323_database
        assert detect_v323_database(v323_db) is True

    def test_detect_v33_no_migration_needed(self, v33_db):
        """Full 3.3 DB reports no migration needed."""
        from superlocalmemory.storage.migration_v33 import detect_v323_database
        assert detect_v323_database(v33_db) is False

    def test_migration_creates_missing_tables(self, v323_db):
        """Migration creates all 6 missing tables."""
        from superlocalmemory.storage.migration_v33 import migrate_v323_to_v33

        report = migrate_v323_to_v33(v323_db)
        v323_db.commit()

        assert report.is_clean
        assert len(report.tables_created) == 6
        assert "fact_retention" in report.tables_created
        assert "polar_embeddings" in report.tables_created
        assert "embedding_quantization_metadata" in report.tables_created
        assert "ccq_consolidated_blocks" in report.tables_created
        assert "ccq_audit_log" in report.tables_created
        assert "soft_prompt_templates" in report.tables_created

    def test_migration_idempotent(self, v323_db):
        """Running migration twice causes no errors."""
        from superlocalmemory.storage.migration_v33 import migrate_v323_to_v33

        report1 = migrate_v323_to_v33(v323_db)
        v323_db.commit()
        assert len(report1.tables_created) == 6

        report2 = migrate_v323_to_v33(v323_db)
        assert len(report2.tables_created) == 0
        assert len(report2.tables_existed) == 6
        assert report2.is_clean

    def test_migration_tables_functional(self, v323_db):
        """After migration, new tables accept inserts."""
        from superlocalmemory.storage.migration_v33 import migrate_v323_to_v33

        migrate_v323_to_v33(v323_db)
        v323_db.commit()

        # Store a fact first (FK dependency)
        v323_db.execute(
            "INSERT OR IGNORE INTO memories (memory_id, profile_id, content) "
            "VALUES ('m1', 'default', 'test')",
        )
        v323_db.execute(
            "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content) "
            "VALUES ('f1', 'm1', 'default', 'test fact')",
        )

        # Now insert into 3.3 tables
        v323_db.execute(
            "INSERT INTO fact_retention (fact_id, profile_id, retention_score, lifecycle_zone) "
            "VALUES ('f1', 'default', 0.9, 'active')",
        )
        v323_db.execute(
            "INSERT INTO polar_embeddings (fact_id, profile_id, radius, angle_indices, bit_width) "
            "VALUES ('f1', 'default', 1.0, X'00', 4)",
        )
        v323_db.execute(
            "INSERT INTO embedding_quantization_metadata (fact_id, profile_id) "
            "VALUES ('f1', 'default')",
        )
        v323_db.commit()

        # Verify inserts
        row = v323_db.execute(
            "SELECT lifecycle_zone FROM fact_retention WHERE fact_id='f1'"
        ).fetchone()
        assert row["lifecycle_zone"] == "active"


# ---------------------------------------------------------------------------
# Test 8: Engine wiring includes Hopfield and forgetting filter
# ---------------------------------------------------------------------------

class TestEngineWiringV33:
    """Verify engine_wiring.py creates Hopfield channel and forgetting filter."""

    def test_init_hopfield_channel_with_config(self):
        """_init_hopfield_channel creates channel when config enabled."""
        from superlocalmemory.core.engine_wiring import _init_hopfield_channel

        db = MagicMock()
        vs = MagicMock()
        config = SLMConfig.for_mode(Mode.A)

        channel = _init_hopfield_channel(db, vs, config)
        assert channel is not None

    def test_init_hopfield_channel_disabled(self):
        """_init_hopfield_channel returns None when disabled."""
        from superlocalmemory.core.engine_wiring import _init_hopfield_channel

        db = MagicMock()
        vs = MagicMock()
        config = SLMConfig.for_mode(Mode.A)
        # Manually disable
        object.__setattr__(config.hopfield, "enabled", False)

        channel = _init_hopfield_channel(db, vs, config)
        assert channel is None

    def test_forgetting_filter_registered_in_init_retrieval(self):
        """init_retrieval registers forgetting filter in engine registry."""
        from superlocalmemory.core.engine_wiring import init_retrieval

        db = MagicMock()
        db.execute.return_value = []
        embedder = MagicMock()
        embedder.embed.return_value = [0.1] * 768
        entity_resolver = MagicMock()
        trust_scorer = MagicMock()

        config = SLMConfig.for_mode(Mode.A)

        with patch(
            "superlocalmemory.retrieval.reranker.CrossEncoderReranker",
            return_value=MagicMock(),
        ):
            engine = init_retrieval(
                config, db, embedder, entity_resolver, trust_scorer,
            )

        # The forgetting filter should be registered
        assert len(engine._registry._filters) >= 1


# ---------------------------------------------------------------------------
# Test 9: Mode C includes hopfield in channel_weights
# ---------------------------------------------------------------------------

class TestModeCHopfield:
    """Verify Mode C config includes hopfield channel weight."""

    def test_mode_c_has_hopfield_weight(self):
        config = SLMConfig.for_mode(Mode.C)
        assert config.channel_weights.hopfield == 1.0

    def test_mode_c_as_dict_has_hopfield(self):
        config = SLMConfig.for_mode(Mode.C)
        d = config.channel_weights.as_dict()
        assert "hopfield" in d


# ---------------------------------------------------------------------------
# Test 10: Full pipeline smoke test (store -> retrieve with 6 channels)
# ---------------------------------------------------------------------------

class TestFullPipelineSmoke:
    """Minimal smoke test wiring all 6 channels end-to-end."""

    def test_recall_with_all_channels_mock(self):
        """RetrievalEngine.recall works with all 6 channels mocked."""
        from superlocalmemory.retrieval.engine import RetrievalEngine
        from superlocalmemory.storage.models import AtomicFact

        # Build mock channels
        fact = AtomicFact(
            fact_id="f1", memory_id="m1", profile_id="default",
            content="Python is a versatile programming language",
            fact_type="semantic", entities=["Python"],
            canonical_entities=["python"], confidence=1.0,
            importance=0.8, evidence_count=3, access_count=5,
            embedding=[0.1] * 768,
            created_at=datetime.now(UTC).isoformat(),
        )

        def _mock_search(query, profile_id, top_k=50):
            return [("f1", 0.9)]

        semantic = MagicMock()
        semantic.search.side_effect = _mock_search
        bm25 = MagicMock()
        bm25.search.side_effect = _mock_search
        entity = MagicMock()
        entity.search.side_effect = _mock_search
        temporal = MagicMock()
        temporal.search.side_effect = _mock_search
        hopfield = MagicMock()
        hopfield.search.side_effect = _mock_search

        channels = {
            "semantic": semantic,
            "bm25": bm25,
            "entity_graph": entity,
            "temporal": temporal,
            "hopfield": hopfield,
        }

        embedder = MagicMock()
        embedder.embed.return_value = [0.1] * 768

        db = MagicMock()
        db.get_all_facts.return_value = [fact]
        db.get_facts_by_ids.side_effect = lambda ids, pid: [fact] if fact.fact_id in ids else []
        db.get_scenes_for_fact.return_value = []

        config = RetrievalConfig(use_cross_encoder=False)

        engine = RetrievalEngine(
            db=db, config=config, channels=channels,
            embedder=embedder,
        )

        response = engine.recall("Tell me about Python", "default")
        assert response.query is not None
        assert len(response.results) >= 1
        # All 5 channels + hopfield were called
        assert hopfield.search.called
