# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.core.config — SLMConfig + sub-configs.

Covers:
  - SLMConfig.for_mode(Mode.A/B/C) correct defaults
  - Mode A has no LLM
  - Mode C has cloud embeddings
  - EmbeddingConfig.is_cloud property
  - LLMConfig.is_available property
  - ChannelWeights.as_dict()
  - db_path auto-computed from base_dir
"""

from __future__ import annotations

from pathlib import Path

import pytest

from superlocalmemory.core.config import (
    DEFAULT_BASE_DIR,
    DEFAULT_DB_NAME,
    ChannelWeights,
    EmbeddingConfig,
    EncodingConfig,
    LLMConfig,
    MathConfig,
    RetrievalConfig,
    SLMConfig,
)
from superlocalmemory.storage.models import Mode


# ---------------------------------------------------------------------------
# EmbeddingConfig
# ---------------------------------------------------------------------------

class TestEmbeddingConfig:
    def test_default_is_local(self) -> None:
        ec = EmbeddingConfig()
        assert ec.model_name == "nomic-ai/nomic-embed-text-v1.5"
        assert ec.dimension == 768
        assert ec.is_cloud is False

    def test_cloud_when_endpoint_set(self) -> None:
        ec = EmbeddingConfig(api_endpoint="https://example.com", api_key="key")
        assert ec.is_cloud is True

    def test_not_cloud_with_empty_endpoint(self) -> None:
        ec = EmbeddingConfig(api_endpoint="", api_key="key")
        assert ec.is_cloud is False

    def test_frozen_immutability(self) -> None:
        ec = EmbeddingConfig()
        with pytest.raises(AttributeError):
            ec.dimension = 1024  # type: ignore[misc]


# ---------------------------------------------------------------------------
# LLMConfig
# ---------------------------------------------------------------------------

class TestLLMConfig:
    def test_default_not_available(self) -> None:
        lc = LLMConfig()
        assert lc.is_available is False
        assert lc.provider == ""

    def test_available_when_provider_set(self) -> None:
        lc = LLMConfig(provider="ollama", model="phi3:mini")
        assert lc.is_available is True

    def test_frozen_immutability(self) -> None:
        lc = LLMConfig()
        with pytest.raises(AttributeError):
            lc.provider = "openai"  # type: ignore[misc]

    def test_defaults(self) -> None:
        lc = LLMConfig()
        assert lc.temperature == 0.0
        assert lc.max_tokens == 4096
        assert lc.timeout_seconds == 60.0


# ---------------------------------------------------------------------------
# ChannelWeights
# ---------------------------------------------------------------------------

class TestChannelWeights:
    def test_defaults(self) -> None:
        cw = ChannelWeights()
        assert cw.semantic == 1.5
        assert cw.bm25 == 1.0
        # V3.3.12: rebalanced for conversational retrieval — semantic dominates
        assert cw.entity_graph == 1.0
        # S24: temporal raised from 0.8 to 1.0 — temporal_events now
        # populated (F2 fix), temporal channel contributes real results
        assert cw.temporal == 1.0

    def test_as_dict(self) -> None:
        cw = ChannelWeights(semantic=2.0, bm25=1.5)
        d = cw.as_dict()
        assert d["semantic"] == 2.0
        assert d["bm25"] == 1.5
        assert set(d.keys()) == {"semantic", "bm25", "entity_graph", "temporal", "spreading_activation", "hopfield"}


# ---------------------------------------------------------------------------
# EncodingConfig / RetrievalConfig / MathConfig (default checks)
# ---------------------------------------------------------------------------

class TestSubConfigs:
    def test_encoding_defaults(self) -> None:
        ec = EncodingConfig()
        assert ec.chunk_size == 10
        assert ec.min_fact_confidence == 0.3
        assert ec.entropy_threshold == 0.95

    def test_retrieval_defaults(self) -> None:
        rc = RetrievalConfig()
        assert rc.rrf_k == 15
        assert rc.top_k == 20
        assert rc.use_cross_encoder is True
        assert rc.cross_encoder_model == "cross-encoder/ms-marco-MiniLM-L-12-v2"
        assert rc.cross_encoder_backend == ""
        assert rc.agentic_max_rounds == 3

    def test_math_defaults(self) -> None:
        mc = MathConfig()
        assert mc.fisher_temperature == 15.0
        assert mc.sheaf_at_encoding is True


# ---------------------------------------------------------------------------
# SLMConfig — mode-specific factories
# ---------------------------------------------------------------------------

class TestSLMConfigForMode:
    def test_mode_a_no_llm(self) -> None:
        cfg = SLMConfig.for_mode(Mode.A)
        assert cfg.mode == Mode.A
        assert cfg.llm.is_available is False
        assert cfg.llm.provider == ""
        assert cfg.embedding.is_cloud is False
        assert cfg.embedding.dimension == 768

    def test_mode_b_ollama_llm(self) -> None:
        cfg = SLMConfig.for_mode(Mode.B)
        assert cfg.mode == Mode.B
        assert cfg.llm.is_available is True
        assert cfg.llm.provider == "ollama"
        assert cfg.llm.model == "llama3.2"
        assert cfg.embedding.is_cloud is False

    def test_mode_c_cloud_everything(self) -> None:
        cfg = SLMConfig.for_mode(
            Mode.C,
            embedding_endpoint="https://ai.example.com",
            embedding_key="test_key",
            embedding_deployment="embed-deploy",
            llm_api_key="test_llm_key",
            llm_api_base="https://llm.example.com",
        )
        assert cfg.mode == Mode.C
        assert cfg.embedding.is_cloud is True
        assert cfg.embedding.dimension == 3072
        assert cfg.embedding.deployment_name == "embed-deploy"
        assert cfg.llm.is_available is True
        assert cfg.llm.provider == "openrouter"
        # Mode C has boosted channel weights
        assert cfg.channel_weights.semantic == 1.5
        assert cfg.retrieval.semantic_top_k == 80

    def test_mode_b_custom_provider(self) -> None:
        cfg = SLMConfig.for_mode(
            Mode.B, llm_provider="openai", llm_model="gpt-4o",
        )
        assert cfg.llm.provider == "openai"
        assert cfg.llm.model == "gpt-4o"

    def test_mode_c_custom_provider(self) -> None:
        cfg = SLMConfig.for_mode(
            Mode.C, llm_provider="anthropic", llm_model="claude-opus",
        )
        assert cfg.llm.provider == "anthropic"


# ---------------------------------------------------------------------------
# db_path auto-computed
# ---------------------------------------------------------------------------

class TestDbPathComputation:
    def test_db_path_auto_from_base_dir(self, tmp_path: Path) -> None:
        cfg = SLMConfig(base_dir=tmp_path)
        assert cfg.db_path == tmp_path / DEFAULT_DB_NAME

    def test_db_path_explicit_override(self, tmp_path: Path) -> None:
        custom = tmp_path / "custom.db"
        cfg = SLMConfig(base_dir=tmp_path, db_path=custom)
        assert cfg.db_path == custom

    def test_default_base_dir(self) -> None:
        cfg = SLMConfig()
        assert cfg.base_dir == DEFAULT_BASE_DIR
        assert cfg.db_path == DEFAULT_BASE_DIR / DEFAULT_DB_NAME

    def test_for_mode_uses_custom_base_dir(self, tmp_path: Path) -> None:
        cfg = SLMConfig.for_mode(Mode.A, base_dir=tmp_path)
        assert cfg.base_dir == tmp_path


# ---------------------------------------------------------------------------
# V3.3.2: ONNX cross-encoder config + migration
# ---------------------------------------------------------------------------

class TestV332OnnxCrossEncoderConfig:
    """V3.3.2: ONNX cross-encoder enabled for all modes."""

    def test_mode_a_cross_encoder_enabled(self) -> None:
        cfg = SLMConfig.for_mode(Mode.A)
        assert cfg.retrieval.use_cross_encoder is True
        assert cfg.retrieval.cross_encoder_backend == ""
        assert cfg.retrieval.cross_encoder_model == "cross-encoder/ms-marco-MiniLM-L-12-v2"

    def test_mode_b_cross_encoder_enabled(self) -> None:
        cfg = SLMConfig.for_mode(Mode.B)
        assert cfg.retrieval.use_cross_encoder is True
        assert cfg.retrieval.cross_encoder_backend == ""

    def test_mode_c_cross_encoder_unchanged(self) -> None:
        cfg = SLMConfig.for_mode(Mode.C)
        assert cfg.retrieval.use_cross_encoder is True

    def test_save_persists_onnx_fields(self, tmp_path: Path) -> None:
        import json
        cfg = SLMConfig.for_mode(Mode.A, base_dir=tmp_path)
        cfg_path = tmp_path / "config.json"
        cfg.save(cfg_path)
        data = json.loads(cfg_path.read_text())
        assert data["retrieval"]["cross_encoder_model"] == "cross-encoder/ms-marco-MiniLM-L-12-v2"
        assert data["retrieval"]["cross_encoder_backend"] == ""

    def test_load_migrates_pre_332_config_respects_explicit_false(self, tmp_path: Path) -> None:
        """Pre-3.3.2 configs with explicit use_cross_encoder=False keep it.

        V3.3.3 fix: the old migration force-overwrote False→True, causing
        a 15 GB memory leak (sentence_transformers imports torch in main
        process). Now we respect the user's explicit setting.
        """
        import json
        cfg_path = tmp_path / "config.json"
        old_config = {
            "mode": "a",
            "active_profile": "default",
            "llm": {"provider": "", "model": ""},
            "embedding": {
                "model_name": "nomic-ai/nomic-embed-text-v1.5",
                "dimension": 768,
                "provider": "sentence-transformers",
            },
            "retrieval": {
                "use_cross_encoder": False,
            },
        }
        cfg_path.write_text(json.dumps(old_config))
        loaded = SLMConfig.load(cfg_path)
        # Explicit False is RESPECTED — not overwritten
        assert loaded.retrieval.use_cross_encoder is False
        # Backend field still added so migration won't trigger again
        assert loaded.retrieval.cross_encoder_backend == ""
        # Migration preserves old default for backward compat (new installs get BGE)
        assert loaded.retrieval.cross_encoder_model == "cross-encoder/ms-marco-MiniLM-L-12-v2"

    def test_load_auto_enables_ce_when_absent(self, tmp_path: Path) -> None:
        """Pre-3.3.2 configs WITHOUT use_cross_encoder get it auto-enabled."""
        import json
        cfg_path = tmp_path / "config.json"
        old_config = {
            "mode": "a",
            "active_profile": "default",
            "llm": {"provider": "", "model": ""},
            "embedding": {
                "model_name": "nomic-ai/nomic-embed-text-v1.5",
                "dimension": 768,
                "provider": "sentence-transformers",
            },
            "retrieval": {"rrf_k": 60},
        }
        cfg_path.write_text(json.dumps(old_config))
        loaded = SLMConfig.load(cfg_path)
        # When use_cross_encoder is absent, setdefault enables it
        assert loaded.retrieval.use_cross_encoder is True
        assert loaded.retrieval.cross_encoder_backend == ""

    def test_load_respects_post_332_config(self, tmp_path: Path) -> None:
        """Post-3.3.2 configs with explicit backend are respected."""
        import json
        cfg_path = tmp_path / "config.json"
        post_config = {
            "mode": "a",
            "active_profile": "default",
            "llm": {"provider": "", "model": ""},
            "embedding": {
                "model_name": "nomic-ai/nomic-embed-text-v1.5",
                "dimension": 768,
                "provider": "sentence-transformers",
            },
            "retrieval": {
                "use_cross_encoder": False,  # User explicitly disabled
                "cross_encoder_backend": "onnx",
                "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
            },
        }
        cfg_path.write_text(json.dumps(post_config))
        loaded = SLMConfig.load(cfg_path)
        # Should respect user's explicit choice
        assert loaded.retrieval.use_cross_encoder is False
