# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for OpenAI-compatible embedding provider (V3.4.24).

Covers:
- EmbeddingConfig.is_openai_compatible property
- Config round-trip: save/load preserves openai provider settings
- for_mode() honors openai provider in Mode A and B
- EmbeddingService routes to OpenAI-compatible HTTP path
- engine_wiring routes openai provider correctly
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from superlocalmemory.core.config import EmbeddingConfig, SLMConfig
from superlocalmemory.core.embeddings import EmbeddingService
from superlocalmemory.storage.models import Mode

# ---------------------------------------------------------------------------
# EmbeddingConfig.is_openai_compatible
# ---------------------------------------------------------------------------

class TestEmbeddingConfigOpenAI:
    """Test is_openai_compatible property."""

    def test_default_is_not_openai(self) -> None:
        cfg = EmbeddingConfig()
        assert cfg.is_openai_compatible is False

    def test_openai_provider_without_endpoint_is_false(self) -> None:
        cfg = EmbeddingConfig(provider="openai")
        assert cfg.is_openai_compatible is False

    def test_openai_provider_with_endpoint_is_true(self) -> None:
        cfg = EmbeddingConfig(
            provider="openai",
            api_endpoint="http://localhost:8045/v1/embeddings",
        )
        assert cfg.is_openai_compatible is True

    def test_non_openai_provider_with_endpoint_is_false(self) -> None:
        cfg = EmbeddingConfig(
            provider="ollama",
            api_endpoint="http://localhost:11434",
        )
        assert cfg.is_openai_compatible is False

    def test_openai_with_all_fields(self) -> None:
        cfg = EmbeddingConfig(
            provider="openai",
            model_name="Qwen3-Embedding",
            dimension=1024,
            api_endpoint="http://192.168.50.140:8045/v1/embeddings",
            api_key="not-needed",
        )
        assert cfg.is_openai_compatible is True
        assert cfg.is_cloud is False
        assert cfg.is_ollama is False
        assert cfg.dimension == 1024
        assert cfg.model_name == "Qwen3-Embedding"


# ---------------------------------------------------------------------------
# for_mode() with openai embedding provider
# ---------------------------------------------------------------------------

class TestForModeOpenAIEmbedding:
    """Test for_mode() passes through openai embedding config."""

    def test_mode_a_default_is_sentence_transformers(self) -> None:
        cfg = SLMConfig.for_mode(Mode.A)
        assert cfg.embedding.provider == "sentence-transformers"
        assert cfg.embedding.is_openai_compatible is False

    def test_mode_b_default_is_ollama(self) -> None:
        cfg = SLMConfig.for_mode(Mode.B)
        assert cfg.embedding.provider == "ollama"
        assert cfg.embedding.is_openai_compatible is False

    def test_mode_a_with_openai_embedding(self) -> None:
        cfg = SLMConfig.for_mode(
            Mode.A,
            embedding_provider="openai",
            embedding_endpoint="http://localhost:8045/v1",
            embedding_model_name="bge-m3",
            embedding_dimension=1024,
        )
        assert cfg.embedding.provider == "openai"
        assert cfg.embedding.is_openai_compatible is True
        assert cfg.embedding.model_name == "bge-m3"
        assert cfg.embedding.dimension == 1024
        assert cfg.embedding.api_endpoint == "http://localhost:8045/v1"

    def test_mode_b_with_openai_embedding(self) -> None:
        cfg = SLMConfig.for_mode(
            Mode.B,
            llm_provider="ollama",
            llm_model="llama3.2",
            embedding_provider="openai",
            embedding_endpoint="http://localhost:8045/v1",
            embedding_model_name="multilingual-e5-large",
            embedding_dimension=1024,
        )
        assert cfg.embedding.provider == "openai"
        assert cfg.embedding.is_openai_compatible is True
        assert cfg.embedding.dimension == 1024
        assert cfg.llm.provider == "ollama"

    def test_mode_b_openai_requires_endpoint(self) -> None:
        cfg = SLMConfig.for_mode(
            Mode.B,
            embedding_provider="openai",
        )
        assert cfg.embedding.provider == "openai"
        assert cfg.embedding.is_openai_compatible is False


# ---------------------------------------------------------------------------
# Config save/load round-trip
# ---------------------------------------------------------------------------

class TestConfigRoundTrip:
    """Verify openai embedding config survives save/load cycle."""

    def test_round_trip_preserves_openai_embedding(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.json"
        cfg = SLMConfig.for_mode(
            Mode.B,
            llm_provider="ollama",
            llm_model="llama3.2",
            embedding_provider="openai",
            embedding_endpoint="http://localhost:8045/v1",
            embedding_model_name="Qwen3-Embedding",
            embedding_dimension=1024,
            embedding_key="test-key",
        )
        cfg.save(config_path)

        loaded = SLMConfig.load(config_path)
        assert loaded.embedding.provider == "openai"
        assert loaded.embedding.model_name == "Qwen3-Embedding"
        assert loaded.embedding.dimension == 1024
        assert loaded.embedding.api_endpoint == "http://localhost:8045/v1"
        assert loaded.embedding.api_key == "test-key"
        assert loaded.embedding.is_openai_compatible is True

    def test_round_trip_preserves_default_embedding(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.json"
        cfg = SLMConfig.for_mode(Mode.A)
        cfg.save(config_path)

        loaded = SLMConfig.load(config_path)
        assert loaded.embedding.provider == "sentence-transformers"
        assert loaded.embedding.dimension == 768
        assert loaded.embedding.is_openai_compatible is False


# ---------------------------------------------------------------------------
# EmbeddingService routing
# ---------------------------------------------------------------------------

class TestEmbeddingServiceRouting:
    """Test EmbeddingService routes to correct embed method."""

    def test_is_available_with_openai_endpoint(self) -> None:
        cfg = EmbeddingConfig(
            provider="openai",
            api_endpoint="http://localhost:8045/v1",
        )
        svc = EmbeddingService(cfg)
        assert svc.is_available is True

    def test_openai_without_endpoint_falls_back_to_available(self) -> None:
        cfg = EmbeddingConfig(provider="openai")
        svc = EmbeddingService(cfg)
        # Without endpoint, is_openai_compatible=False, falls back to
        # default _available=True (subprocess path).
        assert cfg.is_openai_compatible is False
        assert svc.is_available is True

    def test_embed_calls_openai_compatible_path(self) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"index": 0, "embedding": [0.1] * 768}],
        }
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response

        cfg = EmbeddingConfig(
            provider="openai",
            api_endpoint="http://localhost:8045/v1",
            model_name="test-model",
            dimension=768,
        )
        svc = EmbeddingService(cfg)

        with patch("httpx.Client", return_value=mock_client):
            with patch("httpx.Timeout"):
                result = svc.embed("hello world")

        assert result is not None
        assert len(result) == 768
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "/embeddings" in call_args[0][0]


# ---------------------------------------------------------------------------
# engine_wiring routes openai provider
# ---------------------------------------------------------------------------

class TestEngineWiringOpenAI:
    """Test init_embedder routes openai provider."""

    @patch("superlocalmemory.core.engine_wiring._try_service_embedder")
    def test_openai_provider_routes_to_service(self, mock_try: MagicMock) -> None:
        mock_try.return_value = MagicMock()
        cfg = SLMConfig.for_mode(
            Mode.B,
            embedding_provider="openai",
            embedding_endpoint="http://localhost:8045/v1",
            embedding_model_name="bge-m3",
            embedding_dimension=1024,
        )
        from superlocalmemory.core.engine_wiring import init_embedder
        result = init_embedder(cfg)

        mock_try.assert_called_once()
        assert result is not None
