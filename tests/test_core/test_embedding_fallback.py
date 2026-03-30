# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for embedding provider auto-detection, timeout, and fallback.

Covers:
- Engine auto-detects Ollama when LLM provider=ollama
- Engine falls back to sentence-transformers when Ollama unavailable
- Engine falls back to BM25-only when all embedders fail
- EmbeddingService subprocess timeout does not hang
- Config round-trips provider field
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from superlocalmemory.core.config import EmbeddingConfig, SLMConfig
from superlocalmemory.core.embeddings import EmbeddingService
from superlocalmemory.core.engine_wiring import init_embedder
from superlocalmemory.storage.models import Mode


# ---------------------------------------------------------------------------
# EmbeddingConfig provider field
# ---------------------------------------------------------------------------

class TestEmbeddingConfigProvider:
    """Test new provider field on EmbeddingConfig."""

    def test_default_provider_is_empty(self) -> None:
        cfg = EmbeddingConfig()
        assert cfg.provider == ""

    def test_ollama_provider(self) -> None:
        cfg = EmbeddingConfig(provider="ollama")
        assert cfg.is_ollama is True
        assert cfg.is_cloud is False

    def test_cloud_provider(self) -> None:
        cfg = EmbeddingConfig(provider="cloud", api_endpoint="https://x.com")
        assert cfg.is_cloud is True
        assert cfg.is_ollama is False

    def test_sentence_transformers_provider(self) -> None:
        cfg = EmbeddingConfig(provider="sentence-transformers")
        assert cfg.is_ollama is False
        assert cfg.is_cloud is False

    def test_ollama_defaults(self) -> None:
        cfg = EmbeddingConfig()
        assert cfg.ollama_model == "nomic-embed-text"
        assert cfg.ollama_base_url == "http://localhost:11434"


# ---------------------------------------------------------------------------
# Config persistence (save/load round-trip)
# ---------------------------------------------------------------------------

class TestConfigProviderRoundTrip:
    """Config save/load preserves embedding provider."""

    def test_save_load_preserves_provider(self, tmp_path: Path) -> None:
        cfg = SLMConfig.for_mode(Mode.B, base_dir=tmp_path, embedding_provider="ollama")
        cfg_path = tmp_path / "config.json"
        cfg.save(cfg_path)

        loaded = SLMConfig.load(cfg_path)
        assert loaded.embedding.provider == "ollama"

    def test_save_load_empty_provider(self, tmp_path: Path) -> None:
        cfg = SLMConfig.for_mode(Mode.A, base_dir=tmp_path)
        cfg_path = tmp_path / "config.json"
        cfg.save(cfg_path)

        loaded = SLMConfig.load(cfg_path)
        assert loaded.embedding.provider == ""

    def test_for_mode_passes_provider(self) -> None:
        cfg = SLMConfig.for_mode(Mode.A, embedding_provider="ollama")
        assert cfg.embedding.provider == "ollama"

        cfg_b = SLMConfig.for_mode(Mode.B, embedding_provider="sentence-transformers")
        assert cfg_b.embedding.provider == "sentence-transformers"


# ---------------------------------------------------------------------------
# Engine auto-detection
# ---------------------------------------------------------------------------

class TestEngineEmbedderAutoDetect:
    """Test engine picks the right embedder based on config."""

    def _make_engine(self, mode: Mode = Mode.B, llm_provider: str = "ollama",
                     emb_provider: str = "", tmp_path: Path | None = None):
        from superlocalmemory.core.engine import MemoryEngine
        base = tmp_path or Path("/tmp/slm-test-autodetect")
        base.mkdir(parents=True, exist_ok=True)
        cfg = SLMConfig.for_mode(
            mode, base_dir=base,
            llm_provider=llm_provider,
            embedding_provider=emb_provider,
        )
        return MemoryEngine(cfg)

    @patch("superlocalmemory.core.ollama_embedder.OllamaEmbedder.is_available", new_callable=lambda: property(lambda self: True))
    def test_auto_detects_ollama_when_llm_ollama(self, _mock, tmp_path: Path) -> None:
        """When LLM=ollama and Ollama has embed model → use OllamaEmbedder."""
        from superlocalmemory.core.ollama_embedder import OllamaEmbedder
        engine = self._make_engine(tmp_path=tmp_path)
        embedder = init_embedder(engine._config)
        assert isinstance(embedder, OllamaEmbedder)

    @patch("superlocalmemory.core.ollama_embedder.OllamaEmbedder.is_available", new_callable=lambda: property(lambda self: False))
    def test_falls_back_to_st_when_ollama_unavailable(self, _mock, tmp_path: Path) -> None:
        """When Ollama is down → fall back to sentence-transformers."""
        from superlocalmemory.core.ollama_embedder import OllamaEmbedder
        engine = self._make_engine(tmp_path=tmp_path)
        # Also mock EmbeddingService to avoid real model load
        with patch("superlocalmemory.core.embeddings.EmbeddingService") as MockES:
            mock_instance = MagicMock()
            mock_instance.is_available = True
            MockES.return_value = mock_instance
            embedder = init_embedder(engine._config)
        assert embedder is mock_instance

    @patch("superlocalmemory.core.ollama_embedder.OllamaEmbedder.is_available", new_callable=lambda: property(lambda self: False))
    def test_returns_none_when_all_fail(self, _mock, tmp_path: Path) -> None:
        """When both Ollama and sentence-transformers fail → None (BM25-only)."""
        engine = self._make_engine(tmp_path=tmp_path)
        with patch("superlocalmemory.core.embeddings.EmbeddingService") as MockES:
            mock_instance = MagicMock()
            mock_instance.is_available = False
            MockES.return_value = mock_instance
            embedder = init_embedder(engine._config)
        assert embedder is None

    @patch("superlocalmemory.core.ollama_embedder.OllamaEmbedder.is_available", new_callable=lambda: property(lambda self: True))
    def test_explicit_ollama_provider(self, _mock, tmp_path: Path) -> None:
        """When provider=ollama explicitly → use OllamaEmbedder."""
        from superlocalmemory.core.ollama_embedder import OllamaEmbedder
        engine = self._make_engine(
            llm_provider="", emb_provider="ollama", tmp_path=tmp_path,
        )
        embedder = init_embedder(engine._config)
        assert isinstance(embedder, OllamaEmbedder)

    def test_explicit_st_provider_skips_ollama(self, tmp_path: Path) -> None:
        """When provider=sentence-transformers → skip Ollama detection."""
        engine = self._make_engine(
            emb_provider="sentence-transformers", tmp_path=tmp_path,
        )
        with patch("superlocalmemory.core.embeddings.EmbeddingService") as MockES:
            mock_instance = MagicMock()
            mock_instance.is_available = True
            MockES.return_value = mock_instance
            embedder = init_embedder(engine._config)
        assert embedder is mock_instance

    @patch("superlocalmemory.core.ollama_embedder.OllamaEmbedder.is_available", new_callable=lambda: property(lambda self: True))
    def test_mode_a_no_llm_still_auto_detects_ollama(self, _mock, tmp_path: Path) -> None:
        """Mode A with no LLM but Ollama running → auto-detects Ollama for embeddings."""
        from superlocalmemory.core.ollama_embedder import OllamaEmbedder
        engine = self._make_engine(
            mode=Mode.A, llm_provider="", emb_provider="", tmp_path=tmp_path,
        )
        embedder = init_embedder(engine._config)
        assert isinstance(embedder, OllamaEmbedder)

    @patch("superlocalmemory.core.ollama_embedder.OllamaEmbedder.is_available", new_callable=lambda: property(lambda self: False))
    def test_mode_a_no_ollama_falls_back_to_st(self, _mock, tmp_path: Path) -> None:
        """Mode A with no Ollama → falls back to sentence-transformers."""
        engine = self._make_engine(
            mode=Mode.A, llm_provider="", emb_provider="", tmp_path=tmp_path,
        )
        with patch("superlocalmemory.core.embeddings.EmbeddingService") as MockES:
            mock_instance = MagicMock()
            mock_instance.is_available = True
            MockES.return_value = mock_instance
            embedder = init_embedder(engine._config)
        assert embedder is mock_instance


# ---------------------------------------------------------------------------
# EmbeddingService readline timeout
# ---------------------------------------------------------------------------

class TestEmbeddingServiceTimeout:
    """Test that subprocess readline has a timeout."""

    def test_readline_with_timeout_returns_data(self) -> None:
        """Normal case: data arrives before timeout."""
        import io
        stream = io.StringIO('{"ok": true}\n')
        result = EmbeddingService._readline_with_timeout(stream, timeout_seconds=5.0)
        assert result == '{"ok": true}\n'

    def test_readline_with_timeout_returns_empty_on_timeout(self) -> None:
        """Slow case: no data arrives → returns empty string."""
        # Use a mock stream whose readline blocks for longer than timeout
        slow_stream = MagicMock()

        def _slow_readline():
            time.sleep(5.0)  # Simulate a hang
            return '{"ok": true}\n'

        slow_stream.readline = _slow_readline
        result = EmbeddingService._readline_with_timeout(slow_stream, timeout_seconds=0.3)
        assert result == ""

    def test_readline_with_timeout_propagates_error(self) -> None:
        """Error case: stream raises → exception propagated."""
        stream = MagicMock()
        stream.readline.side_effect = OSError("broken pipe")
        with pytest.raises(OSError, match="broken pipe"):
            EmbeddingService._readline_with_timeout(stream, timeout_seconds=5.0)
