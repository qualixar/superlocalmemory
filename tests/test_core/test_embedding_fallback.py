# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
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

    def test_save_load_mode_a_default_provider(self, tmp_path: Path) -> None:
        """Mode A defaults to sentence-transformers (subprocess-isolated)."""
        cfg = SLMConfig.for_mode(Mode.A, base_dir=tmp_path)
        cfg_path = tmp_path / "config.json"
        cfg.save(cfg_path)

        loaded = SLMConfig.load(cfg_path)
        assert loaded.embedding.provider == "sentence-transformers"

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
        """v3.4.55: When LLM=ollama and provider=ollama, use OllamaEmbedder directly.

        v3.4.55 changed init_embedder: provider=ollama now returns OllamaEmbedder
        (not EmbeddingService) when Ollama is available. Reason: Ollama's
        nomic-embed-text and sentence-transformers' nomic-embed-text-v1.5
        produce DIFFERENT vector spaces — mixing them degrades semantic recall.
        Using OllamaEmbedder when provider=ollama is consistent with stored vectors.
        """
        from superlocalmemory.core.ollama_embedder import OllamaEmbedder
        engine = self._make_engine(tmp_path=tmp_path)
        embedder = init_embedder(engine._config)
        # v3.4.55: provider=ollama + Ollama available → OllamaEmbedder returned directly
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
    def test_explicit_ollama_provider_mode_b_hybrid(self, _mock, tmp_path: Path) -> None:
        """v3.4.55: Explicit provider=ollama → OllamaEmbedder when Ollama is available.

        v3.4.55 fix: provider=ollama now returns OllamaEmbedder directly
        when available. Vectors stored by Ollama's embedding API are incompatible
        with sentence-transformers' embedding space — mixing providers degrades recall.
        EmbeddingService (sentence-transformers) is only the fallback when Ollama is down.
        """
        from superlocalmemory.core.ollama_embedder import OllamaEmbedder
        engine = self._make_engine(
            llm_provider="", emb_provider="ollama", tmp_path=tmp_path,
        )
        embedder = init_embedder(engine._config)
        # v3.4.55: explicit provider=ollama + Ollama up → OllamaEmbedder (not EmbeddingService)
        assert isinstance(embedder, OllamaEmbedder)

    @patch("superlocalmemory.core.ollama_embedder.OllamaEmbedder.is_available", new_callable=lambda: property(lambda self: True))
    def test_explicit_ollama_provider_mode_a_prefers_ollama(self, _mock, tmp_path: Path) -> None:
        """v3.4.55: Mode A with explicit provider=ollama → OllamaEmbedder when Ollama is up.

        init_embedder respects the explicit provider choice regardless of mode.
        When provider=ollama and Ollama is available, OllamaEmbedder is returned
        to keep vector spaces consistent with existing stored vectors.
        """
        from superlocalmemory.core.ollama_embedder import OllamaEmbedder
        engine = self._make_engine(
            mode=Mode.A, llm_provider="", emb_provider="ollama", tmp_path=tmp_path,
        )
        embedder = init_embedder(engine._config)
        # v3.4.55: explicit provider=ollama + Ollama up → OllamaEmbedder (not EmbeddingService)
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

    def test_mode_a_default_uses_subprocess_st(self, tmp_path: Path) -> None:
        """Mode A defaults to sentence-transformers (subprocess-isolated, no Ollama)."""
        engine = self._make_engine(
            mode=Mode.A, llm_provider="", emb_provider="", tmp_path=tmp_path,
        )
        # Mode A now defaults to provider="sentence-transformers" (subprocess)
        assert engine._config.embedding.provider == "sentence-transformers"
        with patch("superlocalmemory.core.embeddings.EmbeddingService") as MockES:
            mock_instance = MagicMock()
            mock_instance.is_available = True
            MockES.return_value = mock_instance
            embedder = init_embedder(engine._config)
        assert embedder is mock_instance

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

    def test_worker_dependency_import_failure_is_reported_not_crashed(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A broken local ML stack must emit a protocol error, not exit silently."""
        import builtins
        from superlocalmemory.core import embedding_worker

        real_import = builtins.__import__

        def _blocked_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ValueError("incompatible local numpy installation")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocked_import)
        assert embedding_worker._load_embedding_model("local-model") == (None, "")

    def test_terminal_worker_error_disables_repeated_respawns(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A reported dependency failure must not churn workers on each recall."""
        import io

        service = EmbeddingService(EmbeddingConfig())
        worker = MagicMock()
        worker.stdin = io.StringIO()
        worker.stdout = io.StringIO(
            '{"ok": false, "error": "sentence-transformers unavailable"}\n',
        )
        service._worker_proc = worker
        monkeypatch.setattr(service, "_ensure_worker", lambda: None)
        killed = MagicMock()
        monkeypatch.setattr(service, "_kill_worker", killed)

        assert service._subprocess_embed(["dependency failure witness"]) is None
        assert service.is_available is False
        killed.assert_called_once()


# ---------------------------------------------------------------------------
# Availability tri-state (regression for v3.7.8 self-heal brick)
# ---------------------------------------------------------------------------

class TestEmbeddingAvailabilityTriState:
    """The ``_available`` flag is a tri-state and recall-health depends on it.

    v3.7.8 added ``if not self._available: return None`` to ``_subprocess_embed``.
    ``recall_health._heal_embedder`` sets ``_available = None`` to force a
    re-probe (the OllamaEmbedder convention). Because ``None`` is falsy, that
    guard made every self-heal tick short-circuit the probe embed and leave the
    flag stuck at ``None`` — permanently bricking the local subprocess worker
    (semantic/hopfield/spreading_activation dead) until the daemon restarted,
    and re-bricking on the first heal tick after each restart.
    """

    def _wire_ok_worker(self, service, monkeypatch, vectors):
        import io
        import json as _json

        worker = MagicMock()
        worker.stdin = io.StringIO()
        worker.stdout = io.StringIO(
            _json.dumps({"ok": True, "vectors": vectors, "dim": len(vectors[0])})
            + "\n",
        )

        def _ensure() -> None:
            service._worker_proc = worker

        monkeypatch.setattr(service, "_ensure_worker", _ensure)
        monkeypatch.setattr(service, "_reset_idle_timer", lambda: None)
        return worker

    def test_none_availability_reprobes_and_succeeds(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """self-heal set ``_available = None`` — the next embed must re-probe the
        worker, not short-circuit to None."""
        service = EmbeddingService(EmbeddingConfig())
        service._available = None  # recall_health._heal_embedder reset
        self._wire_ok_worker(service, monkeypatch, [[0.1, 0.2, 0.3]])

        assert service._subprocess_embed(["heal probe"]) == [[0.1, 0.2, 0.3]]

    def test_successful_embed_resets_available_to_true(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A proven-healthy embed clears the transient ``None`` back to True so
        the flag no longer lingers falsy."""
        service = EmbeddingService(EmbeddingConfig())
        service._available = None
        self._wire_ok_worker(service, monkeypatch, [[1.0, 2.0]])

        service._subprocess_embed(["witness"])
        assert service._available is True

    def test_explicit_false_still_short_circuits(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A terminal disable (``False``) must still block without touching the
        worker — preserves the v3.7.8 "don't churn a dead worker" intent."""
        service = EmbeddingService(EmbeddingConfig())
        service._available = False
        ensure = MagicMock()
        monkeypatch.setattr(service, "_ensure_worker", ensure)

        assert service._subprocess_embed(["blocked"]) is None
        ensure.assert_not_called()
