# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""v3.7.6 (#72): the setup wizard must not force the local 768d nomic download
when a remote/OpenAI-compatible embedding endpoint is configured."""

from __future__ import annotations

from types import SimpleNamespace

from superlocalmemory.cli.setup_wizard import _embedding_is_remote


def _cfg(**embedding) -> SimpleNamespace:
    return SimpleNamespace(embedding=SimpleNamespace(**embedding))


def test_openai_provider_is_remote():
    assert _embedding_is_remote(_cfg(provider="openai", dimension=1024)) is True


def test_endpoint_without_provider_name_is_remote():
    assert _embedding_is_remote(
        _cfg(provider="", api_endpoint="http://192.168.50.140:8045/v1")
    ) is True


def test_local_sentence_transformers_is_not_remote():
    assert _embedding_is_remote(
        _cfg(provider="sentence-transformers", dimension=768, api_endpoint="")
    ) is False


def test_missing_embedding_section_is_not_remote():
    assert _embedding_is_remote(SimpleNamespace()) is False


def test_reconfigure_preserves_existing_custom_embedding(tmp_path, monkeypatch):
    """Re-running setup cannot replace a 1024d endpoint with local 768d."""
    monkeypatch.setenv("SLM_BASE_DIR", str(tmp_path))
    from superlocalmemory.core.config import EmbeddingConfig, SLMConfig
    from superlocalmemory.storage.models import Mode
    from superlocalmemory.cli.setup_wizard import _build_wizard_config

    existing = SLMConfig.for_mode(Mode.C)
    existing.embedding = EmbeddingConfig(
        provider="openai",
        api_endpoint="http://127.0.0.1:8045/v1/embeddings",
        model_name="qwen3-embedding",
        dimension=1024,
    )
    existing.save(mode_change=True)

    rebuilt = _build_wizard_config(Mode.A)

    assert rebuilt.mode is Mode.A
    assert rebuilt.embedding.provider == "openai"
    assert rebuilt.embedding.api_endpoint == "http://127.0.0.1:8045/v1/embeddings"
    assert rebuilt.embedding.model_name == "qwen3-embedding"
    assert rebuilt.embedding.dimension == 1024
