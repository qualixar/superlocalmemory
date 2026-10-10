"""A light (MCP) engine on the managed text provider asks the daemon with the model check."""

from __future__ import annotations

from dataclasses import replace

import pytest

from superlocalmemory.core.daemon_text_embedder import DaemonTextEmbedder
from superlocalmemory.core.engine import MemoryEngine
from superlocalmemory.core.mcp_embedder_proxy import McpEmbedderProxy

EG2 = "google/embeddinggemma-2"


def _engine(mode_a_config, **embedding):
    mode_a_config.embedding = replace(mode_a_config.embedding, **embedding)
    return MemoryEngine(mode_a_config)


def test_the_managed_provider_gets_the_daemon_embedder_even_before_the_daemon_is_up(mode_a_config):
    engine = _engine(mode_a_config, provider="slm-media", model_name=EG2, dimension=768)
    engine._try_init_proxy()
    assert isinstance(engine._embedder, DaemonTextEmbedder)
    assert engine._embedder.model_name == EG2


@pytest.mark.parametrize("provider", ["", "sentence-transformers", "ollama"])
def test_other_providers_keep_the_plain_proxy(mode_a_config, monkeypatch, provider):
    monkeypatch.setattr(McpEmbedderProxy, "is_available", lambda self: True)
    engine = _engine(mode_a_config, provider=provider)
    engine._try_init_proxy()
    assert isinstance(engine._embedder, McpEmbedderProxy)
