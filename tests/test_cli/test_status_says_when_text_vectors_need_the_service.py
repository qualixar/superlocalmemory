"""``slm status`` says why recall is keyword-only when the managed text model needs the service."""

from __future__ import annotations

from types import SimpleNamespace

from superlocalmemory.cli import embedder_cmd
from superlocalmemory.core.config import EmbeddingConfig


def _config(provider: str):
    return SimpleNamespace(embedding=EmbeddingConfig(provider=provider))


def test_the_managed_provider_without_a_daemon_gets_one_clear_line():
    line = embedder_cmd.text_provider_note(_config("slm-media"))
    assert line.startswith("  Text vectors need the SLM service running") and line.endswith("\n")
    assert "slm serve start" in line and line.count("\n") == 1


def test_every_other_provider_prints_nothing():
    for provider in ("", "sentence-transformers", "ollama", "openai", "cloud"):
        assert embedder_cmd.text_provider_note(_config(provider)) == ""
