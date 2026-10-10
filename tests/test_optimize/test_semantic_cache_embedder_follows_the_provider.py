"""The semantic cache embeds with the configured provider, not always with the built-in model."""

from __future__ import annotations

from types import SimpleNamespace

from superlocalmemory.core import embeddings, engine_wiring
from superlocalmemory.core.config import EmbeddingConfig, SLMConfig
from superlocalmemory.optimize.cache.manager import _LazySemanticEmbedder


class _Service:
    kind = "built-in"

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def embed(self, text):
        return [1.0]

    def unload(self):
        pass


def _config(monkeypatch, provider):
    cfg = SimpleNamespace(embedding=EmbeddingConfig(provider=provider))
    monkeypatch.setattr(SLMConfig, "load", classmethod(lambda cls, *a, **k: cfg))
    return cfg


def test_the_managed_provider_goes_through_the_one_embedder_factory(monkeypatch):
    cfg = _config(monkeypatch, "slm-media")
    built = []

    class Managed:
        def embed(self, text):
            return [2.0]

        def unload(self):
            built.append("unloaded")

    monkeypatch.setattr(engine_wiring, "init_embedder", lambda c: built.append(c) or Managed())
    monkeypatch.setattr(embeddings, "EmbeddingService", _Service)
    lazy = _LazySemanticEmbedder()
    assert lazy("hello") == [2.0]
    assert built == [cfg]
    lazy.close()
    assert built[-1] == "unloaded"


def test_every_other_provider_still_uses_the_built_in_service(monkeypatch):
    for provider in ("", "sentence-transformers", "ollama"):
        _config(monkeypatch, provider)
        monkeypatch.setattr(embeddings, "EmbeddingService", _Service)
        monkeypatch.setattr(engine_wiring, "init_embedder",
                            lambda c: (_ for _ in ()).throw(AssertionError("factory used")))
        assert _LazySemanticEmbedder()("hello") == [1.0]
