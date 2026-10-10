"""A model and its provider always agree: the managed model runs only under the managed provider."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from superlocalmemory.core.config import EmbeddingConfig
from superlocalmemory.server.routes import v3_api
from superlocalmemory.server.routes.embedding_reindex import target_config

EG2 = "google/embeddinggemma-2"


def test_the_managed_provider_accepts_only_managed_models():
    live = EmbeddingConfig()
    with pytest.raises(ValueError, match="slm-media serves only: google/embeddinggemma-2"):
        target_config(live, {"provider": "slm-media"})  # the live model is nomic
    with pytest.raises(ValueError, match="slm-media serves only"):
        target_config(live, {"provider": "slm-media", "model_name": "someone/any-model", "dimension": 768})
    ok = target_config(live, {"provider": "slm-media", "model_name": EG2, "dimension": 768})
    assert (ok.provider, ok.model_name) == ("slm-media", EG2)


def test_the_managed_model_without_a_provider_gets_the_managed_provider():
    for live in (EmbeddingConfig(), EmbeddingConfig(provider="sentence-transformers")):
        target = target_config(live, {"model_name": EG2, "dimension": 768})
        assert target.provider == "slm-media" and target.model_name == EG2


def test_the_managed_model_with_another_explicit_provider_is_refused():
    for provider in ("sentence-transformers", "ollama", "openai", "cloud"):
        with pytest.raises(ValueError, match="slm-media"):
            target_config(EmbeddingConfig(), {"provider": provider, "model_name": EG2, "dimension": 768})


def test_leaving_the_managed_provider_for_a_plain_provider_keeps_the_model_check():
    live = EmbeddingConfig(model_name=EG2, dimension=768, provider="slm-media")
    with pytest.raises(ValueError, match="slm-media"):
        target_config(live, {"provider": "sentence-transformers"})  # would keep the managed model
    other = target_config(live, {"provider": "ollama", "model_name": "nomic-embed-text", "dimension": 768})
    assert other.provider == "ollama"



def test_going_back_to_a_plain_model_from_the_managed_one_needs_no_provider():
    """After an upgrade, `slm embedder switch nomic-ai/...` alone must work: the automatic provider."""
    live = EmbeddingConfig(model_name=EG2, dimension=768, provider="slm-media")
    back = target_config(live, {"model_name": "nomic-ai/nomic-embed-text-v1.5", "dimension": 768})
    assert back.provider == "" and back.model_name == "nomic-ai/nomic-embed-text-v1.5"


class _Request:
    def __init__(self, body) -> None:
        self._body = body
        self.app = SimpleNamespace(state=SimpleNamespace())
        self.client = SimpleNamespace(host="127.0.0.1")
        self.headers = {}

    async def json(self):
        return self._body


@pytest.fixture()
def _admin(monkeypatch):
    from superlocalmemory.server import rbac_enforce

    monkeypatch.setattr(rbac_enforce, "require_manage", lambda request: None)
    from superlocalmemory.core.config import SLMConfig

    config = SLMConfig.for_mode(__import__("superlocalmemory.storage.models", fromlist=["Mode"]).Mode.A)
    monkeypatch.setattr(SLMConfig, "load", classmethod(lambda cls, *a, **k: config))
    return config


def _status(response):
    return response.status_code, json.loads(response.body)


def test_the_dashboard_embedding_save_refuses_a_wrong_pair(_admin):
    status, body = _status(asyncio.run(v3_api.set_embedding_config(
        _Request({"provider": "slm-media", "model_name": "someone/any-model", "dimension": 768}))))
    assert status == 400 and "slm-media serves only" in body["error"]
    status, body = _status(asyncio.run(v3_api.set_embedding_config(
        _Request({"provider": "ollama", "model_name": EG2, "dimension": 768}))))
    assert status == 400 and "slm-media" in body["error"]


def test_the_dashboard_mode_save_refuses_a_wrong_pair(_admin):
    status, body = _status(asyncio.run(v3_api.set_full_config(_Request(
        {"mode": "a", "embedding_provider": "slm-media", "embedding_model": "someone/any-model",
         "embedding_dimension": 768}))))
    assert status == 400 and "slm-media serves only" in body["error"]
    status, body = _status(asyncio.run(v3_api.set_full_config(_Request(
        {"mode": "a", "embedding_provider": "ollama", "embedding_model": EG2,
         "embedding_dimension": 768}))))
    assert status == 400 and "slm-media" in body["error"]


def test_a_reindex_to_the_managed_model_builds_the_managed_embedder(monkeypatch):
    from superlocalmemory.core.embedding_reindex_steps import build_embedder

    target = target_config(EmbeddingConfig(), {"model_name": EG2, "dimension": 768})
    emb = build_embedder(target)
    assert type(emb).__name__ != "OllamaEmbedder"
