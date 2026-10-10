"""The managed provider in the catalogue, in live-space inference, and in switch validation."""

from __future__ import annotations

from argparse import Namespace

import pytest

from superlocalmemory.core import model_catalog
from superlocalmemory.core.config import EmbeddingConfig
from superlocalmemory.core.embedding_live import _infer_live
from superlocalmemory.core.embedding_providers import (
    EMBEDDING_PROVIDERS,
    validate_embedding_provider,
)

EG2 = "google/embeddinggemma-2"


def test_the_managed_model_is_in_the_catalogue_with_its_provider():
    entry = model_catalog.find(EG2)
    assert entry is not None and entry.provider == "slm-media" and entry.role == "embedder"
    assert entry.dimension == 768


def test_the_dashboard_catalogue_listing_is_unchanged():
    listed = [e["id"] for e in model_catalog.catalog()["local_embedders"]]
    assert listed == ["nomic-ai/nomic-embed-text-v1.5", "nomic-embed-text"]


def test_a_store_whose_live_space_is_the_managed_model_is_inferred_as_such():
    desired = EmbeddingConfig(model_name="nomic-ai/nomic-embed-text-v1.5", dimension=768,
                              provider="sentence-transformers", api_endpoint="http://x", api_key="k")
    live = _infer_live(desired, f"{EG2}::768")
    assert (live.provider, live.model_name, live.dimension) == ("slm-media", EG2, 768)
    assert live.api_endpoint == "" and live.api_key == "" and live.is_cloud is False


def test_inference_for_the_existing_models_is_unchanged():
    desired = EmbeddingConfig(provider="ollama", model_name="nomic-embed-text")
    assert _infer_live(desired, "nomic-embed-text::768").provider == "ollama"
    assert _infer_live(desired, "nomic-ai/nomic-embed-text-v1.5::768").provider == "sentence-transformers"


def test_the_managed_provider_is_never_cloud_or_ollama():
    cfg = EmbeddingConfig(model_name=EG2, dimension=768, provider="slm-media")
    assert not cfg.is_cloud and not cfg.is_ollama and not cfg.is_openai_compatible


def test_provider_names_are_validated_with_the_known_list():
    for name in EMBEDDING_PROVIDERS:
        assert validate_embedding_provider(name) == name
    assert "slm-media" in EMBEDDING_PROVIDERS
    with pytest.raises(ValueError) as err:
        validate_embedding_provider("slm-medai")
    text = str(err.value)
    assert "slm-medai" in text and "slm-media" in text and "sentence-transformers" in text


def test_the_switch_route_refuses_an_unknown_provider_and_accepts_the_managed_one():
    from superlocalmemory.server.routes.embedding_reindex import target_config

    live = EmbeddingConfig()
    with pytest.raises(ValueError, match="slm-media"):
        target_config(live, {"model_name": EG2, "dimension": 768, "provider": "bogus"})
    target = target_config(live, {"model_name": EG2, "dimension": 768, "provider": "slm-media"})
    assert target.provider == "slm-media" and target.model_name == EG2
    # a request that names no provider keeps working, whatever the live provider is
    legacy = EmbeddingConfig(provider="something-old")
    assert target_config(legacy, {"model_name": "m", "dimension": 384}).provider == "something-old"


def test_the_cli_refuses_an_unknown_provider_before_asking_the_daemon(monkeypatch, capsys):
    from superlocalmemory.cli import embedder_cmd

    def no_request(*a, **k):
        raise AssertionError("the daemon was asked")

    monkeypatch.setattr(embedder_cmd, "daemon_request", no_request)
    with pytest.raises(SystemExit) as exit_info:
        embedder_cmd.cmd_embedder(Namespace(embedder_command="switch", json=False, model="m", dimension=8,
                                            provider="bogus", endpoint="", no_wait=True))
    assert exit_info.value.code == 1
    assert "slm-media" in capsys.readouterr().err
