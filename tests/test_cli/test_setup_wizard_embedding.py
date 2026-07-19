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
