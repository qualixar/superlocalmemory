# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The model catalogue and its recommendations (4.1.22 model management)."""

from __future__ import annotations

import pytest

from superlocalmemory.core import model_catalog as mc

ALL = (*mc.LOCAL_LLMS, *mc.LOCAL_EMBEDDERS, *mc.HOSTED_LLMS, *mc.HOSTED_EMBEDDERS)


@pytest.mark.parametrize("entry", ALL, ids=lambda e: e.id)
def test_every_entry_is_complete(entry) -> None:
    assert entry.id and entry.label and entry.advice
    assert entry.role in ("llm", "embedder")
    assert entry.provider in ("ollama", "sentence-transformers", "openrouter")
    if entry.provider == "openrouter":
        assert "/" in entry.id and entry.price
    else:
        assert entry.size_gb and entry.min_ram_gb
    if entry.damaged is not None:
        assert entry.tested and 0 <= entry.damaged <= entry.tested


def test_ids_are_unique() -> None:
    ids = [e.id for e in ALL]
    assert len(ids) == len(set(ids))


def test_the_stale_defaults_are_gone() -> None:
    assert mc.DEFAULT_LOCAL_LLM != "llama3.2"
    assert mc.DEFAULT_HOSTED_LLM not in ("anthropic/claude-sonnet-4", "openai/gpt-4.1-mini")
    assert mc.find(mc.DEFAULT_LOCAL_LLM) is not None
    assert mc.find(mc.DEFAULT_HOSTED_LLM) is not None


def test_installed_and_tested_come_first_best_first() -> None:
    recs = mc.recommend_local_llms(24, ["llama3.2:latest", "qwen2.5:7b", "gemma3:4b"])
    assert [r.model_id for r in recs[:3]] == ["gemma3:4b", "qwen2.5:7b", "llama3.2"]
    assert all(r.installed and r.fits for r in recs[:3])


def test_an_untested_installed_model_is_listed_and_said_so() -> None:
    recs = mc.recommend_local_llms(24, ["mistral-nemo:12b", "gemma3:4b"])
    ids = [r.model_id for r in recs]
    assert ids[0] == "gemma3:4b" and "mistral-nemo:12b" in ids
    untested = recs[ids.index("mistral-nemo:12b")]
    assert untested.entry is None and "not tested" in untested.reason


def test_embedders_are_not_offered_as_language_models() -> None:
    recs = mc.recommend_local_llms(24, ["nomic-embed-text:latest"])
    assert "nomic-embed-text" not in [r.model_id for r in recs]


def test_nothing_installed_suggests_a_pull_that_fits() -> None:
    recs = mc.recommend_local_llms(8, [])
    assert recs and not recs[0].installed
    assert recs[0].reason.startswith("Not installed: ollama pull ")
    assert all(r.fits for r in recs)
    assert "qwen2.5:7b" not in [r.model_id for r in recs]  # needs 16 GB


def test_a_model_too_large_for_memory_goes_last() -> None:
    recs = mc.recommend_local_llms(8, ["qwen2.5:7b", "llama3.2"])
    assert recs[0].model_id == "llama3.2"
    assert recs[-1].model_id == "qwen2.5:7b" and recs[-1].fits is False


def test_best_local_llm_prefers_installed_then_the_default() -> None:
    assert mc.best_local_llm(24, ["qwen2.5:7b", "llama3.2"]) == "qwen2.5:7b"
    assert mc.best_local_llm(24, []) == mc.DEFAULT_LOCAL_LLM
    assert mc.best_local_llm(None, ["llama3.2"]) == "llama3.2"


def test_recommendations_are_deterministic() -> None:
    args = (16, ["llama3.2", "gemma3:4b", "zeta:1b", "alpha:2b"])
    assert mc.recommend_local_llms(*args) == mc.recommend_local_llms(*args)


def test_catalog_is_plain_data() -> None:
    import json

    data = mc.catalog()
    assert json.loads(json.dumps(data)) == data
    assert data["defaults"]["local_llm"] == mc.DEFAULT_LOCAL_LLM


def test_a_small_clean_sample_does_not_outrank_a_large_nearly_clean_one() -> None:
    recs = mc.recommend_local_llms(24, ["qwen3:8b", "gemma3:4b", "qwen2.5:7b", "llama3.2"])
    assert [r.model_id for r in recs[:4]] == ["gemma3:4b", "qwen3:8b", "qwen2.5:7b", "llama3.2"]
