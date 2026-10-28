# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""#302: an OpenAI-compatible embedding endpoint keeps the model the user chose.

Modes A and B filled a blank model name and width with the local bundle's
identity (nomic-ai/nomic-embed-text-v1.5 at 768) and the next save made that
durable. A provider gets its own model id or none at all, never another
provider's. The save doors now refuse an endpoint with no model or no width,
and a blank width can never make the vector store drop vectors it holds.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from superlocalmemory.core.config import EmbeddingConfig, SLMConfig
from superlocalmemory.storage.models import Mode

NOMIC = "nomic-ai/nomic-embed-text-v1.5"
ENDPOINT = "http://localhost:8045/v1/embeddings"


@pytest.fixture
def base(tmp_path, monkeypatch):
    monkeypatch.setattr("superlocalmemory.core.config.DEFAULT_BASE_DIR", tmp_path)
    monkeypatch.setenv("SLM_BASE_DIR", str(tmp_path))
    return tmp_path


def _write(base: Path, mode: str, embedding: dict) -> Path:
    path = base / "config.json"
    path.write_text(json.dumps({
        "mode": mode,
        "base_dir": str(base),
        "llm": {"provider": "openai", "model": "gpt-4o-mini"},
        "embedding": embedding,
    }), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The config template: blank stays blank
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["a", "b"])
def test_blank_model_and_width_stay_blank_through_load_and_save(base, mode):
    path = _write(base, mode, {
        "provider": "openai", "api_endpoint": ENDPOINT, "model_name": "", "dimension": 0,
    })
    config = SLMConfig.load(path)

    assert config.embedding.provider == "openai"
    assert config.embedding.model_name == ""
    assert config.embedding.dimension == 0
    assert NOMIC not in repr(config.embedding)

    config.save(path)
    on_disk = json.loads(path.read_text(encoding="utf-8"))["embedding"]
    assert on_disk["model_name"] == ""
    assert on_disk["dimension"] == 0


@pytest.mark.parametrize("mode", [Mode.A, Mode.B])
def test_for_mode_does_not_invent_a_model_for_an_endpoint(mode, base):
    config = SLMConfig.for_mode(
        mode, base_dir=base, embedding_provider="openai", embedding_endpoint=ENDPOINT,
    )
    assert config.embedding.model_name == ""
    assert config.embedding.dimension == 0


@pytest.mark.parametrize("mode", ["a", "b"])
def test_a_chosen_model_and_width_round_trip_unchanged(base, mode):
    path = _write(base, mode, {
        "provider": "openai", "api_endpoint": ENDPOINT,
        "model_name": "Qwen3-Embedding", "dimension": 1024,
    })
    config = SLMConfig.load(path)
    config.save(path)
    on_disk = json.loads(path.read_text(encoding="utf-8"))["embedding"]
    assert (on_disk["model_name"], on_disk["dimension"]) == ("Qwen3-Embedding", 1024)


def test_mode_c_still_defaults_to_the_catalogue_model(base):
    path = _write(base, "c", {"provider": "openai", "api_endpoint": ENDPOINT})
    config = SLMConfig.load(path)
    assert config.embedding.model_name == "text-embedding-3-small"
    assert config.embedding.dimension == 1536


def test_the_local_defaults_are_untouched(base):
    config = SLMConfig.for_mode(Mode.A, base_dir=base)
    assert (config.embedding.model_name, config.embedding.dimension) == (NOMIC, 768)
    config = SLMConfig.for_mode(Mode.B, base_dir=base)
    assert (config.embedding.model_name, config.embedding.dimension) == (NOMIC, 768)


@pytest.mark.parametrize("mode", ["a", "b"])
def test_an_already_stored_inferred_pair_loads_as_stored_and_is_not_rewritten(base, mode):
    """4.1.x wrote the guess to disk; loading it must not crash or rewrite the file."""
    path = _write(base, mode, {
        "provider": "openai", "api_endpoint": ENDPOINT, "model_name": NOMIC, "dimension": 768,
    })
    before = path.read_bytes()

    config = SLMConfig.load(path)

    assert (config.embedding.model_name, config.embedding.dimension) == (NOMIC, 768)
    assert path.read_bytes() == before


# ---------------------------------------------------------------------------
# The refusal, shared by every door
# ---------------------------------------------------------------------------

def test_incomplete_endpoint_error_names_the_missing_fields():
    from superlocalmemory.core.embedding_endpoint_check import incomplete_endpoint_error

    both = incomplete_endpoint_error(
        EmbeddingConfig(provider="openai", api_endpoint=ENDPOINT, model_name="", dimension=0))
    assert both is not None and "model name" in both and "dimension" in both

    only_dim = incomplete_endpoint_error(
        EmbeddingConfig(provider="openai", api_endpoint=ENDPOINT, model_name="m", dimension=0))
    assert only_dim is not None and "dimension" in only_dim and "model name" not in only_dim

    only_model = incomplete_endpoint_error(
        EmbeddingConfig(provider="openai", api_endpoint=ENDPOINT, model_name="  ", dimension=512))
    assert only_model is not None and "model name" in only_model and "dimension" not in only_model


def test_complete_and_non_endpoint_configs_are_not_refused():
    from superlocalmemory.core.embedding_endpoint_check import incomplete_endpoint_error

    assert incomplete_endpoint_error(
        EmbeddingConfig(provider="openai", api_endpoint=ENDPOINT, model_name="m", dimension=1024)) is None
    assert incomplete_endpoint_error(EmbeddingConfig()) is None
    assert incomplete_endpoint_error(EmbeddingConfig(provider="ollama", dimension=768)) is None
    # No endpoint yet: "openai" without one is not an OpenAI-compatible config.
    assert incomplete_endpoint_error(EmbeddingConfig(provider="openai", model_name="", dimension=0)) is None


def _request(body: dict):
    request = MagicMock()
    request.json = AsyncMock(return_value=body)
    request.app.state = MagicMock(spec=[])
    request.client = SimpleNamespace(host="127.0.0.1")
    return request


def _call(route, body):
    async def _no_apply(*_a, **_k):
        return None
    with patch("superlocalmemory.server.routes.v3_api._apply_runtime_config", _no_apply), \
            patch("superlocalmemory.server.rbac_enforce.require_manage", lambda _r: None):
        response = asyncio.run(route(_request(body)))
    if hasattr(response, "body"):
        return response.status_code, json.loads(response.body)
    return 200, response


@pytest.fixture
def local_store(base):
    _write(base, "a", {"provider": "", "model_name": NOMIC, "dimension": 768})
    return base


def _saved(base: Path) -> dict:
    return json.loads((base / "config.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("fields", [
    {"embedding_model": "", "embedding_dimension": 1024},
    {"embedding_model": "Qwen3-Embedding", "embedding_dimension": 0},
    {"embedding_model": "", "embedding_dimension": 0},
])
def test_mode_set_refuses_an_endpoint_without_a_model_or_width(local_store, fields):
    from superlocalmemory.server.routes.v3_api import set_full_config

    code, payload = _call(set_full_config, {
        "mode": "a", "embedding_provider": "openai", "embedding_endpoint": ENDPOINT, **fields,
    })

    assert code == 400, payload
    assert payload["code"] == "embedding_endpoint_incomplete"
    assert NOMIC not in json.dumps(payload)
    assert _saved(local_store)["embedding"]["model_name"] == NOMIC  # nothing was persisted


def test_mode_set_does_not_borrow_the_local_width_for_an_endpoint(local_store):
    """Switching from the local model, the old 768 is not the endpoint's width."""
    from superlocalmemory.server.routes.v3_api import set_full_config

    code, payload = _call(set_full_config, {
        "mode": "a", "embedding_provider": "openai", "embedding_endpoint": ENDPOINT,
        "embedding_model": "Qwen3-Embedding",
    })
    assert code == 400, payload
    assert "dimension" in payload["error"]


def test_mode_set_accepts_a_complete_endpoint(local_store):
    from superlocalmemory.server.routes.v3_api import set_full_config

    code, payload = _call(set_full_config, {
        "mode": "a", "embedding_provider": "openai", "embedding_endpoint": ENDPOINT,
        "embedding_model": "Qwen3-Embedding", "embedding_dimension": 1024,
    })
    # A new space is re-indexed in the background, so the live model stays until
    # that is done; what matters here is that the save is not refused.
    assert code == 200, payload
    assert payload["needs_reindex"] is True and "Qwen3-Embedding" in payload["message"]


def test_embedding_config_put_refuses_a_blank_model(local_store):
    from superlocalmemory.server.routes.v3_api import set_embedding_config

    code, payload = _call(set_embedding_config, {
        "provider": "openai", "api_endpoint": ENDPOINT, "model_name": "", "dimension": 1024,
    })
    assert code == 400, payload
    assert payload["code"] == "embedding_endpoint_incomplete"
    assert "model name" in payload["error"]


def test_embedding_config_put_does_not_default_a_blank_width_to_768(local_store):
    from superlocalmemory.server.routes.v3_api import set_embedding_config

    code, payload = _call(set_embedding_config, {
        "provider": "openai", "api_endpoint": ENDPOINT, "model_name": "Qwen3-Embedding",
    })
    assert code == 400, payload
    assert payload["code"] == "embedding_endpoint_incomplete"
    assert "dimension" in payload["error"]


def test_embedding_config_put_still_defaults_the_local_width(local_store):
    from superlocalmemory.server.routes.v3_api import set_embedding_config

    code, payload = _call(set_embedding_config, {"provider": "", "model_name": NOMIC})
    assert code == 200, payload
    assert payload["dimension"] == 768


def test_reindex_target_refuses_an_endpoint_without_a_model_or_width():
    from superlocalmemory.server.routes.embedding_reindex import target_config

    live = EmbeddingConfig(provider="", model_name=NOMIC, dimension=768)

    with pytest.raises(ValueError) as no_model:
        target_config(live, {"provider": "openai", "api_endpoint": ENDPOINT,
                             "model_name": " ", "dimension": 1024})
    assert "embedding_endpoint_incomplete" in str(no_model.value) or "model name" in str(no_model.value)

    with pytest.raises(ValueError) as no_width:
        target_config(live, {"provider": "openai", "api_endpoint": ENDPOINT,
                             "model_name": "Qwen3-Embedding"})
    assert "dimension" in str(no_width.value)


def test_reindex_target_accepts_a_complete_endpoint():
    from superlocalmemory.server.routes.embedding_reindex import target_config

    live = EmbeddingConfig(provider="", model_name=NOMIC, dimension=768)
    target = target_config(live, {"provider": "openai", "api_endpoint": ENDPOINT,
                                  "model_name": "Qwen3-Embedding", "dimension": 1024})
    assert (target.model_name, target.dimension) == ("Qwen3-Embedding", 1024)


# ---------------------------------------------------------------------------
# A blank width never reaches the vector tables
# ---------------------------------------------------------------------------

def _tables(db: Path) -> set[str]:
    conn = sqlite3.connect(str(db))
    try:
        return {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    finally:
        conn.close()


@pytest.mark.parametrize("width", [0, -1])
def test_an_unknown_width_does_not_drop_existing_vector_tables(tmp_path, width):
    pytest.importorskip("sqlite_vec")
    from superlocalmemory.retrieval.vector_store import VectorStore, VectorStoreConfig

    db = tmp_path / "vec.db"
    store = VectorStore(db, VectorStoreConfig(dimension=4))
    assert store.available
    assert store.upsert("f1", "default", [1.0, 0.0, 0.0, 0.0]) is True
    del store

    guarded = VectorStore(db, VectorStoreConfig(dimension=width))

    assert guarded.available is False
    assert {"fact_embeddings", "embedding_metadata"} <= _tables(db)

    reopened = VectorStore(db, VectorStoreConfig(dimension=4))
    assert reopened.available
    assert reopened.count("default") == 1
