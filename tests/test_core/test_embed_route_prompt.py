"""POST /api/v3/embed: ``prompt`` picks document or question vectors; the default is unchanged."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from superlocalmemory.server.routes import v3_api


class _Embedder:
    _available = True
    is_available = True
    is_warm = False
    model_name = "google/embeddinggemma-2"
    dimension = 4

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def embed_batch(self, texts):
        self.calls.append(("document", list(texts)))
        return [[1.0] * 4 for _ in texts]

    def embed_query(self, text):
        self.calls.append(("query", [text]))
        return [2.0] * 4


class _Request:
    def __init__(self, body, embedder) -> None:
        self._body = body
        self.app = SimpleNamespace(state=SimpleNamespace(engine=SimpleNamespace(_embedder=embedder)))

    async def json(self):
        return self._body


def _post(body, embedder):
    response = asyncio.run(v3_api.embed_texts(_Request(body, embedder)))
    return response if isinstance(response, dict) else (response.status_code, json.loads(response.body))


def test_the_default_is_document_vectors_exactly_as_before():
    emb = _Embedder()
    assert _post({"texts": ["a", "b"]}, emb) == {"embeddings": [[1.0] * 4, [1.0] * 4]}
    assert emb.calls == [("document", ["a", "b"])]


def test_query_asks_the_question_prompt_per_text():
    emb = _Embedder()
    assert _post({"texts": ["a", "b"], "prompt": "query"}, emb) == {"embeddings": [[2.0] * 4, [2.0] * 4]}
    assert emb.calls == [("query", ["a"]), ("query", ["b"])]


def test_an_embedder_without_a_question_prompt_embeds_the_same_way():
    class Old:
        def embed_batch(self, texts):
            return [[9.0]] * len(texts)

        def embed(self, text):
            return [9.0]

    assert _post({"texts": ["a"], "prompt": "query"}, Old()) == {"embeddings": [[9.0]]}


def test_an_unknown_prompt_is_refused():
    status, body = _post({"texts": ["a"], "prompt": "poem"}, _Embedder())
    assert status == 400 and "prompt" in body["error"]


def test_the_ping_says_what_the_embedder_is_so_other_processes_never_mix_spaces():
    emb = _Embedder()
    request = _Request({}, emb)
    ping = asyncio.run(v3_api.embed_ping(request))
    assert ping["ok"] is True
    assert ping["embedder"] == {"available": True, "warm": False,
                                "model": "google/embeddinggemma-2", "dimension": 4}
    request.app.state.engine = None  # engine not built yet
    assert asyncio.run(v3_api.embed_ping(request))["embedder"]["available"] is False
