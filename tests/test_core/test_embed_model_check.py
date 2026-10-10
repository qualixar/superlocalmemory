"""Every embed request names its model; a daemon on another model never answers with vectors."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from superlocalmemory.core import daemon_text_embedder as dte
from superlocalmemory.server.routes import v3_api

EG2 = "google/embeddinggemma-2"
NOMIC = "nomic-ai/nomic-embed-text-v1.5"


def _daemon(monkeypatch, model, *, dim=768, embed_reply=None):
    calls = []

    def fake(method, path, body, timeout):
        calls.append((method, path, body))
        if path.endswith("/ping"):
            return {"ok": True, "embedder": {"available": True, "warm": True, "model": model,
                                             "dimension": dim}}
        if embed_reply is not None:
            return embed_reply
        return {"embeddings": [[0.1] * 768 for _ in body["texts"]]}

    monkeypatch.setattr(dte, "_owned_daemon_request", fake)
    return calls


def _embedder():
    return dte.DaemonTextEmbedder(SimpleNamespace(model_name=EG2, dimension=768))


def test_a_daemon_on_another_model_gets_no_embed_request_and_no_vectors(monkeypatch):
    calls = _daemon(monkeypatch, NOMIC)
    emb = _embedder()
    assert emb.embed("a memory") is None
    assert emb.embed_query("what did I say") is None
    assert emb.embed_batch(["a", "b"]) == [None, None]
    assert [c for c in calls if c[0] == "POST"] == []
    assert emb.is_available is False and emb._available is False


def test_an_embed_reply_never_makes_the_embedder_available(monkeypatch):
    _daemon(monkeypatch, NOMIC)
    emb = _embedder()
    emb._ask(["x"], "document")
    assert emb._ok is False and emb.has_loaded_once is False


def test_every_embed_body_names_the_model_and_size(monkeypatch):
    calls = _daemon(monkeypatch, EG2)
    emb = _embedder()
    assert len(emb.embed("x")) == 768
    post = [c for c in calls if c[0] == "POST"][0][2]
    assert post["model"] == EG2 and post["dimension"] == 768


def test_a_model_mismatch_answer_means_unavailable_and_asks_again_next_time(monkeypatch):
    calls = []
    ping = {"ok": True, "embedder": {"available": True, "warm": True, "model": EG2, "dimension": 768}}

    def fake(method, path, body, timeout):  # a 409 reaches the caller as no answer
        calls.append(path)
        return ping if path.endswith("/ping") else None

    monkeypatch.setattr(dte, "_owned_daemon_request", fake)
    emb = _embedder()
    assert emb.embed("x") is None
    assert emb._ok is False
    pings = len([p for p in calls if p.endswith("/ping")])
    emb.is_available  # the next look pings again instead of waiting for the cache to expire
    assert len([p for p in calls if p.endswith("/ping")]) == pings + 1


class _Embedder:
    _available = True
    is_available = True
    is_warm = True
    model_name = EG2
    dimension = 4

    def __init__(self) -> None:
        self.calls = 0

    def embed_batch(self, texts):
        self.calls += 1
        return [[1.0] * 4 for _ in texts]


class _Request:
    def __init__(self, body, embedder) -> None:
        self._body = body
        self.app = SimpleNamespace(state=SimpleNamespace(engine=SimpleNamespace(_embedder=embedder)))

    async def json(self):
        return self._body


def _post(body, embedder):
    response = asyncio.run(v3_api.embed_texts(_Request(body, embedder)))
    return response if isinstance(response, dict) else (response.status_code, json.loads(response.body))


def test_the_route_refuses_a_request_for_another_model_without_embedding():
    emb = _Embedder()
    assert _post({"texts": ["a"], "model": NOMIC, "dimension": 4}, emb) == (409, {"error": "model_mismatch"})
    assert _post({"texts": ["a"], "model": EG2, "dimension": 5}, emb) == (409, {"error": "model_mismatch"})
    assert emb.calls == 0


def test_the_route_embeds_a_matching_request_and_old_callers_unchanged():
    emb = _Embedder()
    assert _post({"texts": ["a"], "model": EG2, "dimension": 4}, emb) == {"embeddings": [[1.0] * 4]}
    assert _post({"texts": ["a"]}, emb) == {"embeddings": [[1.0] * 4]}
    assert _post({"texts": ["a"], "model": NOMIC}, emb) == {"embeddings": [[1.0] * 4]}


def test_the_ping_never_asks_an_embedder_that_probes_for_its_availability():
    class Probing(_Embedder):
        _available = True

        @property
        def is_available(self):
            raise AssertionError("the ping must not probe on the event loop")

    ping = asyncio.run(v3_api.embed_ping(_Request({}, Probing())))
    assert ping["ok"] is True and ping["embedder"]["available"] is True


def test_an_embedder_not_yet_probed_is_reported_unavailable_without_probing():
    class Lazy(_Embedder):
        _available = None

        @property
        def is_available(self):
            raise AssertionError("the ping must not probe on the event loop")

    ping = asyncio.run(v3_api.embed_ping(_Request({}, Lazy())))
    assert ping["embedder"]["available"] is False


def test_a_question_to_an_embedder_without_a_question_prompt_is_one_batch_call():
    class Old:
        def __init__(self) -> None:
            self.calls = []

        def embed_batch(self, texts):
            self.calls.append(list(texts))
            return [[9.0]] * len(texts)

        def embed(self, text):
            raise AssertionError("per-text embed must not be used")

    old = Old()
    assert dte.embed_with_prompt(old, ["a", "  "], "query") == [[9.0], [9.0]]
    assert old.calls == [["a", "  "]]
