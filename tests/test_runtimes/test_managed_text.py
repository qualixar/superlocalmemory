"""The managed text provider inside the daemon: one shared worker, prompts, warm and cold."""

from __future__ import annotations

import subprocess

import pytest

from superlocalmemory.core.config import EmbeddingConfig
from superlocalmemory.runtimes import features, worker_client
from superlocalmemory.runtimes.managed_text import ManagedTextEmbedder
from tests.test_runtimes.conftest import StubEnv

FAKE = "fake:768"


def _config(model: str = FAKE, dim: int = 768) -> EmbeddingConfig:
    return EmbeddingConfig(model_name=model, dimension=dim, provider="slm-media")


class RecordingClient:
    """Stands in for the shared worker client: records the prompt of every call."""

    def __init__(self, dim: int = 768, warm: bool = False) -> None:
        self.calls: list[tuple[list[str], str]] = []
        self.dim, self.warm, self.stopped = dim, warm, 0

    def embed_texts(self, texts, *, prompt):
        self.calls.append((list(texts), prompt))
        self.warm = True
        return [[0.5] * self.dim for _ in texts]

    def is_warm(self) -> bool:
        return self.warm

    def stop(self) -> None:
        self.stopped += 1
        self.warm = False


@pytest.fixture(autouse=True)
def _fresh_clients(monkeypatch):
    worker_client._CLIENTS.clear()
    monkeypatch.setattr(worker_client, "register_media_stop_hook", lambda f: None)
    yield
    for client in list(worker_client._CLIENTS.values()):
        client.stop()
    worker_client._CLIENTS.clear()


def _embedder(tmp_path, client=None, **kw) -> ManagedTextEmbedder:
    supplier = (lambda: client) if client is not None else None
    return ManagedTextEmbedder(_config(), data_root=tmp_path, env=StubEnv(tmp_path),
                               client_supplier=supplier, **kw)


def test_documents_and_queries_reach_the_worker_with_their_own_prompt(tmp_path):
    client = RecordingClient()
    emb = _embedder(tmp_path, client)
    emb.embed("one memory")
    emb.embed_batch(["a", "b"])
    emb.embed_query("what did I save?")
    assert client.calls == [(["one memory"], "Document"), (["a", "b"], "Document"),
                            (["what did I save?"], "SearchQuery")]


def test_a_wrong_width_is_refused_like_the_other_embedders(tmp_path):
    from superlocalmemory.core.embeddings import DimensionMismatchError

    emb = _embedder(tmp_path, RecordingClient(dim=384))
    with pytest.raises(DimensionMismatchError):
        emb.embed("x")


def test_empty_input_is_refused(tmp_path):
    emb = _embedder(tmp_path, RecordingClient())
    with pytest.raises(ValueError):
        emb.embed("  ")
    with pytest.raises(ValueError):
        emb.embed_batch([])


def test_the_interface_the_engine_reads(tmp_path):
    emb = _embedder(tmp_path, RecordingClient())
    assert emb.dimension == 768 and emb.model_name == FAKE
    assert emb.is_available is True and emb._available is True
    assert isinstance(emb.is_warm, bool) and emb.is_warm is False


def test_cold_then_warm_then_unloaded_keeps_has_loaded_once(tmp_path):
    client = RecordingClient()
    emb = _embedder(tmp_path, client)
    assert emb.is_warm is False and emb.has_loaded_once is False
    emb.embed("x")
    assert emb.is_warm is True and emb.has_loaded_once is True
    client.warm = False  # the idle timer stopped the worker
    assert emb.is_warm is False and emb.has_loaded_once is True


def test_a_worker_that_cannot_answer_gives_none_not_an_exception(tmp_path):
    class Broken(RecordingClient):
        def embed_texts(self, texts, *, prompt):
            raise worker_client.MediaWorkerError("The image model could not be loaded.")

    emb = _embedder(tmp_path, Broken())
    assert emb.embed("x") is None
    assert emb.embed_batch(["a", "b"]) == [None, None]
    assert emb.embed_query("q") is None


def test_an_environment_that_is_not_ready_is_unavailable_and_starts_nothing(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a process was started")
    monkeypatch.setattr(subprocess, "Popen", boom)
    emb = ManagedTextEmbedder(_config(), data_root=tmp_path, env=StubEnv(tmp_path, "installing"))
    assert emb.is_available is False and emb._available is False
    assert emb.embed("x") is None and emb.embed_batch(["a"]) == [None]
    assert emb.is_warm is False


def test_fisher_params_match_the_other_embedders(tmp_path):
    from superlocalmemory.core.embeddings import EmbeddingService

    vec = [0.1, -0.4, 0.2, 0.7]
    expected = EmbeddingService(EmbeddingConfig(dimension=4)).compute_fisher_params(vec)
    assert _embedder(tmp_path, RecordingClient()).compute_fisher_params(vec) == expected


def test_text_only_uses_the_text_loadout_and_a_real_fake_worker_answers(tmp_path):
    emb = ManagedTextEmbedder(_config(), data_root=tmp_path, env=StubEnv(tmp_path))
    vec = emb.embed("a memory")
    assert len(vec) == 768
    [client] = worker_client.live_clients()
    assert client.role == "text" and client.is_warm() and emb.is_warm is True
    assert emb.embed_batch(["a", "b"])[0] is not None and emb.embed_query("q") is not None


def test_with_pictures_on_text_and_pictures_share_one_worker(tmp_path):
    features._write_features(tmp_path, {"schema": 1, "media": {"enabled": True}})
    env = StubEnv(tmp_path)
    emb = ManagedTextEmbedder(_config(), data_root=tmp_path, env=env)
    emb.embed("a memory")
    picture = worker_client.media_embedder(env=env, data_root=tmp_path, model_id=FAKE, revision="")
    [client] = worker_client.live_clients()
    assert picture is client and client.role == ""


def test_pictures_turned_on_later_do_not_leave_two_copies(tmp_path):
    env = StubEnv(tmp_path)
    emb = ManagedTextEmbedder(_config(), data_root=tmp_path, env=env)
    emb.embed("a memory")
    [text_client] = worker_client.live_clients()
    assert text_client.role == "text" and text_client.pid is not None
    features._write_features(tmp_path, {"schema": 1, "media": {"enabled": True}})
    emb.embed("another")
    assert text_client.pid is None  # the text-only copy was stopped
    assert [c.role for c in worker_client.live_clients() if c.pid is not None] == [""]


def test_shutdown_stops_the_worker_only_when_pictures_are_off(tmp_path):
    emb = ManagedTextEmbedder(_config(), data_root=tmp_path, env=StubEnv(tmp_path))
    emb.embed("x")
    [client] = worker_client.live_clients()
    emb.shutdown()
    assert client.pid is None
    assert emb.embed("x") is None  # a shut embedder stays shut

    features._write_features(tmp_path, {"schema": 1, "media": {"enabled": True}})
    worker_client._CLIENTS.clear()
    on = ManagedTextEmbedder(_config(), data_root=tmp_path, env=StubEnv(tmp_path))
    on.embed("x")
    [full] = worker_client.live_clients()
    on.shutdown()
    on.unload()
    assert full.pid is not None  # the picture channel still uses it


def test_a_second_wrapper_closing_does_not_stop_the_first_ones_worker(tmp_path):
    env = StubEnv(tmp_path)
    live = ManagedTextEmbedder(_config(), data_root=tmp_path, env=env)
    probe = ManagedTextEmbedder(_config(), data_root=tmp_path, env=env)  # e.g. a re-index probe
    live.embed("x")
    probe.embed("y")
    probe.shutdown()
    [client] = worker_client.live_clients()
    assert client.pid is not None and live.embed("z") is not None
