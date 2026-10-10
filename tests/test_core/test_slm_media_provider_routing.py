"""``init_embedder`` for the managed provider: the daemon runs the model, everyone else asks it."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from superlocalmemory.core import engine_wiring, process_role
from superlocalmemory.core.config import EmbeddingConfig
from superlocalmemory.core.daemon_text_embedder import DaemonTextEmbedder
from superlocalmemory.core.embeddings import EmbeddingService
from superlocalmemory.runtimes.managed_text import ManagedTextEmbedder

EG2 = "google/embeddinggemma-2"


def _cfg(tmp_path, provider="slm-media", model=EG2):
    return SimpleNamespace(
        base_dir=tmp_path,
        embedding=EmbeddingConfig(model_name=model, dimension=768, provider=provider))


@pytest.fixture(autouse=True)
def _not_the_daemon():
    process_role.clear_daemon_process()
    yield
    process_role.clear_daemon_process()


@pytest.fixture()
def no_spawn(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a process was started")
    monkeypatch.setattr(subprocess, "Popen", boom)


def test_a_process_is_not_the_daemon_until_it_says_so():
    assert process_role.is_daemon_process() is False
    process_role.mark_daemon_process()
    assert process_role.is_daemon_process() is True
    process_role.clear_daemon_process()
    assert process_role.is_daemon_process() is False


def test_the_daemon_gets_the_managed_embedder(tmp_path, no_spawn):
    process_role.mark_daemon_process()
    emb = engine_wiring.init_embedder(_cfg(tmp_path))
    assert isinstance(emb, ManagedTextEmbedder)
    assert emb.model_name == EG2 and emb.dimension == 768
    assert emb.is_available is False          # no media environment installed in this folder
    assert emb.embed("x") is None             # keyword-only: no vector, and never another model's


def test_every_other_process_gets_the_daemon_embedder_and_starts_nothing(tmp_path, no_spawn):
    emb = engine_wiring.init_embedder(_cfg(tmp_path))
    assert isinstance(emb, DaemonTextEmbedder) and not isinstance(emb, EmbeddingService)
    assert emb.is_available is False          # no daemon owns this folder
    assert emb.embed("x") is None and emb.embed_batch(["a"]) == [None]


def test_the_provider_never_falls_back_to_the_other_models(tmp_path, monkeypatch, no_spawn):
    taken = []
    monkeypatch.setattr(engine_wiring, "_try_service_embedder", lambda *a: taken.append("st"))
    monkeypatch.setattr(engine_wiring, "_try_ollama_embedder", lambda *a: taken.append("ollama"))
    for daemon in (True, False):
        process_role.clear_daemon_process()
        if daemon:
            process_role.mark_daemon_process()
        assert engine_wiring.init_embedder(_cfg(tmp_path)) is not None
    assert taken == []


def test_existing_providers_route_exactly_as_before_in_the_daemon_too(tmp_path, monkeypatch):
    sentinel = object()
    monkeypatch.setattr(engine_wiring, "_try_service_embedder", lambda cls, cfg: sentinel)
    monkeypatch.setattr(engine_wiring, "_try_ollama_embedder", lambda cfg: sentinel)
    process_role.mark_daemon_process()
    for provider in ("sentence-transformers", "ollama", ""):
        cfg = _cfg(tmp_path, provider=provider, model="nomic-ai/nomic-embed-text-v1.5")
        assert engine_wiring.init_embedder(cfg) is sentinel


def test_the_daemon_marks_itself_before_it_builds_its_engine(monkeypatch):
    from superlocalmemory.server import unified_daemon

    seen = []

    def stop_here():
        seen.append(process_role.is_daemon_process())
        raise RuntimeError("stop")

    monkeypatch.setattr(unified_daemon, "_start_memory_watchdog", stop_here)
    with pytest.raises(RuntimeError, match="stop"):
        unified_daemon._serve_owned(0, "127.0.0.1", None, None)
    assert seen == [True]


def test_a_process_that_does_not_win_the_folder_is_not_marked(monkeypatch):
    from superlocalmemory.server import unified_daemon

    monkeypatch.setattr(unified_daemon, "assert_no_durable_root_conflict", lambda: None)
    monkeypatch.setattr(unified_daemon, "install_thread_dump_signal", lambda: None)
    monkeypatch.setattr(unified_daemon, "get_instance_lock", lambda: SimpleNamespace(path="x"))
    monkeypatch.setattr(unified_daemon, "acquire_with_backoff", lambda *a: False)
    unified_daemon.start_server(port=0)
    assert process_role.is_daemon_process() is False
