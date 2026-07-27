"""Issue #91 regressions: daemon teardown and dashboard config truth."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

from superlocalmemory.core.config import EmbeddingConfig, SLMConfig
from superlocalmemory.core.engine import MemoryEngine
from superlocalmemory.storage.models import Mode


class _SlowPool:
    """A pool which would block teardown if it were waited on."""

    def __init__(self) -> None:
        self.calls: list[tuple[bool, bool]] = []

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        self.calls.append((wait, cancel_futures))
        if wait:
            raise AssertionError("engine close must not wait for an embed task")


class _Unloadable:
    def __init__(self) -> None:
        self.calls = 0

    def unload(self) -> None:
        self.calls += 1

    def close(self) -> None:
        self.calls += 1


class _Reranker:
    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def shutdown(self, *, timeout: float) -> None:
        self.timeouts.append(timeout)


class _Retrieval:
    def __init__(self, reranker: _Reranker) -> None:
        self._reranker = reranker
        self.wait_args: list[bool] = []

    def close(self, *, wait: bool) -> None:
        self.wait_args.append(wait)


def test_close_is_idempotent_and_never_waits_for_background_workers(tmp_path: Path) -> None:
    """Shutdown releases owned resources without waiting on a stuck worker."""
    engine = MemoryEngine(SLMConfig.for_mode(Mode.A, base_dir=tmp_path))
    pool = _SlowPool()
    embedder = _Unloadable()
    reranker = _Reranker()
    retrieval = _Retrieval(reranker)
    db = _Unloadable()

    engine._initialized = True
    engine._store_fast_embed_pool = pool
    engine._store_fast_embed_pool_lock = threading.Lock()
    engine._embedder = embedder
    engine._retrieval_engine = retrieval
    engine._db = db

    engine.close()
    engine.close()

    assert pool.calls == [(False, True)]
    assert embedder.calls == 1
    assert reranker.timeouts == [1.0]
    assert retrieval.wait_args == [False]
    assert db.calls == 1
    assert engine._store_fast_embed_pool is None
    assert engine._embedder is None
    assert engine._retrieval_engine is None
    assert engine._db is None
    assert engine._initialized is False


def test_embedding_service_unload_does_not_wait_for_an_active_embed_lock() -> None:
    """A stalled embedding request must not make daemon cleanup block forever."""
    from superlocalmemory.core.embeddings import EmbeddingService

    service = EmbeddingService.__new__(EmbeddingService)
    service._lock = threading.Lock()
    assert service._lock.acquire(blocking=False)
    try:
        assert service.unload(timeout=0.0) is False
    finally:
        service._lock.release()


def test_embedding_service_shutdown_kills_worker_despite_active_embed_lock() -> None:
    """Engine shutdown must not leave the detached model child behind."""
    from unittest.mock import Mock

    from superlocalmemory.core.embeddings import EmbeddingService

    service = EmbeddingService.__new__(EmbeddingService)
    service._lock = threading.Lock()
    service._idle_timer = None
    service._worker_ready = True
    service._owns_worker_lock = False
    service._http_client = None
    process = Mock()
    process.poll.return_value = None
    process.wait.side_effect = [TimeoutError, None]
    service._worker_proc = process
    assert service._lock.acquire(blocking=False)
    try:
        service.shutdown(timeout=0.0)
    finally:
        service._lock.release()

    process.kill.assert_called_once()
    assert service._worker_proc is None


def test_embedding_config_uses_the_running_daemon_config(tmp_path: Path, monkeypatch) -> None:
    """The dashboard must not replace a running Mode B config with disk defaults."""
    from superlocalmemory.server.routes.v3_api import get_embedding_config

    live = SLMConfig.for_mode(Mode.B, base_dir=tmp_path)
    live.embedding = EmbeddingConfig(
        provider="openai",
        model_name="qwen3-embedding",
        dimension=1024,
        api_endpoint="http://127.0.0.1:8045/v1",
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(config=live)))

    def _disk_load_must_not_run():
        raise AssertionError("endpoint read disk config instead of daemon app state")

    monkeypatch.setattr(SLMConfig, "load", staticmethod(_disk_load_must_not_run))

    body = asyncio.run(get_embedding_config(request))

    assert body["provider"] == "openai"
    assert body["model_name"] == "qwen3-embedding"
    assert body["dimension"] == 1024
    assert body["mode"] == "b"


def test_embedding_config_mutation_starts_from_running_daemon_config(
    tmp_path: Path, monkeypatch,
) -> None:
    """A live profile must not be overwritten by a stale disk read on update."""
    from unittest.mock import AsyncMock

    from superlocalmemory.server import rbac_enforce
    from superlocalmemory.server.routes import v3_api

    live = SLMConfig.for_mode(Mode.B, base_dir=tmp_path)
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(config=live)),
        json=AsyncMock(return_value={"provider": "openai", "model_name": "qwen3"}),
    )
    monkeypatch.setattr(rbac_enforce, "require_manage", lambda request: None)
    monkeypatch.setattr(v3_api, "_apply_runtime_config", AsyncMock())
    monkeypatch.setattr(
        SLMConfig,
        "load",
        staticmethod(lambda: (_ for _ in ()).throw(AssertionError("stale disk config read"))),
    )

    body = asyncio.run(v3_api.set_embedding_config(request))

    assert body["success"] is True
    assert live.embedding.provider == "openai"
    assert live.embedding.model_name == "qwen3"


def test_every_dashboard_settings_surface_exposes_embedding_config_failure() -> None:
    """A failed config fetch must not masquerade as Mode A/default settings."""
    root = Path(__file__).parents[1]
    for relative_path in (
        "src/superlocalmemory/ui/js/auto-settings.js",
        "src/superlocalmemory/ui/js/od-settings.js",
    ):
        source = (root / relative_path).read_text()
        assert "Embedding configuration unavailable" in source, relative_path
