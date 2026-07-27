"""Foreground recall priority for the shared embedding worker."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from superlocalmemory.core.config import EmbeddingConfig
from superlocalmemory.core.embeddings import EmbeddingService
from superlocalmemory.core.recall_gate import (
    background_work,
    begin_recall,
    end_recall,
)


def _service() -> EmbeddingService:
    service = EmbeddingService.__new__(EmbeddingService)
    service._config = SimpleNamespace(
        is_openai_compatible=False,
        is_cloud=False,
        dimension=1,
    )
    return service


def test_background_embed_waits_for_active_recall(monkeypatch) -> None:
    service = _service()
    entered = threading.Event()

    def fake_subprocess(texts):
        entered.set()
        return [[1.0] for _ in texts]

    monkeypatch.setattr(service, "_subprocess_embed", fake_subprocess)
    begin_recall()
    try:
        def run_background() -> None:
            with background_work():
                service.embed("background")

        worker = threading.Thread(target=run_background)
        worker.start()
        time.sleep(0.05)
        assert not entered.is_set()
    finally:
        end_recall()
    worker.join(timeout=1.0)
    assert entered.is_set()
    assert not worker.is_alive()


def test_background_batch_is_sliced_for_preemption(monkeypatch) -> None:
    service = _service()
    batches: list[list[str]] = []

    def fake_subprocess(texts):
        batches.append(list(texts))
        return [[1.0] for _ in texts]

    monkeypatch.setattr(service, "_subprocess_embed", fake_subprocess)
    with background_work():
        assert service.embed_batch(["one", "two", "three"]) == [
            [1.0], [1.0], [1.0],
        ]
    assert batches == [["one"], ["two"], ["three"]]


def test_local_embedding_worker_is_warm_only_after_successful_request() -> None:
    service = EmbeddingService.__new__(EmbeddingService)
    service._worker_proc = None
    service._request_count = 0

    assert service.is_warm is False

    service._worker_proc = SimpleNamespace(poll=lambda: None)
    assert service.is_warm is False

    service._request_count = 1
    assert service.is_warm is True

    service._worker_proc = SimpleNamespace(poll=lambda: 1)
    assert service.is_warm is False


def test_openai_embedding_is_warm_after_successful_http_request(monkeypatch) -> None:
    service = EmbeddingService(EmbeddingConfig(
        provider="openai",
        api_endpoint="http://localhost:8045/v1",
        dimension=1,
    ))
    monkeypatch.setattr(
        service,
        "_openai_compatible_embed_batch",
        lambda texts: [[1.0] for _ in texts],
    )

    assert service.is_warm is False
    assert service.embed("ready") == [1.0]
    assert service.is_warm is True


def test_cloud_embedding_is_warm_after_successful_http_request(monkeypatch) -> None:
    service = EmbeddingService(EmbeddingConfig(
        provider="cloud",
        api_endpoint="https://example.invalid",
        api_key="test-key",
        dimension=1,
    ))
    monkeypatch.setattr(service, "_cloud_embed_single", lambda text: [1.0])

    assert service.is_warm is False
    assert service.embed("ready") == [1.0]
    assert service.is_warm is True
