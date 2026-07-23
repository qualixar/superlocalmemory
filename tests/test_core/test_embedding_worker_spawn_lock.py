# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

from __future__ import annotations

import multiprocessing
import time
from unittest.mock import MagicMock, call, patch

from superlocalmemory.core.config import EmbeddingConfig
from superlocalmemory.core.embeddings import EmbeddingService


def _contend_for_embedding_lock(start, results) -> None:
    from superlocalmemory.core.embeddings import (
        acquire_embedding_lock,
        release_embedding_lock,
    )

    start.wait()
    acquired = acquire_embedding_lock(timeout=0.4)
    results.put(acquired)
    if acquired:
        time.sleep(0.8)
        release_embedding_lock()


def test_embedding_lock_allows_only_one_cold_process() -> None:
    ctx = multiprocessing.get_context("spawn")
    start = ctx.Event()
    results = ctx.Queue()
    processes = [
        ctx.Process(target=_contend_for_embedding_lock, args=(start, results))
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    observed = [results.get(timeout=5) for _ in processes]
    for process in processes:
        process.join(timeout=5)
        assert process.exitcode == 0
    assert sorted(observed) == [False, True]


def test_ensure_worker_rechecks_pid_after_lock() -> None:
    service = EmbeddingService(EmbeddingConfig())
    with (
        patch(
            "superlocalmemory.core.embeddings.acquire_embedding_lock",
            return_value=True,
        ),
        patch(
            "superlocalmemory.core.embeddings._is_embedding_worker_alive",
            return_value=True,
        ),
        patch(
            "superlocalmemory.core.embeddings.release_embedding_lock",
        ) as release,
        patch("superlocalmemory.core.embeddings.subprocess.Popen") as popen,
    ):
        service._ensure_worker()
    release.assert_called_once_with()
    popen.assert_not_called()


def test_ensure_worker_releases_dead_child_lock_before_respawn() -> None:
    service = EmbeddingService(EmbeddingConfig())
    dead = MagicMock()
    dead.poll.return_value = 1
    dead.pid = 101
    service._worker_proc = dead
    service._owns_worker_lock = True

    replacement = MagicMock()
    replacement.pid = 202
    events: list[str] = []
    with (
        patch(
            "superlocalmemory.core.embeddings.release_embedding_lock",
            side_effect=lambda: events.append("release"),
        ),
        patch(
            "superlocalmemory.core.embeddings.acquire_embedding_lock",
            side_effect=lambda: events.append("acquire") or True,
        ),
        patch(
            "superlocalmemory.core.embeddings._is_embedding_worker_alive",
            return_value=False,
        ),
        patch.object(service, "_check_memory_pressure", return_value=True),
        patch(
            "superlocalmemory.core.embeddings.register_embedding_worker_pid",
        ) as register,
        patch(
            "superlocalmemory.core.embeddings.subprocess.Popen",
            side_effect=lambda *args, **kwargs: (
                events.append("spawn") or replacement
            ),
        ),
    ):
        service._ensure_worker()

    assert events[:3] == ["release", "acquire", "spawn"]
    assert service._worker_proc is replacement
    assert service._owns_worker_lock is True
    register.assert_has_calls([call(202)])
