# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""F5 regression: EmbeddingService must evict the worker mid-window when
memory pressure is detected, not only before spawn.

Scenario reproduced: The worker has been resident for 10 of its 30-minute
idle window.  Memory pressure spikes (another heavy process starts).
_check_memory_pressure() now returns False but is never called until the
next _ensure_worker() invocation.  After this fix, _reset_idle_timer()
calls _check_memory_pressure() opportunistically after each successful embed;
if pressure is detected the worker is killed immediately.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from superlocalmemory.core.config import EmbeddingConfig
from superlocalmemory.core.embeddings import EmbeddingService


def _make_service_with_live_worker() -> tuple[EmbeddingService, list[bool]]:
    """Return a service whose worker is pretending to be live, plus a kill-log."""
    svc = EmbeddingService(EmbeddingConfig())

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None  # process is alive

    svc._worker_proc = mock_proc
    svc._worker_ready = True
    svc._owns_worker_lock = False  # skip real file-lock release

    killed: list[bool] = []

    def _recording_kill() -> None:
        killed.append(True)
        # Simulate the real kill side-effects
        svc._worker_proc = None
        svc._worker_ready = False
        if svc._idle_timer is not None:
            svc._idle_timer.cancel()
            svc._idle_timer = None

    svc._kill_worker = _recording_kill  # type: ignore[method-assign]
    return svc, killed


def test_f5_reset_idle_timer_evicts_worker_under_pressure() -> None:
    """_reset_idle_timer must kill the worker when _check_memory_pressure returns False."""
    svc, killed = _make_service_with_live_worker()

    with patch.object(EmbeddingService, "_check_memory_pressure", return_value=False):
        svc._reset_idle_timer()

    assert killed, (
        "F5: _reset_idle_timer must call _kill_worker() when memory pressure is "
        "detected (i.e. _check_memory_pressure() returns False). "
        "Mid-window eviction not implemented."
    )


def test_f5_no_idle_timer_set_after_eviction() -> None:
    """After mid-window eviction, the idle timer must NOT be running."""
    svc, _ = _make_service_with_live_worker()

    with patch.object(EmbeddingService, "_check_memory_pressure", return_value=False):
        svc._reset_idle_timer()

    assert svc._idle_timer is None, (
        "F5: idle timer must be None after eviction under memory pressure "
        "(no new timer should be scheduled when the worker was just killed)."
    )


def test_f5_no_eviction_when_no_pressure() -> None:
    """When memory is available, _reset_idle_timer must start the idle timer normally."""
    svc, killed = _make_service_with_live_worker()

    try:
        with patch.object(EmbeddingService, "_check_memory_pressure", return_value=True):
            svc._reset_idle_timer()

        assert not killed, "Worker must NOT be killed when memory is available"
        assert svc._idle_timer is not None, (
            "Idle timer must be running when memory is OK"
        )
    finally:
        # Clean up the real timer so it doesn't fire during test teardown
        if svc._idle_timer is not None:
            svc._idle_timer.cancel()
            svc._idle_timer = None
