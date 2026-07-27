"""Lifecycle regression tests for best-effort deferred-write workers."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

from superlocalmemory.storage.deferred_writes import (
    DeferredLastSeen,
    get_deferred_last_seen,
    shutdown_deferred_writes,
    submit_background,
)


def _named_threads(name: str) -> list[threading.Thread]:
    return [thread for thread in threading.enumerate() if thread.name == name]


def test_deferred_last_seen_stop_joins_flusher() -> None:
    writer = DeferredLastSeen(MagicMock(), interval_s=60)

    writer.stop(timeout=1)

    assert not writer._thread.is_alive()


def test_shutdown_resets_deferred_singletons_and_background_writer() -> None:
    db = MagicMock()
    writer = get_deferred_last_seen(db)
    completed = threading.Event()

    submit_background(completed.set)
    assert completed.wait(timeout=1)

    shutdown_deferred_writes(timeout=1)

    assert not writer._thread.is_alive()
    assert _named_threads("slm-deferred-lastseen") == []
    assert _named_threads("slm-bg-writer") == []

    replacement = get_deferred_last_seen(db)
    assert replacement is not writer
    shutdown_deferred_writes(timeout=1)
