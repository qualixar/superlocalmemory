# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The interprocess lock every config.json writer takes (4.2.14).

It lives here, not in ``server/config_file.py``, so ``core.config`` can take
it too without importing the server layer (.importlinter ratchet):
``SLMConfig.save()`` and ``slm config set`` must serialize on one lock, or one
silently undoes the other.

One lock file per target, ``.<name>.lock`` next to it, plus one in-process
RLock. Re-entrant per thread: a second ``flock`` on a new open file blocks
even inside the same process (BSD/macOS and Linux alike), so a thread that
already holds the lock for a target takes it again without touching the file.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_PROCESS_LOCK = threading.RLock()
_HELD = threading.local()


def _held_by_this_thread() -> dict[str, int]:
    held = getattr(_HELD, "depths", None)
    if held is None:
        held = {}
        _HELD.depths = held
    return held


def _lock_handle(handle) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        handle.write(b"\0")
        handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock_handle(handle) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def config_file_lock(path: Path) -> Iterator[None]:
    """Serialize config access across daemon, CLI, and worker processes."""
    lock_path = Path(path).with_name(f".{Path(path).name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    key = str(lock_path.resolve())
    with _PROCESS_LOCK:
        held = _held_by_this_thread()
        if held.get(key):
            held[key] += 1
            try:
                yield
            finally:
                held[key] -= 1
            return
        with lock_path.open("a+b") as handle:
            _lock_handle(handle)
            held[key] = 1
            try:
                yield
            finally:
                del held[key]
                _unlock_handle(handle)


__all__ = ("config_file_lock",)
