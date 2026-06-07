"""stampede.py — per-key mutex preventing cache stampede."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Generator


class StampedeShield:
    """Per-key mutex preventing cache stampede (thundering-herd) on cold miss."""

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout
        self._locks: dict[str, threading.RLock] = {}
        self._refcounts: dict[str, int] = {}
        self._meta_lock = threading.Lock()

    def _acquire_key_lock(self, key: str) -> threading.RLock:
        with self._meta_lock:
            if key not in self._locks:
                self._locks[key] = threading.RLock()
                self._refcounts[key] = 0
            self._refcounts[key] += 1
            return self._locks[key]

    def _release_key_lock(self, key: str) -> None:
        with self._meta_lock:
            count = self._refcounts.get(key, 0) - 1
            if count <= 0:
                self._locks.pop(key, None)
                self._refcounts.pop(key, None)
            else:
                self._refcounts[key] = count

    @contextmanager
    def lock(self, key: str) -> Generator[None, None, None]:
        rlock = self._acquire_key_lock(key)
        acquired = rlock.acquire(timeout=self._timeout)
        if not acquired:
            try:
                yield
            finally:
                self._release_key_lock(key)
            return
        try:
            yield
        finally:
            rlock.release()
            self._release_key_lock(key)
