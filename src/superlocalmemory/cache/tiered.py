# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""Memory in front of disk: the common repeat is answered without touching SQLite."""

from __future__ import annotations

import threading

from superlocalmemory.cache.keys import CacheKey
from superlocalmemory.cache.lru import LruCache
from superlocalmemory.cache.port import check_payload
from superlocalmemory.cache.sqlite_store import SqliteDeriveCache

L1_ENTRIES = 2048


def _l1_key(key: CacheKey) -> str:
    return "|".join(key.as_tuple())


class TieredCache:
    def __init__(self, l2: SqliteDeriveCache) -> None:
        self.l2 = l2
        self.l1 = LruCache(max_size=L1_ENTRIES, ttl_seconds=None, thread_safe=True)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: CacheKey) -> bytes | None:
        hit = self.l1.get(_l1_key(key))
        if hit is not None:
            self._count(True)
            return hit[1]
        payload = self.l2.get(key)
        if payload is None:
            self._count(False)
            return None
        self.l1.set(_l1_key(key), ("", payload))
        self._count(True)
        return payload

    def _count(self, hit: bool) -> None:
        with self._lock:
            if hit:
                self._hits += 1
            else:
                self._misses += 1

    def put(self, key: CacheKey, payload: bytes, *, kind: str) -> None:
        check_payload(payload, kind)
        data = bytes(payload)
        self.l2.put(key, data, kind=kind)
        self.l1.set(_l1_key(key), (kind, data))

    def invalidate(self, *, deriver_id: str | None = None, model_id: str | None = None) -> int:
        if deriver_id is None and model_id is None:
            raise ValueError("invalidate needs deriver_id or model_id; use clear() for all")
        self.l1.clear()  # simple and always correct
        return self.l2.invalidate(deriver_id=deriver_id, model_id=model_id)

    def invalidate_content(self, content_sha256: str) -> int:
        self.l1.clear()
        return self.l2.invalidate_content(content_sha256)

    def clear(self) -> None:
        self.l1.clear()
        self.l2.clear()

    def stats(self) -> dict:
        with self._lock:
            hits, misses = self._hits, self._misses
        return {"backend": "tiered", "hits": hits, "misses": misses,
                "l1": self.l1.get_stats(), "l2": self.l2.stats()}
