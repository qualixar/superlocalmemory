# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""LRU cache with optional TTL for search-result caching.

Key features:
    * O(1) get / set via ``OrderedDict``
    * Time-to-live expiry per entry
    * Size-based eviction (LRU)
    * Optional thread safety
    * Hit / miss / eviction statistics
"""

import hashlib
import json
import logging
import time
from collections import OrderedDict
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("superlocalmemory.cache")


class CacheEntry:
    """Single cache entry with metadata."""

    __slots__ = ["value", "timestamp", "access_count", "size_estimate"]

    def __init__(self, value: Any, size_estimate: int = 0) -> None:
        self.value = value
        self.timestamp = time.time()
        self.access_count = 0
        self.size_estimate = size_estimate

    def is_expired(self, ttl_seconds: Optional[float]) -> bool:
        """Return ``True`` when the entry has exceeded *ttl_seconds*."""
        if ttl_seconds is None:
            return False
        return (time.time() - self.timestamp) > ttl_seconds

    def mark_accessed(self) -> None:
        """Increment access counter."""
        self.access_count += 1


class CacheManager:
    """LRU cache manager with TTL support.

    Args:
        max_size: Maximum number of entries before LRU eviction.
        ttl_seconds: Per-entry time-to-live (``None`` = never expire).
        thread_safe: Wrap operations in a reentrant lock.
    """

    def __init__(
        self,
        max_size: int = 100,
        ttl_seconds: Optional[float] = 300,
        thread_safe: bool = False,
    ) -> None:
        self.max_size = max(1, max_size)
        self.ttl_seconds = ttl_seconds
        self.thread_safe = thread_safe

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock: Optional[RLock] = RLock() if thread_safe else None

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._total_size_estimate = 0

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_key(query: str, **kwargs: Any) -> str:
        """Deterministic cache key from *query* + keyword args."""
        key_data = {"query": query, **kwargs}
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    @staticmethod
    def _estimate_size(value: Any) -> int:
        """Rough byte-size estimate for analytics."""
        try:
            if isinstance(value, list):
                return len(value) * 100
            return len(json.dumps(value, default=str))
        except Exception:
            return 1000

    # ------------------------------------------------------------------
    # Core API (simple key-based)
    # ------------------------------------------------------------------

    def set(self, key: str, value: Any) -> None:
        """Store *value* under a plain string *key*.

        This is the simple interface expected by the test suite.
        """
        self._put_internal(key, value)

    def get(self, key: str, **kwargs: Any) -> Optional[Any]:
        """Retrieve value by plain string *key*.

        Returns ``None`` on cache miss or expiry.
        """
        if kwargs:
            key = self._hash_key(key, **kwargs)
        return self._get_internal(key)

    # ------------------------------------------------------------------
    # Query-oriented API (hashed keys)
    # ------------------------------------------------------------------

    def put(self, query: str, value: Any, **kwargs: Any) -> None:
        """Store *value* keyed by the hash of *query* + *kwargs*."""
        hashed = self._hash_key(query, **kwargs)
        self._put_internal(hashed, value)

    def get_by_query(self, query: str, **kwargs: Any) -> Optional[Any]:
        """Retrieve value keyed by the hash of *query* + *kwargs*."""
        hashed = self._hash_key(query, **kwargs)
        return self._get_internal(hashed)

    # ------------------------------------------------------------------
    # Internal get / put (shared logic)
    # ------------------------------------------------------------------

    def _get_internal(self, key: str) -> Optional[Any]:
        if self._lock:
            self._lock.acquire()
        try:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            if entry.is_expired(self.ttl_seconds):
                del self._cache[key]
                self._total_size_estimate -= entry.size_estimate
                self._misses += 1
                return None

            self._cache.move_to_end(key)
            entry.mark_accessed()
            self._hits += 1
            return entry.value
        finally:
            if self._lock:
                self._lock.release()

    def _put_internal(self, key: str, value: Any) -> None:
        size_estimate = self._estimate_size(value)

        if self._lock:
            self._lock.acquire()
        try:
            if key in self._cache:
                old = self._cache[key]
                self._total_size_estimate -= old.size_estimate
                del self._cache[key]

            if len(self._cache) >= self.max_size:
                _, evicted = self._cache.popitem(last=False)
                self._total_size_estimate -= evicted.size_estimate
                self._evictions += 1

            self._cache[key] = CacheEntry(value, size_estimate)
            self._total_size_estimate += size_estimate
        finally:
            if self._lock:
                self._lock.release()

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def invalidate(self, query: str, **kwargs: Any) -> bool:
        """Remove entry for *query*. Returns ``True`` if it existed."""
        key = self._hash_key(query, **kwargs)
        if self._lock:
            self._lock.acquire()
        try:
            if key in self._cache:
                entry = self._cache.pop(key)
                self._total_size_estimate -= entry.size_estimate
                return True
            return False
        finally:
            if self._lock:
                self._lock.release()

    def clear(self) -> None:
        """Remove all entries."""
        if self._lock:
            self._lock.acquire()
        try:
            self._cache.clear()
            self._total_size_estimate = 0
        finally:
            if self._lock:
                self._lock.release()

    def evict_expired(self) -> int:
        """Manually remove all expired entries. Returns count removed."""
        if self.ttl_seconds is None:
            return 0
        if self._lock:
            self._lock.acquire()
        try:
            expired = [
                k for k, e in self._cache.items()
                if e.is_expired(self.ttl_seconds)
            ]
            for k in expired:
                entry = self._cache.pop(k)
                self._total_size_estimate -= entry.size_estimate
            return len(expired)
        finally:
            if self._lock:
                self._lock.release()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics snapshot."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0

        avg_access = 0.0
        if self._cache:
            avg_access = sum(e.access_count for e in self._cache.values()) / len(self._cache)

        return {
            "max_size": self.max_size,
            "current_size": len(self._cache),
            "ttl_seconds": self.ttl_seconds,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "evictions": self._evictions,
            "total_size_estimate_kb": self._total_size_estimate / 1024,
            "avg_access_count": avg_access,
            "thread_safe": self.thread_safe,
        }

    def get_top_queries(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Return the *limit* most-accessed cache keys."""
        if self._lock:
            self._lock.acquire()
        try:
            items = [
                (k, e.access_count) for k, e in self._cache.items()
            ]
            items.sort(key=lambda x: x[1], reverse=True)
            return items[:limit]
        finally:
            if self._lock:
                self._lock.release()


#: The name the tiered cache and later callers use.
LruCache = CacheManager

__all__ = ["CacheEntry", "CacheManager", "LruCache"]
