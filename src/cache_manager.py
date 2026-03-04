#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""SuperLocalMemory V2 - Cache Manager

Solution Architect & Original Creator

 (see LICENSE file)

ATTRIBUTION REQUIRED: This notice must be preserved in all copies.
"""
"""
Cache Manager - LRU Cache for Search Results

Implements Least Recently Used (LRU) cache for search query results to reduce
redundant computation and improve response times.

Key Features:
1. LRU Eviction Policy: Automatically removes least recently used entries
2. TTL Support: Optional time-to-live for cache entries
3. Size-Based Eviction: Maximum cache size in number of entries
4. Memory-Efficient: Uses OrderedDict for O(1) access and updates
5. Thread-Safe: Optional thread safety for concurrent access

Performance Impact:
- Cache hit: ~0.1ms (negligible overhead)
- Cache miss: Standard search time
- Target cache hit rate: 30-50% for typical usage

Usage:
    cache = CacheManager(max_size=100, ttl_seconds=300)

    # Try cache first
    result = cache.get("python web")
    if result is None:
        # Cache miss - perform search
        result = search_engine.search("python web")
        cache.put("python web", result)
"""

import time
import hashlib
import json
from collections import OrderedDict
from typing import Any, Optional, Dict, Tuple
from threading import RLock


class CacheEntry:
    """
    Single cache entry with metadata.

    Stores:
    - value: Cached result
    - timestamp: Creation time for TTL validation
    - access_count: Number of times accessed (for analytics)
    - size_estimate: Memory size estimate in bytes
    """

    __slots__ = ['value', 'timestamp', 'access_count', 'size_estimate']

    def __init__(self, value: Any, size_estimate: int = 0):
        """
        Create cache entry.

        Args:
            value: Value to cache
            size_estimate: Estimated size in bytes
        """
        self.value = value
        self.timestamp = time.time()
        self.access_count = 0
        self.size_estimate = size_estimate

    def is_expired(self, ttl_seconds: Optional[float]) -> bool:
        """
        Check if entry has exceeded TTL.

        Args:
            ttl_seconds: Time-to-live in seconds (None = no expiry)

        Returns:
            True if expired, False otherwise
        """
        if ttl_seconds is None:
            return False

        age = time.time() - self.timestamp
        return age > ttl_seconds

    def mark_accessed(self) -> None:
        """Mark entry as accessed (increment counter)."""
        self.access_count += 1


class CacheManager:
    """
    LRU cache manager for search results with TTL support.

    Uses OrderedDict to maintain insertion/access order efficiently.
    When cache is full, least recently used entry is evicted.

    Thread-safe when thread_safe=True.
    """

    def __init__(
        self,
        max_size: int = 100,
        ttl_seconds: Optional[float] = 300,
        thread_safe: bool = False
    ):
        """
        Initialize cache manager.

        Args:
            max_size: Maximum number of cache entries
            ttl_seconds: Time-to-live for entries (None = no expiry)
            thread_safe: Enable thread-safe operations
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.thread_safe = thread_safe

        # LRU cache storage
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Thread safety lock
        self._lock = RLock() if thread_safe else None

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._total_size_estimate = 0

    def _hash_key(self, query: str, **kwargs) -> str:
        """
        Generate cache key from query and parameters.

        Args:
            query: Search query
            **kwargs: Additional parameters to include in key

        Returns:
            Hash string for cache key
        """
        # Create deterministic key from query + parameters
        key_data = {
            'query': query,
            **kwargs
        }

        # Sort keys for deterministic hashing
        key_str = json.dumps(key_data, sort_keys=True)

        # Hash for compact key
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def _estimate_size(self, value: Any) -> int:
        """
        Estimate memory size of cached value.

        Rough estimate for monitoring memory usage.

        Args:
            value: Value to estimate

        Returns:
            Estimated size in bytes
        """
        try:
            # For lists of tuples (typical search results)
            if isinstance(value, list):
                # Rough estimate: 100 bytes per result
                return len(value) * 100

            # For other types, try JSON serialization size
            return len(json.dumps(value, default=str))
        except Exception:
            # Fallback: assume moderate size
            return 1000

    def get(
        self,
        query: str,
        **kwargs
    ) -> Optional[Any]:
        """
        Get cached result for query.

        Args:
            query: Search query
            **kwargs: Additional parameters used in cache key

        Returns:
            Cached result if found and valid, None otherwise
        """
        key = self._hash_key(query, **kwargs)

        # Thread-safe access
        if self._lock:
            self._lock.acquire()

        try:
            # Check if key exists
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check TTL expiry
            if entry.is_expired(self.ttl_seconds):
                # Remove expired entry
                del self._cache[key]
                self._total_size_estimate -= entry.size_estimate
                self._misses += 1
                return None

            # Move to end (mark as recently used)
            self._cache.move_to_end(key)
            entry.mark_accessed()

            self._hits += 1
            return entry.value

        finally:
            if self._lock:
                self._lock.release()

    def put(
        self,
        query: str,
        value: Any,
        **kwargs
    ) -> None:
        """
        Store result in cache.

        Args:
            query: Search query
            value: Result to cache
            **kwargs: Additional parameters used in cache key
        """
        key = self._hash_key(query, **kwargs)
        size_estimate = self._estimate_size(value)

        # Thread-safe access
        if self._lock:
            self._lock.acquire()

        try:
            # Check if key already exists (update)
            if key in self._cache:
                old_entry = self._cache[key]
                self._total_size_estimate -= old_entry.size_estimate
                del self._cache[key]

            # Check if cache is full
            if len(self._cache) >= self.max_size:
                # Evict least recently used (first item)
                evicted_key, evicted_entry = self._cache.popitem(last=False)
                self._total_size_estimate -= evicted_entry.size_estimate
                self._evictions += 1

            # Add new entry (at end = most recently used)
            entry = CacheEntry(value, size_estimate)
            self._cache[key] = entry
            self._total_size_estimate += size_estimate

        finally:
            if self._lock:
                self._lock.release()

    def invalidate(self, query: str, **kwargs) -> bool:
        """
        Remove specific entry from cache.

        Args:
            query: Search query
            **kwargs: Additional parameters

        Returns:
            True if entry was removed, False if not found
        """
        key = self._hash_key(query, **kwargs)

        if self._lock:
            self._lock.acquire()

        try:
            if key in self._cache:
                entry = self._cache[key]
                del self._cache[key]
                self._total_size_estimate -= entry.size_estimate
                return True
            return False

        finally:
            if self._lock:
                self._lock.release()

    def clear(self) -> None:
        """Clear entire cache."""
        if self._lock:
            self._lock.acquire()

        try:
            self._cache.clear()
            self._total_size_estimate = 0

        finally:
            if self._lock:
                self._lock.release()

    def evict_expired(self) -> int:
        """
        Manually evict all expired entries.

        Returns:
            Number of entries evicted
        """
        if self.ttl_seconds is None:
            return 0

        if self._lock:
            self._lock.acquire()

        try:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired(self.ttl_seconds)
            ]

            for key in expired_keys:
                entry = self._cache[key]
                del self._cache[key]
                self._total_size_estimate -= entry.size_estimate

            return len(expired_keys)

        finally:
            if self._lock:
                self._lock.release()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        # Average access count
        avg_access_count = 0.0
        if self._cache:
            avg_access_count = sum(
                entry.access_count for entry in self._cache.values()
            ) / len(self._cache)

        return {
            'max_size': self.max_size,
            'current_size': len(self._cache),
            'ttl_seconds': self.ttl_seconds,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'evictions': self._evictions,
            'total_size_estimate_kb': self._total_size_estimate / 1024,
            'avg_access_count': avg_access_count,
            'thread_safe': self.thread_safe
        }

    def get_top_queries(self, limit: int = 10) -> list:
        """
        Get most frequently accessed queries.

        Args:
            limit: Maximum number of queries to return

        Returns:
            List of (query_hash, access_count) tuples
        """
        if self._lock:
            self._lock.acquire()

        try:
            queries = [
                (key, entry.access_count)
                for key, entry in self._cache.items()
            ]

            queries.sort(key=lambda x: x[1], reverse=True)
            return queries[:limit]

        finally:
            if self._lock:
                self._lock.release()


# CLI interface for testing
if __name__ == "__main__":
    import random

    print("Cache Manager - Demo")
    print("=" * 60)

    # Initialize cache
    cache = CacheManager(max_size=5, ttl_seconds=10)

    print("\nCache Configuration:")
    stats = cache.get_stats()
    print(f"  Max Size: {stats['max_size']}")
    print(f"  TTL: {stats['ttl_seconds']}s")

    # Simulate search queries
    queries = [
        "python programming",
        "javascript web",
        "machine learning",
        "database sql",
        "api rest"
    ]

    # Mock search results
    def mock_search(query: str) -> None:
        """Simulate search result."""
        return [
            (f"doc_{i}", random.random())
            for i in range(3)
        ]

    print("\n" + "=" * 60)
    print("Simulating Search Operations:")
    print("=" * 60)

    # First pass - all cache misses
    print("\nPass 1 (Cold Cache):")
    for query in queries:
        result = cache.get(query)
        if result is None:
            print(f"  MISS: '{query}' - performing search")
            result = mock_search(query)
            cache.put(query, result)
        else:
            print(f"  HIT:  '{query}'")

    # Second pass - all cache hits
    print("\nPass 2 (Warm Cache):")
    for query in queries:
        result = cache.get(query)
        if result is None:
            print(f"  MISS: '{query}' - performing search")
            result = mock_search(query)
            cache.put(query, result)
        else:
            print(f"  HIT:  '{query}'")

    # Third pass - add more queries to trigger eviction
    print("\nPass 3 (Cache Overflow - LRU Eviction):")
    extra_queries = [
        "neural networks",
        "cloud computing",
        "devops kubernetes"
    ]

    for query in extra_queries:
        result = cache.get(query)
        if result is None:
            print(f"  MISS: '{query}' - performing search")
            result = mock_search(query)
            cache.put(query, result)

    # Check if old queries were evicted
    print("\nPass 4 (Check Evictions):")
    for query in queries[:3]:
        result = cache.get(query)
        if result is None:
            print(f"  EVICTED: '{query}'")
        else:
            print(f"  RETAINED: '{query}'")

    # Display statistics
    print("\n" + "=" * 60)
    print("Cache Statistics:")
    print("=" * 60)

    stats = cache.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # Test TTL expiry
    print("\n" + "=" * 60)
    print("Testing TTL Expiry:")
    print("=" * 60)

    cache_ttl = CacheManager(max_size=10, ttl_seconds=2)
    cache_ttl.put("test query", mock_search("test"))

    print("\n  Immediately after cache:")
    result = cache_ttl.get("test query")
    print(f"    Result: {'HIT' if result else 'MISS'}")

    print("\n  After 3 seconds (exceeds TTL):")
    time.sleep(3)
    result = cache_ttl.get("test query")
    print(f"    Result: {'HIT' if result else 'MISS (expired)'}")

    print("\n" + "=" * 60)
    print("Performance Impact:")
    print("  Cache hit: ~0.1ms overhead")
    print("  Cache miss: Standard search time + 0.1ms")
    print("  Target hit rate: 30-50% for typical usage")
