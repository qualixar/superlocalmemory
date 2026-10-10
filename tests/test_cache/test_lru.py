# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The in-memory LRU layer."""

def test_cache_manager_set_and_get():
    from superlocalmemory.cache.lru import CacheManager
    cache = CacheManager(max_size=100)
    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"


def test_cache_manager_missing_key():
    from superlocalmemory.cache.lru import CacheManager
    cache = CacheManager(max_size=100)
    assert cache.get("nonexistent") is None


def test_cache_manager_eviction():
    from superlocalmemory.cache.lru import CacheManager
    cache = CacheManager(max_size=2)
    cache.set("k1", "v1")
    cache.set("k2", "v2")
    cache.set("k3", "v3")  # should evict k1
    assert cache.get("k1") is None
    assert cache.get("k3") == "v3"


def test_cache_manager_put_and_get_by_query():
    from superlocalmemory.cache.lru import CacheManager
    cache = CacheManager(max_size=50)
    cache.put("python web", [1, 2, 3])
    assert cache.get_by_query("python web") == [1, 2, 3]


def test_cache_manager_stats():
    from superlocalmemory.cache.lru import CacheManager
    cache = CacheManager(max_size=10, ttl_seconds=60)
    cache.set("a", 1)
    cache.get("a")
    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["current_size"] == 1


def test_cache_manager_clear():
    from superlocalmemory.cache.lru import CacheManager
    cache = CacheManager(max_size=10)
    cache.set("x", 1)
    cache.set("y", 2)
    cache.clear()
    assert cache.get("x") is None
    assert cache.get("y") is None


def test_cache_manager_thread_safe():
    from superlocalmemory.cache.lru import CacheManager
    cache = CacheManager(max_size=10, thread_safe=True)
    cache.set("ts", "ok")
    assert cache.get("ts") == "ok"
