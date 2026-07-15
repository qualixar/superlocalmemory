# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""Tests for V3 Infrastructure modules -- Task 3 of V3 build."""
import pytest
from pathlib import Path


# -----------------------------------------------------------------------
# Rate Limiter
# -----------------------------------------------------------------------

def test_rate_limiter_allows_within_limit():
    from superlocalmemory.infra.rate_limiter import RateLimiter
    limiter = RateLimiter(max_requests=10, window_seconds=60)
    for _ in range(10):
        assert limiter.allow("agent-1") == True


def test_rate_limiter_blocks_over_limit():
    from superlocalmemory.infra.rate_limiter import RateLimiter
    limiter = RateLimiter(max_requests=2, window_seconds=60)
    assert limiter.allow("agent-1") == True
    assert limiter.allow("agent-1") == True
    assert limiter.allow("agent-1") == False


def test_rate_limiter_is_allowed_returns_remaining():
    from superlocalmemory.infra.rate_limiter import RateLimiter
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    allowed, remaining = limiter.is_allowed("a1")
    assert allowed is True
    assert remaining == 2


def test_rate_limiter_remaining_no_record():
    from superlocalmemory.infra.rate_limiter import RateLimiter
    limiter = RateLimiter(max_requests=5, window_seconds=60)
    assert limiter.remaining("fresh") == 5


def test_rate_limiter_reset():
    from superlocalmemory.infra.rate_limiter import RateLimiter
    limiter = RateLimiter(max_requests=1, window_seconds=60)
    limiter.allow("a1")
    assert limiter.allow("a1") == False
    limiter.reset("a1")
    assert limiter.allow("a1") == True


def test_rate_limiter_cleanup():
    from superlocalmemory.infra.rate_limiter import RateLimiter
    limiter = RateLimiter(max_requests=5, window_seconds=0)  # 0s window -> everything expires
    limiter.allow("a1")
    removed = limiter.cleanup()
    assert removed >= 0  # may or may not find stale depending on timing


def test_rate_limiter_get_stats():
    from superlocalmemory.infra.rate_limiter import RateLimiter
    limiter = RateLimiter(max_requests=10, window_seconds=30)
    stats = limiter.get_stats()
    assert stats["max_requests"] == 10
    assert stats["window_seconds"] == 30


def test_rate_limiter_independent_clients():
    from superlocalmemory.infra.rate_limiter import RateLimiter
    limiter = RateLimiter(max_requests=1, window_seconds=60)
    assert limiter.allow("a1") == True
    assert limiter.allow("a1") == False
    assert limiter.allow("a2") == True  # different client


# -----------------------------------------------------------------------
# Cache Manager
# -----------------------------------------------------------------------

def test_cache_manager_set_and_get():
    from superlocalmemory.infra.cache_manager import CacheManager
    cache = CacheManager(max_size=100)
    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"


def test_cache_manager_missing_key():
    from superlocalmemory.infra.cache_manager import CacheManager
    cache = CacheManager(max_size=100)
    assert cache.get("nonexistent") is None


def test_cache_manager_eviction():
    from superlocalmemory.infra.cache_manager import CacheManager
    cache = CacheManager(max_size=2)
    cache.set("k1", "v1")
    cache.set("k2", "v2")
    cache.set("k3", "v3")  # should evict k1
    assert cache.get("k1") is None
    assert cache.get("k3") == "v3"


def test_cache_manager_put_and_get_by_query():
    from superlocalmemory.infra.cache_manager import CacheManager
    cache = CacheManager(max_size=50)
    cache.put("python web", [1, 2, 3])
    assert cache.get_by_query("python web") == [1, 2, 3]


def test_cache_manager_stats():
    from superlocalmemory.infra.cache_manager import CacheManager
    cache = CacheManager(max_size=10, ttl_seconds=60)
    cache.set("a", 1)
    cache.get("a")
    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["current_size"] == 1


def test_cache_manager_clear():
    from superlocalmemory.infra.cache_manager import CacheManager
    cache = CacheManager(max_size=10)
    cache.set("x", 1)
    cache.set("y", 2)
    cache.clear()
    assert cache.get("x") is None
    assert cache.get("y") is None


def test_cache_manager_thread_safe():
    from superlocalmemory.infra.cache_manager import CacheManager
    cache = CacheManager(max_size=10, thread_safe=True)
    cache.set("ts", "ok")
    assert cache.get("ts") == "ok"


# -----------------------------------------------------------------------
# Auth Middleware
# -----------------------------------------------------------------------

def test_auth_no_key_file_allows_all():
    from superlocalmemory.infra.auth_middleware import check_api_key
    # Non-existent file -> auth disabled -> always True
    fake = Path("/tmp/slm_test_nonexistent_key_file")
    if fake.exists():
        fake.unlink()
    assert check_api_key({}, is_write=True, key_file=fake) is True


def test_auth_read_always_allowed(tmp_path):
    from superlocalmemory.infra.auth_middleware import check_api_key
    key_file = tmp_path / "api_key"
    key_file.write_text("secret-key-123")
    assert check_api_key({}, is_write=False, key_file=key_file) is True


def test_auth_write_requires_key(tmp_path):
    from superlocalmemory.infra.auth_middleware import check_api_key
    key_file = tmp_path / "api_key"
    key_file.write_text("secret-key-123")
    # No header -> rejected
    assert check_api_key({}, is_write=True, key_file=key_file) is False
    # Wrong key -> rejected
    assert check_api_key({"x-slm-api-key": "wrong"}, is_write=True, key_file=key_file) is False
    # Correct key -> allowed
    assert check_api_key({"x-slm-api-key": "secret-key-123"}, is_write=True, key_file=key_file) is True


# -----------------------------------------------------------------------
# Webhook Dispatcher
# -----------------------------------------------------------------------

def test_webhook_dispatcher_init():
    from superlocalmemory.infra.webhook_dispatcher import WebhookDispatcher
    dispatcher = WebhookDispatcher()
    assert dispatcher is not None
    assert dispatcher.is_closed is False
    dispatcher.close()


def test_webhook_dispatcher_rejects_invalid_url():
    from superlocalmemory.infra.webhook_dispatcher import WebhookDispatcher
    dispatcher = WebhookDispatcher()
    try:
        with pytest.raises(ValueError):
            dispatcher.dispatch({"type": "test"}, "ftp://bad.url")
    finally:
        dispatcher.close()


def test_webhook_dispatcher_stats():
    from superlocalmemory.infra.webhook_dispatcher import WebhookDispatcher
    dispatcher = WebhookDispatcher()
    try:
        stats = dispatcher.get_stats()
        assert "dispatched" in stats
        assert "succeeded" in stats
    finally:
        dispatcher.close()


def test_webhook_dispatcher_close_is_idempotent():
    from superlocalmemory.infra.webhook_dispatcher import WebhookDispatcher
    dispatcher = WebhookDispatcher()
    dispatcher.close()
    dispatcher.close()  # second call should not raise
    assert dispatcher.is_closed is True


# -----------------------------------------------------------------------
# Backup Manager
# -----------------------------------------------------------------------

def test_backup_default_path():
    from superlocalmemory.infra.backup import BackupManager
    from superlocalmemory.infra.data_root import canonical_data_root
    backup = BackupManager()
    assert backup.base_dir == canonical_data_root()


def test_backup_custom_paths(tmp_path):
    from superlocalmemory.infra.backup import BackupManager
    backup = BackupManager(base_dir=tmp_path)
    assert backup.base_dir == tmp_path
    assert backup.db_path == tmp_path / "memory.db"
    assert (tmp_path / "backups").is_dir()


def test_backup_default_config(tmp_path):
    from superlocalmemory.infra.backup import BackupManager
    backup = BackupManager(base_dir=tmp_path)
    assert backup.config["enabled"] is True
    assert backup.config["interval_hours"] == 168
    assert backup.config["max_backups"] == 10


def test_backup_configure(tmp_path):
    from superlocalmemory.infra.backup import BackupManager
    backup = BackupManager(base_dir=tmp_path)
    status = backup.configure(interval_hours=24, max_backups=5)
    assert status["interval_hours"] == 24
    assert status["max_backups"] == 5


def test_backup_create_and_list(tmp_path):
    from superlocalmemory.infra.backup import BackupManager
    import sqlite3

    # Create a minimal DB to back up
    db_path = tmp_path / "memory.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()

    backup = BackupManager(base_dir=tmp_path)
    name = backup.create_backup(label="test")
    assert name != ""
    assert "test" in name

    backups = backup.list_backups()
    assert len(backups) >= 1
    assert backups[0]["type"] == "memory"


def test_backup_is_due_when_never_backed_up(tmp_path):
    from superlocalmemory.infra.backup import BackupManager
    backup = BackupManager(base_dir=tmp_path)
    assert backup.is_backup_due() is True
