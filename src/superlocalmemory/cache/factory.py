# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""Where the cache lives, which backend is used, and the model-change hook."""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Callable

from superlocalmemory.cache.keys import CacheKey
from superlocalmemory.cache.port import CachePort
from superlocalmemory.cache.sqlite_store import SqliteDeriveCache, set_initial_meta
from superlocalmemory.cache.tiered import TieredCache

logger = logging.getLogger(__name__)

FILE_NAME = "derive_cache.db"
_lock = threading.Lock()
_instances: dict[str, CachePort] = {}


def derive_cache_path(data_root: str | Path | None = None) -> Path:
    if data_root is None:
        from superlocalmemory.infra.data_root import canonical_data_root

        data_root = canonical_data_root()
    return Path(data_root) / FILE_NAME


def _backend() -> str:
    raw = os.environ.get("SLM_CACHE_BACKEND", "").strip().lower()
    if raw in ("", "tiered", "sqlite"):
        return raw or "tiered"
    logger.warning("unknown SLM_CACHE_BACKEND %r; using the tiered cache", raw)
    return "tiered"


def default_cache() -> CachePort:
    """The process-wide cache for the current data root. Creates no file."""
    path = derive_cache_path()
    backend = _backend()
    ident = f"{backend}:{path}"
    with _lock:
        cache = _instances.get(ident)
        if cache is None:
            disk = SqliteDeriveCache(path)
            cache = disk if backend == "sqlite" else TieredCache(disk)
            _instances[ident] = cache
        return cache


def invalidate_for_model(model_id: str) -> int:
    """Drop everything derived with a model. Does nothing when no cache file exists."""
    if not model_id or not derive_cache_path().exists():
        return 0
    return default_cache().invalidate(model_id=model_id)


def invalidate_content(content_sha256: str, data_root: str | Path | None = None) -> int:
    """Drop everything derived from one file. Does nothing (and creates nothing) without a cache file."""
    path = derive_cache_path(data_root)
    if not content_sha256 or not path.exists():
        return 0
    if data_root is None or path == derive_cache_path():
        return default_cache().invalidate_content(content_sha256)
    return SqliteDeriveCache(path).invalidate_content(content_sha256)


def clear_derived_cache(reason: str) -> bool:
    """Delete the derivation cache file and drop every in-memory copy of it.

    Returns True when a cache file was removed, False when there was none or the
    removal failed. Never raises. The reason is logged; no cached content is.
    """
    try:
        path = derive_cache_path()
        if not path.exists():
            return False
        with _lock:
            for cache in list(_instances.values()):
                cache.clear()
            _instances.clear()
        SqliteDeriveCache(path).clear()  # covers a file no live instance owned
        logger.info("derivation cache cleared: %s", reason)
        return True
    except Exception as exc:
        logger.warning("derivation cache could not be cleared (%s): %s", reason, exc)
        return False


_POLICY_KEY = "redaction_policy"


def reconcile_redaction_policy(enabled: bool) -> bool:
    """Clear the cache when it was filled under a different redaction setting.

    The setting in force is stored in the cache file's metadata. Creates no file;
    a file created later starts with the current setting recorded. Never raises.
    """
    state = "on" if enabled else "off"
    set_initial_meta(_POLICY_KEY, state)
    try:
        path = derive_cache_path()
        if not path.exists():
            return False
        conn = sqlite3.connect(str(path), timeout=10.0)
        try:
            row = conn.execute("SELECT value FROM cache_meta WHERE key=?",
                               (_POLICY_KEY,)).fetchone()
            if row is None and not enabled:
                conn.execute("INSERT OR REPLACE INTO cache_meta (key, value) VALUES (?, ?)",
                             (_POLICY_KEY, state))
                conn.commit()
                return False
        finally:
            conn.close()
        if row is not None and row[0] == state:
            return False
        return clear_derived_cache("redaction setting changed")
    except Exception as exc:
        logger.warning("redaction policy check of the derivation cache failed: %s", exc)
        return False


def get_or_compute(cache: CachePort, key: CacheKey, kind: str,
                   compute: Callable[[], bytes]) -> bytes:
    """Return the cached payload, or compute and store it. A cache fault only costs a recompute."""
    try:
        hit = cache.get(key)
    except Exception as exc:
        logger.debug("cache read skipped: %s", exc)
        hit = None
    if hit is not None:
        return hit
    payload = compute()
    try:
        cache.put(key, payload, kind=kind)
    except Exception as exc:
        logger.debug("cache write skipped: %s", exc)
    return payload


def _reset_for_tests() -> None:
    from superlocalmemory.cache import sqlite_store

    sqlite_store._initial_meta.clear()
    with _lock:
        _instances.clear()
