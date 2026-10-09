# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""Where the cache lives, which backend is used, and the model-change hook."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Callable

from superlocalmemory.cache.keys import CacheKey
from superlocalmemory.cache.port import CachePort
from superlocalmemory.cache.sqlite_store import SqliteDeriveCache
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
    with _lock:
        _instances.clear()
