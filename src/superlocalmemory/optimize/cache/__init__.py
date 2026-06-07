"""optimize/cache — exact-response cache (Phase 1)."""

from __future__ import annotations

from superlocalmemory.optimize.cache.exact import ExactCache
from superlocalmemory.optimize.cache.invalidation import InvalidationEngine
from superlocalmemory.optimize.cache.key_builder import CacheConfig, KeyBuilder
from superlocalmemory.optimize.cache.manager import (
    CacheManager,
    NoOpSemantic,
    SemanticTier,
    _TenantScopedManager,
)
from superlocalmemory.optimize.cache.stampede import StampedeShield

__all__ = [
    "CacheManager",
    "CacheView",
    "ExactCache",
    "InvalidationEngine",
    "KeyBuilder",
    "CacheConfig",
    "StampedeShield",
    "SemanticTier",
    "NoOpSemantic",
    "_TenantScopedManager",
]


# Lazy alias so import order doesn't matter
CacheView = _TenantScopedManager
