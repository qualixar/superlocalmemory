"""invalidation.py — tag-based bulk eviction."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from superlocalmemory.optimize.storage.db import CacheDB


class InvalidationEngine:
    """Tag-based bulk eviction for llmcache.db."""

    def __init__(self, db: "CacheDB") -> None:
        self._db = db

    def register(self, key: str, tenant_id: str, tags: list[str]) -> None:
        self._db.tag_register(key=key, tenant_id=tenant_id, tags=tags)

    def invalidate_tag(self, tag: str) -> int:
        return self._db.invalidate_by_tag(tag)

    def invalidate_model(self, model_id: str) -> int:
        return self.invalidate_tag(f"model:{model_id}")

    def invalidate_tenant(self, tenant_id: str) -> int:
        return self.invalidate_tag(f"tenant:{tenant_id}")

    def invalidate_key(self, key: str, tenant_id: str) -> None:
        self._db.delete(key, tenant_id)

    def get_tags_for_key(self, key: str, tenant_id: str) -> list[str]:
        row = self._db.get(key, tenant_id)
        if row is None:
            return []
        return row.tags
