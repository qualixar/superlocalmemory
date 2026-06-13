"""exact.py — SQLite-backed exact-match get/set, cacheable-response guard."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from superlocalmemory.optimize.cache.key_builder import CacheConfig

logger = logging.getLogger(__name__)

_NON_CACHEABLE_FINISH_REASONS: frozenset[str] = frozenset({
    "tool_use",
    "tool_calls",
    "length",
    "max_tokens",
})


def _is_cacheable_response(response: dict) -> bool:
    finish = response.get("stop_reason") or response.get("finish_reason") or ""
    if finish in _NON_CACHEABLE_FINISH_REASONS:
        logger.debug("exact: skip cache (finish_reason=%r)", finish)
        return False
    choices = response.get("choices") or []
    for choice in choices:
        fr = (choice.get("finish_reason") or "")
        if fr in _NON_CACHEABLE_FINISH_REASONS:
            logger.debug("exact: skip cache (choice.finish_reason=%r)", fr)
            return False
        msg = choice.get("message") or {}
        if msg.get("tool_calls"):
            logger.debug("exact: skip cache (choice.message.tool_calls present)")
            return False
    for block in response.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            logger.debug("exact: skip cache (tool_use content block)")
            return False
    return True


class ExactCache:
    """Exact-match cache layer backed by llmcache.db."""

    def __init__(self, db: Any, config: CacheConfig | None = None) -> None:
        self._db = db
        self._config = config or CacheConfig()

    def get(self, key: str, tenant_id: str) -> dict | None:
        row = self._db.get(key, tenant_id)
        if row is None:
            return None
        return json.loads(row.value.decode("utf-8"))

    def set(
        self,
        key: str,
        tenant_id: str,
        response: dict,
        tags: list[str],
        model: str,
        ttl: int | None = None,
    ) -> bool:
        if not self.is_cacheable(response):
            return False
        serialized = json.dumps(response, separators=(",", ":"), default=str)
        encoded = serialized.encode("utf-8")
        if len(encoded) > self._config.max_response_bytes:
            return False
        effective_ttl = ttl if ttl is not None else self._config.default_ttl_seconds
        expires_at = time.time() + effective_ttl if effective_ttl > 0 else None
        self._db.set(
            key=key,
            tenant_id=tenant_id,
            value=encoded,
            model=model,
            ttl_expires=expires_at,
            tags=tags,
        )
        return True

    def delete(self, key: str, tenant_id: str) -> None:
        self._db.delete(key, tenant_id)

    def is_cacheable(self, response: dict) -> bool:
        return _is_cacheable_response(response)
