"""key_builder.py — deterministic SHA-256 cache key, tenant-scoped."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

DETERMINISTIC_PARAMS: frozenset[str] = frozenset({
    "max_tokens", "stop", "stop_sequences", "top_p", "top_k",
    "response_format", "tools", "tool_choice",
})

EXCLUDED_PARAMS: frozenset[str] = frozenset({
    "stream", "temperature", "seed", "user", "metadata",
    "request_id", "idempotency_key", "timeout", "max_retries",
})

_KEY_SCHEMA_VERSION = 1
_KEY_PREFIX = "slmcache"

_TENANT_ID_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class CacheConfig:
    """Runtime configuration for the cache engine."""
    default_ttl_seconds: int = 3600
    max_response_bytes: int = 1_048_576  # 1 MB
    allow_nonzero_temperature_cache: bool = False
    stampede_timeout_seconds: float = 30.0


class KeyBuilder:
    """Builds deterministic, tenant-scoped, collision-resistant SHA-256 cache keys."""

    def __init__(self, config: CacheConfig | None = None) -> None:
        self._config = config or CacheConfig()

    def build(
        self,
        *,
        tenant_id: str,
        model_id: str,
        model_version: str,
        system: str,
        messages: list,
        raw_params: dict | None = None,
    ) -> str | None:
        if not _TENANT_ID_RE.fullmatch(tenant_id or ""):
            raise ValueError(
                "tenant_id must be a 64-char lowercase hex SHA-256 digest "
                f"(got {len(tenant_id or '')} chars) — F4 multi-tenant isolation"
            )

        raw_params = raw_params or {}
        try:
            temperature = float(raw_params.get("temperature", 0) or 0)
        except (TypeError, ValueError):
            temperature = 0.0

        if temperature != 0 and not self._config.allow_nonzero_temperature_cache:
            return None

        deterministic_params: dict[str, Any] = {}
        if temperature == 0:
            deterministic_params["temperature"] = 0
        elif self._config.allow_nonzero_temperature_cache:
            deterministic_params["temperature"] = temperature

        for k in sorted(DETERMINISTIC_PARAMS):
            if k in raw_params:
                deterministic_params[k] = raw_params[k]

        payload: dict[str, Any] = {
            "v": _KEY_SCHEMA_VERSION,
            "tenant": tenant_id,
            "model": model_id,
            "model_version": model_version,
            "system": system,
            "messages": messages,
            "params": deterministic_params,
        }

        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, default=str,
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"{_KEY_PREFIX}:v{_KEY_SCHEMA_VERSION}:{tenant_id}:resp:{digest}"

    def tenant_tag(self, tenant_id: str) -> str:
        return f"tenant:{tenant_id}"

    def model_tag(self, model_id: str) -> str:
        return f"model:{model_id}"
