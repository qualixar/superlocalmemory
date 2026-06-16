"""manager.py — CacheManager orchestrator (singleton, fail-open, stampede-shielded)."""

from __future__ import annotations

import hashlib as _hashlib
import json as _json_mod
import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Callable

# C-08: precomputed hash for the common "default" tenant — avoids sha256 on every request
_DEFAULT_TENANT_HASH: str = _hashlib.sha256(b"default").hexdigest()

from superlocalmemory.optimize.cache.exact import ExactCache
from superlocalmemory.optimize.cache.invalidation import InvalidationEngine
from superlocalmemory.optimize.cache.key_builder import CacheConfig, KeyBuilder
from superlocalmemory.optimize.cache.stampede import StampedeShield
from superlocalmemory.optimize.metrics.counters import MetricsCollector
from superlocalmemory.optimize.proxy.lifecycle import (
    CachedResponse,
    ProxyRequest,
    ProviderResponse,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SemanticTier interface seam
# ---------------------------------------------------------------------------

class SemanticTier(ABC):
    """Abstract interface for the semantic cache tier (Phase 3).

    INTERFACE-CONTRACT §4 conformance: lookup/learn/index_entry/is_enabled.
    Phase 3 (LLD-03) implements VCacheSemantic.
    """

    @abstractmethod
    def lookup(self, req, tenant_id: str, embed):
        """Return semantically similar cached response or None. Fail-open."""
        ...

    @abstractmethod
    def learn(self, entry_id: str, similarity: float, was_correct: bool) -> None:
        """Update per-item MLE model with feedback. Fail-open."""
        ...

    @abstractmethod
    def index_entry(
        self, req, tenant_id: str, embed, resp
    ) -> None:
        """Index a new response vector in the ANN index. Fail-open.

        INTERFACE-CONTRACT v2 §4: canonical signature
        (self, req, tenant_id, embed, resp).
        """
        ...

    @abstractmethod
    def is_enabled(self) -> bool: ...


class NoOpSemantic(SemanticTier):
    """Phase 1 placeholder (also used when semantic_enabled=False)."""

    def lookup(self, req, tenant_id: str, embed):
        return None

    def learn(self, entry_id: str, similarity: float, was_correct: bool) -> None:
        return None

    def index_entry(self, req, tenant_id: str, embed, resp) -> None:
        return None

    def is_enabled(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class CacheMetrics:
    """Thread-safe counters. A-20 fix: lock guards the two-counter read."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.exact_hits: int = 0
        self.exact_misses: int = 0
        self.semantic_hits: int = 0
        self.semantic_misses: int = 0
        self.sets: int = 0
        self.skipped_non_cacheable: int = 0
        self.invalidations: int = 0
        self.stampede_contentions: int = 0
        self.errors: int = 0

    def hit_rate(self) -> float:
        with self._lock:
            total = self.exact_hits + self.exact_misses
            return self.exact_hits / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# CacheManager
# ---------------------------------------------------------------------------

class CacheManager:
    """Central orchestrator for the SLM Optimize exact cache."""

    _instance: "CacheManager | None" = None
    _instance_lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        db: Any,
        config: CacheConfig | None = None,
        semantic_tier: SemanticTier | None = None,
    ) -> None:
        self._db = db
        self._config = config or CacheConfig()
        self._key_builder = KeyBuilder(self._config)
        self._exact = ExactCache(db, self._config)
        self._stampede = StampedeShield(timeout=self._config.stampede_timeout_seconds)
        self._invalidation = InvalidationEngine(db)
        self._semantic = semantic_tier or NoOpSemantic()
        self._metrics = CacheMetrics()

    @classmethod
    def get_instance(cls) -> "CacheManager":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    from superlocalmemory.optimize.storage.db import CacheDB as _CacheDB
                    _db = _CacheDB.get_default()
                    cls._instance = cls(db=_db)
        return cls._instance

    @classmethod
    def set_instance(cls, instance: "CacheManager") -> None:
        with cls._instance_lock:
            cls._instance = instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (testing only)."""
        with cls._instance_lock:
            if cls._instance is not None:
                try:
                    cls._instance._db.close()  # type: ignore[attr-defined]
                except Exception:
                    pass
            cls._instance = None

    # ---- INTERFACE-CONTRACT §4 public methods ----

    def build_key(self, req: Any, tenant_id: str) -> str | None:
        """Build a deterministic cache key for req + tenant_id.

        BUG-FIX (v3.6.3): Two bugs repaired here:
        1. tenant_id="default" failed KeyBuilder's 64-char hex SHA-256 validation,
           silently raising ValueError caught by fail-open wrappers → cache never
           stored or retrieved anything via the proxy path. Fix: normalize any
           non-hex tenant_id to its SHA-256 digest before passing to KeyBuilder.
        2. ProxyRequest objects have a `body` dict, not model_id/messages/system
           attributes. The old `getattr(req, "model_id", "")` path silently
           returned empty strings → all proxy requests got the same (invalid) key.
           Fix: detect ProxyRequest and extract fields from .body.
        """
        import hashlib as _hashlib
        import json as _json
        import re as _re
        _HEX64 = _re.compile(r"[0-9a-f]{64}")
        # Normalize tenant_id: KeyBuilder requires a 64-char lowercase hex SHA-256.
        if not _HEX64.fullmatch(tenant_id or ""):
            tenant_id = _hashlib.sha256(tenant_id.encode()).hexdigest()

        if isinstance(req, ProxyRequest) and req.provider == "vertex":
            # CRIT-2 (WP-11, LOCKED): Vertex bodies have NO model/messages/system.
            # Model is in the PATH; prompts are under 'contents'; system under
            # 'systemInstruction'. Without this branch ALL Vertex requests hash to
            # ONE key → first response poisons every subsequent prompt.
            body = req.body or {}
            model_id = _vertex_model_from_path(req.path)
            messages = body.get("contents", []) or []
            system_raw = body.get("systemInstruction", "") or ""
            if isinstance(system_raw, (dict, list)):
                system = _json.dumps(system_raw, sort_keys=True, separators=(",", ":"))
            else:
                system = str(system_raw)
            # params: everything except the fields extracted above and stream flag.
            _SKIP_VERTEX = frozenset({"contents", "systemInstruction", "stream"})
            params = {k: v for k, v in body.items() if k not in _SKIP_VERTEX}
        elif isinstance(req, ProxyRequest):
            # Extract semantic fields from the parsed JSON body.
            body = req.body or {}
            model_id = body.get("model", "") or ""
            messages = body.get("messages", []) or []
            system_raw = body.get("system", "") or ""
            # Anthropic allows system as a list of content blocks — normalise to str.
            if isinstance(system_raw, list):
                system = _json.dumps(system_raw, sort_keys=True, separators=(",", ":"))
            else:
                system = str(system_raw)
            # params: everything except fields extracted above and stream flag.
            _SKIP = frozenset({"model", "messages", "system", "stream"})
            params = {k: v for k, v in body.items() if k not in _SKIP}
        elif isinstance(req, dict):
            model_id = req.get("model", "") or ""
            messages = req.get("messages", []) or []
            params = req.get("params", {}) or {}
            system = req.get("system", "") or ""
        else:
            model_id = getattr(req, "model_id", "") or ""
            messages = getattr(req, "messages", []) or []
            params = getattr(req, "params", {}) or {}
            system = getattr(req, "system", "") or ""
        return self._key_builder.build(
            tenant_id=tenant_id,
            model_id=model_id,
            model_version="",
            system=system,
            messages=messages,
            raw_params=params,
        )

    def get(self, req: Any, tenant_id: str) -> "CachedResponse | None":
        """CacheHook.check() entry point.

        BUG-FIX (v3.6.3): Previously returned None on cache miss, which caused
        _safe_cache_check to return CachedResponse(cache_key="").  The empty
        cache_key is falsy, so the store condition in handle_messages
        (``cache_result.cache_key``) was always False → cache was NEVER
        populated.  Fix: return a miss CachedResponse that carries the computed
        key so the store path can proceed.
        """
        key = self.build_key(req, tenant_id)
        if key is None:
            # Uncacheable (non-zero temperature, etc.) — signal with None.
            return None
        row = self._exact.get(key, tenant_id)
        if row is not None:
            self._metrics.exact_hits += 1
            return CachedResponse(
                hit=True,
                data=json_dumps_bytes(row),
                cache_key=key,
                ttl_seconds=0,
            )
        self._metrics.exact_misses += 1

        # C-01: semantic fallback on exact miss (only when explicitly enabled)
        if self._semantic.is_enabled():
            try:
                sem_result = self._semantic.lookup(req, tenant_id, None)
                if sem_result is not None:
                    self._metrics.semantic_hits += 1
                    if isinstance(sem_result, CachedResponse):
                        return sem_result
                    # VCacheSemantic may return a response dict — wrap it
                    return CachedResponse(
                        hit=True,
                        data=_json_mod.dumps(sem_result, ensure_ascii=False).encode(),
                        cache_key=key,
                        ttl_seconds=0,
                    )
            except Exception as exc:
                logger.warning("SemanticTier.lookup raised (fail-open): %s", exc)
            self._metrics.semantic_misses += 1

        # Return miss WITH the key so callers can use it for cache storage.
        return CachedResponse(hit=False, data=None, cache_key=key, ttl_seconds=0)

    def set(self, req: Any, resp: Any, tenant_id: str) -> None:
        """CacheHook.store() entry point."""
        import json as _json
        key = self.build_key(req, tenant_id)
        if key is None:
            return
        if isinstance(req, ProxyRequest):
            model_id = (req.body or {}).get("model", "") or ""
        elif isinstance(req, dict):
            model_id = req.get("model", "") or ""
        else:
            model_id = getattr(req, "model_id", "") or ""
        tags = [
            self._key_builder.model_tag(model_id),
            self._key_builder.tenant_tag(tenant_id),
        ]
        # resp may be ProviderResponse (proxy) or dict (manager)
        if isinstance(resp, dict):
            response_dict = resp
        else:
            response_dict = _json.loads(resp.body_bytes) if hasattr(resp, "body_bytes") else {}
        self._exact.set(key, tenant_id, response_dict, tags, model_id)
        self._invalidation.register(key, tenant_id, tags)
        self._metrics.sets += 1

        # C-02: index in semantic tier after exact write (fail-open)
        if self._semantic.is_enabled():
            try:
                self._semantic.index_entry(req, tenant_id, None, response_dict)
            except Exception as exc:
                logger.warning("SemanticTier.index_entry raised (fail-open): %s", exc)

    # ---- CacheHook protocol implementation (INTERFACE-CONTRACT §3) ----

    def check(self, req: ProxyRequest) -> "CachedResponse | None":
        """CacheHook.check() — look up by ProxyRequest; fail-open on error.

        BUG-FIX (v3.6.3): on_miss() was never called from the proxy path,
        so MetricsCollector.misses stayed at 0 and the dashboard always showed
        0 misses.  Fixed by calling on_miss() here whenever get() returns a
        cache-miss result.
        """
        try:
            result = self.get(req, tenant_id=_DEFAULT_TENANT_HASH)
            if result is not None and not result.hit:
                MetricsCollector.get_instance().on_miss()
            return result
        except Exception as exc:
            logger.warning("CacheManager.check raised (fail-open): %s", exc)
            return None

    def store(self, req: ProxyRequest, resp: ProviderResponse) -> None:
        """CacheHook.store() — persist response; fail-open on error."""
        try:
            self.set(req, resp, tenant_id=_DEFAULT_TENANT_HASH)
        except Exception as exc:
            logger.warning("CacheManager.store raised (fail-open): %s", exc)

    def on_hit(self, req: ProxyRequest, resp: bytes, tokens_saved: int) -> None:
        """CacheHook.on_hit() — forward token savings to MetricsCollector.

        M-01: compute input tokens from request body when caller passes 0.
        M-02: parse real output tokens from cached response usage field.
        All counts are estimates for display — not billing-accurate.
        """
        import json as _json

        # M-01: estimate input tokens from message content
        if tokens_saved == 0 and isinstance(req, ProxyRequest):
            try:
                body = req.body or {}
                total_chars = 0
                for m in (body.get("messages") or []):
                    c = m.get("content", "")
                    if isinstance(c, str):
                        total_chars += len(c)
                    elif isinstance(c, list):
                        for blk in c:
                            if isinstance(blk, dict):
                                total_chars += len(blk.get("text", "") or "")
                total_chars += len(body.get("system", "") or "")
                tokens_saved = max(0, total_chars // 4)
            except Exception:
                pass

        # M-02: parse real output tokens from stored response
        output_tokens = 0
        if resp:
            try:
                data = _json.loads(resp)
                usage = data.get("usage") or {}
                output_tokens = (
                    usage.get("output_tokens")
                    or usage.get("completion_tokens")
                    or 0
                )
            except Exception:
                output_tokens = len(resp) // 4  # fallback byte-estimate

        MetricsCollector.get_instance().on_hit(
            tokens_saved_input=tokens_saved,
            tokens_saved_output=output_tokens,
        )

    def on_miss(self, req: ProxyRequest) -> None:
        """CacheHook.on_miss() — forward miss event to MetricsCollector."""
        MetricsCollector.get_instance().on_miss()

    def set_semantic_tier(self, tier: SemanticTier) -> None:
        self._semantic = tier

    # ---- core request path ----

    def get_or_call(
        self,
        *,
        tenant_id: str,
        model_id: str,
        model_version: str,
        system: str,
        messages: list,
        raw_params: dict,
        upstream_fn: Callable[[], dict],
        http_status: int = 200,
        ttl: int | None = None,
        extra_tags: list | None = None,
    ) -> dict:
        from types import SimpleNamespace as _NS
        _req = _NS(
            model_id=model_id, model_version=model_version,
            system=system, messages=messages, params=raw_params,
        )
        try:
            return self._get_or_call_inner(
                req=_req,
                tenant_id=tenant_id, model_id=model_id, model_version=model_version,
                system=system, messages=messages, raw_params=raw_params,
                upstream_fn=upstream_fn, http_status=http_status,
                ttl=ttl, extra_tags=extra_tags or [],
            )
        except Exception as exc:
            logger.error(
                "CacheManager.get_or_call failed — falling through to upstream: %s", exc,
                exc_info=True,
            )
            self._metrics.errors += 1
            return upstream_fn()

    def _get_or_call_inner(
        self,
        *,
        req: Any,
        tenant_id: str,
        model_id: str,
        model_version: str,
        system: str,
        messages: list,
        raw_params: dict,
        upstream_fn: Callable[[], dict],
        http_status: int,
        ttl: int | None,
        extra_tags: list,
    ) -> dict:
        key = self._key_builder.build(
            tenant_id=tenant_id, model_id=model_id, model_version=model_version,
            system=system, messages=messages, raw_params=raw_params,
        )
        if key is None:
            self._metrics.exact_misses += 1
            return upstream_fn()

        cached = self._exact.get(key, tenant_id)
        if cached is not None:
            self._metrics.exact_hits += 1
            return cached

        if self._semantic.is_enabled():
            # vCache path: exact miss → optional semantic hit
            # (CacheManager does not pre-embed; the semantic tier embeds
            # lazily via its injected EmbeddingService. The VCache lookup
            # returns a response dict or None on miss/explore.)
            try:
                sem_hit = self._semantic.lookup(req, tenant_id, None)
                if sem_hit is not None:
                    self._metrics.semantic_hits += 1
                    return sem_hit
            except Exception as exc:
                logger.warning("SemanticTier.lookup raised (fail-open): %s", exc)
            self._metrics.semantic_misses += 1

        self._metrics.exact_misses += 1

        with self._stampede.lock(key):
            cached = self._exact.get(key, tenant_id)
            if cached is not None:
                self._metrics.stampede_contentions += 1
                return cached

            response = upstream_fn()

            if 200 <= http_status < 300:
                tags = [
                    self._key_builder.model_tag(model_id),
                    self._key_builder.tenant_tag(tenant_id),
                    *extra_tags,
                ]
                stored = self._exact.set(key, tenant_id, response, tags, model_id, ttl)
                if stored:
                    self._invalidation.register(key, tenant_id, tags)
                    self._metrics.sets += 1
                else:
                    self._metrics.skipped_non_cacheable += 1

        return response

    # ---- tenant scoping ----

    def for_tenant(self, tenant_id: str) -> "_TenantScopedManager":
        return _TenantScopedManager(manager=self, tenant_id=tenant_id)

    # ---- invalidation API ----

    def invalidate_tag(self, tag: str) -> int:
        count = self._invalidation.invalidate_tag(tag)
        self._metrics.invalidations += count
        return count

    def invalidate_by_tag(self, tag: str) -> int:
        """INTERFACE-CONTRACT v2 §4: delegate to invalidation engine by tag."""
        return self.invalidate_tag(tag)

    def invalidate_model(self, model_id: str) -> int:
        return self.invalidate_tag(f"model:{model_id}")

    def invalidate_tenant(self, tenant_id: str) -> int:
        return self.invalidate_tag(f"tenant:{tenant_id}")

    @property
    def metrics(self) -> CacheMetrics:
        return self._metrics


class _TenantScopedManager:
    """Thin view over CacheManager with tenant_id pre-filled."""

    def __init__(self, manager: CacheManager, tenant_id: str) -> None:
        self._m = manager
        self._tenant_id = tenant_id

    def get_or_call(
        self,
        *,
        model_id: str,
        model_version: str,
        system: str,
        messages: list,
        raw_params: dict,
        upstream_fn: Callable[[], dict],
        http_status: int = 200,
        ttl: int | None = None,
        extra_tags: list | None = None,
    ) -> dict:
        return self._m.get_or_call(
            tenant_id=self._tenant_id,
            model_id=model_id,
            model_version=model_version,
            system=system,
            messages=messages,
            raw_params=raw_params,
            upstream_fn=upstream_fn,
            http_status=http_status,
            ttl=ttl,
            extra_tags=extra_tags,
        )

    def get(self, key: str) -> bytes | None:
        row = self._m._exact.get(key, self._tenant_id)
        if row is None:
            return None
        import json as _json
        return _json.dumps(row).encode("utf-8")

    def set(self, key: str, value: bytes) -> None:
        # Adapter-friendly write path: not used in Phase 1.
        import json as _json
        try:
            decoded = _json.loads(value.decode("utf-8"))
        except Exception:
            return
        self._m._exact.set(
            key, self._tenant_id, decoded, [], "", None,
        )

    def invalidate_all(self) -> int:
        return self._m.invalidate_tenant(self._tenant_id)

    @property
    def metrics(self) -> CacheMetrics:
        return self._m.metrics


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def json_dumps_bytes(d: dict) -> bytes:
    import json as _json
    return _json.dumps(d, separators=(",", ":"), default=str).encode("utf-8")


# ---------------------------------------------------------------------------
# Vertex helpers (WP-11 / CRIT-2)
# ---------------------------------------------------------------------------

def _vertex_model_from_path(path: str) -> str:
    """Extract the model name from a Vertex proxy path.

    Handles both the FastAPI path parameter form:
      /v1/projects/{project}/locations/{loc}/publishers/google/models/{model}:{method}
    and the raw vertex_path parameter:
      {project}/locations/{loc}/publishers/google/models/{model}:{method}

    Returns empty string on parse failure (key_builder treats it as uncacheable-neutral).
    """
    import re as _re
    _MODEL_RE = _re.compile(r"/models/([a-zA-Z0-9._\-]{1,128}):")
    m = _MODEL_RE.search(path)
    if m:
        return m.group(1)
    return ""
