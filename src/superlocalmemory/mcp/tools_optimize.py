# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SLM v3.6.11 — Surface B: MCP Optimize Tools.

Five proxy-free tools exposing compression (reversible via CCR) and
routed-result caching WITHOUT touching ANTHROPIC_BASE_URL, so the full
1M context window is preserved on any Claude subscription.

Primary Claude conversation turns CANNOT be cached without a proxy.
These tools cache results the agent explicitly routes through SLM.

Fail-open: every tool body is wrapped in try/except Exception.
Any internal error returns the input unchanged with ok:False — never raises.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time

from mcp.types import ToolAnnotations

from superlocalmemory.mcp.agent_context import get_current_agent_id
from superlocalmemory.optimize.compress.ccr import CCRStore, _UUID4_RE
from superlocalmemory.optimize.compress.router import CompressRouter
from superlocalmemory.optimize.storage.db import CacheDB, _normalize_tenant_id

logger = logging.getLogger("slm.mcp.tools_optimize")

# ─── Size caps (CWE-400 guards) ───────────────────────────────────────────────

_MAX_COMPRESS_BYTES: int = 1_000_000
_MAX_KV_VALUE_BYTES: int = 1_000_000
_MAX_KV_KEY_CHARS: int = 512

# ─── Exported tool name list (used by server.py + tests) ─────────────────────

_OPTIMIZE_TOOL_NAMES = (
    "slm_compress",
    "slm_retrieve",
    "slm_cache_set",
    "slm_cache_get",
    "slm_optimize_stats",
)

# ─── In-module KV counters (thread-safe; MetricsCollector is process-scoped) ─

_kv_lock = threading.Lock()
_kv_hits: int = 0
_kv_misses: int = 0


def _tenant() -> str:
    # get_current_agent_id() never returns ""; "mcp_client" is its stdio sentinel.
    return get_current_agent_id()


# ─── Tool registration ────────────────────────────────────────────────────────


def register_optimize_tools(server) -> None:
    """Register the 5 Surface B optimize tools on *server*.

    *server* is duck-typed: must support @server.tool() decorator pattern.
    Compatible with FastMCP, _FilteredServer, and test _MockServer.
    """

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False))
    async def slm_compress(
        content: str,
        mode: str = "auto",
        reversible: bool = True,
        ttl_seconds: int = 86400,
    ) -> dict:
        """Compress text or tool output to reduce context window usage.

        Returns compressed text. If lossy and reversible=True, also returns a
        ccr_id — pass it to slm_retrieve to recover the exact original.

        Args:
            content: Text to compress (max 1MB).
            mode: "normalize" (lossless whitespace) | "auto" | "aggressive".
            reversible: Store original in CCR for later retrieval.
            ttl_seconds: CCR lifetime in seconds (default 24h).
        """
        try:
            if not isinstance(content, str) or not content:
                return {
                    "ok": False, "compressed": content or "",
                    "strategy": "none", "tokens_before": 0, "tokens_after": 0,
                    "ratio": 1.0, "lossy": False, "ccr_id": None,
                    "note": "empty input",
                }

            note_parts: list[str] = []
            if len(content.encode("utf-8")) > _MAX_COMPRESS_BYTES:
                reversible = False
                note_parts.append("content over 1MB: ccr skipped")

            if mode == "normalize":
                # @staticmethod — lossless whitespace collapse, no config/daemon dep.
                normalized = CompressRouter._normalize_whitespace(content)
                tb = len(content.split())
                ta = len(normalized.split())
                ratio = round(ta / tb, 4) if tb else 1.0
                return {
                    "ok": True, "compressed": normalized, "strategy": "normalize",
                    "tokens_before": tb, "tokens_after": ta, "ratio": ratio,
                    "lossy": False, "ccr_id": None,
                    "note": " | ".join(note_parts) or None,
                }

            if mode == "aggressive":
                note_parts.append(
                    "aggressive mode requires daemon compress_mode=aggressive in config"
                )

            res = CompressRouter.get_instance().compress_text(content)

            ccr_id = None
            if res.lossy and reversible:
                stored = CCRStore.get_instance().store(
                    content.encode("utf-8"),
                    tenant_id=_tenant(),
                    ttl_seconds=ttl_seconds,
                )
                ccr_id = stored or None
                if ccr_id:
                    note_parts.append("reversible: call slm_retrieve with this ccr_id")

            ratio = (
                round(res.tokens_after / res.tokens_before, 4)
                if res.tokens_before else 1.0
            )
            return {
                "ok": True, "compressed": res.compressed_text, "strategy": res.strategy,
                "tokens_before": res.tokens_before, "tokens_after": res.tokens_after,
                "ratio": ratio, "lossy": res.lossy, "ccr_id": ccr_id,
                "note": " | ".join(note_parts) or None,
            }

        except Exception as exc:
            logger.error("slm_compress failed (fail-open): %s", exc)
            t = len(content.split()) if isinstance(content, str) else 0
            return {
                "ok": False,
                "compressed": content if isinstance(content, str) else "",
                "strategy": "none", "tokens_before": t, "tokens_after": t,
                "ratio": 1.0, "lossy": False, "ccr_id": None,
                "note": f"internal error: {exc}",
            }

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    async def slm_retrieve(ccr_id: str) -> dict:
        """Retrieve original text stored during a lossy slm_compress call.

        Do not log or share ccr_ids — they are unguessable session tokens, but
        if exposed they allow retrieval by anyone with the daemon's decryption key.

        Args:
            ccr_id: UUID4 returned by slm_compress when reversible=True.
        """
        try:
            if not ccr_id or not _UUID4_RE.match(ccr_id):
                return {
                    "ok": False, "content": None, "size_bytes": 0,
                    "error": "ccr_id must be a UUID4",
                }
            original = CCRStore.get_instance().retrieve(ccr_id)
            if original is None:
                return {
                    "ok": False, "content": None, "size_bytes": 0,
                    "error": "not found (expired / never stored / wrong id)",
                }
            size = len(original)
            try:
                text = original.decode("utf-8")
            except UnicodeDecodeError:
                text = original.decode("latin-1")
            return {"ok": True, "content": text, "size_bytes": size, "error": None}

        except Exception as exc:
            logger.error("slm_retrieve failed (fail-open): %s", exc)
            return {
                "ok": False, "content": None, "size_bytes": 0,
                "error": f"internal error: {exc}",
            }

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False))
    async def slm_cache_set(key: str, value: str, ttl_seconds: int = 86400) -> dict:
        """Cache a result you want to reuse (tool output, file read, search result).

        This caches results the agent explicitly routes through SLM — NOT the
        Claude conversation turn (impossible without a proxy).

        Do not cache secrets, credentials, or ccr_ids via this tool.

        Args:
            key: Cache key (max 512 chars). Namespaced per agent automatically.
            value: Value to store as string (max 1MB).
            ttl_seconds: Time-to-live in seconds (default 24h).
        """
        try:
            if not key or len(key) > _MAX_KV_KEY_CHARS:
                return {
                    "ok": False, "stored": False,
                    "note": f"key must be 1–{_MAX_KV_KEY_CHARS} chars",
                }
            value_bytes = value.encode("utf-8")
            if len(value_bytes) > _MAX_KV_VALUE_BYTES:
                return {"ok": False, "stored": False, "note": "value exceeds 1MB limit"}

            tenant = _tenant()
            cache_key = hashlib.sha256(f"mcpkv:{tenant}:{key}".encode()).hexdigest()
            norm_tid = _normalize_tenant_id(tenant)
            ttl_exp = time.time() + ttl_seconds

            CacheDB.get_default().set(
                cache_key, norm_tid, value_bytes,
                model="mcp-kv", ttl_expires=ttl_exp, tags=["mcp-kv"],
            )
            return {"ok": True, "stored": True, "note": None}

        except Exception as exc:
            logger.error("slm_cache_set failed (fail-open): %s", exc)
            return {"ok": False, "stored": False, "note": f"internal error: {exc}"}

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    async def slm_cache_get(key: str) -> dict:
        """Retrieve a previously cached result.

        Returns hit:True + value if the key exists and has not expired.
        Returns hit:False (never raises) on miss, expiry, or any error.

        Args:
            key: Cache key used in slm_cache_set.
        """
        global _kv_hits, _kv_misses
        try:
            if not key or len(key) > _MAX_KV_KEY_CHARS:
                return {
                    "ok": False, "hit": False, "value": None,
                    "note": f"key must be 1–{_MAX_KV_KEY_CHARS} chars",
                }
            tenant = _tenant()
            cache_key = hashlib.sha256(f"mcpkv:{tenant}:{key}".encode()).hexdigest()
            norm_tid = _normalize_tenant_id(tenant)

            blob = CacheDB.get_default().get_value(cache_key, norm_tid)
            if blob is None:
                with _kv_lock:
                    _kv_misses += 1
                return {"ok": True, "hit": False, "value": None, "note": None}
            with _kv_lock:
                _kv_hits += 1
            return {"ok": True, "hit": True, "value": blob.decode("utf-8"), "note": None}

        except Exception as exc:
            logger.error("slm_cache_get failed (fail-open): %s", exc)
            return {
                "ok": False, "hit": False, "value": None,
                "note": f"internal error: {exc}",
            }

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    async def slm_optimize_stats() -> dict:
        """Return compression and cache statistics.

        Proxy/compress stats are daemon-persisted (accurate across restarts).
        KV stats are in-module counters for this MCP process session only.
        """
        try:
            snap = CacheDB.get_default().metrics_load()
            with _kv_lock:
                kv_h = _kv_hits
                kv_m = _kv_misses
            return {
                "ok": True,
                "compress_runs": snap.compress_runs,
                "tokens_saved_compress": snap.tokens_saved_compress,
                "cache_proxy_hits": snap.hits,
                "cache_proxy_misses": snap.misses,
                "cache_kv_hits": kv_h,
                "cache_kv_misses": kv_m,
                "ccr_note": (
                    "CCR entry count not tracked per-session; "
                    "see daemon /api/v1/metrics"
                ),
                "note": "proxy stats are daemon-persisted; kv stats are this session only",
            }
        except Exception as exc:
            logger.error("slm_optimize_stats failed (fail-open): %s", exc)
            return {
                "ok": False,
                "compress_runs": 0, "tokens_saved_compress": 0,
                "cache_proxy_hits": 0, "cache_proxy_misses": 0,
                "cache_kv_hits": 0, "cache_kv_misses": 0,
                "ccr_note": None,
                "note": f"internal error: {exc}",
            }
