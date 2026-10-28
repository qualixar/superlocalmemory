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


from superlocalmemory.mcp.tool_hints import ADDS, READ_ONLY, SETS
from superlocalmemory.core.admission import admits
from superlocalmemory.core.operation_request import OperationKind
from superlocalmemory.mcp.agent_context import get_current_agent_id
from superlocalmemory.mcp.remote_caller import current_remote_key_id
from superlocalmemory.optimize.compress.ccr import _UUID4_RE, CCRStore
from superlocalmemory.optimize.compress.router import CompressRouter as CompressRouter
from superlocalmemory.optimize.compress.router import CompressTextResult as CompressTextResult
from superlocalmemory.optimize.storage.db import CacheDB, _normalize_tenant_id
from superlocalmemory.optimize.storage.provenance import active_profile_id

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
    """Whose cache and reversible-compression store this request uses.

    On this computer that is the agent (``/mcp/<agent>``, else ``SLM_AGENT_ID``).
    A caller on another computer chooses its ``/mcp/<agent>`` segment freely, so
    its tenant is also keyed by the remote key that authenticated it: a remote
    key can never read or overwrite a local agent's entries, nor another key's.
    ``:`` never occurs in a URL-derived agent id, so no local agent can name
    this tenant from a URL either.
    """
    # get_current_agent_id() never returns ""; "mcp_client" is its stdio sentinel.
    agent = get_current_agent_id()
    key_id = current_remote_key_id()
    if key_id is None:
        user = _signed_in_user()
        # A signed-in person's entries are theirs: the agent segment is chosen
        # by the caller, so it alone must not open another person's cached
        # originals (4.1.25 audit OPS-08). No session: unchanged address.
        return f"user:{user}:{agent}" if user else agent
    return f"remote:{key_id}:{agent}"


def _signed_in_user() -> str:
    """The signed-in user's id when the caller is held to a user, else ``""``."""
    from superlocalmemory.core.actor_context import Transport
    from superlocalmemory.core.admission import caller_actor

    actor, mode = caller_actor(Transport.MCP)
    return actor.principal_id if mode == "company" and actor.is_authenticated else ""


def _kv_location(key: str) -> tuple[str, str]:
    """``(cache_key, normalized tenant)`` for a ``slm_cache_*`` entry.

    Local entries keep their pre-4.1.20 address, so existing caches still hit.
    Remote entries use their own prefix, so the hashed key cannot collide with
    a local agent's entry whatever the agent or key text.
    """
    tenant = _tenant()
    prefix = "mcpkv-remote" if current_remote_key_id() is not None else "mcpkv"
    cache_key = hashlib.sha256(f"{prefix}:{tenant}:{key}".encode()).hexdigest()
    return cache_key, _normalize_tenant_id(tenant)


# ─── Tool registration ────────────────────────────────────────────────────────


def register_optimize_tools(server) -> None:
    """Register the 5 Surface B optimize tools on *server*.

    *server* is duck-typed: must support @server.tool() decorator pattern.
    Compatible with FastMCP, _FilteredServer, and test _MockServer.
    """

    @server.tool(title="Compress Text", annotations=ADDS)
    @admits(OperationKind.CONSOLIDATE)
    async def slm_compress(
        content: str,
        mode: str = "auto",
        reversible: bool = True,
        ttl_seconds: int = 86400,
    ) -> dict:
        """Compress text with one tenant-scoped, TTL-bound recovery copy.

        mode: normalize, auto or aggressive (daemon configuration controls models).
        reversible keeps changed original bytes; ttl_seconds defaults to one day.
        Returned byte_exact distinguishes unchanged bytes from normalized prose.
        """
        from superlocalmemory.mcp.compression_service import CompressionContext, compress

        ctx = CompressionContext(
            CompressRouter,
            CompressTextResult,
            CCRStore,
            CacheDB,
            _tenant(),
            _MAX_COMPRESS_BYTES,
            active_profile_id(),
        )
        return compress(ctx, content, mode, reversible, ttl_seconds)

    @server.tool(title="Retrieve Compressed Text", annotations=READ_ONLY)
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
                    "ok": False,
                    "content": None,
                    "size_bytes": 0,
                    "error": "ccr_id must be a UUID4",
                }
            original = CCRStore.get_instance().retrieve(ccr_id, tenant_id=_tenant())
            if original is None:
                return {
                    "ok": False,
                    "content": None,
                    "size_bytes": 0,
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
                "ok": False,
                "content": None,
                "size_bytes": 0,
                "error": f"internal error: {exc}",
            }

    @server.tool(title="Cache Value", annotations=SETS)
    @admits(OperationKind.REMEMBER)
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
                    "ok": False,
                    "stored": False,
                    "note": f"key must be 1–{_MAX_KV_KEY_CHARS} chars",
                }
            value_bytes = value.encode("utf-8")
            if len(value_bytes) > _MAX_KV_VALUE_BYTES:
                return {"ok": False, "stored": False, "note": "value exceeds 1MB limit"}

            cache_key, norm_tid = _kv_location(key)
            ttl_exp = time.time() + ttl_seconds

            db = CacheDB.get_default()
            db.set(
                cache_key,
                norm_tid,
                value_bytes,
                model="mcp-kv",
                ttl_expires=ttl_exp,
                tags=["mcp-kv"],
                profile_id=active_profile_id(),
            )
            # ``tag_json`` is descriptive only; tag invalidation joins the
            # normalized llmcache_tags index.  CacheManager registers that
            # index itself, whereas this direct MCP KV surface writes to
            # CacheDB.  Register it here so ``slm cache invalidate --tag
            # mcp-kv`` actually removes entries created by slm_cache_set.
            db.tag_register(cache_key, norm_tid, ["mcp-kv"])
            return {"ok": True, "stored": True, "note": None}

        except Exception as exc:
            logger.error("slm_cache_set failed (fail-open): %s", exc)
            return {"ok": False, "stored": False, "note": f"internal error: {exc}"}

    @server.tool(title="Get Cached Value", annotations=READ_ONLY)
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
                    "ok": False,
                    "hit": False,
                    "value": None,
                    "note": f"key must be 1–{_MAX_KV_KEY_CHARS} chars",
                }
            cache_key, norm_tid = _kv_location(key)

            db = CacheDB.get_default()
            blob = db.get_value(cache_key, norm_tid)
            if blob is None:
                with _kv_lock:
                    _kv_misses += 1
                db.kv_counter_incr("kv_misses")  # M2: durable across restarts
                return {"ok": True, "hit": False, "value": None, "note": None}
            with _kv_lock:
                _kv_hits += 1
            db.kv_counter_incr("kv_hits")  # M2: durable across restarts
            return {"ok": True, "hit": True, "value": blob.decode("utf-8"), "note": None}

        except Exception as exc:
            logger.error("slm_cache_get failed (fail-open): %s", exc)
            return {
                "ok": False,
                "hit": False,
                "value": None,
                "note": f"internal error: {exc}",
            }

    @server.tool(title="Get Optimization Stats", annotations=READ_ONLY)
    async def slm_optimize_stats() -> dict:
        """Return compression and cache statistics.

        Proxy/compress AND KV stats are daemon-persisted (accurate across
        restarts). KV counters fall back to this session's in-memory tally if
        the persisted counters can't be read.
        """
        try:
            db = CacheDB.get_default()
            from superlocalmemory.optimize.metrics.mcp_compression import snapshot_with_mcp

            snap = snapshot_with_mcp(db, db.metrics_load())
            # M2: prefer the durable KV counters; fall back to session counters.
            persisted = db.kv_counters_load()
            with _kv_lock:
                kv_h = persisted.get("kv_hits", _kv_hits)
                kv_m = persisted.get("kv_misses", _kv_misses)
            return {
                "ok": True,
                "compress_runs": snap.compress_runs,
                "tokens_saved_compress": snap.tokens_saved_compress,
                "cache_proxy_hits": snap.hits,
                "cache_proxy_misses": snap.misses,
                "cache_kv_hits": kv_h,
                "cache_kv_misses": kv_m,
                "ccr_note": ("CCR entry count not tracked per-session; see daemon /api/v1/metrics"),
                "note": "proxy and kv stats are daemon-persisted across restarts",
            }
        except Exception as exc:
            logger.error("slm_optimize_stats failed (fail-open): %s", exc)
            return {
                "ok": False,
                "compress_runs": 0,
                "tokens_saved_compress": 0,
                "cache_proxy_hits": 0,
                "cache_proxy_misses": 0,
                "cache_kv_hits": 0,
                "cache_kv_misses": 0,
                "ccr_note": None,
                "note": f"internal error: {exc}",
            }
