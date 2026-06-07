# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Optimize API routes — single canonical route file (INTERFACE-CONTRACT v2 §5 R2-01).

All optimize endpoints live here.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("superlocalmemory.server.routes.optimize")

router = APIRouter(prefix="/api/optimize", tags=["optimize"])

# F-03 fix: module-level singleton — constructed once, shared by /savings + CLI
from superlocalmemory.optimize.metrics.estimator import SavingsEstimator as _SavingsEstimator  # noqa: E402
_savings_estimator = _SavingsEstimator()


class ConfigUpdateRequest(BaseModel):
    """Partial config update — only provided fields are changed."""
    enabled: bool | None = None
    proxy_enabled: bool | None = None
    cache_enabled: bool | None = None
    semantic_enabled: bool | None = None
    compress_enabled: bool | None = None
    compress_mode: Literal['safe', 'aggressive'] | None = None
    compress_code: bool | None = None
    compress_prose: bool | None = None
    compress_ccr: bool | None = None


@router.get("/config")
async def get_config() -> dict[str, Any]:
    """Return current optimize config as JSON."""
    try:
        from superlocalmemory.optimize.config.store import ConfigStore
        cfg = ConfigStore().get()
        return cfg.as_dict()
    except Exception as exc:
        logger.warning("GET /api/optimize/config failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.put("/config")
async def put_config(body: ConfigUpdateRequest) -> dict[str, Any]:
    """Update optimize config (partial). Daemon hot-reloads within 2s."""
    try:
        import dataclasses
        from superlocalmemory.optimize.config.store import ConfigStore
        store = ConfigStore()
        cfg = store.get()
        updates: dict[str, Any] = {}
        for field_name in ConfigUpdateRequest.model_fields:
            val = getattr(body, field_name, None)
            if val is not None:
                updates[field_name] = val
        if updates:
            cfg = dataclasses.replace(cfg, **updates)
            store.save(cfg)
        return {"status": "ok", "updated": list(updates.keys())}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.warning("PUT /api/optimize/config failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/savings")
async def get_savings() -> dict[str, Any]:
    """Return savings estimates from live metrics + CacheDB.

    Field names match INTERFACE-CONTRACT §5 exactly.
    """
    try:
        from superlocalmemory.optimize.config.store import ConfigStore
        from superlocalmemory.optimize.metrics.counters import get_metrics
        from superlocalmemory.optimize.storage.db import CacheDB

        cfg = ConfigStore().get()
        collector = get_metrics()
        db = CacheDB()
        snap = collector.snapshot(
            cache_size_bytes=db.db_size_bytes(),
            cache_entry_count=db.entry_count(),
        )

        # F-03 fix: use module-level singleton instead of per-request instantiation
        active_model = getattr(cfg, "active_model", "anthropic")
        # Determine provider from model
        provider = "anthropic"
        for key in ("anthropic", "openai", "gemini"):
            if key in active_model.lower():
                provider = key
                break
        est = _savings_estimator.estimate(snap, provider=provider)

        # compress_ratio from snap (byte-based, survives restart)
        if snap.compress_bytes_original > 0:
            compress_ratio = snap.compress_bytes_after / snap.compress_bytes_original
        else:
            compress_ratio = None

        total = snap.hits + snap.misses
        hit_rate = snap.hits / total if total > 0 else 0.0

        # F-02 fix: response keys match INTERFACE-CONTRACT §5 exactly
        return {
            "tokens_saved_input": snap.tokens_saved_input,
            "tokens_saved_output": snap.tokens_saved_output,
            "calls_skipped": snap.calls_skipped,
            "compress_ratio": round(compress_ratio, 4) if compress_ratio is not None else None,
            "cost_saved": {"usd": est["usd"], "inr": est["inr"]},
            "hit_rate": round(hit_rate, 4),
            "cache_bytes": snap.cache_size_bytes,
            "entries": snap.cache_entry_count,
            # diagnostic fields — non-conflicting extras
            "hits": snap.hits,
            "misses": snap.misses,
            "tokens_saved_compress": snap.tokens_saved_compress,
            "is_stale": est["is_stale"],
            "pricing_date": est["pricing_date"],
        }
    except Exception as exc:
        logger.warning("GET /api/optimize/savings failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/stats")
async def get_stats() -> dict[str, Any]:
    """Return raw metrics snapshot as JSON."""
    try:
        from superlocalmemory.optimize.metrics.counters import get_metrics
        from superlocalmemory.optimize.storage.db import CacheDB

        collector = get_metrics()
        db = CacheDB()
        snap = collector.snapshot(
            cache_size_bytes=db.db_size_bytes(),
            cache_entry_count=db.entry_count(),
        )
        # Return all 16 fields as dict
        import dataclasses
        return dataclasses.asdict(snap)
    except Exception as exc:
        logger.warning("GET /api/optimize/stats failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/cache/clear")
async def delete_cache_clear(tenant: str = "default") -> dict[str, Any]:
    """Delete all cache entries for a tenant."""
    try:
        from superlocalmemory.optimize.storage.db import CacheDB
        db = CacheDB()
        deleted = db.clear_tenant(tenant)
        return {"success": True, "deleted": deleted}
    except Exception as exc:
        logger.warning("DELETE /api/optimize/cache/clear failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
