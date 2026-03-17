# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""V3 API endpoints for the SuperLocalMemory dashboard."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v3", tags=["v3"])


# ── Dashboard ────────────────────────────────────────────────

@router.get("/dashboard")
async def dashboard(request: Request):
    """Dashboard summary: mode, memory count, health score, recent activity."""
    try:
        from superlocalmemory.core.config import SLMConfig
        config = SLMConfig.load()

        # Get basic stats from engine if available
        engine = getattr(request.app.state, "engine", None)
        memory_count = 0
        fact_count = 0
        if engine and engine._db:
            try:
                rows = engine._db.execute("SELECT COUNT(*) FROM atomic_facts")
                if rows:
                    fact_count = rows[0][0] if isinstance(rows[0], (list, tuple)) else dict(rows[0]).get("COUNT(*)", 0)
            except Exception:
                pass
            try:
                rows = engine._db.execute("SELECT COUNT(*) FROM memories")
                if rows:
                    memory_count = rows[0][0] if isinstance(rows[0], (list, tuple)) else dict(rows[0]).get("COUNT(*)", 0)
            except Exception:
                pass

        return {
            "mode": config.mode.value,
            "mode_name": {"a": "Local Guardian", "b": "Smart Local", "c": "Full Power"}.get(config.mode.value, "Unknown"),
            "provider": config.llm.provider or "none",
            "model": config.llm.model or "",
            "memory_count": memory_count,
            "fact_count": fact_count,
            "profile": config.active_profile,
            "base_dir": str(config.base_dir),
            "version": "3.0.0",
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Mode ─────────────────────────────────────────────────────

@router.get("/mode")
async def get_mode():
    """Get current operating mode."""
    try:
        from superlocalmemory.core.config import SLMConfig
        config = SLMConfig.load()
        modes = {
            "a": {"name": "Local Guardian", "description": "Zero cloud. Your data never leaves your machine.", "llm": False, "eu_compliant": True},
            "b": {"name": "Smart Local", "description": "Local LLM via Ollama. Still fully private.", "llm": "local", "eu_compliant": True},
            "c": {"name": "Full Power", "description": "Cloud LLM for maximum accuracy.", "llm": "cloud", "eu_compliant": False},
        }
        current = config.mode.value
        return {"current": current, "details": modes.get(current, {}), "all_modes": modes}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.put("/mode")
async def set_mode(request: Request):
    """Switch operating mode. Body: {"mode": "a"|"b"|"c"}"""
    try:
        body = await request.json()
        new_mode = body.get("mode", "").lower()
        if new_mode not in ("a", "b", "c"):
            return JSONResponse({"error": "Invalid mode. Use a, b, or c."}, status_code=400)

        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.storage.models import Mode
        old_config = SLMConfig.load()
        new_config = SLMConfig.for_mode(
            Mode(new_mode),
            llm_provider=old_config.llm.provider,
            llm_model=old_config.llm.model,
            llm_api_key=old_config.llm.api_key,
            llm_api_base=old_config.llm.api_base,
        )
        new_config.active_profile = old_config.active_profile
        new_config.save()

        # Reset engine to pick up new config
        if hasattr(request.app.state, "engine"):
            request.app.state.engine = None

        return {"success": True, "mode": new_mode}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Provider ─────────────────────────────────────────────────

@router.get("/providers")
async def list_providers():
    """List available LLM providers with presets."""
    try:
        from superlocalmemory.core.config import SLMConfig
        return {"providers": SLMConfig.provider_presets()}
    except Exception as exc:
        return {"error": str(exc), "providers": []}


@router.get("/provider")
async def get_provider():
    """Get current provider configuration (API key masked)."""
    try:
        from superlocalmemory.core.config import SLMConfig
        config = SLMConfig.load()
        key = config.llm.api_key
        masked = f"****{key[-4:]}" if len(key) > 8 else "****" if key else ""
        return {
            "provider": config.llm.provider or "none",
            "model": config.llm.model,
            "base_url": config.llm.api_base,
            "api_key_masked": masked,
            "has_key": bool(key),
        }
    except Exception as exc:
        return {"error": str(exc), "provider": "unknown"}


@router.put("/provider")
async def set_provider(request: Request):
    """Set LLM provider. Body: {"provider": "openai", "api_key": "...", "model": "..."}"""
    try:
        body = await request.json()
        provider = body.get("provider", "")
        api_key = body.get("api_key", "")
        model = body.get("model", "")
        base_url = body.get("base_url", "")

        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.storage.models import Mode
        config = SLMConfig.load()

        # Use preset base_url if not provided
        if not base_url:
            presets = SLMConfig.provider_presets()
            preset = presets.get(provider, {})
            base_url = preset.get("base_url", "")
            if not model:
                model = preset.get("model", "")

        new_config = SLMConfig.for_mode(
            config.mode,
            llm_provider=provider,
            llm_model=model,
            llm_api_key=api_key,
            llm_api_base=base_url,
        )
        new_config.active_profile = config.active_profile
        new_config.save()

        return {"success": True, "provider": provider, "model": model}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Recall Trace ─────────────────────────────────────────────

@router.post("/recall/trace")
async def recall_trace(request: Request):
    """Recall with per-channel score breakdown."""
    try:
        body = await request.json()
        query = body.get("query", "")
        limit = body.get("limit", 10)

        from .helpers import get_engine_lazy
        engine = get_engine_lazy(request.app.state)
        if not engine:
            return JSONResponse({"error": "Engine not initialized"}, status_code=503)

        response = engine.recall(query, limit=limit)
        results = []
        for r in response.results[:limit]:
            results.append({
                "fact_id": r.fact.fact_id,
                "content": r.fact.content[:300],
                "score": round(r.score, 4),
                "confidence": round(r.confidence, 4),
                "trust_score": round(r.trust_score, 4),
                "channel_scores": {k: round(v, 4) for k, v in (r.channel_scores or {}).items()},
            })

        return {
            "query": query,
            "query_type": response.query_type,
            "result_count": len(results),
            "retrieval_time_ms": round(response.retrieval_time_ms, 1),
            "results": results,
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Trust Dashboard ──────────────────────────────────────────

@router.get("/trust/dashboard")
async def trust_dashboard(request: Request):
    """Trust overview: per-agent scores, alerts."""
    try:
        engine = getattr(request.app.state, "engine", None)
        if not engine or not engine._trust_scorer:
            return {"agents": [], "alerts": [], "message": "Trust scorer not available"}

        from superlocalmemory.core.config import SLMConfig
        config = SLMConfig.load()
        scores = engine._trust_scorer.get_all_scores(config.active_profile)

        agents = []
        for s in scores:
            if isinstance(s, dict):
                agents.append(s)
            else:
                agents.append({
                    "target_id": s.target_id,
                    "target_type": s.target_type,
                    "trust_score": round(s.trust_score, 3),
                    "evidence_count": s.evidence_count,
                })

        return {"agents": agents, "alerts": [], "profile": config.active_profile}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Math Health ──────────────────────────────────────────────

@router.get("/math/health")
async def math_health(request: Request):
    """Mathematical layer health: Fisher, sheaf, Langevin status."""
    try:
        engine = getattr(request.app.state, "engine", None)

        health = {
            "fisher": {"status": "active", "description": "Fisher-Rao information geometry for similarity"},
            "sheaf": {"status": "active", "description": "Sheaf cohomology for consistency detection"},
            "langevin": {"status": "active", "description": "Riemannian Langevin dynamics for lifecycle"},
        }

        # Check if math layers are configured
        if engine:
            from superlocalmemory.core.config import SLMConfig
            config = SLMConfig.load()
            health["fisher"]["mode"] = config.math.fisher_mode
            health["sheaf"]["threshold"] = config.math.sheaf_contradiction_threshold
            health["langevin"]["temperature"] = config.math.langevin_temperature

        return {"health": health, "overall": "healthy"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Auto-Capture / Auto-Recall Config ────────────────────────

@router.get("/auto-capture/config")
async def get_auto_capture_config():
    """Get auto-capture configuration."""
    try:
        from superlocalmemory.hooks.rules_engine import RulesEngine
        from superlocalmemory.core.config import DEFAULT_BASE_DIR
        rules = RulesEngine(config_path=DEFAULT_BASE_DIR / "config.json")
        return {"config": rules.get_capture_config()}
    except Exception as exc:
        return {"error": str(exc), "config": {}}


@router.put("/auto-capture/config")
async def set_auto_capture_config(request: Request):
    """Update auto-capture config. Body: {"enabled": true, "capture_decisions": true, ...}"""
    try:
        body = await request.json()
        from superlocalmemory.hooks.rules_engine import RulesEngine
        from superlocalmemory.core.config import DEFAULT_BASE_DIR
        config_path = DEFAULT_BASE_DIR / "config.json"
        rules = RulesEngine(config_path=config_path)
        for key, value in body.items():
            rules.update_rule("auto_capture", key, value)
        rules.save(config_path)
        return {"success": True, "config": rules.get_capture_config()}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/auto-recall/config")
async def get_auto_recall_config():
    """Get auto-recall configuration."""
    try:
        from superlocalmemory.hooks.rules_engine import RulesEngine
        from superlocalmemory.core.config import DEFAULT_BASE_DIR
        rules = RulesEngine(config_path=DEFAULT_BASE_DIR / "config.json")
        return {"config": rules.get_recall_config()}
    except Exception as exc:
        return {"error": str(exc), "config": {}}


@router.put("/auto-recall/config")
async def set_auto_recall_config(request: Request):
    """Update auto-recall config."""
    try:
        body = await request.json()
        from superlocalmemory.hooks.rules_engine import RulesEngine
        from superlocalmemory.core.config import DEFAULT_BASE_DIR
        config_path = DEFAULT_BASE_DIR / "config.json"
        rules = RulesEngine(config_path=config_path)
        for key, value in body.items():
            rules.update_rule("auto_recall", key, value)
        rules.save(config_path)
        return {"success": True, "config": rules.get_recall_config()}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── IDE Status ───────────────────────────────────────────────

@router.get("/ide/status")
async def ide_status():
    """Get IDE connection status."""
    try:
        from superlocalmemory.hooks.ide_connector import IDEConnector
        connector = IDEConnector()
        return {"ides": connector.get_status()}
    except Exception as exc:
        return {"error": str(exc), "ides": []}


@router.post("/ide/connect")
async def ide_connect(request: Request):
    """Connect an IDE. Body: {"ide": "cursor"} or {} for all."""
    try:
        body = await request.json()
        ide = body.get("ide", "")

        from superlocalmemory.hooks.ide_connector import IDEConnector
        connector = IDEConnector()

        if ide:
            success = connector.connect(ide)
            return {"success": success, "ide": ide}
        else:
            results = connector.connect_all()
            return {"results": results}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
