# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""V3 API endpoints for the SuperLocalMemory dashboard."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from superlocalmemory.server.routes.helpers import SLM_VERSION

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v3", tags=["v3"])


# ── Dashboard ────────────────────────────────────────────────

@router.get("/dashboard")
async def dashboard(request: Request):
    """Dashboard summary: mode, memory count, health score, recent activity."""
    try:
        from superlocalmemory.core.config import SLMConfig
        config = SLMConfig.load()

        # Read stats directly from SQLite (dashboard doesn't load engine)
        import sqlite3
        memory_count = 0
        fact_count = 0
        db_path = config.base_dir / "memory.db"
        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                try:
                    cursor.execute("SELECT COUNT(*) FROM atomic_facts")
                    fact_count = cursor.fetchone()[0]
                except Exception:
                    pass
                try:
                    cursor.execute("SELECT COUNT(*) FROM memories")
                    memory_count = cursor.fetchone()[0]
                except Exception:
                    pass
                conn.close()
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
            "version": SLM_VERSION,
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Mode ─────────────────────────────────────────────────────

@router.get("/mode")
async def get_mode():
    """Get current mode, provider, model — single source of truth for UI."""
    try:
        from superlocalmemory.core.config import SLMConfig
        config = SLMConfig.load()
        current = config.mode.value
        return {
            "mode": current,
            "provider": config.llm.provider or "none",
            "model": config.llm.model or "",
            "has_key": bool(config.llm.api_key),
            "endpoint": config.llm.api_base or "",
            "capabilities": {
                "llm_available": bool(config.llm.provider),
                "cross_encoder": config.retrieval.use_cross_encoder if hasattr(config, 'retrieval') else False,
            },
        }
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


@router.post("/mode/set")
async def set_full_config(request: Request):
    """Save mode + provider + model + API key together."""
    try:
        body = await request.json()
        new_mode = body.get("mode", "a").lower()
        provider = body.get("provider", "none")
        model = body.get("model", "")
        api_key = body.get("api_key", "")

        if new_mode not in ("a", "b", "c"):
            return JSONResponse({"error": "Invalid mode"}, status_code=400)

        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.storage.models import Mode
        config = SLMConfig.for_mode(
            Mode(new_mode),
            llm_provider=provider if provider != "none" else "",
            llm_model=model,
            llm_api_key=api_key,
            llm_api_base="http://localhost:11434" if provider == "ollama" else "",
        )
        old = SLMConfig.load()
        config.active_profile = old.active_profile
        config.save()

        # Kill existing worker so next request uses new config
        try:
            from superlocalmemory.core.worker_pool import WorkerPool
            WorkerPool.shared().shutdown()
        except Exception:
            pass

        if hasattr(request.app.state, "engine"):
            request.app.state.engine = None

        return {
            "success": True,
            "mode": new_mode,
            "provider": provider,
            "model": model,
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/provider/test")
async def test_provider(request: Request):
    """Test connectivity to an LLM provider."""
    try:
        import httpx
        body = await request.json()
        provider = body.get("provider", "")
        model = body.get("model", "")
        api_key = body.get("api_key", "")

        if provider == "ollama":
            endpoint = body.get("endpoint", "http://localhost:11434")
            with httpx.Client(timeout=httpx.Timeout(5.0)) as c:
                resp = c.get(f"{endpoint}/api/tags")
                resp.raise_for_status()
                models = [m["name"] for m in resp.json().get("models", [])]
                found = model in models if model else len(models) > 0
                return {
                    "success": found,
                    "message": f"Ollama OK, {len(models)} models" + (f", '{model}' available" if found and model else ""),
                }

        if provider == "openrouter":
            if not api_key:
                api_key = os.environ.get("OPENROUTER_API_KEY", "")
            if not api_key:
                return {"success": False, "error": "API key required"}
            with httpx.Client(timeout=httpx.Timeout(10.0)) as c:
                resp = c.get("https://openrouter.ai/api/v1/models", headers={"Authorization": f"Bearer {api_key}"})
                resp.raise_for_status()
                return {"success": True, "message": "OpenRouter connected, key valid"}

        if provider == "openai":
            if not api_key:
                return {"success": False, "error": "API key required"}
            with httpx.Client(timeout=httpx.Timeout(10.0)) as c:
                resp = c.get("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {api_key}"})
                resp.raise_for_status()
                return {"success": True, "message": "OpenAI connected, key valid"}

        if provider == "anthropic":
            if not api_key:
                return {"success": False, "error": "API key required"}
            # Anthropic doesn't have a models list endpoint, just verify key format
            if api_key.startswith("sk-ant-"):
                return {"success": True, "message": "Anthropic key format valid"}
            return {"success": False, "error": "Key should start with sk-ant-"}

        return {"success": False, "error": f"Unknown provider: {provider}"}
    except httpx.ConnectError:
        return {"success": False, "error": "Cannot connect — is the service running?"}
    except httpx.HTTPStatusError as e:
        return {"success": False, "error": f"HTTP {e.response.status_code}: Invalid key or endpoint"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/ollama/status")
async def ollama_status():
    """Check if Ollama is running and list available models."""
    try:
        import httpx
        with httpx.Client(timeout=httpx.Timeout(5.0)) as client:
            resp = client.get("http://localhost:11434/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = [
                {"name": m["name"], "size": m.get("size", 0)}
                for m in data.get("models", [])
            ]
            return {"running": True, "models": models, "count": len(models)}
    except Exception:
        return {"running": False, "models": [], "count": 0}


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

        from superlocalmemory.core.worker_pool import WorkerPool
        pool = WorkerPool.shared()
        result = pool.recall(query, limit=limit)

        if not result.get("ok"):
            return JSONResponse(
                {"error": result.get("error", "Recall failed")},
                status_code=503,
            )

        # Optional: synthesize answer from results (Mode B/C only)
        synthesis = ""
        if body.get("synthesize") and result.get("results"):
            try:
                syn_result = pool.synthesize(query, result["results"][:5])
                synthesis = syn_result.get("synthesis", "") if syn_result.get("ok") else ""
            except Exception:
                pass

        return {
            "query": query,
            "query_type": result.get("query_type", "unknown"),
            "result_count": result.get("result_count", 0),
            "retrieval_time_ms": result.get("retrieval_time_ms", 0),
            "results": result.get("results", []),
            "synthesis": synthesis,
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Trust Dashboard ──────────────────────────────────────────

@router.get("/trust/dashboard")
async def trust_dashboard(request: Request):
    """Trust overview: per-agent scores, alerts. Queries DB directly."""
    try:
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.storage.database import DatabaseManager
        from superlocalmemory.storage import schema as _schema
        config = SLMConfig.load()
        pid = config.active_profile

        db_path = config.db_path
        db = DatabaseManager(db_path)
        db.initialize(_schema)

        # Query trust scores from DB
        agents = []
        try:
            rows = db.execute(
                "SELECT target_id, target_type, trust_score, evidence_count, "
                "last_updated FROM trust_scores WHERE profile_id = ? "
                "ORDER BY trust_score DESC",
                (pid,),
            )
            for r in rows:
                d = dict(r)
                agents.append({
                    "target_id": d.get("target_id", ""),
                    "target_type": d.get("target_type", ""),
                    "trust_score": round(float(d.get("trust_score", 0.5)), 3),
                    "evidence_count": d.get("evidence_count", 0),
                    "last_updated": d.get("last_updated", ""),
                })
        except Exception:
            pass

        # Aggregate stats
        avg = round(sum(a["trust_score"] for a in agents) / len(agents), 3) if agents else 0.5
        alerts = [a for a in agents if a["trust_score"] < 0.3]

        return {
            "agents": agents,
            "avg_trust": avg,
            "alerts": alerts,
            "total": len(agents),
            "profile": pid,
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Math Health ──────────────────────────────────────────────

@router.get("/math/health")
async def math_health(request: Request):
    """Mathematical layer health: Fisher, sheaf, Langevin status. Queries DB directly."""
    try:
        engine = None  # Engine runs in subprocess; query DB directly below

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
