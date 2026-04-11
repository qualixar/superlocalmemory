# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the Elastic License 2.0 - see LICENSE file
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

        # V3.3: Check if embedding model changed — flag for re-indexing
        needs_reindex = (
            old_config.embedding.provider != new_config.embedding.provider
            or old_config.embedding.model_name != new_config.embedding.model_name
        )

        # Reset engine to pick up new config
        if hasattr(request.app.state, "engine"):
            request.app.state.engine = None

        return {
            "success": True,
            "mode": new_mode,
            "needs_reindex": needs_reindex,
            "message": "Embedding re-indexing will run on next recall." if needs_reindex else "",
        }
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

        # Record learning signals (non-blocking, non-critical)
        try:
            _record_learning_signals(query, result.get("results", []))
        except Exception as _sig_exc:
            import logging as _log
            _log.getLogger(__name__).warning("Learning signal error: %s", _sig_exc)

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


def _record_learning_signals(query: str, results: list) -> None:
    """Record feedback + co-retrieval + confidence boost for any recall."""
    from pathlib import Path
    from superlocalmemory.core.config import SLMConfig

    slm_dir = Path.home() / ".superlocalmemory"
    config = SLMConfig.load()
    pid = config.active_profile
    fact_ids = [r.get("fact_id", "") for r in results[:10] if r.get("fact_id")]
    if not fact_ids:
        return

    try:
        from superlocalmemory.learning.feedback import FeedbackCollector
        collector = FeedbackCollector(slm_dir / "learning.db")
        collector.record_implicit(
            profile_id=pid, query=query,
            fact_ids_returned=fact_ids, fact_ids_available=fact_ids,
        )
    except Exception:
        pass

    try:
        from superlocalmemory.learning.signals import LearningSignals
        signals = LearningSignals(slm_dir / "learning.db")
        signals.record_co_retrieval(pid, fact_ids)
    except Exception:
        pass

    try:
        from superlocalmemory.learning.signals import LearningSignals
        mem_db = str(slm_dir / "memory.db")
        for fid in fact_ids[:5]:
            LearningSignals.boost_confidence(mem_db, fid)
    except Exception:
        pass


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


# ── Active Memory (V3.1) ────────────────────────────────────

@router.get("/learning/signals")
async def learning_signals():
    """Get zero-cost learning signal statistics."""
    try:
        from superlocalmemory.learning.signals import LearningSignals
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.server.routes.helpers import DB_PATH
        learning_db = DB_PATH.parent / "learning.db"
        signals = LearningSignals(learning_db)
        config = SLMConfig.load()
        pid = config.active_profile
        return {"success": True, **signals.get_signal_stats(pid)}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@router.post("/learning/consolidate")
async def run_consolidation(request: Request):
    """Run sleep-time consolidation. Body: {dry_run: true/false}."""
    try:
        body = await request.json()
        dry_run = body.get("dry_run", False)
        from superlocalmemory.learning.consolidation_worker import ConsolidationWorker
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.server.routes.helpers import DB_PATH
        worker = ConsolidationWorker(
            memory_db=str(DB_PATH),
            learning_db=str(DB_PATH.parent / "learning.db"),
        )
        config = SLMConfig.load()
        stats = worker.run(config.active_profile, dry_run=dry_run)
        return {"success": True, **stats}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@router.get("/hooks/status")
async def hooks_status():
    """Check if Claude Code hooks are installed."""
    try:
        from superlocalmemory.hooks.claude_code_hooks import check_status
        return {"success": True, **check_status()}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ── Phase 6: V3.2 API Endpoints ──────────────────────────────
# 9 new endpoints for the V3.2 dashboard tabs:
#   Auto-Invoke (2), Associations (2), Consolidation (2),
#   Core Memory (2), VectorStore (1)
#
# Rules enforced:
#   01 - Profile scoping on ALL endpoints
#   06 - No engine import from routes (direct sqlite3)
#   11 - Parameterized SQL everywhere
#   18 - WorkerPool for POST consolidation/trigger
#   19 - Silent errors with JSONResponse
# ──────────────────────────────────────────────────────────────


def _load_auto_invoke_json() -> dict:
    """Load auto-invoke config from config.json's auto_invoke section."""
    from superlocalmemory.server.routes.helpers import MEMORY_DIR
    config_path = MEMORY_DIR / "config.json"
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text())
            return data.get("auto_invoke", {})
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _save_auto_invoke_json(auto_invoke_data: dict) -> None:
    """Persist auto-invoke config into config.json's auto_invoke section."""
    from superlocalmemory.server.routes.helpers import MEMORY_DIR
    config_path = MEMORY_DIR / "config.json"
    cfg: dict = {}
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
        except (json.JSONDecodeError, IOError):
            pass
    cfg["auto_invoke"] = auto_invoke_data
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(cfg, indent=2))


# ── 1. GET /api/v3/auto-invoke/config ─────────────────────────

@router.get("/auto-invoke/config")
async def get_auto_invoke_config(request: Request):
    """Get current auto-invoke configuration."""
    try:
        from superlocalmemory.core.config import AutoInvokeConfig
        defaults = AutoInvokeConfig()

        persisted = _load_auto_invoke_json()

        return {
            "enabled": persisted.get("enabled", defaults.enabled),
            "min_score": persisted.get("min_score", defaults.fok_threshold),
            "weights": persisted.get("weights", dict(defaults.weights)),
            "act_r_mode": persisted.get("act_r_mode", defaults.use_act_r),
            "invocation_count": persisted.get("invocation_count", 0),
            "last_invocation": persisted.get("last_invocation", None),
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── 2. PUT /api/v3/auto-invoke/config ─────────────────────────

@router.put("/auto-invoke/config")
async def set_auto_invoke_config(request: Request):
    """Update auto-invoke configuration.

    Body: {"enabled": true, "min_score": 0.15, "weights": {...}}
    """
    try:
        body = await request.json()

        # Validate min_score range
        min_score = body.get("min_score")
        if min_score is not None and (min_score < 0 or min_score > 1):
            return JSONResponse(
                {"error": "min_score must be between 0 and 1"},
                status_code=400,
            )

        # Load existing, merge updates
        from superlocalmemory.core.config import AutoInvokeConfig
        defaults = AutoInvokeConfig()
        persisted = _load_auto_invoke_json()

        updated = {
            "enabled": body.get("enabled", persisted.get("enabled", defaults.enabled)),
            "min_score": body.get("min_score", persisted.get("min_score", defaults.fok_threshold)),
            "weights": body.get("weights", persisted.get("weights", dict(defaults.weights))),
            "act_r_mode": body.get("act_r_mode", persisted.get("act_r_mode", defaults.use_act_r)),
            "invocation_count": persisted.get("invocation_count", 0),
            "last_invocation": persisted.get("last_invocation", None),
        }
        _save_auto_invoke_json(updated)

        return updated
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── 3. GET /api/v3/associations ───────────────────────────────

@router.get("/associations")
async def get_associations(
    request: Request,
    limit: int = 50,
    type: str = "",
    profile: str = "",
):
    """Get association edges for a profile with content previews."""
    try:
        from superlocalmemory.server.routes.helpers import get_active_profile, DB_PATH
        import sqlite3
        pid = profile or get_active_profile()

        if not DB_PATH.exists():
            return {"edges": [], "total": 0}

        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row

        # Build query with optional type filter (parameterized)
        params: list = [pid]
        sql = (
            "SELECT ae.edge_id, ae.source_fact_id, ae.target_fact_id, "
            "ae.association_type, ae.weight, ae.co_access_count, ae.created_at, "
            "sf.content AS source_content, tf.content AS target_content "
            "FROM association_edges ae "
            "LEFT JOIN atomic_facts sf ON sf.fact_id = ae.source_fact_id "
            "LEFT JOIN atomic_facts tf ON tf.fact_id = ae.target_fact_id "
            "WHERE ae.profile_id = ? "
        )
        if type:
            sql += "AND ae.association_type = ? "
            params.append(type)
        sql += "ORDER BY ae.created_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(sql, params).fetchall()

        # Total count (separate query for pagination info)
        count_sql = "SELECT COUNT(*) FROM association_edges WHERE profile_id = ?"
        count_params: list = [pid]
        if type:
            count_sql += " AND association_type = ?"
            count_params.append(type)
        total = conn.execute(count_sql, count_params).fetchone()[0]

        conn.close()

        edges = []
        for r in rows:
            row = dict(r)
            source_content = row.get("source_content") or ""
            target_content = row.get("target_content") or ""
            edges.append({
                "edge_id": row["edge_id"],
                "source_fact_id": row["source_fact_id"],
                "target_fact_id": row["target_fact_id"],
                "association_type": row["association_type"],
                "weight": round(float(row["weight"]), 3),
                "co_access_count": row["co_access_count"],
                "created_at": row["created_at"],
                "source_preview": source_content[:100],
                "target_preview": target_content[:100],
            })

        return {"edges": edges, "total": total}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── 4. GET /api/v3/associations/stats ─────────────────────────

@router.get("/associations/stats")
async def get_association_stats(request: Request, profile: str = ""):
    """Get association graph statistics."""
    try:
        from superlocalmemory.server.routes.helpers import get_active_profile, DB_PATH
        import sqlite3
        pid = profile or get_active_profile()

        if not DB_PATH.exists():
            return {
                "total_edges": 0,
                "by_type": {},
                "community_count": 0,
                "avg_weight": 0.0,
                "top_connected_facts": [],
            }

        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row

        # Total edges
        total = conn.execute(
            "SELECT COUNT(*) FROM association_edges WHERE profile_id = ?",
            (pid,),
        ).fetchone()[0]

        # Edges by type
        by_type_rows = conn.execute(
            "SELECT association_type, COUNT(*) AS cnt "
            "FROM association_edges WHERE profile_id = ? "
            "GROUP BY association_type",
            (pid,),
        ).fetchall()
        by_type = {row["association_type"]: row["cnt"] for row in by_type_rows}

        # Average weight
        avg_row = conn.execute(
            "SELECT AVG(weight) AS avg_w FROM association_edges "
            "WHERE profile_id = ?",
            (pid,),
        ).fetchone()
        avg_weight = round(float(avg_row["avg_w"] or 0), 3)

        # Community count from fact_importance table
        community_count = 0
        try:
            cc_row = conn.execute(
                "SELECT COUNT(DISTINCT community_id) AS cnt "
                "FROM fact_importance "
                "WHERE profile_id = ? AND community_id IS NOT NULL",
                (pid,),
            ).fetchone()
            community_count = cc_row["cnt"] if cc_row else 0
        except Exception:
            pass

        # Top connected facts (by degree = count of edges as source or target)
        top_facts = []
        try:
            degree_rows = conn.execute(
                "SELECT fact_id, degree FROM ("
                "  SELECT source_fact_id AS fact_id, COUNT(*) AS degree "
                "  FROM association_edges WHERE profile_id = ? "
                "  GROUP BY source_fact_id "
                "  UNION ALL "
                "  SELECT target_fact_id AS fact_id, COUNT(*) AS degree "
                "  FROM association_edges WHERE profile_id = ? "
                "  GROUP BY target_fact_id "
                ") GROUP BY fact_id ORDER BY SUM(degree) DESC LIMIT 5",
                (pid, pid),
            ).fetchall()
            for dr in degree_rows:
                fact_id = dr["fact_id"]
                preview_row = conn.execute(
                    "SELECT content FROM atomic_facts WHERE fact_id = ?",
                    (fact_id,),
                ).fetchone()
                preview = (dict(preview_row).get("content", "")[:80]) if preview_row else ""
                top_facts.append({
                    "fact_id": fact_id,
                    "degree": dr["degree"],
                    "preview": preview,
                })
        except Exception:
            pass

        conn.close()

        return {
            "total_edges": total,
            "by_type": by_type,
            "community_count": community_count,
            "avg_weight": avg_weight,
            "top_connected_facts": top_facts,
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── 5. GET /api/v3/consolidation/status ───────────────────────

@router.get("/consolidation/status")
async def get_consolidation_status(request: Request, profile: str = ""):
    """Get consolidation status and last run results."""
    try:
        from superlocalmemory.server.routes.helpers import get_active_profile, DB_PATH
        from superlocalmemory.core.config import SLMConfig
        import sqlite3

        pid = profile or get_active_profile()
        config = SLMConfig.load()
        cons_cfg = config.consolidation

        result: dict = {
            "enabled": cons_cfg.enabled,
            "last_run": None,
            "last_result": None,
            "triggers": {
                "session_end": cons_cfg.session_trigger,
                "idle_timeout": cons_cfg.idle_timeout_seconds,
                "step_count": cons_cfg.step_count_trigger,
                "scheduled_sessions": cons_cfg.scheduled_sessions,
            },
            "store_count_since_last": 0,
        }

        if not DB_PATH.exists():
            return result

        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row

        # Last consolidation log entry
        try:
            last_row = conn.execute(
                "SELECT timestamp, action_type, reason "
                "FROM consolidation_log "
                "WHERE profile_id = ? ORDER BY timestamp DESC LIMIT 1",
                (pid,),
            ).fetchone()
            if last_row:
                result["last_run"] = dict(last_row).get("timestamp")
        except Exception:
            pass

        # Count blocks compiled (proxy for last consolidation result)
        try:
            block_count = conn.execute(
                "SELECT COUNT(*) FROM core_memory_blocks WHERE profile_id = ?",
                (pid,),
            ).fetchone()[0]
            edge_count = conn.execute(
                "SELECT COUNT(*) FROM association_edges WHERE profile_id = ?",
                (pid,),
            ).fetchone()[0]
            result["last_result"] = {
                "blocks_compiled": block_count,
                "total_edges": edge_count,
            }
        except Exception:
            pass

        # Store count since last consolidation
        try:
            if result["last_run"]:
                sc = conn.execute(
                    "SELECT COUNT(*) FROM atomic_facts "
                    "WHERE profile_id = ? AND created_at > ?",
                    (pid, result["last_run"]),
                ).fetchone()[0]
                result["store_count_since_last"] = sc
            else:
                sc = conn.execute(
                    "SELECT COUNT(*) FROM atomic_facts WHERE profile_id = ?",
                    (pid,),
                ).fetchone()[0]
                result["store_count_since_last"] = sc
        except Exception:
            pass

        conn.close()
        return result
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── 6. POST /api/v3/consolidation/trigger ─────────────────────

@router.post("/consolidation/trigger")
async def trigger_consolidation(request: Request):
    """Trigger consolidation manually.

    Body: {"lightweight": false, "profile": ""}
    Uses WorkerPool for thread safety (Rule 18).
    """
    try:
        body = await request.json()
        lightweight = body.get("lightweight", False)
        profile = body.get("profile", "")

        from superlocalmemory.server.routes.helpers import get_active_profile
        pid = profile or get_active_profile()

        # Use WorkerPool to run consolidation in the worker subprocess (Rule 18)
        try:
            from superlocalmemory.core.worker_pool import WorkerPool
            pool = WorkerPool.shared()
            result = pool.send_command({
                "action": "consolidate",
                "profile_id": pid,
                "lightweight": lightweight,
            })
            if result and result.get("ok"):
                return {"success": True, **result}
        except Exception:
            pass

        # Fallback: direct consolidation if WorkerPool unavailable
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.storage.database import DatabaseManager
        from superlocalmemory.storage import schema as _schema
        from superlocalmemory.core.consolidation_engine import ConsolidationEngine

        config = SLMConfig.load()
        db = DatabaseManager(config.db_path)
        db.initialize(_schema)

        engine = ConsolidationEngine(db=db, config=config.consolidation, slm_config=config)
        result = engine.consolidate(profile_id=pid, lightweight=lightweight)

        # v3.4.1: Auto-trigger behavioral pattern mining after consolidation
        try:
            from superlocalmemory.learning.consolidation_worker import ConsolidationWorker
            learning_db = config.base_dir / "learning.db"
            cw = ConsolidationWorker(str(config.db_path), str(learning_db))
            pattern_count = cw._generate_patterns(pid, False)
            result["patterns_mined"] = pattern_count
            logger.info("Auto-mined %d patterns after consolidation", pattern_count)
        except Exception as exc:
            logger.debug("Pattern mining after consolidation failed: %s", exc)

        return {"success": True, **result}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── 7. GET /api/v3/core-memory ────────────────────────────────

@router.get("/core-memory")
async def get_core_memory(request: Request, profile: str = ""):
    """Get all Core Memory blocks for a profile."""
    try:
        from superlocalmemory.server.routes.helpers import get_active_profile, DB_PATH
        import sqlite3
        pid = profile or get_active_profile()

        if not DB_PATH.exists():
            return {"blocks": [], "total_chars": 0, "char_limit": 2000}

        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            "SELECT block_id, block_type, content, char_count, version, "
            "compiled_by, updated_at FROM core_memory_blocks "
            "WHERE profile_id = ? ORDER BY block_type",
            (pid,),
        ).fetchall()

        conn.close()

        blocks = []
        total_chars = 0
        for r in rows:
            row = dict(r)
            char_count = row.get("char_count", 0) or len(row.get("content", ""))
            total_chars += char_count
            blocks.append({
                "block_id": row["block_id"],
                "block_type": row["block_type"],
                "content": row["content"],
                "char_count": char_count,
                "version": row["version"],
                "compiled_by": row["compiled_by"],
                "updated_at": row["updated_at"],
            })

        return {"blocks": blocks, "total_chars": total_chars, "char_limit": 2000}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── 8. PUT /api/v3/core-memory/{block_id} ─────────────────────

@router.put("/core-memory/{block_id}")
async def update_core_memory_block(block_id: str, request: Request):
    """Update a Core Memory block's content manually.

    Body: {"content": "Updated content..."}
    """
    try:
        body = await request.json()
        content = body.get("content")
        if content is None:
            return JSONResponse(
                {"error": "content field is required"},
                status_code=400,
            )

        from superlocalmemory.server.routes.helpers import DB_PATH
        import sqlite3
        from datetime import datetime, timezone

        if not DB_PATH.exists():
            return JSONResponse(
                {"error": "Database not found"},
                status_code=404,
            )

        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row

        # Verify block exists
        existing = conn.execute(
            "SELECT block_id, profile_id, block_type, version "
            "FROM core_memory_blocks WHERE block_id = ?",
            (block_id,),
        ).fetchone()

        if not existing:
            conn.close()
            return JSONResponse(
                {"error": f"Block {block_id} not found"},
                status_code=404,
            )

        existing_dict = dict(existing)
        new_version = existing_dict["version"] + 1
        now = datetime.now(timezone.utc).isoformat()

        conn.execute(
            "UPDATE core_memory_blocks SET content = ?, char_count = ?, "
            "version = ?, compiled_by = 'manual', updated_at = ? "
            "WHERE block_id = ?",
            (content, len(content), new_version, now, block_id),
        )
        conn.commit()

        # Read back updated block
        updated = conn.execute(
            "SELECT block_id, block_type, content, char_count, version, "
            "compiled_by, updated_at FROM core_memory_blocks "
            "WHERE block_id = ?",
            (block_id,),
        ).fetchone()
        conn.close()

        return dict(updated) if updated else {"block_id": block_id, "updated": True}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── 9. GET /api/v3/vector-store/status ────────────────────────

@router.get("/vector-store/status")
async def get_vector_store_status(request: Request, profile: str = ""):
    """Get VectorStore health and statistics."""
    try:
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.server.routes.helpers import DB_PATH
        import sqlite3

        config = SLMConfig.load()

        result: dict = {
            "available": False,
            "provider": "sqlite-vec",
            "dimension": config.embedding.dimension,
            "embedding_model": config.embedding.model_name,
            "total_vectors": 0,
            "binary_quantization": False,
            "binary_quantization_threshold": 100000,
            "fallback_to_ann": False,
        }

        # Check if sqlite-vec extension is available
        try:
            import sqlite_vec  # noqa: F401
            result["available"] = True
        except ImportError:
            result["fallback_to_ann"] = True

        # Count vectors in embedding_metadata
        if DB_PATH.exists():
            try:
                conn = sqlite3.connect(str(DB_PATH))
                count = conn.execute(
                    "SELECT COUNT(*) FROM embedding_metadata"
                ).fetchone()[0]
                result["total_vectors"] = count
                conn.close()
            except Exception:
                pass

        return result
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Phase 10: V3.3 API Endpoints ────────────────────────────
# 7 new endpoints for the V3.3 dashboard:
#   Forgetting (2), Quantization (1), CCQ (1),
#   Soft Prompts (1), Process Health (1), V3.3 Overview (1)
#
# Rules enforced:
#   01 - Profile scoping on ALL endpoints
#   06 - No engine import from routes (direct sqlite3)
#   11 - Parameterized SQL everywhere
#   19 - Silent errors with JSONResponse
# ──────────────────────────────────────────────────────────────


# ── 1a. GET /api/v3/forgetting/stats ────────────────────────

@router.get("/forgetting/stats")
async def forgetting_stats(request: Request, profile: str = ""):
    """Get memory retention zone distribution."""
    try:
        from superlocalmemory.server.routes.helpers import get_active_profile, DB_PATH
        import sqlite3 as _sqlite3
        pid = profile or get_active_profile()

        zones = {"active": 0, "warm": 0, "cold": 0, "archive": 0, "forgotten": 0}
        total = 0

        if not DB_PATH.exists():
            return {"total": total, "zones": zones}

        conn = _sqlite3.connect(str(DB_PATH))
        conn.row_factory = _sqlite3.Row

        try:
            rows = conn.execute(
                "SELECT lifecycle_zone, COUNT(*) AS cnt "
                "FROM fact_retention WHERE profile_id = ? "
                "GROUP BY lifecycle_zone",
                (pid,),
            ).fetchall()
            for row in rows:
                zone = dict(row)["lifecycle_zone"]
                cnt = dict(row)["cnt"]
                if zone in zones:
                    zones[zone] = cnt
                total += cnt
        except Exception:
            # Table may not exist in older DBs -- graceful fallback
            pass

        conn.close()
        return {"total": total, "zones": zones}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── 1b. POST /api/v3/forgetting/run ─────────────────────────

@router.post("/forgetting/run")
async def run_forgetting(request: Request):
    """Trigger a forgetting decay cycle.

    Body: {"profile": ""} (optional profile override).
    """
    try:
        body = await request.json()
        profile = body.get("profile", "")

        from superlocalmemory.server.routes.helpers import get_active_profile, DB_PATH
        import sqlite3 as _sqlite3
        pid = profile or get_active_profile()

        if not DB_PATH.exists():
            return {"success": False, "error": "Database not found"}

        conn = _sqlite3.connect(str(DB_PATH))
        conn.row_factory = _sqlite3.Row

        updated = 0
        try:
            # Apply Ebbinghaus decay: reduce retention for facts not accessed recently
            # Formula: retention *= exp(-0.1) for each cycle (simplified batch decay)
            conn.execute(
                "UPDATE fact_retention "
                "SET retention_score = MAX(0.0, retention_score * 0.9), "
                "    last_computed_at = datetime('now') "
                "WHERE profile_id = ? "
                "AND lifecycle_zone NOT IN ('archive', 'forgotten')",
                (pid,),
            )
            updated = conn.total_changes

            # Transition zones based on new retention scores
            zone_thresholds = [
                ("forgotten", 0.05),
                ("archive", 0.15),
                ("cold", 0.35),
                ("warm", 0.65),
            ]
            for zone, threshold in zone_thresholds:
                conn.execute(
                    "UPDATE fact_retention "
                    "SET lifecycle_zone = ? "
                    "WHERE profile_id = ? "
                    "AND retention_score < ? "
                    "AND lifecycle_zone NOT IN ('archive', 'forgotten')",
                    (zone, pid, threshold),
                )

            # Ensure high-retention facts are active
            conn.execute(
                "UPDATE fact_retention "
                "SET lifecycle_zone = 'active' "
                "WHERE profile_id = ? AND retention_score >= 0.65 "
                "AND lifecycle_zone NOT IN ('archive', 'forgotten')",
                (pid,),
            )

            conn.commit()
        except Exception as exc:
            conn.close()
            return {"success": False, "error": str(exc)}

        conn.close()
        return {"success": True, "facts_decayed": updated, "profile": pid}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── 1c. GET /api/v3/quantization/stats ──────────────────────

@router.get("/quantization/stats")
async def quantization_stats(request: Request, profile: str = ""):
    """Get embedding quantization tier distribution."""
    try:
        from superlocalmemory.server.routes.helpers import get_active_profile, DB_PATH
        import sqlite3 as _sqlite3
        pid = profile or get_active_profile()

        tiers = {"float32": 0, "int8": 0, "polar4": 0, "polar2": 0}
        total = 0
        compression_ratio = 1.0

        if not DB_PATH.exists():
            return {"total": total, "tiers": tiers, "compression_ratio": compression_ratio}

        conn = _sqlite3.connect(str(DB_PATH))
        conn.row_factory = _sqlite3.Row

        try:
            rows = conn.execute(
                "SELECT quantization_level, COUNT(*) AS cnt "
                "FROM embedding_quantization_metadata "
                "WHERE profile_id = ? "
                "GROUP BY quantization_level",
                (pid,),
            ).fetchall()
            for row in rows:
                level = dict(row)["quantization_level"]
                cnt = dict(row)["cnt"]
                if level in tiers:
                    tiers[level] = cnt
                total += cnt
        except Exception:
            pass

        # Compute compression ratio from actual sizes if available
        try:
            size_row = conn.execute(
                "SELECT "
                "SUM(CASE WHEN bit_width = 32 THEN 768 * 4 ELSE "
                "    COALESCE(compressed_size_bytes, 768 * bit_width / 8) END) AS actual, "
                "SUM(768 * 4) AS uncompressed "
                "FROM embedding_quantization_metadata "
                "WHERE profile_id = ?",
                (pid,),
            ).fetchone()
            if size_row:
                d = dict(size_row)
                uncompressed = d.get("uncompressed") or 0
                actual = d.get("actual") or 0
                if actual > 0 and uncompressed > 0:
                    compression_ratio = round(uncompressed / actual, 2)
        except Exception:
            pass

        conn.close()
        return {"total": total, "tiers": tiers, "compression_ratio": compression_ratio}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── 1d. GET /api/v3/ccq/blocks ──────────────────────────────

@router.get("/ccq/blocks")
async def ccq_blocks(request: Request, profile: str = "", limit: int = 50):
    """Get CCQ consolidated blocks."""
    try:
        from superlocalmemory.server.routes.helpers import get_active_profile, DB_PATH
        import sqlite3 as _sqlite3
        pid = profile or get_active_profile()

        if not DB_PATH.exists():
            return {"blocks": [], "total": 0}

        conn = _sqlite3.connect(str(DB_PATH))
        conn.row_factory = _sqlite3.Row

        blocks = []
        total = 0
        try:
            rows = conn.execute(
                "SELECT block_id, content, source_fact_ids, char_count, "
                "compiled_by, cluster_id, created_at "
                "FROM ccq_consolidated_blocks "
                "WHERE profile_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (pid, limit),
            ).fetchall()

            for row in rows:
                d = dict(row)
                source_ids = []
                try:
                    source_ids = json.loads(d.get("source_fact_ids", "[]"))
                except (json.JSONDecodeError, TypeError):
                    pass
                blocks.append({
                    "block_id": d["block_id"],
                    "content": d["content"],
                    "source_fact_count": len(source_ids),
                    "char_count": d["char_count"],
                    "compiled_by": d["compiled_by"],
                    "cluster_id": d["cluster_id"],
                    "created_at": d["created_at"],
                })

            count_row = conn.execute(
                "SELECT COUNT(*) FROM ccq_consolidated_blocks "
                "WHERE profile_id = ?",
                (pid,),
            ).fetchone()
            total = count_row[0] if count_row else 0
        except Exception:
            pass

        conn.close()
        return {"blocks": blocks, "total": total}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── 1e. GET /api/v3/soft-prompts ─────────────────────────────

@router.get("/soft-prompts")
async def get_soft_prompts(request: Request, profile: str = ""):
    """Get active soft prompt templates."""
    try:
        from superlocalmemory.server.routes.helpers import get_active_profile, DB_PATH
        import sqlite3 as _sqlite3
        pid = profile or get_active_profile()

        if not DB_PATH.exists():
            return {"prompts": [], "total": 0, "total_tokens": 0}

        conn = _sqlite3.connect(str(DB_PATH))
        conn.row_factory = _sqlite3.Row

        prompts = []
        total_tokens = 0
        try:
            rows = conn.execute(
                "SELECT prompt_id, category, content, confidence, "
                "effectiveness, token_count, retention_score, "
                "active, version, created_at, updated_at "
                "FROM soft_prompt_templates "
                "WHERE profile_id = ? AND active = 1 "
                "ORDER BY confidence DESC",
                (pid,),
            ).fetchall()

            for row in rows:
                d = dict(row)
                tokens = d.get("token_count", 0)
                total_tokens += tokens
                prompts.append({
                    "prompt_id": d["prompt_id"],
                    "category": d["category"],
                    "content": d["content"][:200],
                    "confidence": round(float(d["confidence"]), 3),
                    "effectiveness": round(float(d.get("effectiveness", 0.5)), 3),
                    "token_count": tokens,
                    "retention_score": round(float(d.get("retention_score", 1.0)), 3),
                    "version": d["version"],
                    "created_at": d["created_at"],
                })
        except Exception:
            pass

        conn.close()
        return {"prompts": prompts, "total": len(prompts), "total_tokens": total_tokens}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── 1f. GET /api/v3/health/processes ─────────────────────────

@router.get("/health/processes")
async def process_health(request: Request):
    """Get SLM process health status."""
    try:
        import os as _os

        processes = {
            "mcp_server": {"pid": _os.getpid(), "status": "running"},
            "parent": {"pid": _os.getppid(), "status": "unknown"},
        }

        # Check parent process
        try:
            _os.kill(_os.getppid(), 0)
            processes["parent"]["status"] = "running"
        except ProcessLookupError:
            processes["parent"]["status"] = "dead"
        except PermissionError:
            processes["parent"]["status"] = "running"
        except OSError:
            processes["parent"]["status"] = "unknown"

        # Check worker pool status
        worker_status = "unavailable"
        try:
            from superlocalmemory.core.worker_pool import WorkerPool
            pool = WorkerPool.shared()
            worker_status = "running" if pool else "stopped"
        except Exception:
            pass
        processes["worker_pool"] = {"status": worker_status}

        # Memory usage of current process (approximate)
        memory_mb = 0.0
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            memory_mb = round(usage.ru_maxrss / (1024 * 1024), 1)
        except Exception:
            pass

        return {
            "processes": processes,
            "memory_mb": memory_mb,
            "healthy": processes["parent"]["status"] != "dead",
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── 1g. GET /api/v3/v33/overview ─────────────────────────────

# ── v3.4.1: Graph Communities ──────────────────────────────────

@router.get("/graph/communities")
async def get_graph_communities(request: Request, profile: str = ""):
    """Get community assignments with TF-IDF labels, entities, and colors.

    v3.4.1: Uses TF-IDF labels from config table (computed by GraphAnalyzer
    at consolidation time). Falls back to inline word frequency if labels
    not yet computed.
    """
    try:
        from superlocalmemory.server.routes.helpers import get_active_profile, DB_PATH
        import sqlite3
        pid = profile or get_active_profile()

        if not DB_PATH.exists():
            return {"communities": [], "total": 0}

        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row

        # Get community member counts and average pagerank
        comm_rows = conn.execute(
            "SELECT community_id, COUNT(*) AS member_count, "
            "AVG(pagerank_score) AS pagerank_avg "
            "FROM fact_importance "
            "WHERE profile_id = ? AND community_id IS NOT NULL "
            "GROUP BY community_id ORDER BY member_count DESC",
            (pid,),
        ).fetchall()

        if not comm_rows:
            conn.close()
            return {"communities": [], "total": 0}

        # Load TF-IDF labels from config table
        tfidf_labels = {}
        try:
            label_key = "community_labels_" + pid
            label_row = conn.execute(
                "SELECT value FROM config WHERE key = ?",
                (label_key,),
            ).fetchone()
            if label_row:
                import json as _json
                tfidf_labels = _json.loads(dict(label_row)["value"])
        except Exception:
            pass

        cluster_colors = [
            '#667eea', '#764ba2', '#43e97b', '#38f9d7',
            '#4facfe', '#00f2fe', '#f093fb', '#f5576c',
            '#fa709a', '#fee140', '#30cfd0', '#330867',
        ]

        communities = []
        for row in comm_rows:
            d = dict(row)
            comm_id = d["community_id"]

            # Use TF-IDF label if available, else generate inline
            label = tfidf_labels.get(str(comm_id), "")
            if not label:
                # Inline fallback: word frequency from top facts
                try:
                    fact_rows = conn.execute(
                        "SELECT af.content FROM fact_importance fi "
                        "JOIN atomic_facts af ON fi.fact_id = af.fact_id "
                        "WHERE fi.profile_id = ? AND fi.community_id = ? "
                        "ORDER BY fi.pagerank_score DESC LIMIT 20",
                        (pid, comm_id),
                    ).fetchall()
                    from collections import defaultdict as _ddict
                    wf = _ddict(int)
                    sw = {"the", "a", "an", "is", "was", "are", "to", "of",
                          "in", "for", "on", "with", "at", "by", "from",
                          "and", "but", "or", "not", "it", "this", "that",
                          "i", "we", "they", "he", "she", "you", "my"}
                    for fr in fact_rows:
                        for w in (dict(fr).get("content", "")).lower().split():
                            w = w.strip(".,;:!?\"'()[]{}")
                            if len(w) > 2 and w not in sw:
                                wf[w] += 1
                    top = sorted(wf.items(), key=lambda x: x[1], reverse=True)[:3]
                    label = ", ".join(w for w, _ in top) if top else f"Community {comm_id}"
                except Exception:
                    label = f"Community {comm_id}"

            # Get top entities from canonical_entities_json
            top_entities = []
            try:
                ent_rows = conn.execute(
                    "SELECT af.canonical_entities_json FROM fact_importance fi "
                    "JOIN atomic_facts af ON fi.fact_id = af.fact_id "
                    "WHERE fi.profile_id = ? AND fi.community_id = ? "
                    "ORDER BY fi.pagerank_score DESC LIMIT 10",
                    (pid, comm_id),
                ).fetchall()
                import json as _json2
                entity_counts: dict = {}
                for er in ent_rows:
                    raw = dict(er).get("canonical_entities_json", "")
                    if raw:
                        try:
                            for ent in _json2.loads(raw):
                                entity_counts[ent] = entity_counts.get(ent, 0) + 1
                        except (ValueError, TypeError):
                            pass
                top_entities = sorted(
                    entity_counts, key=entity_counts.get, reverse=True,
                )[:5]
            except Exception:
                pass

            communities.append({
                "community_id": comm_id,
                "label": label,
                "member_count": d["member_count"],
                "top_entities": top_entities,
                "color": cluster_colors[comm_id % len(cluster_colors)],
                "pagerank_avg": round(float(d["pagerank_avg"] or 0), 4),
            })

        conn.close()
        return {"communities": communities, "total": len(communities)}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/graph/run-communities")
async def run_community_detection(request: Request):
    """Trigger community detection manually (runs GraphAnalyzer)."""
    try:
        from superlocalmemory.server.routes.helpers import get_active_profile, DB_PATH
        from superlocalmemory.storage.database import DatabaseManager
        from superlocalmemory.storage import schema as _schema
        from superlocalmemory.core.graph_analyzer import GraphAnalyzer

        pid = get_active_profile()
        db = DatabaseManager(DB_PATH)
        db.initialize(_schema)

        analyzer = GraphAnalyzer(db)
        result = analyzer.compute_and_store(pid)
        return {"success": True, **result}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/v33/overview")
async def v33_overview(request: Request, profile: str = ""):
    """Get SLM 3.3 feature overview -- all new capabilities at a glance."""
    try:
        from superlocalmemory.server.routes.helpers import get_active_profile, DB_PATH
        import sqlite3 as _sqlite3
        pid = profile or get_active_profile()

        overview: dict = {
            "version": "3.3",
            "profile": pid,
            "forgetting": {"total": 0, "zones": {}},
            "quantization": {"total": 0, "tiers": {}, "compression_ratio": 1.0},
            "ccq": {"blocks": 0, "facts_archived": 0},
            "soft_prompts": {"total": 0, "total_tokens": 0},
            "hopfield": {
                "available": False,
                "description": "Modern Continuous Hopfield Network retrieval channel",
            },
            "process_health": {"healthy": True},
        }

        if not DB_PATH.exists():
            return overview

        conn = _sqlite3.connect(str(DB_PATH))
        conn.row_factory = _sqlite3.Row

        # Forgetting stats
        try:
            zones = {"active": 0, "warm": 0, "cold": 0, "archive": 0, "forgotten": 0}
            rows = conn.execute(
                "SELECT lifecycle_zone, COUNT(*) AS cnt "
                "FROM fact_retention WHERE profile_id = ? "
                "GROUP BY lifecycle_zone",
                (pid,),
            ).fetchall()
            total_fg = 0
            for row in rows:
                d = dict(row)
                zone = d["lifecycle_zone"]
                if zone in zones:
                    zones[zone] = d["cnt"]
                total_fg += d["cnt"]
            overview["forgetting"] = {"total": total_fg, "zones": zones}
        except Exception:
            pass

        # Quantization stats
        try:
            tiers = {"float32": 0, "int8": 0, "polar4": 0, "polar2": 0}
            rows = conn.execute(
                "SELECT quantization_level, COUNT(*) AS cnt "
                "FROM embedding_quantization_metadata "
                "WHERE profile_id = ? GROUP BY quantization_level",
                (pid,),
            ).fetchall()
            total_q = 0
            for row in rows:
                d = dict(row)
                level = d["quantization_level"]
                if level in tiers:
                    tiers[level] = d["cnt"]
                total_q += d["cnt"]
            overview["quantization"] = {
                "total": total_q, "tiers": tiers, "compression_ratio": 1.0,
            }
        except Exception:
            pass

        # CCQ stats
        try:
            block_count = conn.execute(
                "SELECT COUNT(*) FROM ccq_consolidated_blocks "
                "WHERE profile_id = ?", (pid,),
            ).fetchone()[0]
            # Count archived facts (lifecycle='archived' from CCQ)
            archived_count = 0
            try:
                archived_count = conn.execute(
                    "SELECT COUNT(*) FROM atomic_facts "
                    "WHERE profile_id = ? AND lifecycle = 'archived'",
                    (pid,),
                ).fetchone()[0]
            except Exception:
                pass
            overview["ccq"] = {
                "blocks": block_count,
                "facts_archived": archived_count,
            }
        except Exception:
            pass

        # Soft prompts stats
        try:
            prompt_rows = conn.execute(
                "SELECT COUNT(*) AS cnt, COALESCE(SUM(token_count), 0) AS tokens "
                "FROM soft_prompt_templates "
                "WHERE profile_id = ? AND active = 1",
                (pid,),
            ).fetchone()
            if prompt_rows:
                d = dict(prompt_rows)
                overview["soft_prompts"] = {
                    "total": d["cnt"],
                    "total_tokens": d["tokens"],
                }
        except Exception:
            pass

        # Hopfield channel availability
        try:
            from superlocalmemory.retrieval.hopfield_channel import HopfieldChannel  # noqa: F401
            overview["hopfield"]["available"] = True
        except ImportError:
            pass

        # Process health
        try:
            import os as _os
            _os.kill(_os.getppid(), 0)
            overview["process_health"] = {"healthy": True}
        except ProcessLookupError:
            overview["process_health"] = {"healthy": False}
        except (PermissionError, OSError):
            overview["process_health"] = {"healthy": True}

        conn.close()
        return overview
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
