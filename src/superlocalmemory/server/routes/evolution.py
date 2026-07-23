# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Evolution API routes — dashboard endpoints for skill evolution engine.

Routes: /api/evolution/status, /api/evolution/enable, /api/evolution/run
"""

import logging
from types import SimpleNamespace
from typing import Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

from superlocalmemory.server.config_file import read_config, update_config

from .helpers import MEMORY_DIR, get_active_profile

logger = logging.getLogger("superlocalmemory.routes.evolution")
router = APIRouter()


def _require_read(request: Request) -> None:
    """Guard evolution telemetry with READ on the active profile."""
    from superlocalmemory.access.rbac import Permission
    from superlocalmemory.server.rbac_enforce import require_permission

    require_permission(request, Permission.READ, profile=get_active_profile())


def _require_manage(request: Request) -> None:
    """Guard evolution mutations with the same RBAC boundary as v3 settings."""
    from superlocalmemory.server.rbac_enforce import require_manage

    require_manage(request)


def _read_evolution_config() -> dict:
    """Read one process-safe evolution config snapshot."""
    return dict(read_config(MEMORY_DIR / "config.json").get("evolution", {}))


def _update_evolution_config(update) -> dict:
    """Atomically update evolution without losing other config sections."""
    config_path = MEMORY_DIR / "config.json"

    def mutate(cfg: dict) -> None:
        evolution = cfg.setdefault("evolution", {})
        update(evolution)

    cfg = update_config(config_path, mutate)
    return dict(cfg.get("evolution", {}))


def _enable_evolution(config: dict) -> None:
    config["enabled"] = True
    config.setdefault("backend", "auto")


@router.get("/api/evolution/status")
def evolution_status(request: Request):
    """Get evolution engine status, backend, and recent history."""
    _require_read(request)
    try:
        from superlocalmemory.evolution.evolution_store import EvolutionStore
        from superlocalmemory.evolution.skill_evolver import detect_backend

        evo_cfg = _read_evolution_config()
        enabled = evo_cfg.get("enabled", False)
        backend_setting = evo_cfg.get("backend", "auto")
        backend = (
            detect_backend() if enabled and backend_setting == "auto"
            else backend_setting if enabled
            else "none"
        )
        db_path = str(MEMORY_DIR / "memory.db")

        profile_id = get_active_profile()
        store = EvolutionStore(db_path)
        stats = store.get_stats(profile_id)
        recent = store.get_recent(profile_id, limit=10)

        return {
            "enabled": enabled,
            "backend": backend,
            "config": {
                "backend_setting": backend_setting,
                "max_per_cycle": evo_cfg.get("max_evolutions_per_cycle", 3),
                "mutation_model": evo_cfg.get("mutation_model", ""),
                "verify_model": evo_cfg.get("verify_model", ""),
                "confirm_model": evo_cfg.get("confirm_model", ""),
            },
            "stats": {
                "total": stats.get("total", 0),
                "promoted": stats.get("by_status", {}).get("promoted", 0),
                "rejected": stats.get("by_status", {}).get("rejected", 0),
                "failed": stats.get("by_status", {}).get("failed", 0),
                "cycle_budget_remaining": stats.get("cycle_budget_remaining", 3),
            },
            "recent": [
                {
                    "id": r.id,
                    "skill_name": r.skill_name,
                    "evolution_type": r.evolution_type.value,
                    "trigger": r.trigger.value,
                    "status": r.status.value,
                    "mutation_summary": r.mutation_summary,
                    "blind_verified": r.blind_verified,
                    "created_at": r.created_at,
                }
                for r in recent
            ],
        }
    except Exception:
        logger.exception("evolution_status error")
        return {"enabled": False, "backend": "none", "error": "Internal server error"}


@router.post("/api/evolution/enable")
def evolution_enable(request: Request):
    """Enable evolution without replacing the user's selected backend."""
    _require_manage(request)
    try:
        evolution = _update_evolution_config(_enable_evolution)
        return {
            "ok": True,
            "message": f"Evolution enabled with {evolution['backend']} backend.",
        }
    except Exception:
        logger.exception("evolution_enable error")
        return {"ok": False, "error": "Internal server error"}


@router.post("/api/evolution/disable")
def evolution_disable(request: Request):
    """Disable skill evolution engine.  Mirrors /api/evolution/enable."""
    _require_manage(request)
    try:
        _update_evolution_config(lambda cfg: cfg.update({"enabled": False}))

        return {"ok": True, "message": "Evolution disabled."}
    except Exception:
        logger.exception("evolution_disable error")
        return {"ok": False, "error": "Internal server error"}


@router.post("/api/evolution/run")
def evolution_run(request: Request):
    """Manually trigger an evolution cycle."""
    _require_manage(request)
    try:
        from superlocalmemory.evolution.skill_evolver import SkillEvolver

        evo_cfg = _read_evolution_config()
        if not evo_cfg.get("enabled", False):
            return {"ok": False, "error": "Evolution is disabled. Enable first."}

        profile = get_active_profile()
        db_path = str(MEMORY_DIR / "memory.db")

        # Build a minimal config object for the evolver. Must carry the
        # per-step model fields (v3.7.9) or a dashboard-triggered run would
        # silently ignore the user's configured models and fall back to the
        # cheapest defaults.
        evolution_config = SimpleNamespace(
            enabled=True,
            backend=evo_cfg.get("backend", "auto"),
            max_evolutions_per_cycle=evo_cfg.get("max_evolutions_per_cycle", 3),
            mutation_model=evo_cfg.get("mutation_model", ""),
            verify_model=evo_cfg.get("verify_model", ""),
            confirm_model=evo_cfg.get("confirm_model", ""),
        )
        evolver = SkillEvolver(
            db_path, SimpleNamespace(evolution=evolution_config)
        )
        result = evolver.run_consolidation_cycle(profile)

        return {"ok": True, **result}
    except Exception:
        logger.exception("evolution_run error")
        return {"ok": False, "error": "Internal server error"}


class EvolutionConfigUpdate(BaseModel):
    enabled: Optional[bool] = None
    backend: Optional[str] = None
    max_evolutions_per_cycle: Optional[int] = None
    mutation_model: Optional[str] = None
    verify_model: Optional[str] = None
    confirm_model: Optional[str] = None


@router.post("/api/evolution/config")
def evolution_config(request: Request, body: EvolutionConfigUpdate):
    """Update evolution config from the dashboard (v3.7.9).

    Validates model + backend values against the same allow-list the CLI uses
    (``slm config set evolution.*``) and persists to config.json atomically.
    Only fields provided in the body are changed.
    """
    _require_manage(request)
    try:
        from superlocalmemory.evolution.model_selection import _MODEL_ALIASES

        accepted_models = set(_MODEL_ALIASES) | {"", "auto"}
        accepted_backends = {"auto", "claude", "ollama", "anthropic", "openai"}

        for field in ("mutation_model", "verify_model", "confirm_model"):
            val = getattr(body, field)
            if val is not None and val not in accepted_models:
                allowed = ", ".join(["auto", *sorted(_MODEL_ALIASES)])
                return {"ok": False, "error": f"{field} must be one of: {allowed}"}
        if body.backend is not None and body.backend not in accepted_backends:
            return {
                "ok": False,
                "error": f"backend must be one of: {', '.join(sorted(accepted_backends))}",
            }
        if (body.max_evolutions_per_cycle is not None
                and not 0 < body.max_evolutions_per_cycle <= 50):
            return {"ok": False, "error": "max_evolutions_per_cycle must be 1..50"}

        def _apply(evo: dict) -> None:
            for field in (
                "enabled",
                "backend",
                "max_evolutions_per_cycle",
                "mutation_model",
                "verify_model",
                "confirm_model",
            ):
                value = getattr(body, field)
                if value is None:
                    continue
                if field.endswith("_model") and value == "auto":
                    value = ""
                evo[field] = value

        evo = _update_evolution_config(_apply)

        return {"ok": True, "config": {
            "enabled": evo.get("enabled", False),
            "backend": evo.get("backend", "auto"),
            "max_evolutions_per_cycle": evo.get("max_evolutions_per_cycle", 3),
            "mutation_model": evo.get("mutation_model", ""),
            "verify_model": evo.get("verify_model", ""),
            "confirm_model": evo.get("confirm_model", ""),
        }}
    except Exception:
        logger.exception("evolution_config error")
        return {"ok": False, "error": "Internal server error"}


@router.get("/api/evolution/lineage")
def evolution_lineage(request: Request, skill_name: str = ""):
    """Get evolution lineage for a skill or all skills.

    Returns lineage records and a tree structure grouped by root skill.
    """
    _require_read(request)
    conn = None
    try:
        import sqlite3 as _sqlite3

        db_path = str(MEMORY_DIR / "memory.db")
        profile_id = get_active_profile()
        conn = _sqlite3.connect(db_path, timeout=10)
        conn.row_factory = _sqlite3.Row

        if skill_name:
            rows = conn.execute(
                "SELECT id, skill_name, parent_skill_id, evolution_type, "
                "trigger_type, generation, status, mutation_summary, "
                "blind_verified, created_at, completed_at "
                "FROM skill_evolution_log "
                "WHERE profile_id = ? AND (skill_name = ? OR parent_skill_id = ?) "
                "ORDER BY created_at ASC",
                (profile_id, skill_name, skill_name),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, skill_name, parent_skill_id, evolution_type, "
                "trigger_type, generation, status, mutation_summary, "
                "blind_verified, created_at, completed_at "
                "FROM skill_evolution_log "
                "WHERE profile_id = ? "
                "ORDER BY created_at DESC LIMIT 100",
                (profile_id,),
            ).fetchall()

        lineage = [
            {
                "id": dict(r)["id"],
                "skill_name": dict(r)["skill_name"],
                "parent_skill_id": dict(r).get("parent_skill_id", ""),
                "evolution_type": dict(r)["evolution_type"],
                "trigger": dict(r)["trigger_type"],
                "generation": dict(r).get("generation", 0),
                "status": dict(r)["status"],
                "mutation_summary": dict(r).get("mutation_summary", ""),
                "blind_verified": bool(dict(r).get("blind_verified", 0)),
                "created_at": dict(r).get("created_at", ""),
                "completed_at": dict(r).get("completed_at", ""),
            }
            for r in rows
        ]

        # Build tree structure: group by root skill
        tree: dict = {}
        for entry in lineage:
            root = entry.get("parent_skill_id") or entry["skill_name"]
            if root not in tree:
                tree[root] = {"root": root, "evolutions": []}
            tree[root]["evolutions"].append({
                "id": entry["id"],
                "skill_name": entry["skill_name"],
                "evolution_type": entry["evolution_type"],
                "status": entry["status"],
                "generation": entry["generation"],
                "created_at": entry["created_at"],
            })

        return {
            "lineage": lineage,
            "lineage_count": len(lineage),
            "tree": tree,
        }
    except Exception:
        logger.exception("evolution_lineage error")
        return {"lineage": [], "lineage_count": 0, "tree": {}, "error": "Internal server error"}
    finally:
        if conn is not None:
            conn.close()
