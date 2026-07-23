# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Dashboard config endpoints — storage, daemon, mesh, trust, forgetting.

Each section provides GET (read current) and PUT (validate + persist) routes.
Writes use the same direct-JSON approach as the evolution config endpoint:
  1. Read config.json as a raw dict (fail-open → defaults when absent/corrupt).
  2. Update only the targeted keys; all other keys — including 'mode' — survive.
  3. Atomic write via a temp file + os.replace (no torn writes).

Auth: follows the same pattern as the auto-capture/auto-recall/auto-invoke
config endpoints — no route-level auth guard.  Write-identity is enforced by
the middleware layer wired in unified_daemon.py / ui.py for non-localhost
callers.  Tests use a bare FastAPI app (no middleware) and monkeypatch
MEMORY_DIR to a tmp_path.

Restart-required semantics:
  - graph_backend / vector_backend: changes take effect only on daemon restart.
  - daemon_port / daemon_legacy_port: changes take effect only on daemon restart.
  - mesh, trust, and forgetting are persisted safely but require restart
    because no complete supported worker-rebind transaction exists for them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from superlocalmemory.server.config_file import read_config, update_config
from superlocalmemory.server.routes.helpers import MEMORY_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v3", tags=["config"])


def _require_admin(request: Request) -> None:
    """SEC-H-01: system-configuration mutations are admin-only (MANAGE).

    Machine auth (loopback / credential) is enforced by the daemon middleware;
    this adds the RBAC layer so a logged-in non-admin (viewer/member) in company
    mode cannot change the daemon port, swap the LLM key, or alter the forgetting
    curve. The machine owner keeps MANAGE (personal mode is unaffected). Call
    this BEFORE the handler's try/except so the 401/403 is not swallowed into a
    500.
    """
    from superlocalmemory.server.rbac_enforce import require_manage
    require_manage(request)

# ---------------------------------------------------------------------------
# Allowed backend values
# ---------------------------------------------------------------------------

_GRAPH_BACKENDS = frozenset({"auto", "sqlite", "cozo"})
_VECTOR_BACKENDS = frozenset({"auto", "lancedb", "sqlite-vec"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _config_path() -> Path:
    """Return the config.json path, resolved from the active MEMORY_DIR."""
    return MEMORY_DIR / "config.json"


def _read_config() -> dict:
    """Read one coherent config snapshot."""
    p = _config_path()
    try:
        return read_config(p)
    except (ValueError, OSError) as exc:
        logger.warning("config_api: could not read config.json: %s", exc)
        raise


def _update_config(mutator) -> dict:
    """Run one interprocess-locked read/modify/replace transaction."""
    return update_config(_config_path(), mutator)


# ---------------------------------------------------------------------------
# Pydantic request models  (extra="forbid" → 422 on unknown keys)
# ---------------------------------------------------------------------------


class StorageConfigUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    graph_backend: Optional[Annotated[str, Field(pattern=r"^(auto|sqlite|cozo)$")]] = None
    vector_backend: Optional[Annotated[str, Field(pattern=r"^(auto|lancedb|sqlite-vec)$")]] = None


class DaemonConfigUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idle_timeout: Optional[int] = Field(None, ge=0)
    port: Optional[int] = Field(None, ge=1, le=65535)
    legacy_port: Optional[int] = Field(None, ge=1, le=65535)
    enable_legacy_port: Optional[StrictBool] = None


class MeshConfigUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: StrictBool


class TrustConfigUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_trust_weighting: Optional[StrictBool] = None
    trust_first_party: Optional[StrictBool] = None
    promotion_min_trust: Optional[float] = Field(None, ge=0.0, le=1.0)


class ForgettingConfigUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: Optional[StrictBool] = None
    alpha: Optional[float] = Field(None, gt=0.0)
    beta: Optional[float] = Field(None, gt=0.0)
    gamma: Optional[float] = Field(None, gt=0.0)
    delta: Optional[float] = Field(None, gt=0.0)
    min_strength: Optional[float] = Field(None, gt=0.0)
    max_strength: Optional[float] = Field(None, gt=0.0)
    archive_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    forget_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    learning_rate: Optional[float] = Field(None, gt=0.0)
    forgetting_drift_scale: Optional[float] = Field(None, gt=0.0)
    trust_kappa: Optional[float] = Field(None, gt=0.0)
    scheduler_interval_minutes: Optional[int] = Field(None, ge=1)
    core_memory_immune: Optional[StrictBool] = None


# ---------------------------------------------------------------------------
# Default value constants (mirrors ForgettingConfig dataclass defaults)
# ---------------------------------------------------------------------------

_FORGETTING_DEFAULTS: dict = {
    "enabled": True,
    "alpha": 2.0,
    "beta": 1.5,
    "gamma": 1.0,
    "delta": 0.5,
    "min_strength": 0.1,
    "max_strength": 100.0,
    "archive_threshold": 0.2,
    "forget_threshold": 0.05,
    "learning_rate": 1.0,
    "forgetting_drift_scale": 0.5,
    "trust_kappa": 2.0,
    "scheduler_interval_minutes": 30,
    "core_memory_immune": True,
}


# ---------------------------------------------------------------------------
# GET /api/v3/storage/config
# ---------------------------------------------------------------------------


@router.get("/storage/config")
def get_storage_config():
    """Return current storage backend configuration.

    base_dir is read-only — it is derived from the process namespace and
    cannot be changed via this endpoint.
    """
    try:
        data = _read_config()
        return {
            "graph_backend": data.get("graph_backend", "auto"),
            "vector_backend": data.get("vector_backend", "auto"),
            "base_dir": data.get("base_dir", str(MEMORY_DIR)),
        }
    except Exception:
        logger.exception("get_storage_config failed")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


# ---------------------------------------------------------------------------
# PUT /api/v3/storage/config
# ---------------------------------------------------------------------------


@router.put("/storage/config")
def put_storage_config(request: Request, body: StorageConfigUpdate):
    """Update graph_backend and/or vector_backend.

    Both fields require a daemon restart to take effect.
    Returns restart_required: true unconditionally.
    """
    _require_admin(request)
    try:
        def mutate(data: dict) -> None:
            if body.graph_backend is not None:
                data["graph_backend"] = body.graph_backend
            if body.vector_backend is not None:
                data["vector_backend"] = body.vector_backend

        data = _update_config(mutate)
        return {
            "graph_backend": data.get("graph_backend", "auto"),
            "vector_backend": data.get("vector_backend", "auto"),
            "base_dir": data.get("base_dir", str(MEMORY_DIR)),
            "restart_required": True,
        }
    except Exception:
        logger.exception("put_storage_config failed")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


# ---------------------------------------------------------------------------
# GET /api/v3/daemon/config
# ---------------------------------------------------------------------------


@router.get("/daemon/config")
def get_daemon_config():
    """Return current daemon configuration."""
    try:
        data = _read_config()
        return {
            "idle_timeout": data.get("daemon_idle_timeout", 0),
            "port": data.get("daemon_port", 8765),
            "legacy_port": data.get("daemon_legacy_port", 8767),
            "enable_legacy_port": data.get("daemon_enable_legacy_port", True),
        }
    except Exception:
        logger.exception("get_daemon_config failed")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


# ---------------------------------------------------------------------------
# PUT /api/v3/daemon/config
# ---------------------------------------------------------------------------


@router.put("/daemon/config")
def put_daemon_config(request: Request, body: DaemonConfigUpdate):
    """Update daemon configuration.

    Port / legacy_port changes require a daemon restart.
    restart_required is True if any port field is included in the request.
    """
    _require_admin(request)
    try:
        port_changed = False

        def mutate(data: dict) -> None:
            nonlocal port_changed
            if body.idle_timeout is not None:
                data["daemon_idle_timeout"] = body.idle_timeout
            if body.port is not None:
                data["daemon_port"] = body.port
                port_changed = True
            if body.legacy_port is not None:
                data["daemon_legacy_port"] = body.legacy_port
                port_changed = True
            if body.enable_legacy_port is not None:
                data["daemon_enable_legacy_port"] = body.enable_legacy_port

        data = _update_config(mutate)
        return {
            "idle_timeout": data.get("daemon_idle_timeout", 0),
            "port": data.get("daemon_port", 8765),
            "legacy_port": data.get("daemon_legacy_port", 8767),
            "enable_legacy_port": data.get("daemon_enable_legacy_port", True),
            "restart_required": port_changed,
        }
    except Exception:
        logger.exception("put_daemon_config failed")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


# ---------------------------------------------------------------------------
# GET /api/v3/mesh/config
# ---------------------------------------------------------------------------


@router.get("/mesh/config")
def get_mesh_config():
    """Return current mesh configuration."""
    try:
        data = _read_config()
        return {"enabled": data.get("mesh_enabled", True)}
    except Exception:
        logger.exception("get_mesh_config failed")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


# ---------------------------------------------------------------------------
# PUT /api/v3/mesh/config
# ---------------------------------------------------------------------------


@router.put("/mesh/config")
def put_mesh_config(request: Request, body: MeshConfigUpdate):
    """Persist mesh state; restart is required to rebuild the mesh worker."""
    _require_admin(request)
    try:
        _update_config(
            lambda data: data.update({"mesh_enabled": body.enabled}),
        )
        return {"enabled": body.enabled, "restart_required": True}
    except Exception:
        logger.exception("put_mesh_config failed")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


# ---------------------------------------------------------------------------
# GET /api/v3/trust/config
# ---------------------------------------------------------------------------


@router.get("/trust/config")
def get_trust_config():
    """Return current trust configuration.

    Fields are spread across three config sections:
      - retrieval.use_trust_weighting   (Bayesian trust in retrieval ranking)
      - injection.trust_first_party     (framing of injected context)
      - consolidation.promotion_min_trust (min trust required for promotion)
    """
    try:
        data = _read_config()
        retrieval = data.get("retrieval", {})
        injection = data.get("injection", {})
        consolidation = data.get("consolidation", {})
        return {
            "use_trust_weighting": retrieval.get("use_trust_weighting", True),
            "trust_first_party": injection.get("trust_first_party", False),
            "promotion_min_trust": consolidation.get("promotion_min_trust", 0.5),
        }
    except Exception:
        logger.exception("get_trust_config failed")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


# ---------------------------------------------------------------------------
# PUT /api/v3/trust/config
# ---------------------------------------------------------------------------


@router.put("/trust/config")
def put_trust_config(request: Request, body: TrustConfigUpdate):
    """Update trust configuration.

    Each field is stored in its canonical config.json sub-section.
    Unrelated keys in those sub-sections are preserved.
    """
    _require_admin(request)
    try:
        def mutate(data: dict) -> None:
            if body.use_trust_weighting is not None:
                retrieval = data.setdefault("retrieval", {})
                retrieval["use_trust_weighting"] = body.use_trust_weighting
            if body.trust_first_party is not None:
                injection = data.setdefault("injection", {})
                injection["trust_first_party"] = body.trust_first_party
            if body.promotion_min_trust is not None:
                consolidation = data.setdefault("consolidation", {})
                consolidation["promotion_min_trust"] = body.promotion_min_trust

        data = _update_config(mutate)
        retrieval = data.get("retrieval", {})
        injection = data.get("injection", {})
        consolidation = data.get("consolidation", {})
        return {
            "use_trust_weighting": retrieval.get("use_trust_weighting", True),
            "trust_first_party": injection.get("trust_first_party", False),
            "promotion_min_trust": consolidation.get("promotion_min_trust", 0.5),
            "restart_required": True,
        }
    except Exception:
        logger.exception("put_trust_config failed")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


# ---------------------------------------------------------------------------
# GET /api/v3/forgetting/config
# ---------------------------------------------------------------------------


@router.get("/forgetting/config")
def get_forgetting_config():
    """Return all Ebbinghaus forgetting configuration fields."""
    try:
        data = _read_config()
        stored = data.get("forgetting", {})
        # Return defaults for any field not yet in config.json
        result = {**_FORGETTING_DEFAULTS, **stored}
        # Keep only known fields
        return {k: result[k] for k in _FORGETTING_DEFAULTS}
    except Exception:
        logger.exception("get_forgetting_config failed")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


# ---------------------------------------------------------------------------
# PUT /api/v3/forgetting/config
# ---------------------------------------------------------------------------


@router.put("/forgetting/config")
def put_forgetting_config(request: Request, body: ForgettingConfigUpdate):
    """Update Ebbinghaus forgetting configuration.

    Only provided fields are changed; all other forgetting fields are
    preserved. Changes take effect after the daemon restarts.
    """
    _require_admin(request)
    try:
        updates = body.model_dump(exclude_none=True)

        def mutate(data: dict) -> None:
            stored = data.get("forgetting", {})
            merged = {**_FORGETTING_DEFAULTS, **stored, **updates}
            data["forgetting"] = merged

        data = _update_config(mutate)
        merged = data["forgetting"]
        return {
            **{k: merged[k] for k in _FORGETTING_DEFAULTS},
            "restart_required": True,
        }
    except Exception:
        logger.exception("put_forgetting_config failed")
        return JSONResponse({"error": "Internal server error"}, status_code=500)
