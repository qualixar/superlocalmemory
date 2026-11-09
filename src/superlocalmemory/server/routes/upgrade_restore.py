# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Updates & restore -- the dashboard's "Restore memories to before the update".

Over 75% of SLM's users are non-technical, so everything a restore or a
downgrade needs is reachable from here; nobody is asked to run a command.

Reads (GET) are open to whoever can read the dashboard (READ) and change
nothing. Every change needs the product's own credential (the install token the
dashboard sends, the daemon capability, or an API key), MANAGE permission, and
a typed confirmation word -- "RESTORE" to restore, "DOWNGRADE" to prepare for
an older version. A restore is recorded here and runs at the restart this
route starts, before anything opens the store.

Handlers are plain ``def`` (worker threads: checksums and copies are slow).
Failures are logged server-side and answered with a generic message.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from superlocalmemory.storage import upgrade_restore as ur
from superlocalmemory.storage._restore_public import public_view
from superlocalmemory.storage._restore_types import (
    OUTCOME_NAME,
    REPAIR_RECORD,
    DowngradeRefusedError,
    RestoreRefusedError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/upgrade", tags=["upgrade"])

RESTORE_WORD = "RESTORE"
DOWNGRADE_WORD = "DOWNGRADE"


class _PointBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    restore_point_id: str = Field(min_length=1, max_length=100)


class _RestoreBody(_PointBody):
    confirm: str = Field(default="", max_length=32)
    reimport: StrictBool = True


class _ConfirmBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confirm: str = Field(default="", max_length=32)


def _error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


def _internal_error() -> JSONResponse:
    return _error("Something went wrong. Your memories were not changed.", 500)


def _require_read(request: Request) -> None:
    from superlocalmemory.access.rbac import Permission
    from superlocalmemory.server.rbac_enforce import require_permission

    require_permission(request, Permission.READ)


def _require_credential(request: Request) -> None:
    from superlocalmemory.server.write_identity import require_write_actor

    require_write_actor(request, getattr(request.app.state, "daemon_descriptor", None),
                        actor_kind="upgrade-restore")


def _require_manage(request: Request) -> dict:
    from superlocalmemory.server.rbac_enforce import require_global_manage

    return require_global_manage(request)


def _gate(request: Request) -> str:
    _require_credential(request)
    principal = _require_manage(request) or {}
    return str(principal.get("username") or principal.get("user_id") or "owner")


def _paths(request: Request) -> tuple[Path, Path, Path]:
    root = getattr(request.app.state, "upgrade_data_root", None)
    if root is None:
        # The same resolution the daemon uses for the store it migrates and
        # serves (unified_daemon._memory_db_for_config), so a configured
        # base_dir is honoured here too.
        from superlocalmemory.infra.data_root import canonical_data_root

        try:
            from superlocalmemory.core.config import SLMConfig

            base = getattr(SLMConfig.load(), "base_dir", None)
        except Exception:  # noqa: BLE001 - fall back to the canonical root
            base = None
        root = canonical_data_root(configured_base_dir=base)
    root = Path(root)
    return root, root / "memory.db", root / "learning.db"


def _spawn_restart() -> bool:
    """The same detached ``slm restart`` as /api/daemon/restart."""
    import subprocess
    import sys

    from superlocalmemory.core.platform_utils import popen_platform_kwargs

    try:
        subprocess.Popen(
            [sys.executable, "-m", "superlocalmemory.cli.main", "restart", "--json"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            **popen_platform_kwargs())
    except Exception:  # noqa: BLE001
        logger.exception("Failed to spawn restart for a restore")
        return False
    return True


def _read(path: Path) -> dict[str, Any] | None:
    from superlocalmemory.storage._durable_json import read_json

    return read_json(path)


@router.get("/status")
def upgrade_status(request: Request):
    _require_read(request)
    try:
        from superlocalmemory.storage._snapshot_manifest import package_version

        root, _memory_db, _learning = _paths(request)
        intent = ur.pending_restore(root)
        last = root / ".last_version"
        return public_view({
            "version": package_version(),
            "last_version": last.read_text(encoding="utf-8").strip() if last.exists() else None,
            "restore_points": len(ur.list_restore_points(root)),
            "pending_restore": ({k: intent.get(k) for k in
                                 ("point_id", "requested_by", "requested_at", "stage")}
                                if intent else None),
            "last_restore": _read(root / OUTCOME_NAME),
            "downgrade": ur.downgrade_status(root),
            "repair": _read(root / REPAIR_RECORD),
        }, root)
    except Exception:  # noqa: BLE001
        logger.exception("upgrade status failed")
        return _internal_error()


@router.get("/restore-points")
def restore_points(request: Request):
    _require_read(request)
    try:
        root, _m, _l = _paths(request)
        return {"restore_points": public_view(
                    [p.as_dict() for p in ur.list_restore_points(root)], root),
                "kept": "Copies from the last two updates are kept."}
    except Exception:  # noqa: BLE001
        logger.exception("listing restore points failed")
        return _internal_error()


@router.post("/restore/preview")
def restore_preview(request: Request, body: _PointBody):
    _gate(request)
    root, memory_db, _l = _paths(request)
    try:
        return public_view(ur.preview_restore(body.restore_point_id, data_root=root,
                                              memory_db=memory_db).as_dict(), root)
    except RestoreRefusedError as exc:
        return _error(public_view(str(exc), root), 404)
    except Exception:  # noqa: BLE001
        logger.exception("restore preview failed")
        return _internal_error()


@router.post("/restore")
def restore(request: Request, body: _RestoreBody):
    actor = _gate(request)
    if body.confirm != RESTORE_WORD:
        return _error(f"Type {RESTORE_WORD} to confirm the restore.")
    root, memory_db, _l = _paths(request)
    try:
        intent = ur.request_restore(body.restore_point_id, requested_by=actor,
                                    reimport=body.reimport, data_root=root,
                                    memory_db=memory_db)
    except RestoreRefusedError as exc:
        return _error(public_view(str(exc), root), 409)
    except Exception:  # noqa: BLE001
        logger.exception("restore request failed")
        return _internal_error()
    started = _spawn_restart()
    return {
        "success": True,
        "status": "restarting" if started else "waiting_for_restart",
        "message": ("SuperLocalMemory is restarting to restore your memories."
                    if started else "Restart SuperLocalMemory to finish the restore."),
        "intent": public_view(intent.as_dict(), root),
    }


@router.post("/restore/cancel")
def restore_cancel(request: Request):
    _gate(request)
    root, _m, _l = _paths(request)
    try:
        return {"cancelled": ur.cancel_restore(root)}
    except RestoreRefusedError as exc:      # part-way through: it must finish (L1-03)
        return _error(public_view(str(exc), root), 409)
    except Exception:  # noqa: BLE001
        logger.exception("restore cancel failed")
        return _internal_error()


@router.post("/prepare-downgrade")
def prepare_downgrade(request: Request, body: _ConfirmBody):
    actor = _gate(request)
    if body.confirm != DOWNGRADE_WORD:
        return _error(f"Type {DOWNGRADE_WORD} to confirm.")
    root, memory_db, learning_db = _paths(request)
    try:
        return public_view(ur.prepare_downgrade(
            requested_by=actor, data_root=root, memory_db=memory_db,
            learning_db=learning_db).as_dict(), root)
    except DowngradeRefusedError as exc:
        return _error(public_view(str(exc), root), 409)
    except Exception:  # noqa: BLE001
        logger.exception("prepare downgrade failed")
        return _internal_error()


@router.post("/prepare-downgrade/cancel")
def prepare_downgrade_cancel(request: Request):
    _gate(request)
    root, _m, _l = _paths(request)
    try:
        return {"cancelled": ur.cancel_downgrade(root)}
    except Exception:  # noqa: BLE001
        logger.exception("cancel downgrade failed")
        return _internal_error()


def register(app: FastAPI, *, data_root: Path | None = None) -> None:
    """Mount the routes. ``data_root`` overrides the canonical data directory."""
    if data_root is not None:
        app.state.upgrade_data_root = Path(data_root)
    app.include_router(router)


__all__ = ["DOWNGRADE_WORD", "RESTORE_WORD", "register", "router"]
