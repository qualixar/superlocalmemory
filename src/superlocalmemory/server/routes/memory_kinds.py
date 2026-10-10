# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Memory kinds over HTTP: status, settings, "Classify my memories", review.

The dashboard and ``slm kinds`` both use these routes, so every behaviour has
one implementation. Who may do what (LLD §7.2):

* READ   — status, settings, suggestions.
* WRITE  — set or confirm a fact's kind, only on a fact the profile owns
  (``memories._authorize_memory_mutation``); the change goes through the
  canonical mutation writer like every other memory edit.

Status, suggestions and kind changes take an optional ``profile_id`` (query for
reads, body for writes): that profile is served for this one request,
authorized on THAT profile before its existence is revealed, and the active
profile is not moved. Without it, the active profile.
* MANAGE — start, pause, resume, cancel or undo a classification run, and
  change the settings. Also needs a credential the product issued, even from
  this machine: these decide whether memory text may be sent online.

A run that would send memory text off the device (Jev) is refused with 409
and the numbers unless the request carries ``confirm_data_leaves_device:
true`` — a real boolean; ``"true"`` is rejected. Errors are plain messages;
a traceback never leaves the server.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from superlocalmemory.access.rbac import Permission
from superlocalmemory.core import memory_kind_wiring as wiring
from superlocalmemory.core.memory_kind_backfill import BackfillRefused, BackfillRunner
from superlocalmemory.core.memory_kind_config import (
    save_memory_kind_settings,
    settings_dict,
)
from superlocalmemory.core.kind_query import InvalidKind
from superlocalmemory.retrieval import remote_view
from superlocalmemory.server import write_identity
from superlocalmemory.server.kind_error import invalid_kind_http
from superlocalmemory.storage.memory_kind_store import MemoryKindStore
from superlocalmemory.storage.memory_kinds import is_confirmed, parse_kind

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/memory-kinds", tags=["memory-kinds"])

_STATE_ATTR = "memory_kind_backfill"
_MAX_PROFILE_CHARS = 200
_REFUSAL_STATUS = {"needs_confirmation": 409, "schema_not_ready": 409, "disabled": 409,
                   "run_active": 409, "bad_state": 409, "not_found": 404,
                   "bad_request": 422, "not_ready": 503}


class SettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: StrictBool | None = None
    backend: str | None = Field(None, pattern="^(auto|rules|laya|jev|llm|off)$")
    jev_consent: StrictBool | None = None
    standing_rules_in_session: StrictBool | None = None


class BackfillRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["untyped", "refresh"] = "untyped"
    confirm_data_leaves_device: StrictBool = False


class ConfirmItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # L3-11: no upper length bound here. A fact_id this field itself rejects
    # (Pydantic validates the whole ``items`` list before the route ever
    # runs) 422s the ENTIRE batch -- contradicting this route's own contract
    # ("one bad item does not abort the rest"), since a real id this long
    # simply will not be found and already comes back a graceful per-item
    # {"ok": false, "error": "not found"} (storage.memory_kind_writes.set_kinds).
    # The generous cap still exists -- at the request-body size limit the
    # ingest gate enforces, not a per-field one.
    fact_id: str = Field(..., min_length=1)
    kind: str | None = Field(None, max_length=64)


class ConfirmRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ConfirmItem] = Field(..., min_length=1, max_length=200)
    profile_id: str | None = Field(None, max_length=_MAX_PROFILE_CHARS)


class KindEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = Field(..., min_length=1, max_length=64)
    profile_id: str | None = Field(None, max_length=_MAX_PROFILE_CHARS)


# ---------------------------------------------------------------------------
# Wiring for the daemon (WP-INT calls these)
# ---------------------------------------------------------------------------


def register(app: Any) -> None:
    app.include_router(router)


def _runner_for(app: Any) -> BackfillRunner:
    runner = getattr(app.state, _STATE_ATTR, None)
    if runner is None:
        def lease():
            runtime = getattr(app.state, "profile_runtime", None)
            return runtime.operation_nowait() if runtime is not None \
                else contextlib.nullcontext(True)

        runner = BackfillRunner(
            engine_supplier=lambda: getattr(app.state, "engine", None),
            lease=lease,
            preempt=lambda: bool(getattr(getattr(app.state, "profile_runtime", None),
                                         "transitioning", False)),
        )
        setattr(app.state, _STATE_ATTR, runner)
    return runner


def start_backfill(app: Any) -> BackfillRunner:
    """Start the runner thread (idempotent). Runs left 'running' resume."""
    runner = _runner_for(app)
    runner.start()
    return runner


def stop_backfill(app: Any, timeout_s: float = 5.0) -> bool:
    """Stop the runner thread; True when it has exited. Call before engine close."""
    runner = getattr(app.state, _STATE_ATTR, None)
    return True if runner is None else runner.stop(timeout_s=timeout_s)


# ---------------------------------------------------------------------------
# Gates and helpers
# ---------------------------------------------------------------------------


def _profile() -> str:
    from superlocalmemory.server.routes.helpers import get_active_profile

    return get_active_profile()


def _read_gate(request: Request) -> None:
    from superlocalmemory.server.rbac_enforce import require_permission

    require_permission(request, Permission.READ)


def _read_profile(request: Request, profile_id: str) -> str:
    """The profile a read serves: the named one, else the active one.

    A named profile is authorized (READ on it) before its existence is
    revealed; an unknown one raises ``_UnknownRoutedProfile``.
    """
    from superlocalmemory.server.rbac_enforce import require_permission
    from superlocalmemory.server.routed_profile import RoutedProfileError, routed_profile_id
    from superlocalmemory.server.routes.memories import _UnknownRoutedProfile

    try:
        profile = routed_profile_id(profile_id)
    except RoutedProfileError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    if profile is None:
        _read_gate(request)
        return _profile()
    require_permission(request, Permission.READ, profile=profile)
    db = getattr(_engine(request), "_db", None)
    if not db.execute("SELECT 1 AS one FROM profiles WHERE profile_id = ?", (profile,)):
        raise _UnknownRoutedProfile(profile)
    return profile


def _unknown_profile(profile_id: str) -> JSONResponse:
    from superlocalmemory.server.routes.memories import _unknown_profile_response

    return _unknown_profile_response(profile_id)


def _manage_gate(request: Request) -> str:
    from superlocalmemory.server.rbac_enforce import require_permission

    actor = write_identity.require_write_actor(
        request, getattr(request.app.state, "daemon_descriptor", None),
        actor_kind="memory-kinds")
    require_permission(request, Permission.MANAGE)
    return str(actor or "dashboard")


def _internal_error() -> JSONResponse:
    logger.exception("memory kinds: request failed")
    return JSONResponse({"error": "Internal server error"}, status_code=500)


def _refused(exc: BackfillRefused) -> JSONResponse:
    body = {"code": exc.code, "detail": exc.message, **exc.payload}
    return JSONResponse(body, status_code=_REFUSAL_STATUS.get(exc.code, 409))


def _engine(request: Request) -> Any:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(503, detail="The memory engine is starting; try again shortly.")
    return engine


def _settings_view(engine: Any) -> dict[str, Any]:
    """The settings plus what actually runs: for new memories (as they are
    saved) and for a classification run over existing ones."""
    from superlocalmemory.core.memory_kind_backfill_plan import resolve_backfill_backend
    from superlocalmemory.encoding.memory_kind_classifier import resolve_kind_backend

    cfg = wiring.current_config(engine)
    inputs = (wiring.engine_mode(engine), wiring.live_judge(engine),
              wiring.llm_available(engine))
    choice = resolve_backfill_backend(cfg, *inputs)
    return {**settings_dict(cfg), "active_backend": choice.active,
            "active_reason": choice.reason, "leaves_device": choice.leaves_device,
            "new_memories_backend": resolve_kind_backend(cfg, *inputs).value}


def _base_dir(request: Request, engine: Any) -> Any:
    for holder in (getattr(engine, "_config", None), getattr(request.app.state, "config", None)):
        base = getattr(holder, "base_dir", None)
        if base is not None:
            return base
    from superlocalmemory.infra.data_root import canonical_data_root

    return canonical_data_root()


def _set_kinds(request: Request, profile_id: str, pairs: list[tuple[str, str]]) -> list:
    from superlocalmemory.server.routes.memories import (
        _canonical_mutation_error,
        _canonical_mutation_runtime,
        _mutation_idempotency_key,
    )

    runtime = _canonical_mutation_runtime(request)
    try:
        receipt = runtime.set_fact_kinds(profile_id, pairs,
                                         idempotency_key=_mutation_idempotency_key(request))
    except ValueError as exc:
        raise invalid_kind_http(InvalidKind(str(exc))) from exc
    except Exception as exc:  # noqa: BLE001 — typed mapping, never a traceback
        raise _canonical_mutation_error(exc, "Could not change the memory kind") from exc
    if not receipt.get("ok", False) and not receipt.get("facts"):
        raise HTTPException(409, detail="Memory kinds are not available on this store yet.")
    return list(receipt.get("facts") or [])


def _authorize(request: Request, fact_id: str,
               profile: str | None = None) -> tuple[Any, str]:
    """Authorize a kind change: WRITE on a fact the profile owns.

    ``profile`` is a routed profile (None = the active one); an unknown one
    raises ``memories._UnknownRoutedProfile`` after the permission check.

    Passes ``admission_kind=OperationKind.REMEMBER`` so the admission layer
    evaluates this as the WRITE-level operation the route is documented as
    (same policy tier as ``remember(kind=...)``), not the owner/admin-only
    CORRECT contract ``operation="update"`` would otherwise default to for a
    content edit. The RBAC permission check and the hook name are unaffected
    — both still key off ``operation="update"`` (Permission.WRITE; the
    trust-gate pre-hook still fires under its existing "update" name).
    """
    from superlocalmemory.core.operation_request import OperationKind
    from superlocalmemory.server.routes.memories import _authorize_memory_mutation

    engine, profile_id, _context = _authorize_memory_mutation(
        request, "update", fact_id, admission_kind=OperationKind.REMEMBER,
        profile=profile)
    return engine, profile_id


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/status")
def get_status(request: Request,
               profile_id: str = Query("", max_length=_MAX_PROFILE_CHARS)):
    from superlocalmemory.server.routes.memories import _UnknownRoutedProfile

    try:
        profile = _read_profile(request, profile_id)
    except _UnknownRoutedProfile as exc:
        return _unknown_profile(exc.profile_id)
    try:
        return _runner_for(request.app).status(profile)
    except Exception:  # noqa: BLE001
        return _internal_error()


@router.get("/settings")
def get_settings(request: Request):
    _read_gate(request)
    try:
        return _settings_view(_engine(request))
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001
        return _internal_error()


@router.post("/settings")
def post_settings(request: Request, body: SettingsUpdate):
    _manage_gate(request)
    try:
        engine = _engine(request)
        changes = body.model_dump(exclude_none=True)
        current = wiring.current_config(engine)
        cfg = save_memory_kind_settings(_base_dir(request, engine), changes,
                                        seed=settings_dict(current))
        for holder in (getattr(engine, "_config", None),
                       getattr(request.app.state, "config", None)):
            if holder is not None:
                holder.memory_kinds = cfg
        return {**_settings_view(engine), "message": "Memory kind settings saved."}
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001
        return _internal_error()


@router.post("/backfill")
def post_backfill(request: Request, body: BackfillRequest):
    actor = _manage_gate(request)
    try:
        return _runner_for(request.app).create_run(
            _profile(), mode=body.mode, requested_by=actor,
            confirm_data_leaves_device=body.confirm_data_leaves_device)
    except BackfillRefused as exc:
        return _refused(exc)
    except Exception:  # noqa: BLE001
        return _internal_error()


@router.post("/backfill/{run_id}/{action}")
def post_backfill_action(request: Request, run_id: str,
                         action: Literal["pause", "resume", "cancel", "revert"]):
    actor = _manage_gate(request)
    try:
        method = getattr(_runner_for(request.app), action)
        return method(run_id, requested_by=actor, profile_id=_profile())
    except BackfillRefused as exc:
        return _refused(exc)
    except Exception:  # noqa: BLE001
        return _internal_error()


@router.get("/suggestions")
def get_suggestions(request: Request, kind: str | None = Query(None, max_length=64),
                    limit: int = Query(50, ge=1, le=100),
                    offset: int = Query(0, ge=0, le=1_000_000),
                    profile_id: str = Query("", max_length=_MAX_PROFILE_CHARS),
                    # How a remote caller came in; it can only hide more.
                    caller_view: str = Query("", max_length=32)):
    from superlocalmemory.server.routes.memories import _UnknownRoutedProfile

    try:
        profile = _read_profile(request, profile_id)
    except _UnknownRoutedProfile as exc:
        return _unknown_profile(exc.profile_id)
    parsed = parse_kind(kind) if kind else None
    if kind and parsed is None:
        raise invalid_kind_http(InvalidKind(kind))
    try:
        engine = _engine(request)
        db = getattr(engine, "db", None) or getattr(engine, "_db", None)
        cfg = wiring.current_config(engine)
        items = MemoryKindStore(db).suggestions(
            profile, kind=parsed, limit=limit, offset=offset,
            display_min_confidence=cfg.display_min_confidence)
        view = remote_view.parse_view(caller_view)
        if view:  # a remote caller: never a memory it may not see
            hidden = remote_view.hidden_among(view, db, profile, [i["fact_id"] for i in items])
            items = [i for i in items if i["fact_id"] not in hidden]
        return {"items": items}
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001
        return _internal_error()


def _stored_suggestion(engine: Any, profile_id: str, fact_id: str) -> str | None:
    db = getattr(engine, "db", None) or getattr(engine, "_db", None)
    rows = db.execute("SELECT memory_kind, memory_kind_source FROM atomic_facts "
                      "WHERE fact_id = ? AND profile_id = ?", (fact_id, profile_id))
    if not rows:
        return None
    row = dict(rows[0])
    kind = parse_kind(row.get("memory_kind"))
    if kind is None or is_confirmed(row.get("memory_kind_source")):
        return None
    return kind.value


@router.post("/confirm")
def post_confirm(request: Request, body: ConfirmRequest):
    """Confirm (or set) the kind of 1-200 facts at once.

    Permission: WRITE on the profile (``profile_id``, else the active one),
    and the caller must own each
    fact (``_authorize`` -> ``_authorize_memory_mutation``, admitted as
    OperationKind.REMEMBER) — the same tier ``remember(kind=...)`` runs
    under, not the owner/admin-only CORRECT tier a content edit requires.
    """
    from superlocalmemory.server.routed_profile import routed_profile_id
    from superlocalmemory.server.routes.memories import _UnknownRoutedProfile

    results: list[dict[str, Any] | None] = [None] * len(body.items)
    pairs: list[tuple[str, str]] = []
    slots: list[int] = []
    profile_id = ""
    routed = routed_profile_id(body.profile_id)
    for index, item in enumerate(body.items):
        try:
            engine, profile_id = _authorize(request, item.fact_id, routed)
        except _UnknownRoutedProfile as exc:
            return _unknown_profile(exc.profile_id)
        if item.kind is not None:
            parsed = parse_kind(item.kind)
            value = parsed.value if parsed is not None else None
            error = "Not a memory kind."
        else:
            value = _stored_suggestion(engine, profile_id, item.fact_id)
            error = "There is no suggestion to accept for this memory."
        if value is None:
            results[index] = {"fact_id": item.fact_id, "ok": False, "error": error}
            continue
        pairs.append((item.fact_id, value))
        slots.append(index)
    if pairs:
        applied = _set_kinds(request, profile_id, pairs)
        for index, outcome in zip(slots, applied):
            entry = dict(outcome)
            if not entry.get("ok"):
                entry["error"] = "Memory not found."
            results[index] = entry
    return {"items": results}


@router.patch("/fact/{fact_id}")
def patch_fact_kind(request: Request, fact_id: str, body: KindEdit):
    """Set one fact's kind.

    Permission: WRITE on the profile (``profile_id``, else the active one),
    and the caller must own the
    fact (``_authorize`` -> ``_authorize_memory_mutation``, admitted as
    OperationKind.REMEMBER) — the same tier ``remember(kind=...)`` runs
    under, not the owner/admin-only CORRECT tier a content edit requires.
    """
    from superlocalmemory.server.routed_profile import routed_profile_id
    from superlocalmemory.server.routes.memories import _UnknownRoutedProfile

    parsed = parse_kind(body.kind)
    if parsed is None:
        raise invalid_kind_http(InvalidKind(body.kind))
    try:
        _engine_obj, profile_id = _authorize(request, fact_id,
                                             routed_profile_id(body.profile_id))
    except _UnknownRoutedProfile as exc:
        return _unknown_profile(exc.profile_id)
    applied = _set_kinds(request, profile_id, [(fact_id, parsed.value)])
    if not applied or not applied[0].get("ok"):
        raise HTTPException(404, detail="Memory not found")
    return dict(applied[0])


__all__ = ["register", "router", "start_backfill", "stop_backfill"]
