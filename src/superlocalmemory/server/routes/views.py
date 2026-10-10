# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory | https://qualixar.com

"""Saved views over HTTP — the routes the dashboard and ``slm view`` both use.

    GET  /api/v3/views                 list the active profile's views
    GET  /api/v3/views/show?name=      one view's definition
    GET  /api/v3/views/run?name=       run it: recall's answer, with memory ids
    POST /api/v3/views                 create   {name, query, filters?, limit?}
    POST /api/v3/views/rename          rename   {name, new_name}
    POST /api/v3/views/delete          delete   {name}

Names travel in the query string or body, never in the path: a name may hold
a "/" and must still reach the right view.

WHO MAY DO WHAT
---------------
Reads (list, show, run) need READ on the workspace — the same permission as
recall — and are listed as sensitive reads in ``server/read_gates.py``.
Writes need a credential this product issued (``require_write_actor``) and
WRITE on the workspace, so a viewer can run views but not change them. Every
call is scoped to the active profile, except ``run``, which takes an optional
``profile_id``: that profile's view is run against that profile's memories,
READ is checked on THAT profile before its existence is revealed, and the
active profile is not moved. (The MCP ``run_view`` tool uses it for a remote
key bound to another profile; listing and editing views there read the store
directly.)

RUNNING IS RECALL
-----------------
``run`` is the one run path for every surface: the dashboard calls it, ``slm
view run`` calls it, and the MCP ``run_view`` tool calls it through the daemon.
It hands the view's arguments to ``server.recall_core.run_recall`` — the
function ``GET /recall`` itself calls — so ranking, the answer check, the
budget and the keyword fallback are those of recall, and the same view on an
unchanged store returns the same memories in the same order everywhere. The
recall runs under a synthetic ``view:`` session, which continuity ignores, and
is labelled in the Answer Check history by where it was run from (``via``).
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from superlocalmemory.access.rbac import Permission
from superlocalmemory.retrieval import remote_view
from superlocalmemory.views import (
    ViewError,
    ViewStore,
    default_store,
    recall_arguments,
    shape_run,
    view_session_id,
)
from superlocalmemory.views import model

logger = logging.getLogger("superlocalmemory.routes.views")
router = APIRouter(prefix="/api/v3/views", tags=["views"])

_Name = Annotated[str, Query(min_length=1, max_length=model.MAX_NAME_CHARS * 4)]

#: Refusal code -> HTTP status. "Not ready" is a 409 (state), not a 503, so a
#: client never mistakes it for a daemon that is down and retries.
_STATUS = {model.VIEW_NOT_FOUND: 404, model.VIEW_EXISTS: 409,
           model.TOO_MANY_VIEWS: 409, model.VIEWS_UNAVAILABLE: 409}


class CreateBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Bounds here are generous outer limits for the request size; the exact
    # rules (and their plain-English refusals) live in views.model.
    name: Any = Field(...)
    query: Any = Field(...)
    filters: dict[str, Any] | None = Field(None, max_length=16)
    limit: Any = None


class RenameBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Any = Field(...)
    new_name: Any = Field(...)


class DeleteBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Any = Field(...)


# -- gates and helpers --------------------------------------------------------


def _profile() -> str:
    from superlocalmemory.server.routes.helpers import get_active_profile

    return get_active_profile()


def _read_gate(request: Request) -> str:
    from superlocalmemory.server.rbac_enforce import require_permission

    profile = _profile()
    require_permission(request, Permission.READ, profile=profile)
    return profile


def _write_gate(request: Request) -> str:
    from superlocalmemory.server import write_identity
    from superlocalmemory.server.rbac_enforce import require_permission

    write_identity.require_write_actor(
        request, getattr(request.app.state, "daemon_descriptor", None),
        actor_kind="saved-views")
    profile = _profile()
    require_permission(request, Permission.WRITE, profile=profile)
    return profile


def _store() -> ViewStore:
    return default_store()


def _refusal(exc: ViewError) -> JSONResponse:
    status = 422 if exc.code in model.INPUT_CODES else _STATUS.get(exc.code, 409)
    if status == 409:
        # The CLI's daemon client reads a 409's ``detail`` as text.
        return JSONResponse({"detail": exc.message, "code": exc.code}, status_code=status)
    return JSONResponse({"detail": exc.as_dict()}, status_code=status)


def _internal_error() -> JSONResponse:
    logger.exception("saved views: request failed")
    return JSONResponse({"detail": "Internal server error"}, status_code=500)


# -- reads --------------------------------------------------------------------


@router.get("")
def list_views(request: Request):
    profile = _read_gate(request)
    try:
        views = _store().list(profile)
    except ViewError as exc:
        return _refusal(exc)
    except Exception:  # noqa: BLE001 — typed refusals above; never a traceback
        return _internal_error()
    return {"profile": profile, "views": [v.to_dict() for v in views],
            "count": len(views), "limits": _limits()}


@router.get("/show")
def show_view(request: Request, name: _Name):
    profile = _read_gate(request)
    try:
        return {"profile": profile, "view": _store().get(profile, name).to_dict()}
    except ViewError as exc:
        return _refusal(exc)
    except Exception:  # noqa: BLE001
        return _internal_error()


def _run_profile(request: Request, profile_id: str) -> str:
    """The profile a run serves: the named one (READ on it, then it must
    exist), else the active one."""
    from superlocalmemory.server.rbac_enforce import require_permission
    from superlocalmemory.server.routed_profile import RoutedProfileError, routed_profile_id
    from superlocalmemory.server.routes.helpers import get_engine_lazy
    from superlocalmemory.server.routes.memories import _UnknownRoutedProfile

    try:
        named = routed_profile_id(profile_id)
    except RoutedProfileError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    if named is None:
        return _read_gate(request)
    require_permission(request, Permission.READ, profile=named)
    engine = get_engine_lazy(request.app.state)
    if engine is None:
        raise HTTPException(503, detail="The memory engine is starting; try again shortly.")
    if not engine._db.execute("SELECT 1 AS one FROM profiles WHERE profile_id = ?", (named,)):
        raise _UnknownRoutedProfile(named)
    return named


#: Where a view was run from, for the Answer Check history. A label only: it
#: changes nothing about the recall, so a closed set is all it needs.
_VIA = Annotated[str, Query(pattern="^(dashboard|cli|mcp)$")]


@router.get("/run")
async def run_view(request: Request, name: _Name, via: _VIA = "dashboard",
                   profile_id: Annotated[str, Query(max_length=200)] = "",
                   # How a remote caller came in (set by the MCP side); it can only hide more.
                   caller_view: Annotated[str, Query(max_length=32)] = ""):
    """Run a view: the recall ``GET /recall`` runs, with the view's arguments.

    The dashboard, ``slm view run`` and the MCP ``run_view`` tool all land here,
    and from here on the path is ``server.recall_core.run_recall`` — the same
    function ``/recall`` calls — so a view gives the same answer on every
    surface, keyword fallback included.
    """
    from superlocalmemory.server.routes.memories import (
        _UnknownRoutedProfile,
        _unknown_profile_response,
    )

    try:
        profile = _run_profile(request, profile_id)
    except _UnknownRoutedProfile as exc:
        return _unknown_profile_response(exc.profile_id)
    try:
        view = _store().get(profile, name)
    except ViewError as exc:
        return _refusal(exc)
    except Exception:  # noqa: BLE001
        return _internal_error()

    from superlocalmemory.server.recall_core import run_recall
    from superlocalmemory.server.routes.helpers import get_engine_lazy
    from superlocalmemory.server.write_identity import require_http_mutation_actor

    engine = get_engine_lazy(request.app.state)
    if engine is None:
        raise HTTPException(503, detail="The memory engine is starting; try again shortly.")
    # The same principal rule /recall applies: a recall records outcomes.
    actor = require_http_mutation_actor(
        request, getattr(request.app.state, "daemon_descriptor", None),
        actor_kind="saved-view")
    try:
        call = view_recall_call(engine, view, profile, actor=actor, via=via,
                                caller_view=remote_view.parse_view(caller_view))
        return shape_run(view, await run_recall(engine, call, app_state=request.app.state))
    except Exception:  # noqa: BLE001
        return _internal_error()


def view_recall_call(engine: Any, view: model.SavedView, profile: str, *,
                     actor: str, via: str, caller_view: str = "") -> Any:
    """The ``RecallCall`` for a view: its arguments, resolved as ``/recall`` would.

    ``profile_id`` is the profile the view was read from, named explicitly so a
    profile switch between reading the view and running it cannot run one
    profile's view against another profile's memories. The session id is
    synthetic (``view:``), so continuity ignores it.
    """
    from superlocalmemory.core.admission import enforce_read_scope
    from superlocalmemory.core.answer_check_history import ORIGIN_VIEW_DASHBOARD
    from superlocalmemory.core.recall_pipeline import resolve_hot_path_fast
    from superlocalmemory.retrieval.facets import Facets
    from superlocalmemory.server.recall_core import RecallCall

    args = recall_arguments(view)
    facets = Facets.of(kind=args.get("kind"))
    include_global, include_shared = enforce_read_scope(None, None)
    return RecallCall(
        query=args["query"], limit=args["limit"], session_id=view_session_id(view),
        agent_id=actor, fast=resolve_hot_path_fast(None, getattr(engine, "_config", None)),
        profile_id=profile, include_global=include_global, include_shared=include_shared,
        window=args.get("window", ""), as_of=args.get("as_of", ""),
        facets=None if facets.empty else facets,
        origin=f"view-{via}" if via in ("cli", "mcp") else ORIGIN_VIEW_DASHBOARD,
        caller_view=caller_view,
    )


def _limits() -> dict[str, Any]:
    """The rules a client builds its form from, so the form cannot drift from them."""
    from superlocalmemory.storage.memory_kinds import LABELS

    return {"max_views": model.MAX_VIEWS_PER_PROFILE, "max_name_chars": model.MAX_NAME_CHARS,
            "max_query_chars": model.MAX_QUERY_CHARS, "max_results": model.MAX_LIMIT,
            "default_results": model.DEFAULT_LIMIT, "filters": sorted(model.FILTERS),
            "kinds": [{"value": kind.value, "label": label} for kind, label in LABELS.items()]}


# -- writes -------------------------------------------------------------------


@router.post("")
def create_view(request: Request, body: CreateBody):
    profile = _write_gate(request)
    try:
        view = _store().create(profile, name=body.name, query=body.query,
                               filters=body.filters, limit=body.limit)
    except ViewError as exc:
        return _refusal(exc)
    except Exception:  # noqa: BLE001
        return _internal_error()
    return {"profile": profile, "view": view.to_dict(),
            "message": f"Saved the view {view.name!r}."}


@router.post("/rename")
def rename_view(request: Request, body: RenameBody):
    profile = _write_gate(request)
    try:
        view = _store().rename(profile, body.name, body.new_name)
    except ViewError as exc:
        return _refusal(exc)
    except Exception:  # noqa: BLE001
        return _internal_error()
    return {"profile": profile, "view": view.to_dict(),
            "message": f"Renamed the view to {view.name!r}."}


@router.post("/delete")
def delete_view(request: Request, body: DeleteBody):
    profile = _write_gate(request)
    try:
        view = _store().delete(profile, body.name)
    except ViewError as exc:
        return _refusal(exc)
    except Exception:  # noqa: BLE001
        return _internal_error()
    return {"profile": profile, "deleted": view.to_dict(),
            "message": f"Deleted the view {view.name!r}. No memory was changed."}


__all__ = ["router"]
