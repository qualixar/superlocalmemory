# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory | https://qualixar.com

"""MCP surface for saved views (issue #113).

Two tools, split by what they may change, because a remote read-only key may
run views but never edit them (``server/remote_tool_policy.py``):

* ``run_view`` (read) — no name: list the profile's views. A name: run that
  view through the daemon's run route, the one run path the dashboard and the
  CLI use too, and return recall's answer in recall's order, with every
  memory's id.
* ``manage_view`` (write) — create, rename or delete a view.

Both work on the active profile, or on ``profile_id`` when one is named (it
must exist; the active profile is not moved). Remote access always names the
key's own profile (``server/remote_profile_binding``), so a key bound to one
profile sees and edits only that profile's views, whatever profile this
computer is using. Inputs are validated by ``views.model`` with the same codes
the CLI and HTTP routes return.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from mcp.types import ToolAnnotations

from superlocalmemory.core.admission import admits
from superlocalmemory.core.operation_request import OperationKind
from superlocalmemory.views import ViewError, default_store

logger = logging.getLogger("superlocalmemory.mcp.views")

_ACTIONS = ("create", "rename", "delete")


def _refused(exc: ViewError) -> dict[str, Any]:
    return {"success": False, "retryable": False, **exc.as_dict(), "error": exc.message}


async def _profile(get_engine: Callable[[], Any],
                   profile_id: str) -> tuple[str, dict[str, Any] | None]:
    from superlocalmemory.mcp.tools_core import _call_profile

    return await _call_profile(get_engine, profile_id)


def _run_through_daemon(name: str, profile_id: str = "") -> dict[str, Any]:
    """Run a view through the daemon's own run route — the one run path.

    The dashboard and ``slm view run`` call ``GET /api/v3/views/run`` too, and
    that route hands the view to the recall core ``GET /recall`` uses, so a
    view gives the same answer here as on every other surface.
    """
    from urllib.parse import quote

    from superlocalmemory.cli.daemon import (
        DaemonConflict,
        DaemonNotFound,
        DaemonRefused,
        DaemonUnprocessable,
        daemon_request,
    )
    from superlocalmemory.mcp._daemon_proxy import daemon_unavailable_error

    path = f"/api/v3/views/run?name={quote(name, safe='')}&via=mcp"
    if profile_id:
        path += f"&profile_id={quote(profile_id, safe='')}"
    from superlocalmemory.mcp.remote_visibility import with_view

    try:
        data = daemon_request(
            "GET", with_view(path),
            timeout_seconds=60.0, preserve_conflict=True, preserve_not_found=True,
            preserve_unprocessable=True)
    except DaemonNotFound as exc:
        return {"success": False, "retryable": False, "code": exc.code, "error": exc.message}
    except DaemonUnprocessable as exc:
        return {"success": False, "retryable": False, "code": exc.code, "error": exc.message}
    except DaemonConflict as exc:
        return {"success": False, "retryable": False, "code": "view_refused",
                "error": exc.detail}
    except DaemonRefused as exc:
        return {"success": False, "retryable": False, "code": "not_allowed",
                "error": str(exc)}
    if not isinstance(data, dict):
        return {"success": False, "retryable": True, "code": "DAEMON_UNAVAILABLE",
                "error": daemon_unavailable_error()}
    return data


def register_view_tools(server: Any, get_engine: Callable[[], Any]) -> None:
    """Register ``run_view`` and ``manage_view``."""

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    @admits(OperationKind.RECALL)
    async def run_view(name: str = "", profile_id: str = "") -> dict[str, Any]:
        """Run one of your saved views, or list them.

        A saved view is a named recall query the person saved (e.g. "Work log":
        "what did I ship" over the last 7 days). Leave ``name`` empty to list
        the views in this profile. With a name, the view's query runs through
        the normal recall path and the answer comes back in recall's order;
        every result carries ``fact_id``, the id ``fetch`` takes. Honour
        ``no_confident_match`` exactly as for ``recall``. ``profile_id`` lists
        or runs another profile's views (empty = the active profile).
        """
        try:
            profile, refused = await _profile(get_engine, profile_id)
            if refused:
                return refused
            store = default_store()
            if not (name or "").strip():
                views = await asyncio.to_thread(store.list, profile)
                return {"success": True, "profile": profile, "count": len(views),
                        "views": [v.to_dict() for v in views]}
            named = (profile_id or "").strip()
            return await asyncio.to_thread(_run_through_daemon, name.strip(), named)
        except ViewError as exc:
            return _refused(exc)
        except Exception as exc:  # noqa: BLE001 — reported, never a traceback
            logger.exception("run_view failed")
            return {"success": False, "error": f"run_view failed: {exc}"}

    @server.tool(annotations=ToolAnnotations(destructiveHint=True))
    @admits(OperationKind.REMEMBER)
    async def manage_view(
        action: str,
        name: str,
        query: str = "",
        filters: dict[str, str] | None = None,
        limit: int | None = None,
        new_name: str = "",
        profile_id: str = "",
    ) -> dict[str, Any]:
        """Create, rename or delete a saved view. No memory is ever changed.

        Args:
            action: "create", "rename" or "delete".
            name: the view's name (at most 80 characters).
            query: for "create": what to look for, as you would ask ``recall``
                (at most 1000 characters).
            filters: for "create", optional: ``kind`` (a memory kind),
                ``window`` ("7d", "30d", "2026-07-01..2026-07-31") and/or
                ``as_of`` (ISO-8601). Anything else is refused.
            limit: for "create": results to show, 1-50 (default 10).
            new_name: for "rename": the new name.
            profile_id: the profile whose views change (empty = the active one).
        """
        verb = (action or "").strip().lower()
        if verb not in _ACTIONS:
            return {"success": False, "code": "invalid_view_action", "retryable": False,
                    "error": f"action must be one of: {', '.join(_ACTIONS)}"}
        try:
            profile, refused = await _profile(get_engine, profile_id)
            if refused:
                return refused
            store = default_store()
            if verb == "create":
                view = await asyncio.to_thread(
                    lambda: store.create(profile, name=name, query=query,
                                         filters=filters, limit=limit))
            elif verb == "rename":
                view = await asyncio.to_thread(store.rename, profile, name, new_name)
            else:
                view = await asyncio.to_thread(store.delete, profile, name)
        except ViewError as exc:
            return _refused(exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("manage_view failed")
            return {"success": False, "error": f"manage_view failed: {exc}"}
        return {"success": True, "action": verb, "profile": profile, "view": view.to_dict()}


__all__ = ["register_view_tools"]
