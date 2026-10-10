# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V4 — Memory-kind MCP tools (4 tools).

set_memory_kind, memory_kinds_status, review_memory_kinds, confirm_memory_kinds.

Every tool is a thin client of ``/api/memory-kinds`` — the exact routes the
dashboard and ``slm kinds`` (cli/kinds_cmd.py) use
(server/routes/memory_kinds.py) — through the resident daemon. None of them
touches a database directly: the canonical mutation writer and its RBAC
gates (READ/WRITE/MANAGE) are the single place a kind is ever written, so
MCP, CLI and HTTP cannot disagree about what a kind is or who may set one.

Every tool takes an optional ``profile_id``: that profile is served for this
one call (the daemon authorizes it on THAT profile) and the active profile is
not moved. Empty = the active profile, the request unchanged.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
import urllib.parse
from typing import Any, Callable

from mcp.types import ToolAnnotations

from superlocalmemory.core.admission import admits
from superlocalmemory.core.operation_request import OperationKind

logger = logging.getLogger(__name__)

_BASE = "/api/memory-kinds"
_MAX_CONFIRM_ITEMS = 200


def _invalid_kind_error() -> dict[str, Any]:
    from superlocalmemory.storage.memory_kinds import MemoryKind

    return {
        "success": False,
        "code": "INVALID_KIND",
        "retryable": False,
        "error": "Unknown memory kind. Use one of: " + ", ".join(k.value for k in MemoryKind),
    }


async def _kinds_request(method: str, path: str, body: dict | None = None) -> dict:
    """The daemon's ``/api/memory-kinds`` answer, or a structured failure.

    Mirrors ``cli/kinds_cmd.py``'s ``_request``: a deterministic refusal
    (409/404/422) is DATA, not an outage — it is returned as a structured
    envelope rather than collapsed into a generic "unavailable" the caller
    might retry forever.
    """
    import asyncio

    from superlocalmemory.cli.daemon import (
        DaemonConflict,
        DaemonNotFound,
        DaemonRefused,
        DaemonUnprocessable,
        daemon_request,
        is_daemon_running,
    )
    from superlocalmemory.mcp._daemon_proxy import daemon_unavailable_error

    try:
        if not await asyncio.to_thread(is_daemon_running):
            return {"success": False, "code": "DAEMON_UNAVAILABLE", "retryable": True,
                    "error": daemon_unavailable_error()}
        result = await asyncio.to_thread(
            daemon_request, method, _BASE + path, body,
            preserve_conflict=True, preserve_not_found=True, preserve_unprocessable=True,
        )
    except DaemonConflict as exc:
        return {"success": False, "code": "CONFLICT", "retryable": False, "error": exc.detail}
    except DaemonNotFound as exc:
        return {"success": False, "code": exc.code or "NOT_FOUND", "retryable": False,
                "error": exc.message}
    except DaemonUnprocessable as exc:
        return {"success": False, "code": exc.code or "INVALID_REQUEST", "retryable": False,
                "error": exc.message}
    except DaemonRefused as exc:
        # 401/403 is an answer, not an outage (L3-04): the daemon refused this
        # caller, every retry will refuse it the same way, and reporting it as
        # DAEMON_UNAVAILABLE/retryable=True — as the bare except below used to,
        # since DaemonRefused is a RuntimeError — invited an endless retry loop
        # instead of surfacing the refusal, the same fix `remember`'s daemon
        # path and `_daemon_proxy.store()` already apply.
        return {"success": False, "code": "NOT_AUTHORIZED", "retryable": False, "error": str(exc)}
    except Exception:
        logger.exception("memory-kinds request failed: %s %s", method, path)
        return {"success": False, "code": "DAEMON_UNAVAILABLE", "retryable": True,
                "error": daemon_unavailable_error()}
    if result is None:
        from superlocalmemory.mcp._daemon_proxy import daemon_unavailable_error as _dmsg

        return {"success": False, "code": "DAEMON_UNAVAILABLE", "retryable": True,
                "error": _dmsg()}
    return {"success": True, **result}


def _with_profile(path: str, profile_id: str) -> str:
    """``path`` with ``profile_id`` in its query string, only when one is named."""
    named = (profile_id or "").strip()
    if not named:
        return path
    joiner = "&" if "?" in path else "?"
    return f"{path}{joiner}profile_id={urllib.parse.quote(named, safe='')}"


def _body_with_profile(body: dict, profile_id: str) -> dict:
    named = (profile_id or "").strip()
    return {**body, "profile_id": named} if named else body


def register_kind_tools(server, get_engine: Callable) -> None:
    """Register the 4 memory-kind MCP tools on *server*.

    ``get_engine`` is accepted for signature parity with the other
    ``register_*_tools`` functions (and so a future tool here can read local
    engine state); every tool today reaches the kind system only through the
    daemon's HTTP routes, never through ``get_engine()`` directly.
    """

    @server.tool()
    @admits(OperationKind.CORRECT)
    async def set_memory_kind(fact_id: str, kind: str, profile_id: str = "") -> dict:
        """Set (confirm) the kind of one memory you already know the type of.

        ``kind`` is one of the nine memory kinds — the same values
        ``remember``'s ``kind`` parameter takes — or a known alias. Refused
        (``INVALID_KIND``) before any request reaches the daemon if it does
        not parse. The fact must belong to the profile changed: ``profile_id``,
        or the active profile when it is empty.

        Returns the fact's kind_fields — the same five fields every surface
        shows (``memory_kind``, ``memory_kind_label``, ``memory_kind_state``,
        ``memory_kind_source``, ``memory_kind_confidence``) — once applied.
        """
        if not fact_id or not isinstance(fact_id, str):
            return {"success": False, "code": "INVALID_FACT_ID", "retryable": False,
                    "error": "fact_id is required"}
        from superlocalmemory.storage.memory_kinds import parse_kind

        if parse_kind(kind) is None:
            return _invalid_kind_error()
        from superlocalmemory.mcp.remote_visibility import with_view

        # A remote app is marked, so the daemon changes only what that app may see.
        path = with_view("/fact/" + urllib.parse.quote(fact_id, safe=""))
        result = await _kinds_request("PATCH", path,
                                      _body_with_profile({"kind": kind}, profile_id))
        if result.get("success") and not result.get("ok", True):
            return {"success": False, "code": "NOT_FOUND", "retryable": False,
                    "error": result.get("error", "Memory not found.")}
        return result

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    async def memory_kinds_status(profile_id: str = "") -> dict:
        """Memory-kind status for the active profile (or ``profile_id``): counts
        per kind (confirmed/suggested), which backend classifies new memories,
        and any classification run in progress."""
        return await _kinds_request("GET", _with_profile("/status", profile_id))

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    async def review_memory_kinds(kind: str = "", limit: int = 20,
                                  profile_id: str = "") -> dict:
        """List memory-kind suggestions awaiting confirmation.

        ``kind`` (optional) narrows to suggestions of one kind; empty lists
        every pending suggestion. Refused (``INVALID_KIND``) before any
        request reaches the daemon if ``kind`` is set but does not parse.
        """
        from superlocalmemory.storage.memory_kinds import parse_kind

        if kind and parse_kind(kind) is None:
            return _invalid_kind_error()
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            return {"success": False, "code": "INVALID_LIMIT", "retryable": False,
                    "error": "limit must be an integer from 1 to 100"}
        qs = f"?limit={int(limit)}"
        if kind:
            qs += f"&kind={urllib.parse.quote(kind)}"
        from superlocalmemory.mcp.remote_visibility import with_view

        return await _kinds_request(
            "GET", with_view(_with_profile("/suggestions" + qs, profile_id)))

    @server.tool()
    @admits(OperationKind.CORRECT)
    async def confirm_memory_kinds(items: list[dict], profile_id: str = "") -> dict:
        """Confirm kinds for 1-200 facts at once.

        Each item is ``{"fact_id": "...", "kind": "..."}``; omit ``kind`` (or
        set it to ``null``) to accept the kind SLM already suggested for that
        fact rather than naming one yourself. One bad item does not abort the
        rest — each is reported with its own ``ok``/``error`` in the
        returned ``items`` list, in the same order as the request.
        """
        if not isinstance(items, list) or not items:
            return {"success": False, "code": "INVALID_ITEMS", "retryable": False,
                    "error": "items must be a non-empty list of {fact_id, kind?} objects"}
        if len(items) > _MAX_CONFIRM_ITEMS:
            return {"success": False, "code": "INVALID_ITEMS", "retryable": False,
                    "error": f"confirm at most {_MAX_CONFIRM_ITEMS} items at a time"}
        cleaned: list[dict] = []
        for raw in items:
            fact_id = raw.get("fact_id") if isinstance(raw, dict) else None
            if not isinstance(fact_id, str) or not fact_id:
                return {"success": False, "code": "INVALID_ITEMS", "retryable": False,
                        "error": f"not a {{fact_id, kind?}} item: {raw!r}"}
            entry: dict[str, Any] = {"fact_id": fact_id}
            if raw.get("kind") is not None:
                entry["kind"] = raw["kind"]
            cleaned.append(entry)
        from superlocalmemory.mcp.remote_visibility import with_view

        return await _kinds_request("POST", with_view("/confirm"),
                                    _body_with_profile({"items": cleaned}, profile_id))


__all__ = ["register_kind_tools"]
