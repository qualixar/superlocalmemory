# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Image and document tools for AI apps on this computer: ``remember_media``, ``get_media``,
``remember_document`` and ``media_status``.

All go through the local daemon's routes, so the profile, permission and
governance checks live in one place. All are host-only: a caller on another
computer is refused here, before any daemon call, as well as by the remote tool
policy. A thumbnail goes back only as an MCP ``image`` content block, never
inside ``structuredContent``. An error never repeats a link's query string.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import re
from typing import Any
from urllib.parse import quote

from mcp.types import CallToolResult, ImageContent, TextContent, ToolAnnotations

from superlocalmemory.core.admission import admits
from superlocalmemory.core.operation_request import OperationKind
from superlocalmemory.mcp.remote_caller import current_remote_key_id

logger = logging.getLogger("slm.mcp.tools_media")

MAX_THUMB_BYTES = 32 * 1024
NOT_FOR_REMOTE = "Images are not available to remote apps."
NOT_FOR_REMOTE_DOCS = "Documents are not available to remote apps."
_ID = re.compile(r"[0-9a-f]{32}")
_LINK = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://\S+")
_SOURCES = ("path", "download_url", "base64")
_DOC_SOURCES = ("path", "base64")


def _clean(text: object) -> str:
    """Short error text with any link removed (a link can carry a secret query)."""
    return _LINK.sub("[link]", str(text))[:300]


def _fail(code: str, message: str, *, retryable: bool = False) -> dict[str, Any]:
    return {"success": False, "status": "refused", "retryable": retryable,
            "code": code, "error": _clean(message)}


def _error_result(message: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=_clean(message))], is_error=True)


def _remember_via_daemon(body: dict[str, Any]) -> dict[str, Any]:
    from superlocalmemory.cli import daemon
    from superlocalmemory.mcp._daemon_proxy import daemon_unavailable_error

    try:
        data = daemon.daemon_request("POST", "/api/v3/media/remember", body,
                                     timeout_seconds=60.0, preserve_unprocessable=True)
    except daemon.DaemonUnprocessable as exc:
        return _fail("refused", exc.message)
    except daemon.DaemonRefused:
        return _fail("not_allowed", "The daemon did not allow this image to be saved.")
    if not isinstance(data, dict):
        return {"success": False, "status": "unavailable", "retryable": True,
                "code": "DAEMON_UNAVAILABLE", "error": daemon_unavailable_error()}
    if data.get("media_id"):
        data = {**data, "resource": f"slm://media/{data['media_id']}"}
    return {k: (_clean(v) if k in ("reason", "error") else v) for k, v in data.items()}


def _document_via_daemon(body: dict[str, Any]) -> dict[str, Any]:
    from superlocalmemory.cli import daemon
    from superlocalmemory.mcp._daemon_proxy import daemon_unavailable_error

    try:
        data = daemon.daemon_request("POST", "/api/v3/documents", body,
                                     timeout_seconds=60.0, preserve_unprocessable=True)
    except daemon.DaemonUnprocessable as exc:
        return _fail("refused", exc.message)
    except daemon.DaemonRefused:
        return _fail("not_allowed", "The daemon did not allow this document to be saved.")
    if not isinstance(data, dict):
        return {"success": False, "status": "unavailable", "retryable": True,
                "code": "DAEMON_UNAVAILABLE", "error": daemon_unavailable_error()}
    return {k: (_clean(v) if k in ("reason", "error", "detail") else v) for k, v in data.items()}


def _job_via_daemon(job_id: str, profile_id: str) -> dict[str, Any]:
    from superlocalmemory.cli import daemon
    from superlocalmemory.mcp._daemon_proxy import daemon_unavailable_error

    path = f"/api/v3/jobs/{job_id}"
    if profile_id:
        path += f"?profile_id={quote(profile_id, safe='')}"
    try:
        data = daemon.daemon_request("GET", path, timeout_seconds=30.0, preserve_not_found=True)
    except daemon.DaemonNotFound:
        return _fail("not_found", "Job not found.")
    except daemon.DaemonRefused:
        return _fail("not_allowed", "The daemon did not allow this job to be read.")
    if not isinstance(data, dict):
        return {"success": False, "status": "unavailable", "retryable": True,
                "code": "DAEMON_UNAVAILABLE", "error": daemon_unavailable_error()}
    return {k: (_clean(v) if k == "error" and v else v) for k, v in data.items()}


def thumb_via_daemon(media_id: str, profile_id: str = "") -> tuple[bytes | None, str]:
    """``(webp bytes, "")`` for a picture's thumbnail, else ``(None, short reason)``."""
    from superlocalmemory.cli import daemon

    if not _ID.fullmatch(media_id or ""):
        return None, "That is not a valid image id."
    path = f"/api/v3/media/{media_id}/thumb?format=json"
    if profile_id:
        path += f"&profile_id={quote(profile_id, safe='')}"
    try:
        data = daemon.daemon_request("GET", path, timeout_seconds=30.0, preserve_not_found=True)
    except daemon.DaemonNotFound:
        return None, "Image not found."
    except daemon.DaemonRefused:
        return None, "The daemon did not allow this image to be read."
    if not isinstance(data, dict):
        return None, "The memory service is not reachable right now; try again shortly."
    try:
        raw = base64.b64decode(str(data.get("base64") or ""), validate=True)
    except (binascii.Error, ValueError):
        return None, "The thumbnail could not be read."
    if not raw:
        return None, "Image not found."
    if len(raw) > MAX_THUMB_BYTES:
        return None, "The thumbnail is too large to send."
    return raw, ""


MAX_RECALL_IMAGES = 3


def _recall_media_ids(payload: dict[str, Any]) -> list[str]:
    """The first few distinct picture ids among a recall's results, in result order."""
    found: list[str] = []
    for row in payload.get("results") or []:
        block = row.get("media") if isinstance(row, dict) else None
        media_id = block.get("media_id") if isinstance(block, dict) else None
        if isinstance(media_id, str) and _ID.fullmatch(media_id) and media_id not in found:
            found.append(media_id)
    return found[:MAX_RECALL_IMAGES]


def _recall_thumbs(ids: list[str], profile_id: str) -> list[bytes]:
    thumbs: list[bytes] = []
    for media_id in ids:
        try:
            raw, _ = thumb_via_daemon(media_id, profile_id)
        except Exception as exc:  # noqa: BLE001 - a missing picture never fails a recall
            logger.debug("recall thumbnail skipped (%s)", type(exc).__name__)
            continue
        if raw:
            thumbs.append(raw)
    return thumbs


def _without_media(payload: dict[str, Any]) -> dict[str, Any]:
    """``payload`` with each result's ``media`` key removed. Nothing to remove: the same object."""
    rows = payload.get("results")
    if not isinstance(rows, list) or not any(isinstance(r, dict) and "media" in r for r in rows):
        return payload
    kept = [{k: v for k, v in r.items() if k != "media"} if isinstance(r, dict) else r for r in rows]
    return {**payload, "results": kept}


async def with_recall_images(payload: dict[str, Any], profile_id: str = "") -> Any:
    """What the recall tool returns: ``payload`` itself, untouched, unless a local caller's
    results include pictures and at least one thumbnail could be read. Then the same text the
    framework would have made from ``payload``, followed by up to three image blocks. The
    payload (and so ``structuredContent``) never carries a thumbnail. A remote caller gets
    the payload without any ``media`` block: picture ids and links are not for remote apps."""
    if current_remote_key_id() is not None:
        return _without_media(payload)
    try:
        ids = _recall_media_ids(payload)
        if not ids:
            return payload
        thumbs = await asyncio.to_thread(_recall_thumbs, ids, (profile_id or "").strip())
        if not thumbs:
            return payload
        # The framework's own conversion (compact JSON included), so the first block is
        # exactly what a plain dict return would have produced.
        from mcp.server.mcpserver.utilities import func_metadata

        return CallToolResult(content=[
            *func_metadata._convert_to_content(payload),
            *[ImageContent(type="image", data=base64.b64encode(raw).decode("ascii"),
                           mime_type="image/webp") for raw in thumbs],
        ])
    except Exception as exc:  # noqa: BLE001
        logger.debug("recall images skipped (%s)", type(exc).__name__)
        return payload


async def _remember(args: dict[str, Any]) -> dict[str, Any]:
    if current_remote_key_id() is not None:
        return _fail("not_for_remote", NOT_FOR_REMOTE)
    if sum(bool(args[k]) for k in _SOURCES) != 1:
        return _fail("invalid_request", "Give exactly one of path, download_url or base64.")
    body = {k: v for k, v in args.items() if v}
    try:
        return await asyncio.to_thread(_remember_via_daemon, body)
    except Exception as exc:  # noqa: BLE001 - a tool answer, never a traceback
        logger.warning("remember_media failed: %s", type(exc).__name__)
        return _fail("error", "The image could not be saved.", retryable=True)


async def _remember_document(args: dict[str, Any]) -> dict[str, Any]:
    if current_remote_key_id() is not None:
        return _fail("not_for_remote", NOT_FOR_REMOTE_DOCS)
    if sum(bool(args[k]) for k in _DOC_SOURCES) != 1:
        return _fail("invalid_request", "Give exactly one of path or base64.")
    body = {k: v for k, v in args.items() if v}
    try:
        return await asyncio.to_thread(_document_via_daemon, body)
    except Exception as exc:  # noqa: BLE001 - a tool answer, never a traceback
        logger.warning("remember_document failed: %s", type(exc).__name__)
        return _fail("error", "The document could not be saved.", retryable=True)


async def _status(job_id: str, profile_id: str) -> dict[str, Any]:
    if current_remote_key_id() is not None:
        return _fail("not_for_remote", NOT_FOR_REMOTE_DOCS)
    if not _ID.fullmatch(job_id or ""):
        return _fail("invalid_request", "That is not a valid job id.")
    try:
        return await asyncio.to_thread(_job_via_daemon, job_id, (profile_id or "").strip())
    except Exception as exc:  # noqa: BLE001
        logger.warning("media_status failed: %s", type(exc).__name__)
        return _fail("error", "The job could not be read.", retryable=True)


async def _get(media_id: str, variant: str, profile_id: str) -> CallToolResult:
    if current_remote_key_id() is not None:
        return _error_result(NOT_FOR_REMOTE)
    if variant != "thumb":
        return _error_result("Only the thumb variant is available.")
    raw, reason = await asyncio.to_thread(thumb_via_daemon, media_id, profile_id)
    if raw is None:
        return _error_result(reason)
    note = f"Thumbnail of image {media_id}. Full image: slm://media/{media_id}"
    return CallToolResult(content=[
        TextContent(type="text", text=note),
        ImageContent(type="image", data=base64.b64encode(raw).decode("ascii"), mime_type="image/webp"),
    ])


def register_media_tools(server: Any) -> None:
    """Register ``remember_media`` and ``get_media`` on *server*."""

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False))
    @admits(OperationKind.REMEMBER)
    async def remember_media(
        path: str = "", download_url: str = "", base64: str = "",
        content: str = "", tags: str = "", profile_id: str = "",
        idempotency_key: str = "",
    ) -> dict:
        """Save an image as a memory, from a file on this computer, an https link or base64.

        Give exactly one of path, download_url or base64. Only for apps on this computer.
        """
        return await _remember({
            "path": path, "download_url": download_url, "base64": base64, "content": content,
            "tags": tags, "profile_id": profile_id, "idempotency_key": idempotency_key})

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
    @admits(OperationKind.RECALL)
    async def get_media(media_id: str, variant: str = "thumb", profile_id: str = "") -> CallToolResult:
        """Show the thumbnail of a saved image as an image. Only for apps on this computer."""
        return await _get(media_id, variant, profile_id)


def register_document_tools(server: Any) -> None:
    """Register ``remember_document`` and ``media_status`` on *server*."""

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False))
    @admits(OperationKind.REMEMBER)
    async def remember_document(
        path: str = "", base64: str = "", file_name: str = "",
        content: str = "", tags: str = "", profile_id: str = "",
        idempotency_key: str = "",
    ) -> dict:
        """Save a PDF as a memory, from a file on this computer or base64. Work continues in the
        background: the answer has a job_id to follow with media_status.

        Give exactly one of path or base64. Only for apps on this computer.
        """
        return await _remember_document({
            "path": path, "base64": base64, "file_name": file_name, "content": content,
            "tags": tags, "profile_id": profile_id, "idempotency_key": idempotency_key})

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
    @admits(OperationKind.RECALL)
    async def media_status(job_id: str, profile_id: str = "") -> dict:
        """How far a saved document's background work has got. Only for apps on this computer."""
        return await _status(job_id, profile_id)


def register_media_resources(server: Any) -> None:
    """Register the ``slm://media/{media_id}`` and ``.../thumb`` resource templates."""

    @server.resource("slm://media/{media_id}/thumb", mime_type="image/webp")
    async def media_thumb(media_id: str) -> bytes:
        """Thumbnail (WebP) of a saved image in the active profile."""
        if current_remote_key_id() is not None:
            raise PermissionError(NOT_FOR_REMOTE)
        raw, reason = await asyncio.to_thread(thumb_via_daemon, media_id)
        if raw is None:
            raise ValueError(reason)
        return raw

    @server.resource("slm://media/{media_id}")
    async def media_card(media_id: str) -> str:
        """A saved image: its id and where to read its thumbnail."""
        if current_remote_key_id() is not None:
            raise PermissionError(NOT_FOR_REMOTE)
        raw, reason = await asyncio.to_thread(thumb_via_daemon, media_id)
        if raw is None:
            raise ValueError(reason)
        return f"Image {media_id}. Thumbnail: slm://media/{media_id}/thumb"


__all__ = ["register_media_tools", "register_document_tools", "register_media_resources", "thumb_via_daemon"]
