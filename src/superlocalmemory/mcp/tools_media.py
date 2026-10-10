# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Image and document tools for AI apps on this computer: ``remember_media``, ``get_media``,
``remember_document`` and ``media_status``.

All go through the local daemon's routes, so the profile, permission and
governance checks live in one place. Apps on this computer always work; a caller on another
computer is refused here, before any daemon call, as well as by the remote tool
policy, unless the remote app has the signed media grant and its key allows media; such an app
cannot name a file on this computer, sends at most 512 KB of pasted data, and its links go through
the remote fetch rules (the daemon is told, so it can only become stricter). A thumbnail goes back only as an MCP ``image`` content block, never
inside ``structuredContent``. An error never repeats a link's query string.

``remember_media`` and ``remember_document`` also take a ``file`` object, which ChatGPT fills in for
an attachment (``_meta["openai/fileParams"]``): its ``download_url`` is fetched like any other link,
marked as coming from a file so the chat app's own file hosts are trusted for it and nothing else.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import re
from typing import Annotated, Any
from urllib.parse import quote

from mcp.types import CallToolResult, ImageContent, TextContent, ToolAnnotations
from pydantic import WithJsonSchema

from superlocalmemory.core.admission import admits
from superlocalmemory.core.operation_request import OperationKind
from superlocalmemory.core.media_fetch import MAX_REMOTE_BASE64_BYTES, too_large_for_remote
from superlocalmemory.mcp.remote_caller import current_remote_key_id, current_remote_media_allowed
from superlocalmemory.mcp.remote_visibility import with_view

logger = logging.getLogger("slm.mcp.tools_media")

MAX_THUMB_BYTES = 32 * 1024
NOT_FOR_REMOTE = "Images are not available to remote apps."
NOT_FOR_REMOTE_DOCS = "Documents are not available to remote apps."
_ID = re.compile(r"[0-9a-f]{32}")
_LINK = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://\S+")
_SOURCES = ("path", "download_url", "base64")
_DOC_SOURCES = ("path", "download_url", "base64")
#: A link to a whole document may take a while to fetch on the computer; a short call does not.
LINK_TIMEOUT_S = 150.0
SHORT_TIMEOUT_S = 60.0
#: Tells ChatGPT which input fields are attachments (OpenAI Apps SDK, ``_meta["openai/fileParams"]``).
FILE_PARAMS_META: dict[str, Any] = {"openai/fileParams": ["file"]}
#: What ChatGPT sends for an attachment: all four properties declared, two always present.
FILE_OBJECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "A file the person attached in the chat. ChatGPT fills this in; do not write it by hand.",
    "properties": {
        "download_url": {"type": "string", "description": "Temporary link to the file."},
        "file_id": {"type": "string", "description": "The chat app's id for the file."},
        "mime_type": {"type": "string", "description": "The file's type, if known."},
        "file_name": {"type": "string", "description": "The file's name, if known."},
    },
    "required": ["download_url", "file_id"],
}
FileArg = Annotated[dict[str, Any] | None, WithJsonSchema(FILE_OBJECT_SCHEMA)]


def _remote_refused() -> bool:
    """A remote caller that is not allowed images and documents."""
    return current_remote_key_id() is not None and not current_remote_media_allowed()


def _is_remote() -> bool:
    return current_remote_key_id() is not None


def _remote_source_refusal(args: dict[str, Any]) -> dict[str, Any] | None:
    """The refusal for a source an allowed remote app may not use, else ``None``."""
    if args.get("path"):
        return _fail("path_not_for_remote",
                     "Remote apps cannot name a file on this computer. Send the data or a link.")
    if too_large_for_remote(args.get("base64")):
        return _fail("too_large_for_remote",
                     f"Pasted data from a remote app is limited to {MAX_REMOTE_BASE64_BYTES // 1024} KB.")
    return None


def _unpack_file(args: dict[str, Any], *, name_it: bool) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """``args`` with a ``file`` object turned into a link marked as coming from a file, and a
    refusal when the object lacks its ``download_url`` or ``file_id``. Without one: ``args`` as is."""
    file = args.get("file")
    rest = {k: v for k, v in args.items() if k != "file"}
    if file is None:
        return rest, None
    link, file_id = file.get("download_url"), file.get("file_id")
    if not (isinstance(link, str) and link.strip() and isinstance(file_id, str) and file_id.strip()):
        return rest, _fail("invalid_request", "The attached file needs a download_url and a file_id.")
    rest["download_url"], rest["from_file"] = link.strip(), True
    label = file.get("file_name")
    if name_it and not rest.get("file_name") and isinstance(label, str) and label.strip():
        rest["file_name"] = label.strip()[:255]
    return rest, None


def _source_count(args: dict[str, Any], names: tuple[str, ...]) -> int:
    return sum(bool(args.get(k)) for k in names) + (args.get("file") is not None)


def _profiles(text: str) -> list[str]:
    """The profile ids in a comma-separated ``shared_with``, as ``remember`` reads it."""
    return [p.strip() for p in (text or "").split(",") if p.strip()]


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
                                     timeout_seconds=LINK_TIMEOUT_S if body.get("download_url") else SHORT_TIMEOUT_S,
                                     preserve_unprocessable=True)
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
    path = with_view(path)  # a remote caller's view reaches the daemon
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
    if _remote_refused():
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


def _prepare_save(args: dict[str, Any], sources: tuple[str, ...], *, name_it: bool
                  ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """``(daemon body, None)`` for a save that may go ahead, else ``(None, refusal)``."""
    if _source_count(args, sources) != 1:
        return None, _fail("invalid_request", "Give exactly one of path, download_url, base64 or file.")
    args, refusal = _unpack_file(args, name_it=name_it)
    if refusal is None and _is_remote():
        refusal = _remote_source_refusal(args)
    if refusal is not None:
        return None, refusal
    body = {k: v for k, v in args.items() if v}
    if _is_remote():
        body["origin"] = "remote"
    return body, None


async def _remember(args: dict[str, Any]) -> dict[str, Any]:
    if _remote_refused():
        return _fail("not_for_remote", NOT_FOR_REMOTE)
    body, refusal = _prepare_save(args, _SOURCES, name_it=False)
    if body is None:
        return refusal or {}
    try:
        return await asyncio.to_thread(_remember_via_daemon, body)
    except Exception as exc:  # noqa: BLE001 - a tool answer, never a traceback
        logger.warning("remember_media failed: %s", type(exc).__name__)
        return _fail("error", "The image could not be saved.", retryable=True)


async def _remember_document(args: dict[str, Any]) -> dict[str, Any]:
    if _remote_refused():
        return _fail("not_for_remote", NOT_FOR_REMOTE_DOCS)
    body, refusal = _prepare_save(args, _DOC_SOURCES, name_it=True)
    if body is None:
        return refusal or {}
    try:
        return await asyncio.to_thread(_document_via_daemon, body)
    except Exception as exc:  # noqa: BLE001 - a tool answer, never a traceback
        logger.warning("remember_document failed: %s", type(exc).__name__)
        return _fail("error", "The document could not be saved.", retryable=True)


async def _status(job_id: str, profile_id: str) -> dict[str, Any]:
    if _remote_refused():
        return _fail("not_for_remote", NOT_FOR_REMOTE_DOCS)
    if not _ID.fullmatch(job_id or ""):
        return _fail("invalid_request", "That is not a valid job id.")
    try:
        return await asyncio.to_thread(_job_via_daemon, job_id, (profile_id or "").strip())
    except Exception as exc:  # noqa: BLE001
        logger.warning("media_status failed: %s", type(exc).__name__)
        return _fail("error", "The job could not be read.", retryable=True)


async def _get(media_id: str, variant: str, profile_id: str) -> CallToolResult:
    if _remote_refused():
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

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
                 meta=FILE_PARAMS_META)
    @admits(OperationKind.REMEMBER)
    async def remember_media(
        path: str = "", download_url: str = "", base64: str = "",
        content: str = "", tags: str = "", profile_id: str = "",
        idempotency_key: str = "", scope: str = "", shared_with: str = "",
        file: FileArg = None,
    ) -> dict:
        """Save an image as a memory, from a file on this computer, an https link, base64 or a chat attachment.

        Give exactly one of path, download_url, base64 or file. ``file`` is the picture attached in
        the chat (ChatGPT fills it in). ``scope`` (personal, project, shared or
        global) and ``shared_with`` (comma-separated profile ids) work as they do for remember. Apps on other computers need the
        owner's permission.
        """
        return await _remember({
            "path": path, "download_url": download_url, "base64": base64, "file": file, "content": content,
            "tags": tags, "profile_id": profile_id, "idempotency_key": idempotency_key,
            "scope": scope, "shared_with": _profiles(shared_with)})

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
    @admits(OperationKind.RECALL)
    async def get_media(media_id: str, variant: str = "thumb", profile_id: str = "") -> CallToolResult:
        """Show the thumbnail of a saved image as an image. Apps on other computers need the owner's permission."""
        return await _get(media_id, variant, profile_id)


def register_document_tools(server: Any) -> None:
    """Register ``remember_document`` and ``media_status`` on *server*."""

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
                 meta=FILE_PARAMS_META)
    @admits(OperationKind.REMEMBER)
    async def remember_document(
        path: str = "", base64: str = "", file_name: str = "",
        content: str = "", tags: str = "", profile_id: str = "",
        idempotency_key: str = "", scope: str = "", shared_with: str = "",
        download_url: str = "", file: FileArg = None,
    ) -> dict:
        """Save a PDF as a memory, from a file on this computer, an https link, base64 or a chat attachment.
        Work continues in the background: the answer has a job_id to follow with media_status.

        Give exactly one of path, download_url, base64 or file. ``file`` is the PDF attached in the chat
        (ChatGPT fills it in). ``scope`` and ``shared_with`` (comma-separated profile ids) work as they do for remember.
        Apps on other computers need the owner's permission.
        """
        return await _remember_document({
            "path": path, "base64": base64, "download_url": download_url, "file": file,
            "file_name": file_name, "content": content,
            "tags": tags, "profile_id": profile_id, "idempotency_key": idempotency_key,
            "scope": scope, "shared_with": _profiles(shared_with)})

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
    @admits(OperationKind.RECALL)
    async def media_status(job_id: str, profile_id: str = "") -> dict:
        """How far a saved document's background work has got. Apps on other computers need the owner's permission."""
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
