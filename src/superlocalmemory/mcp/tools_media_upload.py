# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""``media_upload_link``: how a web app gets a picture or a PDF onto this computer.

A web app cannot type a file into a tool call. It asks for a link instead; the
person opens the link in any browser, picks the file and presses Save, and the
file travels to this computer over the connection that is already open. The
link works once, for ten minutes, and nothing is stored in the cloud on the way.
It is for the person alone and must not be shared with anyone.

Only a remote app can ask: an app on this computer passes a path to
``remember_media`` or ``remember_document``. A remote app needs what the saving
tools need: the signed media and write permissions in its grant, and a write key
that has opted in to media. The link is bound to that connection, key and
profile and to the app (authorization) that asked for it, so taking that app's
access away ends its links; the profile is the key's own, never an argument.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from mcp.types import ToolAnnotations

from superlocalmemory.core.admission import admits
from superlocalmemory.core.operation_request import OperationKind
from superlocalmemory.mcp.remote_caller import (
    current_remote_grant, current_remote_key_id, current_remote_media_allowed,
)
from superlocalmemory.media.upload_links import UPLOAD_BASE_URL, UploadError, default_links

logger = logging.getLogger("slm.mcp.tools_media_upload")

_WORDS = {"image": "picture", "document": "document"}


def _fail(code: str, message: str) -> dict[str, Any]:
    return {"success": False, "status": "refused", "retryable": False, "code": code, "error": message}


def _key_store() -> Any:
    from superlocalmemory.server.remote_keys import default_store

    return default_store()


def _writer_key(key_id: str, connection_id: str) -> Any:
    """The connection's own write key that allows media, or ``None``."""
    key = next((k for k in _key_store().list() if k.key_id == key_id), None)
    if (key is None or not key.active or key.name != "web-" + connection_id or key.scope != "write"
            or "media" not in key.extras or not key.profile):
        return None
    return key


def _link(kind: str, note: str) -> dict[str, Any]:
    key_id, grant = current_remote_key_id(), current_remote_grant()
    if key_id is None:
        return _fail("not_for_local", "Upload links are for apps on other computers. "
                     "Pass a file path to remember_media or remember_document instead.")
    if (not current_remote_media_allowed() or grant is None
            or not {"slm:write", "slm:media"} <= set(grant.scopes)):
        return _fail("not_for_remote", "This app is not allowed to add pictures or documents.")
    key = _writer_key(key_id, grant.connection_id)
    if key is None:
        return _fail("not_allowed", "This computer does not let this app add pictures or documents.")
    from superlocalmemory.remote_connections import companion

    if "upload-v1" not in companion.CONNECTOR_FEATURES:
        return _fail("update_required", "This computer's SuperLocalMemory must be updated to use upload links.")
    try:
        minted = default_links().mint(grant.connection_id, key.key_id, key.profile, kind, note or "",
                                      authorization_id=grant.authorization_id)
    except UploadError as refused:
        return _fail(refused.code, refused.message)
    url = f"{UPLOAD_BASE_URL}/u/{grant.connection_id}/{minted.token}"
    expires = datetime.fromtimestamp(minted.expires_at, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"success": True, "kind": kind, "url": url, "expires_at": expires,
            "max_mb": minted.max_bytes // (1024 * 1024),
            "message": (f"Tell the person: open this link to add the {_WORDS[kind]}, pick the file and press Save. "
                        "The link works once, expires in 10 minutes, and is only for them: do not share it "
                        f"with anyone. {url}")}


def register_upload_link_tool(server: Any) -> None:
    """Register ``media_upload_link`` on *server*."""

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False))
    @admits(OperationKind.REMEMBER)
    async def media_upload_link(kind: str = "image", note: str = "", profile_id: str = "") -> dict:
        """Get a one-time link the person opens to add a picture or a PDF from their own device.

        Use this when you cannot send the file yourself. ``kind`` is "image" or "document";
        ``note`` is the words to save with it. Show the person the link and ask them to open it,
        pick the file and press Save. Only for apps on other computers: it works once, expires
        in 10 minutes, and must not be shared with anyone.
        """
        try:
            return await asyncio.to_thread(_link, kind, note)
        except Exception as exc:  # noqa: BLE001 - a tool answer, never a traceback
            logger.warning("media_upload_link failed: %s", type(exc).__name__)
            return _fail("error", "The upload link could not be made.")


__all__ = ["register_upload_link_tool"]
