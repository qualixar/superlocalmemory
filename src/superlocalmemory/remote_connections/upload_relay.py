# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""The laptop's side of the one-time upload link (``media/upload_links``).

The gateway sends a person's file as ordinary relay request frames, each with
an ``x-slm-upload`` header: ``<op> <token> <index> <total>`` where ``op`` is
``info`` (is this link alive, and what is it for), ``chunk`` (the body is the
next piece of the file, at most 700 KB) or ``finish`` (save what arrived).
:class:`~remote_connections.origin.CanonicalMcpOrigin` hands such a frame here
instead of to the MCP app, so an upload frame can reach exactly these three
operations and nothing else, and an MCP call can never carry one.

The token is the only capability, and it is checked here, never by the gateway.
Each step also re-checks that this connection's key still exists, is a write key
for the profile the link was made for, and still allows media: taking access away
stops an upload that is already running. The save itself runs inside the daemon
through its own local route, so the profile, permission and trust checks are the
ones every image and document save passes.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from collections.abc import Callable
from typing import Any

from superlocalmemory.media.upload_links import UploadError, UploadLinks, UploadRow, default_links
from superlocalmemory.remote_connections.credentials import ConnectorCredential
from superlocalmemory.remote_connections.session import OriginResponse

logger = logging.getLogger(__name__)

UPLOAD_HEADER = "x-slm-upload"
#: How long one ``finish`` frame waits for the save; the gateway asks again until it ends.
FINISH_WAIT_S = 15.0
_HEADER = re.compile(r"(info|chunk|finish) ([A-Za-z0-9_-]{43}) (\d{1,10}) (\d{1,12})")
_LINK = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://\S+")
_SAVED = {
    ("image", "stored"): "Saved to your memory.",
    ("image", "duplicate"): "That picture was already in your memory.",
    ("document", "processing"): "Saved. SuperLocalMemory is reading the document in the background.",
    ("document", "duplicate"): "That document was already in your memory.",
}
_CANNOT = "The file could not be saved. Ask the app for a new link and try again."


def upload_op(request: dict) -> str | None:
    """The raw ``x-slm-upload`` value of a frame, or ``None`` when it is not an upload frame."""
    for name, value in request["headers"]:
        if name.lower() == UPLOAD_HEADER:
            return value
    return None


def _answer(payload: dict[str, Any]) -> OriginResponse:
    return OriginResponse(200, (("content-type", "application/json"),),
                          json.dumps(payload, separators=(",", ":")).encode())


def _refusal(code: str) -> dict[str, Any]:
    return {"ok": False, "code": code, "message": UploadError(code).message}


def _plain(text: object) -> str:
    return _LINK.sub("[link]", " ".join(str(text or "").split()))[:300]


def daemon_finisher(upload_id: str) -> dict[str, Any]:
    """Ask this computer's own daemon to save the finished upload; its receipt as a dict."""
    from superlocalmemory.cli import daemon

    try:
        data = daemon.daemon_request("POST", f"/api/v3/media/uploads/{upload_id}/finish", {},
                                     timeout_seconds=300.0, preserve_unprocessable=True)
    except daemon.DaemonUnprocessable as exc:
        return {"status": "refused", "reason": exc.message}
    except daemon.DaemonRefused:
        return {"status": "refused", "reason": "The daemon did not allow this file to be saved."}
    return data if isinstance(data, dict) else {"status": "unavailable"}


class UploadRelay:
    def __init__(self, links: Callable[[], UploadLinks] = default_links, *, keys: Any = None,
                 finisher: Callable[[str], dict[str, Any]] = daemon_finisher,
                 finish_wait_s: float = FINISH_WAIT_S) -> None:
        self._links, self._keys, self._finisher = links, keys, finisher
        self._wait = finish_wait_s
        self._running: dict[str, asyncio.Task] = {}

    def _key_store(self) -> Any:
        if self._keys is None:
            from superlocalmemory.server.remote_keys import default_store

            self._keys = default_store()
        return self._keys

    async def handle(self, request: dict, credential: ConnectorCredential) -> OriginResponse:
        found = _HEADER.fullmatch(upload_op(request) or "")
        if found is None:
            return _answer(_refusal("invalid_request"))
        op, token, index, total = found[1], found[2], int(found[3]), int(found[4])
        body = base64.b64decode(request["bodyBase64"], validate=True)
        if op != "chunk" and body:
            return _answer(_refusal("invalid_request"))
        try:
            return _answer(await self._dispatch(op, token, index, total, body, credential))
        except UploadError as refused:
            return _answer(_refusal(refused.code))
        except Exception as exc:  # noqa: BLE001 - a plain refusal, never a traceback or a path
            logger.warning("upload step failed (%s)", type(exc).__name__)
            return _answer({"ok": False, "code": "error", "message": _CANNOT})

    async def _dispatch(self, op: str, token: str, index: int, total: int, body: bytes,
                        credential: ConnectorCredential) -> dict[str, Any]:
        links = self._links()
        row = await asyncio.to_thread(links.find, token, credential.connection_id)
        await asyncio.to_thread(self._authorize, credential, row)
        if op == "info":
            info = await asyncio.to_thread(links.info, token, credential.connection_id)
            return {"ok": True, "kind": info.kind, "max_bytes": info.max_bytes, "expires_at": info.expires_at}
        if op == "chunk":
            held = await asyncio.to_thread(links.accept_chunk, token, credential.connection_id,
                                           index, total, body)
            return {"ok": True, "received": held}
        return await self._finish(links, token, credential.connection_id)

    def _authorize(self, credential: ConnectorCredential, row: UploadRow) -> None:
        key = self._key_store().verify(credential.origin_key)
        if (key is None or key.name != "web-" + credential.connection_id or key.key_id != row.key_id
                or key.scope != "write" or "media" not in key.extras or key.profile != row.profile_id):
            raise UploadError("not_allowed")

    async def _finish(self, links: UploadLinks, token: str, connection_id: str) -> dict[str, Any]:
        plan = await asyncio.to_thread(links.begin_finish, token, connection_id)
        if plan.action == "result":
            return plan.result or _refusal("used")
        task = self._running.get(plan.row.upload_id)
        if plan.action == "run":
            task = asyncio.create_task(self._save(links, plan.row), name="slm-upload-save")
            self._running[plan.row.upload_id] = task
            task.add_done_callback(lambda _t, key=plan.row.upload_id: self._running.pop(key, None))
        if task is None:
            return {"ok": True, "done": False}
        try:
            return await asyncio.wait_for(asyncio.shield(task), self._wait)
        except TimeoutError:
            return {"ok": True, "done": False}

    async def _save(self, links: UploadLinks, row: UploadRow) -> dict[str, Any]:
        try:
            receipt = await asyncio.to_thread(self._finisher, row.upload_id)
        except Exception as exc:  # noqa: BLE001 - the person sees a plain line only
            logger.warning("upload save failed (%s)", type(exc).__name__)
            receipt = {}
        status = receipt.get("status") if isinstance(receipt, dict) else None
        saved = _SAVED.get((row.kind, str(status)))
        if saved:
            result = {"ok": True, "done": True, "message": saved}
            await asyncio.to_thread(links.finish_done, row.upload_id, result)
            return result
        reason = _plain(receipt.get("reason") or receipt.get("detail") or receipt.get("error")) \
            if isinstance(receipt, dict) else ""
        result = {"ok": False, "code": "refused", "message": reason or _CANNOT}
        if status == "warming":
            result["code"] = "warming"
            await asyncio.to_thread(links.finish_retry, row.upload_id)
        else:
            await asyncio.to_thread(links.finish_failed, row.upload_id, result)
        return result


