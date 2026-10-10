# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""A daemon that answered with a server error (5xx) is not a daemon that is down."""

from __future__ import annotations

import json

_GENERIC = "The SLM daemon could not finish the request."


class DaemonServerError(RuntimeError):
    """The daemon answered HTTP 5xx.

    Raised only when the caller passes ``preserve_server_error=True``. Without
    it a 5xx collapses to ``None``, which callers read as "the daemon is not
    running" - wrong advice for a daemon that is running and said why it failed.
    """

    def __init__(self, status: int, code: str, message: str) -> None:
        self.status = int(status)
        self.code = code or ""
        self.message = message or _GENERIC
        super().__init__(self.message)


def server_error_from(exc) -> DaemonServerError:
    """Read ``{"detail": {"code", "message"}}`` or ``{"detail": "text"}`` from a 5xx body."""
    try:
        detail = json.loads(exc.read().decode()).get("detail")
    except Exception:  # noqa: BLE001 - an unreadable body still means "the daemon answered"
        detail = None
    if isinstance(detail, dict):
        return DaemonServerError(exc.code, str(detail.get("code") or ""), str(detail.get("message") or ""))
    return DaemonServerError(exc.code, "", detail if isinstance(detail, str) else "")
