"""GET /internal/token — serve install token to the local dashboard.

Non-technical user fix: the browser dashboard runs on the same machine
that owns the install-token file. Forcing the user to open a terminal
and paste ``cat ~/.superlocalmemory/.install_token`` into a browser
prompt is pure UX friction with no real security gain — anyone who can
open the dashboard can read the token file (both are user-owned on
loopback). Non-browser clients (Cursor, Antigravity, Copilot, MCP, CLI)
continue to read the token file directly and send ``X-Install-Token``
on their requests; their security model is unchanged.

Gates (narrower than prewarm's 4):
  1. Loopback-only client (127.0.0.1 / ::1).
  2. Origin header, if present, must be a loopback URL.
  3. No install-token requirement — this endpoint PROVIDES the token.

On gate failure or unreadable token file, responds with a fixed-tag
error and non-200 status. Never echoes attacker-supplied material.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["internal"])


_ALLOWED_ORIGIN_PREFIXES = (
    "http://127.0.0.1",
    "https://127.0.0.1",
    "http://localhost",
    "https://localhost",
    "http://[::1]",
    "https://[::1]",
)


def _origin_is_loopback(origin: str) -> bool:
    """Return True iff ``origin`` is absent or a loopback URL."""
    if not origin:
        return True
    return any(origin.startswith(p) for p in _ALLOWED_ORIGIN_PREFIXES)


@router.get("/internal/token")
async def get_token(request: Request) -> JSONResponse:
    """Return the install token for browser-based local dashboard use."""
    try:
        from superlocalmemory.core.security_primitives import (
            _install_token_path,
        )
        from superlocalmemory.hooks.prewarm_auth import is_loopback
    except Exception as exc:  # pragma: no cover
        logger.debug("token: primitives unimportable: %s", exc)
        return JSONResponse({"error": "server_error"}, status_code=500)

    # v3.6.12 (issue #39): in SLM_REMOTE mode, also serve the token to
    # explicitly-allowlisted LAN clients so a remote-browser dashboard can load
    # the Brain page. Default stays loopback-only — remote_mode helpers return
    # False unless SLM_REMOTE=1 AND the client IP is in SLM_MCP_ALLOWED_HOSTS.
    from superlocalmemory.core.remote_mode import (
        is_lan_client_allowed,
        is_remote_origin_allowed,
    )

    client_host = request.client.host if request.client else ""
    if not is_loopback(client_host) and not is_lan_client_allowed(client_host):
        return JSONResponse({"error": "loopback only"}, status_code=403)

    headers = {k.lower(): v for k, v in request.headers.items()}
    origin = headers.get("origin", "")
    if not _origin_is_loopback(origin) and not is_remote_origin_allowed(origin):
        return JSONResponse(
            {"error": "origin not allowed"}, status_code=403,
        )

    try:
        tok_path = _install_token_path()
        tok = Path(tok_path).read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.debug("token: file read failed: %s", exc)
        return JSONResponse(
            {"error": "token_unavailable"}, status_code=500,
        )

    if not tok:
        return JSONResponse(
            {"error": "token_unavailable"}, status_code=500,
        )

    return JSONResponse({"token": tok})


__all__ = ("router",)
