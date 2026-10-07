"""Private in-process delivery to the daemon's existing ASGI MCP boundary.

There is no extra HTTP listener, memory engine, SQLite writer or network hop.
The remote wrapper still authenticates a profile-bound named SLM key and applies
remote tool policy. Real network listeners retain their existing TLS checks.
"""

from __future__ import annotations

import base64
from typing import Any

import httpx

from superlocalmemory.remote_connections.codec import (
    MAX_RESPONSE_BYTES,
    RESPONSE_HEADERS,
    decode_frame,
    encode_frame,
)
from superlocalmemory.remote_connections.credentials import (
    ConnectorCredential,
    CredentialError,
    _validate,
)
from superlocalmemory.remote_connections.session import OriginResponse
from superlocalmemory.server.remote_listener import RemoteListenerASGI


class CanonicalMcpOrigin:
    def __init__(self, app: Any, *, param_headers: tuple[str, ...] = ()):
        descriptor = getattr(getattr(app, "state", None), "daemon_descriptor", None)
        port = getattr(descriptor, "port", 8765)
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("invalid_origin_port")
        self._base_url = f"https://127.0.0.1:{port}"
        self._app = RemoteListenerASGI(app, ("127.0.0.1",))
        self._params = tuple(param_headers)

    async def __call__(self, frame: dict, credential: ConnectorCredential) -> OriginResponse:
        try:
            _validate(credential)
        except CredentialError:
            raise ValueError("invalid_origin_credential") from None
        request = decode_frame(
            encode_frame(frame, param_headers=self._params), param_headers=self._params
        )
        if request["kind"] != "request":
            raise ValueError("invalid_origin_request")
        total = 0

        async def bounded_app(scope, receive, send):
            async def bounded_send(message):
                nonlocal total
                if message.get("type") == "http.response.body":
                    total += len(message.get("body", b""))
                    if total > MAX_RESPONSE_BYTES:
                        raise ValueError("origin_response_too_large")
                await send(message)

            await self._app(scope, receive, bounded_send)

        # These are virtual transport parameters, never an outbound HTTP URL.
        # Credentials never enter a URL, process argument or public relay frame.
        transport = httpx.ASGITransport(app=bounded_app, raise_app_exceptions=True)
        headers = dict(request["headers"])
        headers["Authorization"] = "Bearer " + credential.origin_key
        try:
            async with httpx.AsyncClient(
                transport=transport,
                base_url=self._base_url,
                follow_redirects=False,
                trust_env=False,
            ) as client:
                response = await client.post(
                    "/mcp",
                    headers=headers,
                    content=base64.b64decode(request["bodyBase64"], validate=True),
                )
        except Exception:
            raise ValueError("origin_unavailable") from None
        if 300 <= response.status_code <= 399:
            raise ValueError("origin_redirect_denied")
        allowed = tuple(
            (name, value) for name, value in response.headers.items() if name in RESPONSE_HEADERS
        )
        return OriginResponse(response.status_code, allowed, response.content)
