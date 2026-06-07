"""gemini_surface.py — Gemini native and OpenAI-compat surfaces."""

from __future__ import annotations

import logging
import re
import urllib.parse

from fastapi.requests import Request
from fastapi.responses import Response

from superlocalmemory.optimize.proxy._helpers import (
    _GEMINI_NATIVE_FORWARD_HEADERS,
    _GEMINI_OPENAI_COMPAT_FORWARD_HEADERS,
    _fail_open_forward,
    _filter_response_headers,
    _stream_forward,
)

logger = logging.getLogger("slm.optimize.proxy.gemini")

_GEMINI_UPSTREAM_BASE = "https://generativelanguage.googleapis.com"

# SSRF guard: accept BOTH `models/<name>:<method>` AND `<name>:<method>`
# (FastAPI's `:path` converter may strip the `models/` segment depending on
# route declaration; we normalize on the server side either way).
_GEMINI_PATH_RE = re.compile(
    r"^(?:models/)?[a-zA-Z0-9._\-]{1,128}:(generateContent|streamGenerateContent|countTokens)$"
)

_GEMINI_ALLOWED_QUERY_PARAMS = frozenset(["pagesize", "pagetoken"])


def _validate_gemini_path(model_and_method: str) -> bool:
    return bool(_GEMINI_PATH_RE.match(model_and_method))


async def handle_gemini_native(
    proxy: object,
    request: Request,
    model_and_method: str,
) -> Response:
    request_id = await proxy.next_request_id()

    if not _validate_gemini_path(model_and_method):
        logger.warning(
            "[%s] handle_gemini_native: rejected invalid path param=%r (SSRF guard)",
            request_id, model_and_method,
        )
        return Response(
            content=b'{"error":{"code":400,"message":"Invalid model/method path",'
                    b'"status":"INVALID_ARGUMENT"}}',
            status_code=400,
            media_type="application/json",
        )

    upstream_url = f"{_GEMINI_UPSTREAM_BASE}/v1beta/{model_and_method}"

    try:
        body_bytes = await request.body()
        fwd_headers = {
            k: v for k, v in request.headers.items()
            if k.lower() in _GEMINI_NATIVE_FORWARD_HEADERS
        }
        fwd_headers["content-length"] = str(len(body_bytes))

        stream = "streamGenerateContent" in model_and_method
        if stream:
            allowed = {
                k: v for k, v in request.query_params.items()
                if k.lower() in _GEMINI_ALLOWED_QUERY_PARAMS
            }
            allowed["alt"] = "sse"
            upstream_url = f"{upstream_url}?{urllib.parse.urlencode(allowed)}"
            return await _stream_forward(
                proxy, request_id, fwd_headers, body_bytes, upstream_url
            )

        upstream_resp = await proxy.http_client.post(
            upstream_url, content=body_bytes, headers=fwd_headers,
        )
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            media_type="application/json",
            headers=_filter_response_headers(dict(upstream_resp.headers)),
        )
    except Exception as exc:
        logger.error("[%s] handle_gemini_native exc=%r — fail-open", request_id, exc)
        return await _fail_open_forward(proxy, request, upstream_url)


async def handle_gemini_openai_compat(proxy: object, request: Request) -> Response:
    local_path = request.url.path
    upstream_url = f"{_GEMINI_UPSTREAM_BASE}{local_path}"
    request_id = await proxy.next_request_id()
    try:
        body_bytes = await request.body()
        fwd_headers = {
            k: v for k, v in request.headers.items()
            if k.lower() in _GEMINI_OPENAI_COMPAT_FORWARD_HEADERS
        }
        if body_bytes:
            fwd_headers["content-length"] = str(len(body_bytes))
        upstream_resp = await proxy.http_client.request(
            method=request.method,
            url=upstream_url,
            content=body_bytes if body_bytes else None,
            headers=fwd_headers,
        )
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            media_type="application/json",
            headers=_filter_response_headers(dict(upstream_resp.headers)),
        )
    except Exception as exc:
        logger.error(
            "[%s] handle_gemini_openai_compat exc=%r — fail-open", request_id, exc
        )
        return await _fail_open_forward(proxy, request, upstream_url)
