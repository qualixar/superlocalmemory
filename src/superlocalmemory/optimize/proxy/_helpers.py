"""_helpers.py — Shared HTTP utilities used by all surface modules."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

import httpx
from fastapi.requests import Request
from fastapi.responses import Response, StreamingResponse

from superlocalmemory.optimize.proxy.lifecycle import (
    CachedResponse,
    HookChain,
    ProviderResponse,
    ProxyRequest,
)

_get_running_loop = asyncio.get_running_loop

logger = logging.getLogger("slm.optimize.proxy.helpers")

# SEC-M-02 (CWE-400): reject oversized bodies to prevent compression-bomb DoS.
_MAX_REQUEST_BODY_BYTES = 10 * 1024 * 1024  # 10 MB


# Test helper: in-test httpx mock transport. NOT used in production code.
from typing import Callable as _Callable
import httpx as _httpx


class _MockTransport(_httpx.AsyncBaseTransport):
    """Minimal in-test mock for httpx that records requests and returns canned responses.

    Handler may be sync (returns Response) or async (returns coroutine).
    """

    def __init__(self, handler: _Callable) -> None:
        self._handler = handler
        self.requests: list = []

    async def handle_async_request(self, request: _httpx.Request) -> _httpx.Response:
        self.requests.append(request)
        result = self._handler(request)
        if hasattr(result, "__await__"):
            return await result
        return result

_HOP_BY_HOP = frozenset([
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "host",
    "x-forwarded-for", "x-forwarded-host", "x-forwarded-proto",
    "x-real-ip", "x-original-forwarded-for",
])

_ANTHROPIC_FORWARD_HEADERS = frozenset([
    "x-api-key",
    "anthropic-version",
    "anthropic-beta",
    "x-claude-code-session-id",
    "x-claude-code-agent-id",
    "x-claude-code-parent-agent-id",
    "authorization",
    "content-type",
])

_OPENAI_FORWARD_HEADERS = frozenset([
    "authorization",
    "content-type",
    "openai-beta",
    "openai-organization",
])

_GEMINI_NATIVE_FORWARD_HEADERS = frozenset([
    "x-goog-api-key",
    "content-type",
])

_GEMINI_OPENAI_COMPAT_FORWARD_HEADERS = frozenset([
    "authorization",
    "content-type",
])

_SENSITIVE_HEADER_KEYS = frozenset([
    "authorization",
    "x-api-key",
    "x-goog-api-key",
])
_REDACTED = "[REDACTED]"


def _redact_headers(headers: dict) -> dict:
    return {
        k: (_REDACTED if k.lower() in _SENSITIVE_HEADER_KEYS else v)
        for k, v in headers.items()
    }


def _body_has_tools(body: dict) -> bool:
    tools = body.get("tools")
    return isinstance(tools, list) and len(tools) > 0


def _build_forward_headers(request: Request, allowed: frozenset) -> dict:
    result: dict = {}
    for k, v in request.headers.items():
        kl = k.lower()
        if kl in _HOP_BY_HOP:
            continue
        if kl in allowed:
            result[kl] = v
    return result


def _filter_response_headers(headers) -> dict:
    if hasattr(headers, "items"):
        items = headers.items()
    else:
        items = headers
    return {k: v for k, v in items if k.lower() not in _HOP_BY_HOP}


async def _fail_open_forward(proxy: Any, request: Request, upstream_url: str) -> Response:
    if proxy.http_client is None:
        logger.error(
            "fail_open_forward: http_client is None — startup() was not called. "
            "Check daemon lifespan wiring."
        )
        return Response(
            content=b'{"type":"error","error":{"type":"api_error",'
                    b'"message":"SLM proxy not started - lifespan wiring error"}}',
            status_code=502,
            media_type="application/json",
        )
    try:
        body_bytes = await request.body()
        fwd_headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in _HOP_BY_HOP
        }
        upstream_resp = await proxy.http_client.request(
            method=request.method,
            url=upstream_url,
            headers=fwd_headers,
            content=body_bytes,
        )
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers=_filter_response_headers(dict(upstream_resp.headers)),
        )
    except Exception as exc:
        logger.error("fail_open_forward failed upstream=%s exc=%r", upstream_url, exc)
        return Response(
            content=b'{"type":"error","error":{"type":"api_error",'
                    b'"message":"SLM proxy unreachable - check upstream"}}',
            status_code=502,
            media_type="application/json",
        )


async def _stream_forward(
    proxy: Any,
    request_id: str,
    fwd_headers: dict,
    body_bytes: bytes,
    upstream_url: str,
) -> Response | StreamingResponse:
    if proxy.http_client is None:
        logger.error(
            "[%s] _stream_forward: http_client is None - startup() was not called. "
            "Check daemon lifespan wiring.",
            request_id,
        )
        return Response(
            content=b'{"type":"error","error":{"type":"api_error",'
                    b'"message":"SLM proxy not started - lifespan wiring error"}}',
            status_code=502,
            media_type="application/json",
        )

    async def _generate() -> AsyncIterator[bytes]:
        try:
            async with proxy.http_client.stream(
                "POST", upstream_url, content=body_bytes, headers=fwd_headers,
            ) as upstream_resp:
                async for chunk in upstream_resp.aiter_bytes():
                    if chunk:
                        yield chunk
        except httpx.RemoteProtocolError as exc:
            logger.warning("[%s] upstream stream closed early: %r", request_id, exc)
            yield (
                b'event: error\ndata: {"type":"error","error":{'
                b'"type":"api_error","message":"upstream stream closed"}}\n\n'
            )
        except Exception as exc:
            logger.error("[%s] stream forward error: %r", request_id, exc)
            yield (
                b'event: error\ndata: {"type":"error","error":{'
                b'"type":"api_error","message":"SLM proxy stream error"}}\n\n'
            )

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _safe_cache_check(hooks: HookChain, ctx: ProxyRequest) -> CachedResponse:
    try:
        result = hooks.cache.check(ctx)
        return result if result is not None else CachedResponse(
            hit=False, data=None, cache_key="", ttl_seconds=0
        )
    except Exception as exc:
        logger.warning("cache.check failed (fail-open): %s", exc)
        return CachedResponse(hit=False, data=None, cache_key="", ttl_seconds=0)


async def _safe_cache_store(
    hooks: HookChain,
    ctx: ProxyRequest,
    resp: ProviderResponse,
) -> None:
    try:
        hooks.cache.store(ctx, resp)
    except Exception as exc:
        logger.warning("cache.store failed (fail-open): %s", exc)


async def _safe_cache_hit_callbacks(
    hooks: HookChain,
    ctx: ProxyRequest,
    response_bytes: bytes,
    tokens_saved: int,
) -> None:
    try:
        hooks.cache.on_hit(ctx, response_bytes, tokens_saved)
    except Exception as exc:
        logger.warning("cache.on_hit failed (fail-open): %s", exc)


async def _safe_compress(hooks: HookChain, ctx: ProxyRequest) -> ProxyRequest:
    try:
        result: ProxyRequest = hooks.compress.compress(ctx)
    except Exception as exc:
        logger.warning("compress.compress failed (fail-open): %s", exc)
        return ctx
    # on_compress is fired internally by CompressRouter.compress() with
    # actual token counts — no duplicate fire here.
    return result
