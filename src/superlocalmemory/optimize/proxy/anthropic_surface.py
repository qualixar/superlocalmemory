"""anthropic_surface.py — Anthropic Messages + count_tokens + models surfaces."""

from __future__ import annotations

import json
import logging

from fastapi.requests import Request
from fastapi.responses import Response

from superlocalmemory.optimize.proxy._helpers import (
    _ANTHROPIC_FORWARD_HEADERS,
    _MAX_REQUEST_BODY_BYTES,
    _body_has_tools,
    _build_forward_headers,
    _fail_open_forward,
    _filter_response_headers,
    _redact_headers,
    _safe_cache_check,
    _safe_cache_hit_callbacks,
    _safe_cache_store,
    _stream_forward,
)
from superlocalmemory.optimize.proxy.lifecycle import ProviderResponse, ProxyRequest

logger = logging.getLogger("slm.optimize.proxy.anthropic")

_UPSTREAM_BASE = "https://api.anthropic.com"


async def handle_messages(proxy: object, request: Request) -> Response:
    request_id = await proxy.next_request_id()
    upstream_url = f"{_UPSTREAM_BASE}/v1/messages"

    cl_header = request.headers.get("content-length")
    if cl_header is not None:
        try:
            if int(cl_header) > _MAX_REQUEST_BODY_BYTES:
                return Response(
                    content=b'{"type":"error","error":{"type":"invalid_request_error",'
                            b'"message":"Request body too large (max 10 MB)"}}',
                    status_code=413,
                    media_type="application/json",
                )
        except ValueError:
            pass

    try:
        body_bytes = await request.body()
        if len(body_bytes) > _MAX_REQUEST_BODY_BYTES:
            return Response(
                content=b'{"type":"error","error":{"type":"invalid_request_error",'
                        b'"message":"Request body too large (max 10 MB)"}}',
                status_code=413,
                media_type="application/json",
            )

        try:
            body = json.loads(body_bytes)
        except json.JSONDecodeError as exc:
            logger.warning("[%s] body parse failed, raw forward: %s", request_id, exc)
            return await _fail_open_forward(proxy, request, upstream_url)

        stream = bool(body.get("stream", False))
        has_tools = _body_has_tools(body)
        ctx = ProxyRequest(
            provider="anthropic",
            method="POST",
            path="/v1/messages",
            headers=_redact_headers(dict(request.headers)),
            body=body,
            body_bytes=body_bytes,
            request_id=request_id,
            stream=stream,
            has_tools=has_tools,
        )

        if stream:
            fwd_headers = _build_forward_headers(request, _ANTHROPIC_FORWARD_HEADERS)
            fwd_headers["content-length"] = str(len(body_bytes))
            return await _stream_forward(
                proxy, request_id, fwd_headers, body_bytes, upstream_url
            )

        cache_result = None
        if not has_tools and proxy.hooks.cache:
            cache_result = await _safe_cache_check(proxy.hooks, ctx)
            if cache_result.hit and cache_result.data:
                logger.debug("[%s] cache HIT key=%s", request_id, cache_result.cache_key)
                await _safe_cache_hit_callbacks(
                    proxy.hooks, ctx, cache_result.data, tokens_saved=0
                )
                return Response(
                    content=cache_result.data,
                    status_code=200,
                    media_type="application/json",
                )

        outbound_bytes = body_bytes
        if proxy.hooks.compress:
            compress_result = await _safe_compress(proxy.hooks, ctx)
            if compress_result.body_bytes != body_bytes:
                outbound_bytes = compress_result.body_bytes

        fwd_headers = _build_forward_headers(request, _ANTHROPIC_FORWARD_HEADERS)
        fwd_headers["content-length"] = str(len(outbound_bytes))

        upstream_resp = await proxy.http_client.post(
            upstream_url, content=outbound_bytes, headers=fwd_headers,
        )
        resp_bytes = upstream_resp.content

        if (
            upstream_resp.status_code == 200
            and not has_tools
            and proxy.hooks.cache
            and cache_result is not None
            and cache_result.cache_key
        ):
            _prov_resp = ProviderResponse(
                modified=False, body={}, body_bytes=resp_bytes,
                tokens_before=0, tokens_after=0, strategy="none",
            )
            await _safe_cache_store(proxy.hooks, ctx, _prov_resp)

        return Response(
            content=resp_bytes,
            status_code=upstream_resp.status_code,
            media_type="application/json",
            headers=_filter_response_headers(dict(upstream_resp.headers)),
        )

    except Exception as exc:
        logger.error("[%s] handle_messages unhandled exc=%r — fail-open", request_id, exc)
        return await _fail_open_forward(proxy, request, upstream_url)


async def handle_count_tokens(proxy: object, request: Request) -> Response:
    request_id = await proxy.next_request_id()
    upstream_url = f"{_UPSTREAM_BASE}/v1/messages/count_tokens"
    try:
        body_bytes = await request.body()
        fwd_headers = _build_forward_headers(request, _ANTHROPIC_FORWARD_HEADERS)
        fwd_headers["content-length"] = str(len(body_bytes))
        upstream_resp = await proxy.http_client.post(
            upstream_url, content=body_bytes, headers=fwd_headers,
        )
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            media_type="application/json",
        )
    except Exception as exc:
        logger.error("[%s] handle_count_tokens exc=%r — fail-open", request_id, exc)
        return await _fail_open_forward(proxy, request, upstream_url)


async def handle_models(proxy: object, request: Request) -> Response:
    request_id = await proxy.next_request_id()
    upstream_url = f"{_UPSTREAM_BASE}/v1/models"
    try:
        fwd_headers = _build_forward_headers(request, _ANTHROPIC_FORWARD_HEADERS)
        upstream_resp = await proxy.http_client.get(upstream_url, headers=fwd_headers)
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            media_type="application/json",
        )
    except Exception as exc:
        logger.error("[%s] handle_models exc=%r — fail-open", request_id, exc)
        return await _fail_open_forward(proxy, request, upstream_url)
