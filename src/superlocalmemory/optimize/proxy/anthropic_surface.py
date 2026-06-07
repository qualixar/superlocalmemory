"""anthropic_surface.py — Anthropic Messages + count_tokens + models surfaces."""

from __future__ import annotations

import json
import logging

from fastapi.requests import Request
from fastapi.responses import Response

from fastapi.responses import StreamingResponse

from superlocalmemory.optimize.proxy._helpers import (
    _ANTHROPIC_FORWARD_HEADERS,
    _MAX_REQUEST_BODY_BYTES,
    _body_has_tools,
    _build_forward_headers,
    _fail_open_forward,
    _filter_response_headers,
    _parse_sse_to_json,
    _redact_headers,
    _safe_cache_check,
    _safe_cache_hit_callbacks,
    _safe_cache_store,
    _safe_compress,
    _stream_and_cache_forward,
    _stream_forward,
)
from superlocalmemory.optimize.proxy.lifecycle import ProviderResponse, ProxyRequest

logger = logging.getLogger("slm.optimize.proxy.anthropic")

_UPSTREAM_BASE = "https://api.anthropic.com"


def _sse_from_cached_json(cached_bytes: bytes) -> "StreamingResponse | None":
    """Convert a cached non-streaming Anthropic response into an SSE StreamingResponse.

    Used when a streaming request hits the cache — we replay the stored JSON as
    the sequence of SSE events the Anthropic API would have emitted, preserving
    the streaming contract with the client (e.g. Claude Code).

    Returns None if the bytes are not a parseable Anthropic message object so the
    caller can fall back to a live upstream request.
    """
    try:
        resp = json.loads(cached_bytes)
    except (json.JSONDecodeError, ValueError):
        return None

    if resp.get("type") != "message":
        return None

    async def _generate():
        # 1. message_start
        msg_start = {
            "type": "message_start",
            "message": {
                "id": resp.get("id", ""),
                "type": "message",
                "role": resp.get("role", "assistant"),
                "content": [],
                "model": resp.get("model", ""),
                "stop_reason": None,
                "stop_sequence": None,
                "usage": resp.get("usage", {}),
            },
        }
        yield f"event: message_start\ndata: {json.dumps(msg_start)}\n\n".encode()
        yield b"data: {\"type\":\"ping\"}\n\n"

        # 2. content blocks — supports both text and tool_use (BUG-FIX v3.6.4)
        for i, block in enumerate(resp.get("content", [])):
            block_type = block.get("type", "text")

            if block_type == "tool_use":
                # tool_use block: content_block_start carries id + name (not text)
                block_start = {
                    "type": "content_block_start",
                    "index": i,
                    "content_block": {
                        "type": "tool_use",
                        "id": block.get("id", ""),
                        "name": block.get("name", ""),
                        "input": {},
                    },
                }
                yield (
                    f"event: content_block_start\n"
                    f"data: {json.dumps(block_start)}\n\n"
                ).encode()

                # Emit input JSON as input_json_delta chunks (50-char pieces)
                input_str = json.dumps(block.get("input", {}), separators=(",", ":"))
                chunk_size = 50
                for start in range(0, max(len(input_str), 1), chunk_size):
                    piece = input_str[start: start + chunk_size]
                    delta = {
                        "type": "content_block_delta",
                        "index": i,
                        "delta": {"type": "input_json_delta", "partial_json": piece},
                    }
                    yield (
                        f"event: content_block_delta\n"
                        f"data: {json.dumps(delta)}\n\n"
                    ).encode()

            else:
                # text block (default)
                block_start = {
                    "type": "content_block_start",
                    "index": i,
                    "content_block": {"type": block_type, "text": ""},
                }
                yield (
                    f"event: content_block_start\n"
                    f"data: {json.dumps(block_start)}\n\n"
                ).encode()

                if block_type == "text":
                    text = block.get("text", "")
                    # Emit text in chunks of 100 chars to preserve the streaming feel.
                    chunk_size = 100
                    for start in range(0, max(len(text), 1), chunk_size):
                        chunk = text[start: start + chunk_size]
                        delta = {
                            "type": "content_block_delta",
                            "index": i,
                            "delta": {"type": "text_delta", "text": chunk},
                        }
                        yield (
                            f"event: content_block_delta\n"
                            f"data: {json.dumps(delta)}\n\n"
                        ).encode()

            block_stop = {"type": "content_block_stop", "index": i}
            yield (
                f"event: content_block_stop\n"
                f"data: {json.dumps(block_stop)}\n\n"
            ).encode()

        # 3. message_delta
        usage = resp.get("usage", {})
        msg_delta = {
            "type": "message_delta",
            "delta": {
                "stop_reason": resp.get("stop_reason", "end_turn"),
                "stop_sequence": resp.get("stop_sequence", None),
            },
            "usage": {"output_tokens": usage.get("output_tokens", 0)},
        }
        yield f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n".encode()

        # 4. message_stop
        yield b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


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
            # BUG-FIX (v3.6.3): streaming path previously bypassed both the
            # cache and compression. Claude Code exclusively uses streaming, so
            # savings were permanently 0. Fixed below:
            #
            # 1. Cache check — if a prior streaming call's response was
            #    accumulated and stored, return it as a proper SSE stream rather
            #    than forwarding to Anthropic at all.
            if proxy.hooks.cache:
                cache_result = await _safe_cache_check(proxy.hooks, ctx)
                if cache_result and cache_result.hit and cache_result.data:
                    logger.debug(
                        "[%s] streaming cache HIT key=%s",
                        request_id, cache_result.cache_key,
                    )
                    await _safe_cache_hit_callbacks(
                        proxy.hooks, ctx, cache_result.data, tokens_saved=0
                    )
                    sse_resp = _sse_from_cached_json(cache_result.data)
                    if sse_resp is not None:
                        return sse_resp
                    # Fallback: cached bytes unparseable as an Anthropic message —
                    # forward normally rather than returning garbage.

            # 2. Compression — apply to the REQUEST body even for streaming.
            #    Tokens saved in the prompt are real savings regardless of whether
            #    the response is streamed.
            outbound_bytes = body_bytes
            if proxy.hooks.compress:
                compress_result = await _safe_compress(proxy.hooks, ctx)
                if compress_result.body_bytes != body_bytes:
                    outbound_bytes = compress_result.body_bytes

            fwd_headers = _build_forward_headers(request, _ANTHROPIC_FORWARD_HEADERS)
            fwd_headers["content-length"] = str(len(outbound_bytes))

            # 3. Cache store callback — accumulate SSE bytes and store after the
            #    stream completes so future identical requests are served from cache.
            #    BUG-FIX (v3.6.3): previously there was no accumulation at all —
            #    the cache was never populated from streaming calls, which is the
            #    ONLY call type Claude Code makes.
            store_callback = None
            if proxy.hooks.cache:
                _hooks = proxy.hooks
                _ctx = ctx
                async def _store_from_sse(sse_bytes: bytes) -> None:
                    parsed = _parse_sse_to_json(sse_bytes)
                    if parsed is None:
                        return
                    prov = ProviderResponse(
                        modified=False, body={}, body_bytes=parsed,
                        tokens_before=0, tokens_after=0, strategy="none",
                    )
                    await _safe_cache_store(_hooks, _ctx, prov)
                store_callback = _store_from_sse

            return await _stream_and_cache_forward(
                proxy, request_id, fwd_headers, outbound_bytes, upstream_url,
                on_complete=store_callback,
            )

        cache_result = None
        if proxy.hooks.cache:
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
