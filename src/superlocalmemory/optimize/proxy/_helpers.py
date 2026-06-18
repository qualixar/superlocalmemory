"""_helpers.py — Shared HTTP utilities used by all surface modules."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from typing import Any, AsyncIterator, Callable

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

# Per-callable cache of "does this hook method accept a tenant_id kwarg?".
# Keyed by id() of the bound method's __func__ so it is stable per hook class.
_HOOK_TENANT_SUPPORT: dict[int, bool] = {}


def _accepts_tenant_id(fn: Callable) -> bool:
    """Return True if a hook method accepts a ``tenant_id`` keyword argument.

    SECURITY (Stage-9 R1): the original WP-D shim used ``except TypeError`` to
    detect legacy hooks, but that clause is structurally unable to tell a
    signature mismatch from a TypeError raised *inside* a tenant-aware hook —
    so an internal bug silently downgraded an authenticated request onto the
    shared (tenant-less) path, re-opening cross-tenant disclosure.  We instead
    probe the signature ONCE (cached): a hook is treated as tenant-aware if it
    has an explicit ``tenant_id`` parameter or accepts ``**kwargs``.  A
    tenant-aware hook is NEVER retried without the tenant_id; any error it
    raises fails open to a cache MISS, never the shared namespace.
    """
    target = getattr(fn, "__func__", fn)
    key = id(target)
    cached = _HOOK_TENANT_SUPPORT.get(key)
    if cached is None:
        try:
            params = inspect.signature(fn).parameters
            cached = "tenant_id" in params or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
        except (ValueError, TypeError):
            # Builtins / C callables without a signature — assume legacy.
            cached = False
        _HOOK_TENANT_SUPPORT[key] = cached
    return cached

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
    "authorization",  # WP-11a: Antigravity ADC/OAuth bearer was dropped; add it back
])

# WP-11: Vertex AI forward headers — Authorization passed untouched (AC-2).
# x-goog-user-project required for quota attribution on Vertex calls.
_VERTEX_FORWARD_HEADERS = frozenset([
    "authorization",
    "content-type",
    "x-goog-api-key",
    "x-goog-user-project",
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


def _derive_tenant_id(provider: str, raw_credential: "str | None") -> "str | None":
    """Derive a per-tenant isolation key from the raw (un-redacted) credential.

    SECURITY (WP-D): Called at the surface-handler layer BEFORE _redact_headers
    strips the credential from ProxyRequest.headers.  The derived id is threaded
    into CacheManager.check() / .store() so that two users sharing the same
    prompt but using different API keys receive independent cache namespaces.

    Returns None when no credential is present — callers must SKIP caching
    (never collapse to the default tenant) to prevent cross-tenant disclosure.

    Output: 64-char lowercase hex SHA-256 of ``f"{provider}:{raw_credential}"``.
    Provider is folded in so anthropic:K and openai:K are distinct tenants even
    if the literal key string coincidentally matches.
    """
    import hashlib as _hashlib

    if not raw_credential:
        return None
    return _hashlib.sha256(f"{provider}:{raw_credential}".encode()).hexdigest()


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


# ---------------------------------------------------------------------------
# SSE → JSON converter (for streaming cache hits and post-stream storage)
# ---------------------------------------------------------------------------

def _parse_sse_to_json(sse_bytes: bytes) -> bytes | None:
    """Parse an accumulated Anthropic SSE stream into a single JSON message.

    Used for two purposes:
    1. After a streaming response completes — store the assembled JSON in the
       cache so the next identical streaming request is served from cache.
    2. Cache hit path — convert stored JSON back to SSE via _sse_from_cached_json
       in anthropic_surface.py.

    Returns None if the bytes are not a valid/complete Anthropic SSE stream
    (e.g. error response, client-disconnected partial, or empty). The caller
    MUST treat None as "do not cache".

    BUG-FIX (v3.6.4): Previously returned None for any response containing
    tool_use blocks, which meant Claude Code responses were NEVER cached (Claude
    Code always uses tools). Now handles both text AND tool_use content blocks:
    - text blocks: accumulated via text_delta events
    - tool_use blocks: accumulated via input_json_delta events, stored with
      full {id, name, input} so _sse_from_cached_json can replay them correctly.
    """
    # content_blocks[index] = {"type": "text", "text_parts": [...]} OR
    #                          {"type": "tool_use", "id": "...", "name": "...", "input_parts": [...]}
    content_blocks: dict[int, dict] = {}
    message_id = ""
    model = ""
    role = "assistant"
    input_tokens = 0
    output_tokens = 0
    stop_reason = "end_turn"
    message_start_seen = False
    message_stop_seen = False

    current_event = ""
    for raw_line in sse_bytes.decode("utf-8", errors="replace").split("\n"):
        line = raw_line.rstrip("\r")
        if line.startswith("event: "):
            current_event = line[7:].strip()
        elif line.startswith("data: "):
            data_str = line[6:].strip()
            if not data_str or data_str == "[DONE]":
                continue
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if current_event == "message_start":
                message_start_seen = True
                msg = data.get("message", {})
                message_id = msg.get("id", "")
                model = msg.get("model", "")
                role = msg.get("role", "assistant")
                usage = msg.get("usage", {})
                input_tokens = usage.get("input_tokens", 0)

            elif current_event == "content_block_start":
                idx = data.get("index", 0)
                cb = data.get("content_block", {})
                block_type = cb.get("type", "text")
                if block_type == "tool_use":
                    content_blocks[idx] = {
                        "type": "tool_use",
                        "id": cb.get("id", ""),
                        "name": cb.get("name", ""),
                        "input_parts": [],
                    }
                else:
                    content_blocks[idx] = {"type": "text", "text_parts": []}

            elif current_event == "content_block_delta":
                idx = data.get("index", 0)
                delta = data.get("delta", {})
                block = content_blocks.get(idx)
                if block is None:
                    continue
                delta_type = delta.get("type", "")
                if delta_type == "text_delta":
                    block.setdefault("text_parts", []).append(delta.get("text", ""))
                elif delta_type == "input_json_delta":
                    block.setdefault("input_parts", []).append(delta.get("partial_json", ""))

            elif current_event == "message_delta":
                usage2 = data.get("usage", {})
                output_tokens = usage2.get("output_tokens", 0)
                stop = data.get("delta", {})
                stop_reason = stop.get("stop_reason", stop_reason) or stop_reason

            elif current_event == "message_stop":
                message_stop_seen = True

    if not message_start_seen or not message_stop_seen:
        return None

    # Assemble content array — preserve block ordering by index
    content: list[dict] = []
    for idx in sorted(content_blocks.keys()):
        block = content_blocks[idx]
        if block["type"] == "tool_use":
            input_json_str = "".join(block.get("input_parts", []))
            try:
                input_obj = json.loads(input_json_str) if input_json_str else {}
            except json.JSONDecodeError:
                input_obj = {"_raw": input_json_str}
            content.append({
                "type": "tool_use",
                "id": block["id"],
                "name": block["name"],
                "input": input_obj,
            })
        else:
            text = "".join(block.get("text_parts", []))
            content.append({"type": "text", "text": text})

    result = {
        "id": message_id,
        "type": "message",
        "role": role,
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }
    return json.dumps(result, separators=(",", ":")).encode("utf-8")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

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
    """Simple passthrough streaming — no cache accumulation."""
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


async def _stream_and_cache_forward(
    proxy: Any,
    request_id: str,
    fwd_headers: dict,
    body_bytes: bytes,
    upstream_url: str,
    on_complete: "Callable[[bytes], Any] | None" = None,
    max_accumulate: int | None = None,
) -> Response | StreamingResponse:
    """Stream-forward with optional post-stream cache-store callback.

    BUG-FIX (v3.6.3): Claude Code exclusively uses streaming.  The old
    _stream_forward never accumulated the response body, so the cache was
    NEVER populated — savings were permanently 0.

    This helper tees the stream: each chunk is yielded to the client AND
    appended to an in-memory accumulator.  After the LAST chunk (detected by
    ``message_stop`` in the SSE bytes), ``on_complete`` is awaited with the
    full accumulated bytes.  The caller converts those bytes to a JSON message
    via ``_parse_sse_to_json`` and stores them in the cache.

    on_complete is only called when:
    - the accumulated bytes contain ``b"message_stop"`` (complete response),
    - the response did NOT error mid-stream.
    """
    if proxy.http_client is None:
        logger.error(
            "[%s] _stream_and_cache_forward: http_client is None.",
            request_id,
        )
        return Response(
            content=b'{"type":"error","error":{"type":"api_error",'
                    b'"message":"SLM proxy not started - lifespan wiring error"}}',
            status_code=502,
            media_type="application/json",
        )

    acc: list[bytes] = []
    acc_bytes = 0
    acc_capped = False
    complete_called = False

    async def _generate() -> AsyncIterator[bytes]:
        nonlocal complete_called, acc_bytes, acc_capped
        stream_error = False
        try:
            async with proxy.http_client.stream(
                "POST", upstream_url, content=body_bytes, headers=fwd_headers,
            ) as upstream_resp:
                async for chunk in upstream_resp.aiter_bytes():
                    if chunk:
                        # Always forward to the client; only bound what we hold
                        # in memory for the on_complete callback (CWE-400).
                        if max_accumulate is None or acc_bytes < max_accumulate:
                            acc.append(chunk)
                            acc_bytes += len(chunk)
                        elif not acc_capped:
                            acc_capped = True
                            logger.debug(
                                "[%s] stream accumulator capped at %d bytes",
                                request_id, max_accumulate,
                            )
                        yield chunk
        except httpx.RemoteProtocolError as exc:
            stream_error = True
            logger.warning("[%s] upstream stream closed early: %r", request_id, exc)
            yield (
                b'event: error\ndata: {"type":"error","error":{'
                b'"type":"api_error","message":"upstream stream closed"}}\n\n'
            )
        except Exception as exc:
            stream_error = True
            logger.error("[%s] stream forward error: %r", request_id, exc)
            yield (
                b'event: error\ndata: {"type":"error","error":{'
                b'"type":"api_error","message":"SLM proxy stream error"}}\n\n'
            )
        finally:
            # BUG-FIX (v3.6.4): Removed surface-specific sentinel check
            # ("message_stop" / "[DONE]"). Completeness is now validated
            # inside each parser (_parse_sse_to_json, _parse_openai_sse_to_json,
            # _parse_gemini_sse_to_json) which return None for incomplete
            # streams. This makes _stream_and_cache_forward universal: it
            # calls on_complete whenever the stream ends without error and
            # lets the parser decide whether to store. Gemini SSE has neither
            # sentinel — streams end by connection close after the final chunk
            # containing "finishReason".
            _joined = b"".join(acc)
            if on_complete and acc and not complete_called and not stream_error:
                complete_called = True
                try:
                    await on_complete(_joined)
                except Exception as exc:
                    logger.warning(
                        "[%s] cache store callback raised (fail-open): %r",
                        request_id, exc,
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


async def capture_passthrough_forward(
    proxy: Any,
    request: Request,
    *,
    provider: str,
    upstream_url: str,
    allowed_headers: frozenset,
    request_id: str,
    model_hint: str = "",
    sse_parser: "Callable[[bytes], bytes | None] | None" = None,
    is_stream: bool = False,
) -> Response | StreamingResponse:
    """Shadow-capture passthrough (v3.6.10, plan §7).

    Pure passthrough to upstream + record the exchange to the capture corpus.
    NO cache, NO compression — capture mode observes only authentic traffic.
    Fail-open: a capture or forward error degrades to a normal forward/error
    response; the user's request is never blocked by capture.
    """
    from superlocalmemory.optimize.proxy.capture import (
        extract_usage,
        record_exchange_async,
    )

    body_bytes = await request.body()
    fwd_headers = _build_forward_headers(request, allowed_headers)
    fwd_headers["content-length"] = str(len(body_bytes))

    if is_stream:
        async def _on_complete(acc: bytes) -> None:
            parsed = sse_parser(acc) if sse_parser else None
            payload = parsed if parsed is not None else acc
            itok, otok, mdl = extract_usage(provider, parsed)
            await record_exchange_async(
                provider=provider,
                model=mdl or model_hint,
                request_body=body_bytes,
                response_body=payload,
                content_type="text/event-stream",
                input_tokens=itok,
                output_tokens=otok,
                status_code=200,
                stream=True,
            )

        # Bound the in-memory accumulator (CWE-400): the corpus only keeps the
        # first 1 MB per side anyway, so cap accumulation there.
        from superlocalmemory.optimize.proxy.capture import _MAX_CAPTURE_BODY_BYTES
        return await _stream_and_cache_forward(
            proxy, request_id, fwd_headers, body_bytes, upstream_url,
            on_complete=_on_complete,
            max_accumulate=_MAX_CAPTURE_BODY_BYTES,
        )

    if proxy.http_client is None:
        return await _fail_open_forward(proxy, request, upstream_url)
    try:
        upstream_resp = await proxy.http_client.post(
            upstream_url, content=body_bytes, headers=fwd_headers,
        )
    except Exception as exc:
        logger.error("[%s] capture passthrough upstream error: %r", request_id, exc)
        return await _fail_open_forward(proxy, request, upstream_url)

    resp_bytes = upstream_resp.content
    itok, otok, mdl = extract_usage(provider, resp_bytes)
    await record_exchange_async(
        provider=provider,
        model=mdl or model_hint,
        request_body=body_bytes,
        response_body=resp_bytes,
        content_type="application/json",
        input_tokens=itok,
        output_tokens=otok,
        status_code=upstream_resp.status_code,
        stream=False,
    )
    return Response(
        content=resp_bytes,
        status_code=upstream_resp.status_code,
        media_type="application/json",
        headers=_filter_response_headers(dict(upstream_resp.headers)),
    )


async def _safe_cache_check(
    hooks: HookChain,
    ctx: ProxyRequest,
    tenant_id: "str | None" = None,
) -> CachedResponse:
    """Invoke cache.check(); fail-open on error.

    SECURITY (WP-D): tenant_id is forwarded to the CacheHook so that the
    credential-derived namespace is used.  None signals "skip caching" — the
    cache hook must not collapse to a default tenant for unauthenticated
    requests.  When tenant_id is None, return an empty miss immediately.

    Backward compat: a genuinely legacy hook (no tenant_id parameter and no
    **kwargs) is called without the kwarg.  A tenant-aware hook is NEVER
    downgraded to the tenant-less path — see _accepts_tenant_id (Stage-9 R1).
    """
    miss = CachedResponse(hit=False, data=None, cache_key="", ttl_seconds=0)
    if tenant_id is None:
        return miss
    try:
        if _accepts_tenant_id(hooks.cache.check):
            result = hooks.cache.check(ctx, tenant_id=tenant_id)
        else:
            # Genuine legacy hook — no tenant_id support at all.
            result = hooks.cache.check(ctx)
        return result if result is not None else miss
    except Exception as exc:
        # SECURITY: a tenant-aware hook that errors must fail-open to a MISS,
        # never be retried on the shared (tenant-less) namespace.
        logger.warning("cache.check failed (fail-open miss): %s", exc)
        return miss


async def _safe_cache_store(
    hooks: HookChain,
    ctx: ProxyRequest,
    resp: ProviderResponse,
    tenant_id: "str | None" = None,
) -> None:
    """Invoke cache.store(); fail-open on error.

    SECURITY (WP-D): tenant_id is forwarded.  None means skip (unauthenticated).

    Backward compat: a genuinely legacy hook (no tenant_id parameter and no
    **kwargs) is called without the kwarg.  A tenant-aware hook is NEVER
    downgraded to the tenant-less path — see _accepts_tenant_id (Stage-9 R1).
    """
    if tenant_id is None:
        return
    try:
        if _accepts_tenant_id(hooks.cache.store):
            hooks.cache.store(ctx, resp, tenant_id=tenant_id)
        else:
            hooks.cache.store(ctx, resp)
    except Exception as exc:
        # SECURITY: never retry a tenant-aware hook without tenant_id.
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
