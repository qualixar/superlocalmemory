"""Gap coverage for anthropic_surface.py — SSE replay, streaming paths, cache store."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from superlocalmemory.optimize.config.schema import OptimizeConfig
from superlocalmemory.optimize.proxy._helpers import _MockTransport
from superlocalmemory.optimize.proxy.anthropic_surface import _sse_from_cached_json
from superlocalmemory.optimize.proxy.lifecycle import (
    CachedResponse,
    HookChain,
    ProxyRequest,
    ProviderResponse,
)
from superlocalmemory.optimize.proxy.server import ProxyApp, build_proxy_router


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _anthropic_headers() -> dict:
    return {
        "x-api-key": "sk-ant-test",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }


def _sample_req(stream: bool = False) -> dict:
    d = {"model": "claude-sonnet-4-6", "max_tokens": 100,
         "messages": [{"role": "user", "content": "hi"}]}
    if stream:
        d["stream"] = True
    return d


def _cached_message_bytes(text: str = "cached answer") -> bytes:
    return json.dumps({
        "id": "msg_cached",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": "claude-sonnet-4-6",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }).encode()


def _cached_tool_message_bytes() -> bytes:
    return json.dumps({
        "id": "msg_tool",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_01",
                "name": "bash",
                "input": {"command": "ls -la"},
            }
        ],
        "model": "claude-sonnet-4-6",
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 20, "output_tokens": 10},
    }).encode()


async def _collect_sse(response: StreamingResponse) -> bytes:
    chunks: list[bytes] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode())
    return b"".join(chunks)


# ---------------------------------------------------------------------------
# _sse_from_cached_json — lines 46-157
# ---------------------------------------------------------------------------

def test_sse_from_cached_json_invalid_bytes_returns_none() -> None:
    assert _sse_from_cached_json(b"not json") is None


def test_sse_from_cached_json_wrong_type_returns_none() -> None:
    payload = json.dumps({"type": "error", "error": "something"}).encode()
    assert _sse_from_cached_json(payload) is None


def test_sse_from_cached_json_text_block_returns_streaming_response() -> None:
    """Valid message with text content → StreamingResponse with SSE events."""
    resp = _sse_from_cached_json(_cached_message_bytes("hello"))
    assert isinstance(resp, StreamingResponse)


@pytest.mark.asyncio
async def test_sse_from_cached_json_text_events_contain_content() -> None:
    """SSE output includes message_start, content_block_start, text_delta, message_stop."""
    resp = _sse_from_cached_json(_cached_message_bytes("hello world"))
    assert resp is not None
    body = await _collect_sse(resp)

    assert b"message_start" in body
    assert b"content_block_start" in body
    assert b"content_block_delta" in body
    assert b"message_stop" in body
    assert b"hello world" in body


@pytest.mark.asyncio
async def test_sse_from_cached_json_long_text_emits_multiple_chunks() -> None:
    """Text > 100 chars is split into multiple text_delta events."""
    long_text = "A" * 250  # 3 chunks of 100
    resp = _sse_from_cached_json(_cached_message_bytes(long_text))
    assert resp is not None
    body = await _collect_sse(resp)

    # Count text_delta occurrences — should be at least 3
    delta_count = body.count(b"text_delta")
    assert delta_count >= 3


@pytest.mark.asyncio
async def test_sse_from_cached_json_tool_use_block_emits_input_json_delta() -> None:
    """Tool use content → input_json_delta events are emitted."""
    resp = _sse_from_cached_json(_cached_tool_message_bytes())
    assert resp is not None
    body = await _collect_sse(resp)

    assert b"tool_use" in body
    assert b"input_json_delta" in body
    assert b"bash" in body


@pytest.mark.asyncio
async def test_sse_from_cached_json_empty_content_still_valid() -> None:
    """Message with empty content array → valid SSE (no content blocks emitted)."""
    payload = json.dumps({
        "id": "msg_empty",
        "type": "message",
        "role": "assistant",
        "content": [],
        "model": "claude-3",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 5, "output_tokens": 0},
    }).encode()
    resp = _sse_from_cached_json(payload)
    assert resp is not None
    body = await _collect_sse(resp)
    assert b"message_start" in body
    assert b"message_stop" in body


@pytest.mark.asyncio
async def test_sse_from_cached_json_message_delta_emitted() -> None:
    """message_delta SSE event contains stop_reason."""
    resp = _sse_from_cached_json(_cached_message_bytes())
    assert resp is not None
    body = await _collect_sse(resp)
    assert b"message_delta" in body
    assert b"end_turn" in body


# ---------------------------------------------------------------------------
# handle_messages — body bytes too large (line 180)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_anthropic_messages_body_bytes_too_large_after_body_read() -> None:
    """Body > 10 MB without content-length header → 413 from body-bytes check (line 180)."""
    config = OptimizeConfig.from_dict({})
    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain.empty()
    proxy.http_client = None
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    big = b"x" * (11 * 1024 * 1024)  # 11 MB

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Set content-length to "invalid" so the CL-header check at line 165
        # hits ValueError → pass, and the body-bytes check at 179-180 fires.
        resp = await client.post(
            "/v1/messages",
            content=big,
            headers={**_anthropic_headers(), "content-length": "not-a-number"},
        )

    assert resp.status_code == 413


# ---------------------------------------------------------------------------
# handle_messages streaming — cache hit → SSE replay (lines 216-227)
# ---------------------------------------------------------------------------

class _StreamingCacheHitHook:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def check(self, ctx: ProxyRequest) -> CachedResponse:
        return CachedResponse(
            hit=True, data=self._data, cache_key="k1", ttl_seconds=300
        )

    def store(self, ctx: ProxyRequest, resp: ProviderResponse) -> None:
        pass

    def on_hit(self, ctx: ProxyRequest, resp: bytes, tokens_saved: int) -> None:
        pass

    def on_miss(self, ctx: ProxyRequest) -> None:
        pass


@pytest.mark.asyncio
async def test_anthropic_streaming_cache_hit_returns_sse() -> None:
    """Streaming request with cache hit → SSE replay from _sse_from_cached_json."""
    config = OptimizeConfig.from_dict({"cache_enabled": True})
    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain(
        cache=_StreamingCacheHitHook(_cached_message_bytes("from cache")),
        compress=None,
    )
    proxy.http_client = None  # no upstream should be touched
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/messages",
            json=_sample_req(stream=True),
            headers=_anthropic_headers(),
        )

    assert resp.status_code == 200
    assert b"message_start" in resp.content
    assert b"from cache" in resp.content


@pytest.mark.asyncio
async def test_anthropic_streaming_cache_hit_with_unparseable_data_falls_through() -> None:
    """Streaming cache hit with garbage cached bytes → falls through to upstream."""

    async def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"data: live\n\n",
        )

    class _GarbageCacheHit:
        def check(self, ctx: ProxyRequest) -> CachedResponse:
            return CachedResponse(hit=True, data=b"not valid json", cache_key="k1", ttl_seconds=300)
        def store(self, ctx: ProxyRequest, resp: ProviderResponse) -> None:
            pass
        def on_hit(self, ctx: ProxyRequest, resp: bytes, tokens_saved: int) -> None:
            pass
        def on_miss(self, ctx: ProxyRequest) -> None:
            pass

    config = OptimizeConfig.from_dict({"cache_enabled": True})
    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain(cache=_GarbageCacheHit(), compress=None)
    proxy.http_client = httpx.AsyncClient(transport=_MockTransport(handler=_handler))
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/messages",
            json=_sample_req(stream=True),
            headers=_anthropic_headers(),
        )

    # Falls through to upstream → live content returned
    assert resp.status_code == 200
    await proxy.http_client.aclose()


# ---------------------------------------------------------------------------
# handle_messages streaming — compress modifies outbound_bytes (lines 236-238)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_anthropic_streaming_compress_modifies_body() -> None:
    """Streaming + compress hook that changes body → outbound_bytes updated (line 238)."""
    captured_body: list[bytes] = []

    async def _handler(request: httpx.Request) -> httpx.Response:
        captured_body.append(request.content)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=b"data: ok\n\n"
        )

    original_body = json.dumps(_sample_req(stream=True)).encode()
    compressed_body = b'{"compressed":true}'

    class _CompressHook:
        def compress(self, ctx: ProxyRequest) -> ProxyRequest:
            return ProxyRequest(
                provider=ctx.provider, method=ctx.method, path=ctx.path,
                headers=ctx.headers, body=ctx.body, body_bytes=compressed_body,
                request_id=ctx.request_id, stream=ctx.stream, has_tools=ctx.has_tools,
            )

    config = OptimizeConfig.from_dict({"compress_enabled": True})
    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain(cache=None, compress=_CompressHook())
    proxy.http_client = httpx.AsyncClient(transport=_MockTransport(handler=_handler))
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.post(
            "/v1/messages",
            content=original_body,
            headers={**_anthropic_headers(), "content-length": str(len(original_body))},
        )

    # Upstream should have received the compressed body
    assert compressed_body in captured_body

    await proxy.http_client.aclose()


# ---------------------------------------------------------------------------
# handle_messages streaming — store callback setup (lines 250-261)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_anthropic_streaming_store_callback_called() -> None:
    """Streaming + cache enabled + miss → store callback fires after stream ends."""
    stored: list[bytes] = []

    class _TrackingCacheHook:
        def check(self, ctx: ProxyRequest) -> CachedResponse:
            return CachedResponse(hit=False, data=None, cache_key="k2", ttl_seconds=300)

        def store(self, ctx: ProxyRequest, resp: ProviderResponse) -> None:
            stored.append(resp.body_bytes)

        def on_hit(self, ctx: ProxyRequest, resp: bytes, tokens_saved: int) -> None:
            pass

        def on_miss(self, ctx: ProxyRequest) -> None:
            pass

    # Build a minimal complete Anthropic SSE response
    sse_content = (
        b"event: message_start\n"
        b'data: {"type":"message_start","message":{"id":"m1","model":"claude-3",'
        b'"role":"assistant","content":[],"stop_reason":null,'
        b'"usage":{"input_tokens":5,"output_tokens":0}}}\n\n'
        b"event: content_block_start\n"
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}\n\n'
        b"event: message_delta\n"
        b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}\n\n'
        b"event: message_stop\n"
        b'data: {"type":"message_stop"}\n\n'
    )

    async def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse_content,
        )

    config = OptimizeConfig.from_dict({"cache_enabled": True})
    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain(cache=_TrackingCacheHook(), compress=None)
    proxy.http_client = httpx.AsyncClient(transport=_MockTransport(handler=_handler))
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/messages",
            json=_sample_req(stream=True),
            headers=_anthropic_headers(),
        )

    assert resp.status_code == 200
    # store() must have been called with the parsed SSE JSON
    assert len(stored) == 1
    parsed = json.loads(stored[0])
    assert parsed["type"] == "message"
    assert parsed["content"][0]["text"] == "hi"

    await proxy.http_client.aclose()


# ---------------------------------------------------------------------------
# handle_messages non-streaming — cache store on 200 (line 286)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_anthropic_non_streaming_cache_store_on_200() -> None:
    """Non-streaming 200 with non-empty cache_key → store() called (line 286)."""
    stored: list[ProviderResponse] = []

    class _StoringCacheHook:
        def check(self, ctx: ProxyRequest) -> CachedResponse:
            return CachedResponse(hit=False, data=None, cache_key="key_abc", ttl_seconds=300)

        def store(self, ctx: ProxyRequest, resp: ProviderResponse) -> None:
            stored.append(resp)

        def on_hit(self, ctx: ProxyRequest, resp: bytes, tokens_saved: int) -> None:
            pass

        def on_miss(self, ctx: ProxyRequest) -> None:
            pass

    async def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "id": "msg_01", "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": "answer"}],
            "model": "claude-sonnet-4-6", "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })

    config = OptimizeConfig.from_dict({"cache_enabled": True})
    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain(cache=_StoringCacheHook(), compress=None)
    proxy.http_client = httpx.AsyncClient(transport=_MockTransport(handler=_handler))
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/messages", json=_sample_req(), headers=_anthropic_headers()
        )

    assert resp.status_code == 200
    assert len(stored) == 1  # store() was called

    await proxy.http_client.aclose()


# ---------------------------------------------------------------------------
# handle_messages non-streaming — compress modifies outbound_bytes (line 286)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_anthropic_non_streaming_compress_modifies_body() -> None:
    """Non-streaming + compress hook that changes body → compressed body sent (line 286)."""
    captured_body: list[bytes] = []

    async def _handler(request: httpx.Request) -> httpx.Response:
        captured_body.append(request.content)
        return httpx.Response(200, json={
            "id": "msg_01", "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "model": "claude-sonnet-4-6", "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 3},
        })

    original_body = json.dumps(_sample_req()).encode()
    compressed_body = b'{"compressed":true}'

    class _CompressHook:
        def compress(self, ctx: ProxyRequest) -> ProxyRequest:
            return ProxyRequest(
                provider=ctx.provider, method=ctx.method, path=ctx.path,
                headers=ctx.headers, body=ctx.body, body_bytes=compressed_body,
                request_id=ctx.request_id, stream=ctx.stream, has_tools=ctx.has_tools,
            )

    config = OptimizeConfig.from_dict({"compress_enabled": True})
    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain(cache=None, compress=_CompressHook())
    proxy.http_client = httpx.AsyncClient(transport=_MockTransport(handler=_handler))
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.post(
            "/v1/messages",
            content=original_body,
            headers={**_anthropic_headers(), "content-length": str(len(original_body))},
        )

    # Upstream should have received the compressed body
    assert compressed_body in captured_body
    await proxy.http_client.aclose()
