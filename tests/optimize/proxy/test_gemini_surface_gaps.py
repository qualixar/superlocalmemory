"""Gap coverage for gemini_surface.py — SSE parsers, cache/compress paths, compat surface."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi import FastAPI

from superlocalmemory.optimize.config.schema import OptimizeConfig
from superlocalmemory.optimize.proxy._helpers import _MockTransport
from superlocalmemory.optimize.proxy.gemini_surface import (
    _gemini_sse_from_cached_json,
    _parse_gemini_sse_to_json,
)
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

def _gemini_headers() -> dict:
    return {"x-goog-api-key": "test-key", "content-type": "application/json"}


def _gemini_body() -> dict:
    return {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]}


def _cached_gemini_bytes(text: str = "Hi there", model_version: str = "") -> bytes:
    doc: dict = {
        "candidates": [{
            "content": {"parts": [{"text": text}], "role": "model"},
            "finishReason": "STOP",
            "index": 0,
        }],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3},
    }
    if model_version:
        doc["modelVersion"] = model_version
    return json.dumps(doc).encode()


async def _collect_stream(response) -> bytes:
    from fastapi.responses import StreamingResponse
    assert isinstance(response, StreamingResponse)
    chunks: list[bytes] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode())
    return b"".join(chunks)


# ---------------------------------------------------------------------------
# _parse_gemini_sse_to_json — lines 86-142
# ---------------------------------------------------------------------------

def _build_gemini_sse(text: str = "hello world", finish_reason: str = "STOP") -> bytes:
    chunk1 = {"candidates": [{"content": {"parts": [{"text": text[:5]}], "role": "model"}, "index": 0}]}
    chunk2 = {
        "candidates": [
            {
                "content": {"parts": [{"text": text[5:]}], "role": "model"},
                "finishReason": finish_reason,
                "index": 0,
            }
        ],
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 5},
    }
    return (
        f"data: {json.dumps(chunk1)}\n\n"
        f"data: {json.dumps(chunk2)}\n\n"
    ).encode()


def test_parse_gemini_sse_to_json_complete_stream() -> None:
    """Complete Gemini SSE stream → valid JSON bytes with candidates."""
    sse = _build_gemini_sse("hello world")
    result = _parse_gemini_sse_to_json(sse)

    assert result is not None
    parsed = json.loads(result)
    assert "candidates" in parsed
    text = "".join(p.get("text", "") for p in parsed["candidates"][0]["content"]["parts"])
    assert "hello world" in text


def test_parse_gemini_sse_to_json_no_finish_reason_returns_none() -> None:
    """No finishReason in any candidate → incomplete → None."""
    chunk = {"candidates": [{"content": {"parts": [{"text": "hi"}], "role": "model"}, "index": 0}]}
    sse = f"data: {json.dumps(chunk)}\n\n".encode()
    assert _parse_gemini_sse_to_json(sse) is None


def test_parse_gemini_sse_to_json_empty_bytes_returns_none() -> None:
    assert _parse_gemini_sse_to_json(b"") is None


def test_parse_gemini_sse_to_json_bad_json_line_skipped() -> None:
    """Malformed data: line is skipped; rest of stream still processed."""
    good_chunk = json.dumps({
        "candidates": [{"content": {"parts": [{"text": "ok"}], "role": "model"},
                        "finishReason": "STOP", "index": 0}],
        "usageMetadata": {},
    })
    sse = b"data: {invalid json}\n\ndata: " + good_chunk.encode() + b"\n\n"
    result = _parse_gemini_sse_to_json(sse)
    assert result is not None


def test_parse_gemini_sse_to_json_done_sentinel_skipped() -> None:
    """[DONE] line is skipped without error."""
    sse = _build_gemini_sse("hello") + b"data: [DONE]\n\n"
    result = _parse_gemini_sse_to_json(sse)
    assert result is not None


def test_parse_gemini_sse_to_json_usage_metadata_captured() -> None:
    """usageMetadata from last chunk appears in result."""
    sse = _build_gemini_sse("test")
    result = _parse_gemini_sse_to_json(sse)
    parsed = json.loads(result)
    assert "usageMetadata" in parsed


def test_parse_gemini_sse_to_json_model_version_captured() -> None:
    """modelVersion field is captured when present."""
    chunk = {
        "candidates": [{"content": {"parts": [{"text": "x"}], "role": "model"},
                        "finishReason": "STOP", "index": 0}],
        "usageMetadata": {},
        "modelVersion": "gemini-3.5-flash-001",
    }
    sse = f"data: {json.dumps(chunk)}\n\n".encode()
    result = _parse_gemini_sse_to_json(sse)
    assert result is not None
    assert json.loads(result)["modelVersion"] == "gemini-3.5-flash-001"


def test_parse_gemini_sse_to_json_finish_reason_unspecified_ignored() -> None:
    """FINISH_REASON_UNSPECIFIED is treated as no finish → None."""
    chunk = {
        "candidates": [
            {"content": {"parts": [{"text": "hi"}], "role": "model"},
             "finishReason": "FINISH_REASON_UNSPECIFIED", "index": 0}
        ]
    }
    sse = f"data: {json.dumps(chunk)}\n\n".encode()
    result = _parse_gemini_sse_to_json(sse)
    assert result is None


def test_parse_gemini_sse_to_json_no_candidates_with_finish_returns_none() -> None:
    """Stream has finishReason but no text parts → candidates list is empty → None."""
    chunk = {
        "candidates": [
            {"content": {"parts": [], "role": "model"}, "finishReason": "STOP", "index": 0}
        ],
        "usageMetadata": {},
    }
    sse = f"data: {json.dumps(chunk)}\n\n".encode()
    result = _parse_gemini_sse_to_json(sse)
    # parts_by_idx is empty, so candidates list is empty after assembly → None
    assert result is None


# ---------------------------------------------------------------------------
# _gemini_sse_from_cached_json — lines 151-193
# ---------------------------------------------------------------------------

def test_gemini_sse_from_cached_json_invalid_json_returns_none() -> None:
    assert _gemini_sse_from_cached_json(b"not json") is None


def test_gemini_sse_from_cached_json_no_candidates_returns_none() -> None:
    payload = json.dumps({"result": "ok"}).encode()
    assert _gemini_sse_from_cached_json(payload) is None


@pytest.mark.asyncio
async def test_gemini_sse_from_cached_json_text_events_emitted() -> None:
    """Valid cached JSON → StreamingResponse with data: lines."""
    resp = _gemini_sse_from_cached_json(_cached_gemini_bytes("hello world"))
    body = await _collect_stream(resp)

    assert b"data:" in body
    assert b"hello world" in body
    assert b"STOP" in body


@pytest.mark.asyncio
async def test_gemini_sse_from_cached_json_long_text_multiple_chunks() -> None:
    """Text > 100 chars → multiple intermediate data: chunks."""
    long_text = "W" * 250
    resp = _gemini_sse_from_cached_json(_cached_gemini_bytes(long_text))
    body = await _collect_stream(resp)

    # There should be multiple data: lines for the text
    data_count = body.count(b"data: ")
    assert data_count >= 3


@pytest.mark.asyncio
async def test_gemini_sse_from_cached_json_model_version_in_final_chunk() -> None:
    """modelVersion is included in the final chunk (line 190)."""
    resp = _gemini_sse_from_cached_json(_cached_gemini_bytes("hi", model_version="gemini-1.5-flash"))
    body = await _collect_stream(resp)
    assert b"gemini-1.5-flash" in body


# ---------------------------------------------------------------------------
# handle_gemini_native — body parse fallback (lines 235-236)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gemini_native_invalid_json_body_falls_back_to_empty_dict() -> None:
    """Invalid JSON body → body = {} fallback (lines 235-236), request proceeds."""
    async def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": [], "usageMetadata": {}})

    config = OptimizeConfig.from_dict({})
    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain.empty()
    proxy.http_client = httpx.AsyncClient(transport=_MockTransport(handler=_handler))
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1beta/models/gemini-3.5-flash:generateContent",
            content=b"{bad json",
            headers=_gemini_headers(),
        )

    assert resp.status_code == 200
    await proxy.http_client.aclose()


# ---------------------------------------------------------------------------
# handle_gemini_native streaming — cache hit path (lines 261-272)
# ---------------------------------------------------------------------------

class _GeminiCacheHit:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def check(self, ctx: ProxyRequest) -> CachedResponse:
        return CachedResponse(hit=True, data=self._data, cache_key="gk1", ttl_seconds=300)

    def store(self, ctx: ProxyRequest, resp: ProviderResponse) -> None:
        pass

    def on_hit(self, ctx: ProxyRequest, resp: bytes, tokens_saved: int) -> None:
        pass

    def on_miss(self, ctx: ProxyRequest) -> None:
        pass


@pytest.mark.asyncio
async def test_gemini_native_streaming_cache_hit_replays_sse() -> None:
    """Streaming Gemini native request with cache hit → SSE replay from cached JSON."""
    config = OptimizeConfig.from_dict({"cache_enabled": True})
    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain(
        cache=_GeminiCacheHit(_cached_gemini_bytes("cached text")),
        compress=None,
    )
    proxy.http_client = None
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1beta/models/gemini-3.5-flash:streamGenerateContent",
            json=_gemini_body(),
            headers=_gemini_headers(),
        )

    assert resp.status_code == 200
    assert b"cached text" in resp.content


@pytest.mark.asyncio
async def test_gemini_native_streaming_cache_hit_garbage_falls_through() -> None:
    """Cache hit with unparseable data → falls through to upstream."""
    async def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=b"data: live\n\n")

    class _GarbageHit:
        def check(self, ctx: ProxyRequest) -> CachedResponse:
            return CachedResponse(hit=True, data=b"not json", cache_key="k", ttl_seconds=300)
        def store(self, ctx, resp): pass
        def on_hit(self, ctx, resp, ts): pass
        def on_miss(self, ctx): pass

    config = OptimizeConfig.from_dict({"cache_enabled": True})
    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain(cache=_GarbageHit(), compress=None)
    proxy.http_client = httpx.AsyncClient(transport=_MockTransport(handler=_handler))
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1beta/models/gemini-3.5-flash:streamGenerateContent",
            json=_gemini_body(), headers=_gemini_headers(),
        )

    assert resp.status_code == 200
    await proxy.http_client.aclose()


# ---------------------------------------------------------------------------
# handle_gemini_native streaming — compress path (lines 277-279)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gemini_native_streaming_compress_modifies_body() -> None:
    """Streaming Gemini + compress hook → compressed body sent to upstream."""
    captured_body: list[bytes] = []

    async def _handler(request: httpx.Request) -> httpx.Response:
        captured_body.append(request.content)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=b"data: ok\n\n"
        )

    original = json.dumps(_gemini_body()).encode()
    compressed = b'{"compressed":true}'

    class _CompressHook:
        def compress(self, ctx: ProxyRequest) -> ProxyRequest:
            return ProxyRequest(
                provider=ctx.provider, method=ctx.method, path=ctx.path,
                headers=ctx.headers, body=ctx.body, body_bytes=compressed,
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
            "/v1beta/models/gemini-3.5-flash:streamGenerateContent",
            content=original, headers={**_gemini_headers(), "content-length": str(len(original))},
        )

    assert compressed in captured_body
    await proxy.http_client.aclose()


# ---------------------------------------------------------------------------
# handle_gemini_native streaming — cache store callback (lines 294-307)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gemini_native_streaming_store_callback_fires() -> None:
    """Streaming Gemini + cache miss → store() called after stream completes."""
    stored: list[bytes] = []

    class _TrackingHook:
        def check(self, ctx: ProxyRequest) -> CachedResponse:
            return CachedResponse(hit=False, data=None, cache_key="gk2", ttl_seconds=300)
        def store(self, ctx: ProxyRequest, resp: ProviderResponse) -> None:
            stored.append(resp.body_bytes)
        def on_hit(self, ctx, resp, ts): pass
        def on_miss(self, ctx): pass

    sse_bytes = (
        b'data: {"candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"},'
        b'"finishReason":"STOP","index":0}],"usageMetadata":{}}\n\n'
    )

    async def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=sse_bytes
        )

    config = OptimizeConfig.from_dict({"cache_enabled": True})
    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain(cache=_TrackingHook(), compress=None)
    proxy.http_client = httpx.AsyncClient(transport=_MockTransport(handler=_handler))
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1beta/models/gemini-3.5-flash:streamGenerateContent",
            json=_gemini_body(), headers=_gemini_headers(),
        )

    assert resp.status_code == 200
    assert len(stored) == 1
    parsed = json.loads(stored[0])
    assert "candidates" in parsed

    await proxy.http_client.aclose()


# ---------------------------------------------------------------------------
# handle_gemini_native non-streaming — cache hit (lines 317-322)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gemini_native_non_streaming_cache_hit() -> None:
    """Non-streaming Gemini + cache hit → returns cached bytes directly."""
    config = OptimizeConfig.from_dict({"cache_enabled": True})
    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain(
        cache=_GeminiCacheHit(_cached_gemini_bytes("from cache")),
        compress=None,
    )
    proxy.http_client = None
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1beta/models/gemini-3.5-flash:generateContent",
            json=_gemini_body(), headers=_gemini_headers(),
        )

    assert resp.status_code == 200
    assert b"from cache" in resp.content


# ---------------------------------------------------------------------------
# handle_gemini_native non-streaming — compress path (lines 330-332)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gemini_native_non_streaming_compress_modifies_body() -> None:
    """Non-streaming Gemini + compress → compressed body reaches upstream."""
    captured_body: list[bytes] = []

    async def _handler(request: httpx.Request) -> httpx.Response:
        captured_body.append(request.content)
        return httpx.Response(200, json={"candidates": []})

    original = json.dumps(_gemini_body()).encode()
    compressed = b'{"c":1}'

    class _CompressHook:
        def compress(self, ctx: ProxyRequest) -> ProxyRequest:
            return ProxyRequest(
                provider=ctx.provider, method=ctx.method, path=ctx.path,
                headers=ctx.headers, body=ctx.body, body_bytes=compressed,
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
            "/v1beta/models/gemini-3.5-flash:generateContent",
            content=original, headers={**_gemini_headers(), "content-length": str(len(original))},
        )

    assert compressed in captured_body
    await proxy.http_client.aclose()


# ---------------------------------------------------------------------------
# handle_gemini_native non-streaming — cache store on 200 (lines 347-351)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gemini_native_non_streaming_cache_store_on_200() -> None:
    """Non-streaming Gemini 200 + non-empty cache_key → store() called."""
    stored: list[ProviderResponse] = []

    class _StoreHook:
        def check(self, ctx: ProxyRequest) -> CachedResponse:
            return CachedResponse(hit=False, data=None, cache_key="gk3", ttl_seconds=300)
        def store(self, ctx: ProxyRequest, resp: ProviderResponse) -> None:
            stored.append(resp)
        def on_hit(self, ctx, resp, ts): pass
        def on_miss(self, ctx): pass

    async def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": [
            {"content": {"parts": [{"text": "result"}], "role": "model"},
             "finishReason": "STOP", "index": 0}
        ], "usageMetadata": {}})

    config = OptimizeConfig.from_dict({"cache_enabled": True})
    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain(cache=_StoreHook(), compress=None)
    proxy.http_client = httpx.AsyncClient(transport=_MockTransport(handler=_handler))
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1beta/models/gemini-3.5-flash:generateContent",
            json=_gemini_body(), headers=_gemini_headers(),
        )

    assert resp.status_code == 200
    assert len(stored) == 1

    await proxy.http_client.aclose()


# ---------------------------------------------------------------------------
# handle_gemini_openai_compat — JSON decode exception (lines 404-405)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gemini_openai_compat_post_invalid_json_fail_open() -> None:
    """POST with invalid JSON body → fail_open_forward (lines 404-405)."""
    async def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    config = OptimizeConfig.from_dict({})
    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain.empty()
    proxy.http_client = httpx.AsyncClient(transport=_MockTransport(handler=_handler))
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1beta/openai/chat/completions",
            content=b"{invalid json",
            headers={"authorization": "Bearer test", "content-type": "application/json"},
        )

    assert resp.status_code == 200  # fail-open to upstream
    await proxy.http_client.aclose()


# ---------------------------------------------------------------------------
# handle_gemini_openai_compat — cache hit (lines 425-430)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gemini_openai_compat_cache_hit() -> None:
    """POST compat with cache hit → returns cached bytes (lines 425-430)."""
    cached = json.dumps({"choices": [{"message": {"content": "cached"}}]}).encode()

    class _CompatCacheHit:
        def check(self, ctx: ProxyRequest) -> CachedResponse:
            return CachedResponse(hit=True, data=cached, cache_key="ck1", ttl_seconds=300)
        def store(self, ctx, resp): pass
        def on_hit(self, ctx, resp, ts): pass
        def on_miss(self, ctx): pass

    config = OptimizeConfig.from_dict({"cache_enabled": True})
    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain(cache=_CompatCacheHit(), compress=None)
    proxy.http_client = None
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1beta/openai/chat/completions",
            json={"model": "gemini-flash", "messages": [{"role": "user", "content": "hi"}]},
            headers={"authorization": "Bearer test", "content-type": "application/json"},
        )

    assert resp.status_code == 200
    assert b"cached" in resp.content


# ---------------------------------------------------------------------------
# handle_gemini_openai_compat — compress modifies body (lines 439-441)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gemini_openai_compat_compress_modifies_body() -> None:
    """Compat POST + compress hook → compressed body sent to upstream."""
    captured_body: list[bytes] = []

    async def _handler(request: httpx.Request) -> httpx.Response:
        captured_body.append(request.content)
        return httpx.Response(200, json={"choices": []})

    original = json.dumps({"model": "g", "messages": [{"role": "user", "content": "x"}]}).encode()
    compressed = b'{"cmp":1}'

    class _CompressHook:
        def compress(self, ctx: ProxyRequest) -> ProxyRequest:
            return ProxyRequest(
                provider=ctx.provider, method=ctx.method, path=ctx.path,
                headers=ctx.headers, body=ctx.body, body_bytes=compressed,
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
            "/v1beta/openai/chat/completions",
            content=original,
            headers={"authorization": "Bearer test", "content-type": "application/json",
                     "content-length": str(len(original))},
        )

    assert compressed in captured_body
    await proxy.http_client.aclose()


# ---------------------------------------------------------------------------
# handle_gemini_openai_compat — cache store on 200 (lines 463-467)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gemini_openai_compat_cache_store_on_200() -> None:
    """Compat POST 200 + non-empty cache_key → store() called (lines 463-467)."""
    stored: list[ProviderResponse] = []

    class _StoreHook:
        def check(self, ctx: ProxyRequest) -> CachedResponse:
            return CachedResponse(hit=False, data=None, cache_key="ck2", ttl_seconds=300)
        def store(self, ctx: ProxyRequest, resp: ProviderResponse) -> None:
            stored.append(resp)
        def on_hit(self, ctx, resp, ts): pass
        def on_miss(self, ctx): pass

    async def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    config = OptimizeConfig.from_dict({"cache_enabled": True})
    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain(cache=_StoreHook(), compress=None)
    proxy.http_client = httpx.AsyncClient(transport=_MockTransport(handler=_handler))
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1beta/openai/chat/completions",
            json={"model": "g", "messages": [{"role": "user", "content": "hi"}]},
            headers={"authorization": "Bearer test", "content-type": "application/json"},
        )

    assert resp.status_code == 200
    assert len(stored) == 1

    await proxy.http_client.aclose()


# ---------------------------------------------------------------------------
# handle_gemini_openai_compat — outer exception fail-open (lines 476-480)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gemini_openai_compat_exception_fail_open() -> None:
    """Outer exception in handle_gemini_openai_compat → fail_open_forward (lines 476-480)."""
    async def _broken_handler(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("upstream totally broken")

    config = OptimizeConfig.from_dict({})
    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain.empty()
    proxy.http_client = httpx.AsyncClient(transport=_MockTransport(handler=_broken_handler))
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1beta/openai/chat/completions",
            json={"model": "g", "messages": []},
            headers={"authorization": "Bearer test", "content-type": "application/json"},
        )

    # fail_open_forward re-tries with broken client → 502
    assert resp.status_code in (502, 200)

    await proxy.http_client.aclose()
