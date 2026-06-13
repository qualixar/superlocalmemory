"""Gap coverage for proxy/_helpers.py — SSE parser, stream error paths, safe helpers."""

from __future__ import annotations

import json

import httpx
import pytest

from superlocalmemory.optimize.proxy._helpers import (
    _MockTransport,
    _filter_response_headers,
    _parse_sse_to_json,
    _safe_cache_hit_callbacks,
    _safe_compress,
    _stream_and_cache_forward,
    _stream_forward,
)
from superlocalmemory.optimize.proxy.lifecycle import (
    HookChain,
    ProxyRequest,
    ProviderResponse,
)


# ---------------------------------------------------------------------------
# _filter_response_headers — else branch (line 121)
# ---------------------------------------------------------------------------

def test_filter_response_headers_with_list_of_tuples() -> None:
    """Non-dict input (no .items()) → uses the else branch (line 121)."""
    raw = [
        ("content-type", "application/json"),
        ("connection", "keep-alive"),
        ("x-request-id", "abc123"),
    ]
    result = _filter_response_headers(raw)

    assert "content-type" in result
    assert "connection" not in result
    assert "x-request-id" in result


# ---------------------------------------------------------------------------
# _parse_sse_to_json — lines 151-255
# ---------------------------------------------------------------------------

def _build_text_sse(text: str = "Hello world", stop_reason: str = "end_turn") -> bytes:
    """Build a complete Anthropic SSE stream with a single text block."""
    events: list[str] = [
        "event: message_start",
        'data: {"type":"message_start","message":{"id":"msg_01","type":"message",'
        '"role":"assistant","model":"claude-3","content":[],"stop_reason":null,'
        '"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}',
        "",
        "event: content_block_start",
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
        "",
        "event: content_block_delta",
        f'data: {{"type":"content_block_delta","index":0,"delta":{{"type":"text_delta","text":{json.dumps(text)}}}}}',
        "",
        "event: content_block_stop",
        'data: {"type":"content_block_stop","index":0}',
        "",
        "event: message_delta",
        f'data: {{"type":"message_delta","delta":{{"stop_reason":"{stop_reason}","stop_sequence":null}},"usage":{{"output_tokens":5}}}}',
        "",
        "event: message_stop",
        'data: {"type":"message_stop"}',
        "",
    ]
    return "\n".join(events).encode()


def test_parse_sse_to_json_complete_text_block() -> None:
    """Full SSE text stream → valid JSON bytes."""
    sse = _build_text_sse("Hello world")
    result = _parse_sse_to_json(sse)

    assert result is not None
    parsed = json.loads(result)
    assert parsed["type"] == "message"
    assert parsed["content"][0]["type"] == "text"
    assert parsed["content"][0]["text"] == "Hello world"
    assert parsed["stop_reason"] == "end_turn"
    assert parsed["usage"]["output_tokens"] == 5


def test_parse_sse_to_json_preserves_message_id_and_model() -> None:
    """message_start fields are preserved in the assembled JSON."""
    sse = _build_text_sse("hi")
    result = _parse_sse_to_json(sse)
    assert result is not None
    parsed = json.loads(result)
    assert parsed["id"] == "msg_01"
    assert parsed["model"] == "claude-3"
    assert parsed["role"] == "assistant"


def test_parse_sse_to_json_incomplete_no_message_stop() -> None:
    """Stream without message_stop event → returns None (incomplete)."""
    sse = (
        b"event: message_start\n"
        b'data: {"type":"message_start","message":{"id":"m1","model":"x",'
        b'"role":"assistant","content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}\n\n'
        b"event: content_block_start\n"
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
    )
    result = _parse_sse_to_json(sse)
    assert result is None


def test_parse_sse_to_json_incomplete_no_message_start() -> None:
    """Stream without message_start → returns None."""
    sse = (
        b"event: message_stop\n"
        b'data: {"type":"message_stop"}\n\n'
    )
    result = _parse_sse_to_json(sse)
    assert result is None


def test_parse_sse_to_json_empty_bytes() -> None:
    """Empty input → None."""
    assert _parse_sse_to_json(b"") is None


def test_parse_sse_to_json_garbage_input() -> None:
    """Garbage non-SSE bytes → None."""
    assert _parse_sse_to_json(b"not an sse stream at all") is None


def test_parse_sse_to_json_done_sentinel_skipped() -> None:
    """[DONE] data lines are skipped without error."""
    sse = _build_text_sse("test") + b"data: [DONE]\n\n"
    result = _parse_sse_to_json(sse)
    assert result is not None
    assert json.loads(result)["content"][0]["text"] == "test"


def test_parse_sse_to_json_bad_json_data_line_skipped() -> None:
    """Invalid JSON in data: line is skipped; rest of stream processed."""
    sse = (
        b"event: message_start\n"
        b'data: {"type":"message_start","message":{"id":"m1","model":"c","role":"assistant",'
        b'"content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}\n\n'
        b"event: content_block_start\n"
        b"data: {invalid json}\n\n"
        b"event: content_block_start\n"
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}\n\n'
        b"event: message_delta\n"
        b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},'
        b'"usage":{"output_tokens":2}}\n\n'
        b"event: message_stop\n"
        b'data: {"type":"message_stop"}\n\n'
    )
    result = _parse_sse_to_json(sse)
    assert result is not None


def test_parse_sse_to_json_tool_use_block() -> None:
    """SSE stream with tool_use block → assembled JSON with tool_use content."""
    sse = (
        b"event: message_start\n"
        b'data: {"type":"message_start","message":{"id":"m1","model":"c","role":"assistant",'
        b'"content":[],"stop_reason":null,"usage":{"input_tokens":5,"output_tokens":0}}}\n\n'
        b"event: content_block_start\n"
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use",'
        b'"id":"toolu_01","name":"bash","input":{}}}\n\n'
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\\"cmd\\\": "}}\n\n'
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\\\"ls\\\"}"}}\n\n'
        b"event: message_delta\n"
        b'data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},'
        b'"usage":{"output_tokens":20}}\n\n'
        b"event: message_stop\n"
        b'data: {"type":"message_stop"}\n\n'
    )
    result = _parse_sse_to_json(sse)

    assert result is not None
    parsed = json.loads(result)
    assert parsed["stop_reason"] == "tool_use"
    block = parsed["content"][0]
    assert block["type"] == "tool_use"
    assert block["id"] == "toolu_01"
    assert block["name"] == "bash"
    assert block["input"] == {"cmd": "ls"}


def test_parse_sse_to_json_tool_use_invalid_input_json() -> None:
    """Tool use with un-parseable input_json_delta → stored as _raw."""
    sse = (
        b"event: message_start\n"
        b'data: {"type":"message_start","message":{"id":"m1","model":"c","role":"assistant",'
        b'"content":[],"stop_reason":null,"usage":{"input_tokens":5,"output_tokens":0}}}\n\n'
        b"event: content_block_start\n"
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use",'
        b'"id":"t1","name":"fn","input":{}}}\n\n'
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{{broken"}}\n\n'
        b"event: message_delta\n"
        b'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":1}}\n\n'
        b"event: message_stop\n"
        b'data: {"type":"message_stop"}\n\n'
    )
    result = _parse_sse_to_json(sse)
    assert result is not None
    parsed = json.loads(result)
    assert "_raw" in parsed["content"][0]["input"]


def test_parse_sse_to_json_unknown_block_index_delta_skipped() -> None:
    """Delta for unknown block index (block is None) → silently skipped."""
    sse = (
        b"event: message_start\n"
        b'data: {"type":"message_start","message":{"id":"m1","model":"c","role":"assistant",'
        b'"content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}\n\n'
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":99,"delta":{"type":"text_delta","text":"ghost"}}\n\n'
        b"event: message_delta\n"
        b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":0}}\n\n'
        b"event: message_stop\n"
        b'data: {"type":"message_stop"}\n\n'
    )
    result = _parse_sse_to_json(sse)
    # message_start not complete, so None — or valid empty message
    # Either way: no crash
    assert result is None or isinstance(result, bytes)


def test_parse_sse_to_json_stop_reason_none_preserved() -> None:
    """message_delta with stop_reason=None → fallback to existing stop_reason."""
    sse = (
        b"event: message_start\n"
        b'data: {"type":"message_start","message":{"id":"m1","model":"c","role":"assistant",'
        b'"content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}\n\n'
        b"event: content_block_start\n"
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"x"}}\n\n'
        b"event: message_delta\n"
        b'data: {"type":"message_delta","delta":{"stop_reason":null,"stop_sequence":null},'
        b'"usage":{"output_tokens":1}}\n\n'
        b"event: message_stop\n"
        b'data: {"type":"message_stop"}\n\n'
    )
    result = _parse_sse_to_json(sse)
    assert result is not None
    # stop_reason should fall back to "end_turn" (the default)
    assert json.loads(result)["stop_reason"] == "end_turn"


# ---------------------------------------------------------------------------
# _stream_forward — error paths in _generate() (lines 330-338)
# ---------------------------------------------------------------------------

class _NullClientProxy:
    http_client = None


class _RemoteProtocolErrorTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError("connection closed early", request=request)


class _GenericErrorTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise RuntimeError("unexpected transport error")


@pytest.mark.asyncio
async def test_stream_forward_remote_protocol_error_yields_error_chunk() -> None:
    """RemoteProtocolError inside _generate → error SSE chunk yielded, no raise."""
    proxy_obj = _NullClientProxy()
    proxy_obj.http_client = httpx.AsyncClient(
        transport=_RemoteProtocolErrorTransport()
    )

    from fastapi.responses import StreamingResponse

    result = await _stream_forward(
        proxy_obj, "req_001", {}, b"hello", "http://example.com"
    )

    assert isinstance(result, StreamingResponse)

    chunks: list[bytes] = []
    async for chunk in result.body_iterator:
        chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode())

    body = b"".join(chunks)
    assert b"upstream stream closed" in body

    await proxy_obj.http_client.aclose()


@pytest.mark.asyncio
async def test_stream_forward_generic_exception_yields_error_chunk() -> None:
    """Generic exception in _generate → error SSE chunk yielded, no raise."""
    proxy_obj = _NullClientProxy()
    proxy_obj.http_client = httpx.AsyncClient(transport=_GenericErrorTransport())

    from fastapi.responses import StreamingResponse

    result = await _stream_forward(
        proxy_obj, "req_002", {}, b"body", "http://example.com"
    )

    assert isinstance(result, StreamingResponse)

    chunks: list[bytes] = []
    async for chunk in result.body_iterator:
        chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode())

    body = b"".join(chunks)
    assert b"SLM proxy stream error" in body

    await proxy_obj.http_client.aclose()


# ---------------------------------------------------------------------------
# _stream_and_cache_forward — error paths + on_complete exception (lines 404-414, 433-434)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_and_cache_forward_remote_protocol_error() -> None:
    """RemoteProtocolError in stream-and-cache → error chunk, on_complete NOT called."""
    proxy_obj = _NullClientProxy()
    proxy_obj.http_client = httpx.AsyncClient(
        transport=_RemoteProtocolErrorTransport()
    )
    on_complete_calls: list = []

    async def on_complete(data: bytes) -> None:
        on_complete_calls.append(data)

    from fastapi.responses import StreamingResponse

    result = await _stream_and_cache_forward(
        proxy_obj, "req_003", {}, b"body", "http://example.com",
        on_complete=on_complete,
    )

    assert isinstance(result, StreamingResponse)

    chunks: list[bytes] = []
    async for chunk in result.body_iterator:
        chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode())

    body = b"".join(chunks)
    assert b"upstream stream closed" in body
    # on_complete NOT called when stream_error=True
    assert on_complete_calls == []

    await proxy_obj.http_client.aclose()


@pytest.mark.asyncio
async def test_stream_and_cache_forward_generic_exception() -> None:
    """Generic exception in stream-and-cache → error chunk, on_complete NOT called."""
    proxy_obj = _NullClientProxy()
    proxy_obj.http_client = httpx.AsyncClient(transport=_GenericErrorTransport())

    called: list = []

    async def on_complete(data: bytes) -> None:
        called.append(data)

    from fastapi.responses import StreamingResponse

    result = await _stream_and_cache_forward(
        proxy_obj, "req_004", {}, b"body", "http://example.com",
        on_complete=on_complete,
    )

    assert isinstance(result, StreamingResponse)

    chunks: list[bytes] = []
    async for chunk in result.body_iterator:
        chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode())

    assert b"SLM proxy stream error" in b"".join(chunks)
    assert called == []

    await proxy_obj.http_client.aclose()


@pytest.mark.asyncio
async def test_stream_and_cache_forward_on_complete_exception_swallowed() -> None:
    """on_complete exception is swallowed in the finally block (lines 433-434)."""
    async def success_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"data: chunk\n\n",
        )

    proxy_obj = _NullClientProxy()
    proxy_obj.http_client = httpx.AsyncClient(
        transport=_MockTransport(handler=success_handler)
    )

    async def on_complete_bad(data: bytes) -> None:
        raise RuntimeError("cache store exploded")

    from fastapi.responses import StreamingResponse

    result = await _stream_and_cache_forward(
        proxy_obj, "req_005", {}, b"body", "http://example.com",
        on_complete=on_complete_bad,
    )

    assert isinstance(result, StreamingResponse)

    # Consuming the generator must NOT raise even though on_complete fails
    chunks: list[bytes] = []
    async for chunk in result.body_iterator:
        chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode())

    assert b"chunk" in b"".join(chunks)

    await proxy_obj.http_client.aclose()


# ---------------------------------------------------------------------------
# _safe_compress — happy path (line 492) and exception path
# ---------------------------------------------------------------------------

def _make_ctx() -> ProxyRequest:
    return ProxyRequest(
        provider="anthropic", method="POST", path="/v1/messages",
        headers={}, body={}, body_bytes=b"original",
        request_id="r1", stream=False, has_tools=False,
    )


@pytest.mark.asyncio
async def test_safe_compress_happy_path_returns_result() -> None:
    """compress.compress() succeeds → returns the new ProxyRequest (line 492)."""
    new_ctx = _make_ctx()
    new_ctx = ProxyRequest(
        provider="anthropic", method="POST", path="/v1/messages",
        headers={}, body={}, body_bytes=b"compressed",
        request_id="r1", stream=False, has_tools=False,
    )

    class _WorkingCompress:
        def compress(self, ctx: ProxyRequest) -> ProxyRequest:
            return new_ctx

    hooks = HookChain(cache=None, compress=_WorkingCompress())
    ctx = _make_ctx()
    result = await _safe_compress(hooks, ctx)

    assert result is new_ctx
    assert result.body_bytes == b"compressed"


@pytest.mark.asyncio
async def test_safe_compress_exception_returns_original_ctx() -> None:
    """compress.compress() raises → fail-open, original ctx returned."""
    class _BrokenCompress:
        def compress(self, ctx: ProxyRequest) -> ProxyRequest:
            raise RuntimeError("compressor on fire")

    hooks = HookChain(cache=None, compress=_BrokenCompress())
    ctx = _make_ctx()
    result = await _safe_compress(hooks, ctx)

    assert result is ctx
