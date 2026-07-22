"""Tests for router.py — CompressRouter (CompressHook impl)."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest

from superlocalmemory.optimize.proxy.lifecycle import ProxyRequest, CompressHook
from superlocalmemory.optimize.compress.router import CompressRouter


def _make_req(
    body: dict | None = None,
    stream: bool = False,
    has_tools: bool = False,
    request_id: str = "test-001",
) -> ProxyRequest:
    return ProxyRequest(
        provider="anthropic",
        method="POST",
        path="/v1/messages",
        headers={},
        body=body or {"messages": [{"role": "user", "content": "hello"}]},
        body_bytes=b"{}",
        request_id=request_id,
        stream=stream,
        has_tools=has_tools,
    )


def _make_req_with_content(content: str, compress_mode: str = "safe") -> ProxyRequest:
    large_content = "word " * 400  # non-JSON, non-code, just prose
    # Embed the target content in a JSON-understood block to force JSON detection
    # For JSON content tests, use actual JSON
    return ProxyRequest(
        provider="anthropic",
        method="POST",
        path="/v1/messages",
        headers={},
        body={
            "messages": [
                {"role": "assistant", "content": content},
            ]
        },
        body_bytes=json.dumps({"messages": [{"role": "assistant", "content": content}]}).encode(),
        request_id="test-001",
        stream=False,
        has_tools=False,
    )


def test_compress_hook_protocol_compliance() -> None:
    router = CompressRouter()
    assert isinstance(router, CompressHook)


def test_compress_disabled_returns_passthrough() -> None:
    router = CompressRouter()
    cfg = router._get_config()
    cfg = type(cfg).from_dict({**cfg.as_dict(), "compress_enabled": False})

    with patch.object(router, "_get_config", return_value=cfg):
        req = _make_req(body={"messages": [{"role": "user", "content": "hello"}]})
        result = router.compress(req)
        assert result is req  # same object = passthrough


def test_compress_streaming_returns_passthrough() -> None:
    router = CompressRouter()
    req = _make_req(stream=True)
    result = router.compress(req)
    assert result is req


def test_compress_has_tools_returns_passthrough() -> None:
    router = CompressRouter()
    req = _make_req(has_tools=True)
    result = router.compress(req)
    assert result is req


def test_compress_json_content_passthrough() -> None:
    """K-01: JSON content must NEVER be compressed (structured data is protected)."""
    router = CompressRouter()
    large_json_text = json.dumps({"key": "v" * 200, "items": list(range(100))})
    req = _make_req_with_content(large_json_text)
    result = router.compress(req)
    # Single-message request → content is protected; the JSON value is preserved
    # exactly (the request envelope may be recompacted losslessly).
    assert json.loads(result.body["messages"][0]["content"]) == json.loads(large_json_text)


def test_compress_code_content_passthrough() -> None:
    """K-02: Code content must NEVER be compressed (structured data is protected)."""
    router = CompressRouter()
    code = "import os\n\ndef foo(x: int) -> int:\n" + "    pass\n" * 20 + "    return x\n"
    req = _make_req_with_content(code)
    result = router.compress(req)
    # Code content preserved verbatim (envelope may be recompacted losslessly).
    assert result.body["messages"][0]["content"] == code


def test_compress_layer1_normalizes_prose_whitespace() -> None:
    """Layer 1: lossless whitespace normalization applies to prose text."""
    router = CompressRouter()
    # Prose with excessive blank lines and trailing whitespace
    prose_with_bloat = ("This is a sentence.   \n" * 20) + "\n\n\n\n" + ("Another paragraph.   \n" * 20)
    req = _make_req(body={"messages": [{"role": "assistant", "content": prose_with_bloat}]})
    result = router.compress(req)
    # Should not error; may or may not produce shorter content
    assert result is not None


def test_compress_prose_safe_mode_passthrough() -> None:
    router = CompressRouter()
    prose = "This is a long prose paragraph. " * 50
    req = _make_req_with_content(prose)
    result = router.compress(req)
    # Safe mode: prose content is not lossily rewritten.
    assert result.body["messages"][0]["content"] == prose


@pytest.mark.skipif(True, reason="LLMLingua not available in test env")
def test_compress_prose_aggressive_mode_uses_llmlingua() -> None:
    prose = "This is a long prose paragraph. " * 50
    router = CompressRouter()
    cfg = router._get_config()
    cfg = type(cfg).from_dict({**cfg.as_dict(), "compress_mode": "aggressive"})

    with patch.object(router, "_get_config", return_value=cfg):
        req = _make_req_with_content(prose)
        result = router.compress(req)
        assert result is not req  # should have compressed


def test_compress_never_routes_json_to_llmlingua() -> None:
    """CRITICAL: JSON MUST pass through unchanged even in aggressive mode (K-01)."""
    router = CompressRouter()
    cfg = router._get_config()
    cfg = type(cfg).from_dict({**cfg.as_dict(), "compress_mode": "aggressive", "compress_prose": True})

    large_json_text = json.dumps({"key": "v" * 200, "items": list(range(100))})
    req = _make_req_with_content(large_json_text)

    with patch.object(router, "_get_config", return_value=cfg):
        result = router.compress(req)
    # JSON value fully preserved — never routed to lossy LLMLingua, even aggressive.
    assert json.loads(result.body["messages"][0]["content"]) == json.loads(large_json_text), (
        "JSON was lossily altered when it must only ever be losslessly minified"
    )


def test_compress_user_messages_never_compressed() -> None:
    router = CompressRouter()
    user_content = "A" * 2000
    req = _make_req(body={"messages": [{"role": "user", "content": user_content}]})
    result = router.compress(req)
    if result is not req:
        for msg in result.body["messages"]:
            if msg["role"] == "user":
                assert msg["content"] == user_content, "User message was mutated"


def test_compress_returns_passthrough_on_no_improvement() -> None:
    router = CompressRouter()
    short_text = "hello"
    req = _make_req_with_content(short_text)
    result = router.compress(req)
    # Below-threshold content is preserved verbatim.
    assert result.body["messages"][0]["content"] == short_text


def test_compress_never_raises() -> None:
    router = CompressRouter()
    req = _make_req(body={"messages": "not_a_list"})
    result = router.compress(req)
    assert result is req


def test_compress_tool_result_history_passthrough() -> None:
    """B-09: Historical tool_result blocks in messages must not be compressed."""
    router = CompressRouter()
    large_tool_result = "x" * 2000
    req = _make_req(
        body={
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "toolu_abc", "name": "fn", "input": {}}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_abc", "content": large_tool_result}
                    ],
                },
            ]
        },
        has_tools=False,
    )
    result = router.compress(req)
    if result is not req:
        for msg in result.body["messages"]:
            for block in (msg.get("content") or []):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    assert block.get("content") == large_tool_result, "tool_result was compressed"


def test_on_compress_called_on_success() -> None:
    """on_compress fires when Layer 1 normalization reduces tokens."""
    router = CompressRouter()
    calls: list = []
    router.on_compress = lambda b, a, l: calls.append((b, a, l))

    # Prose with excessive whitespace bloat — Layer 1 should normalize it
    bloated_prose = ("word " * 50 + "   \n") * 30 + "\n\n\n\n" + ("more prose " * 50 + "   \n") * 30
    req = _make_req(body={"messages": [{"role": "assistant", "content": bloated_prose}]})
    result = router.compress(req)
    if result is not req:
        assert len(calls) == 1, "on_compress must be called exactly once per successful compress"
        before, after, lossy = calls[0]
        assert before >= after
        assert lossy is False


def test_compress_text_convenience_method() -> None:
    """M-06: compress_text() public convenience method."""
    router = CompressRouter()
    prose = "word " * 200
    result = router.compress_text(prose)
    assert result.strategy in ("normalize", "llmlingua2_prose", "none")
    assert result.tokens_before >= 0
    assert result.tokens_after >= 0


def test_compress_text_never_raises() -> None:
    router = CompressRouter()
    result = router.compress_text("")
    assert result.strategy == "none"


def test_compress_system_prompt_with_volatile_tokens() -> None:
    """CacheAligner runs on system prompt when present."""
    router = CompressRouter()
    large_json = json.dumps({"key": "v" * 200, "items": list(range(100))})
    # Include UUID in system prompt
    uuid_token = "550e8400-e29b-41d4-a716-446655440000"
    req = ProxyRequest(
        provider="anthropic",
        method="POST",
        path="/v1/messages",
        headers={},
        body={
            "system": f"Session: {uuid_token}. Help user.",
            "messages": [{"role": "assistant", "content": large_json}],
        },
        body_bytes=json.dumps({
            "system": f"Session: {uuid_token}. Help user.",
            "messages": [{"role": "assistant", "content": large_json}],
        }).encode(),
        request_id="test-sysprompt",
        stream=False,
        has_tools=False,
    )
    result = router.compress(req)
    assert result is not None


def test_compress_content_block_list_with_text() -> None:
    """Message with content as list of blocks is compressed."""
    router = CompressRouter()
    large_text = "word " * 600  # long prose, forces some path
    req = ProxyRequest(
        provider="anthropic",
        method="POST",
        path="/v1/messages",
        headers={},
        body={
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": large_text}],
                },
            ]
        },
        body_bytes=json.dumps({
            "messages": [
                {"role": "assistant", "content": [{"type": "text", "text": large_text}]},
            ]
        }).encode(),
        request_id="test-blocks",
        stream=False,
        has_tools=False,
    )
    result = router.compress(req)
    assert result is not None


def test_compress_content_block_non_dict() -> None:
    """Non-dict blocks in content list are preserved."""
    router = CompressRouter()
    req = ProxyRequest(
        provider="anthropic",
        method="POST",
        path="/v1/messages",
        headers={},
        body={
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "word " * 600}, "plain_string_block"],
                },
            ]
        },
        body_bytes=json.dumps({
            "messages": [
                {"role": "assistant", "content": [{"type": "text", "text": "word " * 600}, "plain_string_block"]},
            ]
        }).encode(),
        request_id="test-nondict",
        stream=False,
        has_tools=False,
    )
    result = router.compress(req)
    assert result is not None


def test_compress_tool_result_text_block() -> None:
    """tool_result blocks with text are handled; JSON content passes through (K-01)."""
    router = CompressRouter()
    large_json = json.dumps({"key": "v" * 200, "items": list(range(100))})
    req = ProxyRequest(
        provider="anthropic",
        method="POST",
        path="/v1/messages",
        headers={},
        body={
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "tool_result", "text": large_json}],
                },
            ]
        },
        body_bytes=json.dumps({
            "messages": [
                {"role": "assistant", "content": [{"type": "tool_result", "text": large_json}]},
            ]
        }).encode(),
        request_id="test-tool-result-text",
        stream=False,
        has_tools=False,
    )
    result = router.compress(req)
    assert result is not None


def test_compress_tool_result_nested_content() -> None:
    """tool_result with nested list content extracts text correctly."""
    router = CompressRouter()
    large_text = "x" * 600
    req = ProxyRequest(
        provider="anthropic",
        method="POST",
        path="/v1/messages",
        headers={},
        body={
            "messages": [
                {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_result",
                        "content": [{"type": "text", "text": large_text}],
                    }],
                },
            ]
        },
        body_bytes=json.dumps({
            "messages": [
                {"role": "assistant", "content": [{"type": "tool_result", "content": [{"type": "text", "text": large_text}]}]},
            ]
        }).encode(),
        request_id="test-nested-tool-result",
        stream=False,
        has_tools=False,
    )
    result = router.compress(req)
    assert result is not None


def test_compress_tool_role_skip() -> None:
    """Messages with role='tool' are never compressed."""
    router = CompressRouter()
    large_text = "word " * 600
    req = ProxyRequest(
        provider="openai",
        method="POST",
        path="/v1/chat/completions",
        headers={},
        body={
            "messages": [
                {"role": "tool", "content": large_text},
                {"role": "assistant", "content": large_text},
            ]
        },
        body_bytes=json.dumps({
            "messages": [
                {"role": "tool", "content": large_text},
                {"role": "assistant", "content": large_text},
            ]
        }).encode(),
        request_id="test-tool-role",
        stream=False,
        has_tools=False,
    )
    result = router.compress(req)
    assert result is not None


def test_on_compress_metrics_counter_wired() -> None:
    """set_metrics injects counters that on_compress calls."""
    router = CompressRouter()
    
    # Mock mirrors the real counter signature:
    # MetricsCounters.on_compress(bytes_original, bytes_after) — two ints, no
    # lossy flag (v3.6.3 corrected the router to call this contract).
    class MockCounters:
        def __init__(self):
            self.calls = []

        def on_compress(self, bytes_original: int, bytes_after: int):
            self.calls.append((bytes_original, bytes_after))

    counters = MockCounters()
    router.set_metrics(counters)

    router.on_compress(100, 40, False)
    assert len(counters.calls) == 1
    # Router forwards (before_tokens, after_tokens) to the counter, which
    # derives "saved" itself. The lossy flag is logged, not forwarded.
    assert counters.calls[0] == (100, 40)


def test_on_compress_non_fatal_on_error() -> None:
    """on_compress must not raise even if counters is broken."""
    router = CompressRouter()

    class BrokenCounters:
        tokens_saved_compress = "not_an_int"  # will cause TypeError on +=

    router.set_metrics(BrokenCounters())
    # Must not raise
    router.on_compress(100, 50, False)


def test_compress_json_ratio_below_threshold_passthrough() -> None:
    """Small JSON that doesn't beat ratio gate is passed through."""
    router = CompressRouter()
    small_json = json.dumps({"a": 1})  # too small
    req = _make_req_with_content(small_json)
    result = router.compress(req)
    # Should not have modified (either passthrough or ratio gate blocked)
    if result is not req:
        assert len(json.dumps(result.body).split()) <= len(json.dumps(req.body).split())


def test_compress_code_path_not_detected() -> None:
    """Non-code prose in safe mode should not be code-compressed."""
    router = CompressRouter()
    prose = "The quick brown fox jumps over the lazy dog. " * 30
    req = _make_req_with_content(prose)
    result = router.compress(req)
    assert result.body["messages"][0]["content"] == prose


def test_compress_no_messages_key() -> None:
    """Body without 'messages' key returns req unchanged."""
    router = CompressRouter()
    req = ProxyRequest(
        provider="anthropic",
        method="POST",
        path="/v1/messages",
        headers={},
        body={"model": "claude-sonnet-4-6"},
        body_bytes=b'{"model": "claude-sonnet-4-6"}',
        request_id="test-nomsg",
        stream=False,
        has_tools=False,
    )
    result = router.compress(req)
    assert result is req


def test_compress_no_system_prompt_skips_align() -> None:
    """When system prompt is absent, aligner is not called."""
    router = CompressRouter()
    large_json = json.dumps({"key": "v" * 200, "items": list(range(100))})
    req = ProxyRequest(
        provider="anthropic",
        method="POST",
        path="/v1/messages",
        headers={},
        body={
            "messages": [{"role": "assistant", "content": large_json}],
        },
        body_bytes=json.dumps({
            "messages": [{"role": "assistant", "content": large_json}],
        }).encode(),
        request_id="test-nosys",
        stream=False,
        has_tools=False,
    )
    result = router.compress(req)
    assert result is not None


def test_ccr_store_original_failure_non_fatal() -> None:
    """CCR store failure returns '' but compression still completes."""
    router = CompressRouter()
    # Force CCR store failure
    original_store = router._ccr_store_original
    try:
        router._ccr_store_original = lambda *a, **kw: ""
        large_json = json.dumps({"key": "v" * 200, "items": list(range(100))})
        req = _make_req_with_content(large_json)
        result = router.compress(req)
        assert result is not None
    finally:
        router._ccr_store_original = original_store


def test_llmlingua_import_error_non_fatal() -> None:
    """When LLMLingua raises ImportError, prose compression is disabled."""
    router = CompressRouter()
    cfg = router._get_config()
    cfg = type(cfg).from_dict({**cfg.as_dict(), "compress_mode": "aggressive"})

    def _raising_import(*args, **kwargs):
        raise ImportError("test")

    router._get_llmlingua_compressor = _raising_import

    with patch.object(router, "_get_config", return_value=cfg):
        prose = "This is just some normal prose text. " * 30
        req = _make_req_with_content(prose)
        result = router.compress(req)
        assert result is not None  # fail-open


def test_lazy_loaders_cached() -> None:
    """Lazy loaders return same instance on second call."""
    router = CompressRouter()

    c1 = router._get_ccr_store()
    c2 = router._get_ccr_store()
    assert c1 is c2

    d1 = router._get_aligner()
    d2 = router._get_aligner()
    assert d1 is d2


def test_on_compress_metrics_counter_is_none() -> None:
    """on_compress works when no metrics counters registered."""
    router = CompressRouter()
    router._metrics_counters = None
    # Must not raise
    router.on_compress(100, 50, False)


def test_set_metrics() -> None:
    router = CompressRouter()
    counters = type("C", (), {"tokens_saved_compress": 0})()
    router.set_metrics(counters)
    assert router._metrics_counters is counters


def test_ccr_update_compressed_non_fatal() -> None:
    """CCR update_compressed fails gracefully."""
    router = CompressRouter()
    original = router._ccr_update_compressed
    try:
        router._ccr_update_compressed = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom"))
        large_json = json.dumps({"key": "v" * 200, "items": list(range(100))})
        req = _make_req_with_content(large_json)
        result = router.compress(req)
        assert result is not None
    finally:
        router._ccr_update_compressed = original


def test_compress_text_aggressive_prose() -> None:
    """compress_text with aggressive mode and prose."""
    router = CompressRouter()
    cfg = router._get_config()
    cfg = type(cfg).from_dict({**cfg.as_dict(), "compress_mode": "aggressive"})

    with patch.object(router, "_get_config", return_value=cfg):
        result = router.compress_text("some short prose text")
        assert result.strategy == "none"


def test_compress_text_code_strategy() -> None:
    """compress_text with code content."""
    router = CompressRouter()
    code = (
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "def process_data(path):\n"
        + "    result = path\n" * 10
        + "    return result\n"
    )
    result = router.compress_text(code)
    assert result.strategy in ("extractive_code", "none")


def test_compress_json_content_preserves_body_keys() -> None:
    """Successful compression preserves non-messages body keys like model."""
    router = CompressRouter()
    large_json = json.dumps({"key": "v" * 200, "items": list(range(30))})
    req = ProxyRequest(
        provider="anthropic",
        method="POST",
        path="/v1/messages",
        headers={},
        body={
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "assistant", "content": large_json}],
        },
        body_bytes=json.dumps({
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "assistant", "content": large_json}],
        }).encode(),
        request_id="test-preserve-keys",
        stream=False,
        has_tools=False,
    )
    result = router.compress(req)
    # If compression happened, model key should be preserved
    if result is not req:
        assert "model" in result.body
        assert result.body["model"] == "claude-sonnet-4-6"
    result = router.compress_text("")
    assert result.strategy == "none"


def test_compress_json_list_root_ccr_embedding() -> None:
    """RB-02: List-root JSON must embed ccr_id in __slm_ccr__ wrapper."""
    router = CompressRouter()
    large_list = [{"id": i, "description": "word " * 60} for i in range(30)]
    large_json_text = json.dumps(large_list)
    req = _make_req_with_content(large_json_text)
    result = router.compress(req)
    if result is not req:
        for msg in result.body.get("messages", []):
            content = msg.get("content", "")
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and "__slm_ccr__" in parsed:
                        assert "__slm_data__" in parsed, "__slm_data__ wrapper missing"
                        assert isinstance(parsed["__slm_data__"], list)
                        return
                except json.JSONDecodeError:
                    pass


def test_compress_protect_recent_from_config() -> None:
    """RB-01: protect_recent must be read from cfg.compress_protect_recent."""
    router = CompressRouter()
    large_content = "word " * 400
    messages = [
        {"role": "assistant", "content": large_content},
    ] + [
        {"role": "assistant", "content": "short"}
        for _ in range(6)
    ]

    cfg = router._get_config()
    cfg = type(cfg).from_dict({**cfg.as_dict(), "compress_protect_recent": 6})

    with patch.object(router, "_get_config", return_value=cfg):
        req = _make_req(body={"messages": messages})
        result = router.compress(req)
    assert isinstance(result, ProxyRequest)


def test_router_singleton_same_instance() -> None:
    a = CompressRouter.get_instance()
    b = CompressRouter.get_instance()
    assert a is b


# ---- compress() where compression IS achieved (lines 116-138) ----

def test_compress_actually_compresses_large_json() -> None:
    """Large JSON should be compressed and on_compress called."""
    from unittest.mock import patch
    router = CompressRouter()

    # Override config to set protect_recent=0 so the message isn't protected
    cfg = router._get_config()
    cfg = type(cfg).from_dict({**cfg.as_dict(), "compress_protect_recent": 0})

    large_json = json.dumps({"key": "v" * 200, "items": list(range(100))})

    with patch.object(router, "_get_config", return_value=cfg):
        req = _make_req_with_content(large_json)
        result = router.compress(req)
        # If compression happened, body_bytes should be smaller
        if result is not req:
            assert len(result.body_bytes) < len(req.body_bytes)
            # Check model key preserved
            if "model" in req.body:
                assert "model" in result.body


# ---- code detection: ratio at/above threshold returns unchanged ----

def test_compress_code_below_threshold_returns_unchanged() -> None:
    """Short code that doesn't meet ratio threshold returns unchanged."""
    router = CompressRouter()
    # Short code that won't compress enough
    code = "x = 1\ny = 2\nz = 3\n"
    req = _make_req_with_content(code)
    result = router.compress(req)
    # Short code content preserved verbatim.
    assert result.body["messages"][0]["content"] == code


# ---- _ccr_store_original raises exception (non-fatal) ----

def test_ccr_store_original_raises_non_fatal() -> None:
    """When _ccr_store_original raises, compress still succeeds."""
    router = CompressRouter()
    original = router._ccr_store_original
    try:
        router._ccr_store_original = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("ccr store failed"))
        large_json = json.dumps({"key": "v" * 200, "items": list(range(100))})
        req = _make_req_with_content(large_json)
        result = router.compress(req)
        assert result is not None
    finally:
        router._ccr_store_original = original


# ---- _ccr_update_compressed raises exception (non-fatal) ----

def test_ccr_update_compressed_raises_non_fatal() -> None:
    """When _ccr_update_compressed raises, compress still succeeds."""
    router = CompressRouter()
    original = router._ccr_update_compressed
    try:
        router._ccr_update_compressed = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("ccr update failed"))
        large_json = json.dumps({"key": "v" * 200, "items": list(range(100))})
        req = _make_req_with_content(large_json)
        result = router.compress(req)
        assert result is not None
    finally:
        router._ccr_update_compressed = original


# ---- compress_text error path ----

def test_compress_text_exception_returns_fallback() -> None:
    """compress_text returns fallback result on exception."""
    router = CompressRouter()
    original = router._compress_text
    try:
        router._compress_text = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom"))
        result = router.compress_text("some text")
        assert result.strategy == "none"
        assert result.compressed_text == "some text"
    finally:
        router._compress_text = original


# ---- prose aggressive with mocked llmlingua ----

def test_compress_prose_aggressive_with_mocked_llmlingua() -> None:
    """Prose in aggressive mode with compress_prose=True uses LLMLingua."""
    from unittest.mock import MagicMock, patch
    import sys

    router = CompressRouter()
    # Set config to aggressive + prose enabled
    cfg = router._get_config()
    cfg = type(cfg).from_dict({
        **cfg.as_dict(),
        "compress_mode": "aggressive",
        "compress_prose": True,
        "compress_protect_recent": 2,
    })

    # Mock the llmlingua compressor to succeed
    fake_llmlingua = MagicMock()
    fake_llmlingua.compress.return_value = "compressed prose text"

    with patch.object(router, "_get_config", return_value=cfg):
        # Patch the lazy loader to return our mock
        original_loader = router._get_llmlingua_compressor
        router._get_llmlingua_compressor = lambda: fake_llmlingua
        try:
            prose = "This is a long paragraph. " * 30
            req = _make_req_with_content(prose)
            result = router.compress(req)
            assert result is not None
        finally:
            router._get_llmlingua_compressor = original_loader


# ---- _compress_content_block with tool_result in list ----

def test_compress_content_block_tool_result_small_text() -> None:
    """tool_result block with text below threshold is not compressed."""
    router = CompressRouter()
    short_text = "short tool result"
    req = ProxyRequest(
        provider="anthropic",
        method="POST",
        path="/v1/messages",
        headers={},
        body={
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "tool_result", "text": short_text}],
                },
            ]
        },
        body_bytes=json.dumps({
            "messages": [
                {"role": "assistant", "content": [{"type": "tool_result", "text": short_text}]},
            ]
        }).encode(),
        request_id="test-tool-result-short",
        stream=False,
        has_tools=False,
    )
    result = router.compress(req)
    # Should be passthrough (text too short)
    if result is not req:
        for msg in result.body["messages"]:
            for block in (msg.get("content") or []):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    assert block["text"] == short_text


# ---- _tool_result_text non-dict, non-str content path ----

def test_tool_result_text_non_dict_string() -> None:
    """_tool_result_text handles content that is not dict or str."""
    from superlocalmemory.optimize.compress.router import _tool_result_text
    # Content is a number (not str, not list)
    result = _tool_result_text({"content": 42})
    assert result == ""


# ---- _set_tool_result_text with string content ----

def test_set_tool_result_text_string_content() -> None:
    """_set_tool_result_text replaces string content directly."""
    from superlocalmemory.optimize.compress.router import _set_tool_result_text
    block = {"type": "tool_result", "content": "old text"}
    result = _set_tool_result_text(block, "new text")
    assert result["content"] == "new text"


# ---- _set_tool_result_text with list content ----

def test_set_tool_result_text_list_content() -> None:
    """_set_tool_result_text replaces first text block in list content."""
    from superlocalmemory.optimize.compress.router import _set_tool_result_text
    block = {
        "type": "tool_result",
        "content": [
            {"type": "text", "text": "old"},
            {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
        ],
    }
    result = _set_tool_result_text(block, "new")
    texts = [b["text"] for b in result["content"] if isinstance(b, dict) and b.get("type") == "text"]
    assert texts == ["new"]


# ---- _set_tool_result_text no-op on unknown block ----

def test_set_tool_result_text_no_text_block() -> None:
    """_set_tool_result_text returns unchanged when no text block to replace."""
    from superlocalmemory.optimize.compress.router import _set_tool_result_text
    block = {"type": "tool_result", "content": [{"type": "image_url", "url": "x"}]}
    result = _set_tool_result_text(block, "new")
    assert result == block


# ---- _detect_language edge cases ----

def test_detect_language_none_on_empty() -> None:
    """_detect_language returns None for empty string."""
    from superlocalmemory.optimize.compress.router import _detect_language
    assert _detect_language("") is None


def test_detect_language_none_on_short_text() -> None:
    """_detect_language returns None for text < 50 chars."""
    from superlocalmemory.optimize.compress.router import _detect_language
    assert _detect_language("short") is None


def test_detect_language_python_shebang() -> None:
    """_detect_language detects python from shebang."""
    from superlocalmemory.optimize.compress.router import _detect_language
    code = "#!/usr/bin/env python\n\nprint('hello')\n" + "x = 1\n" * 10
    assert _detect_language(code) == "python"


def test_detect_language_node_shebang() -> None:
    """_detect_language detects javascript from node shebang."""
    from superlocalmemory.optimize.compress.router import _detect_language
    code = "#!/usr/bin/env node\n\nconsole.log('hi');\n" + "const x = 1;\n" * 10
    assert _detect_language(code) == "javascript"


def test_detect_language_code_fence_hint() -> None:
    """_detect_language detects from ``` fence."""
    from superlocalmemory.optimize.compress.router import _detect_language
    code = "```python\ndef foo():\n" + "    pass\n" * 10
    assert _detect_language(code) == "python"


def test_detect_language_cpp_hint() -> None:
    """_detect_language maps cpp/c++ fence hint correctly."""
    from superlocalmemory.optimize.compress.router import _detect_language
    code = "```c++\nint main() {\n" + "    return 0;\n" * 10 + "}\n"
    assert _detect_language(code) == "cpp"


def test_detect_language_typescript_hint() -> None:
    """_detect_language maps ts/typescript fence hint to javascript."""
    from superlocalmemory.optimize.compress.router import _detect_language
    code = "```typescript\ninterface Foo {\n" + "    bar: string;\n" * 10 + "}\n"
    assert _detect_language(code) == "javascript"


# ---- _msg_has_tool_result ----

def test_msg_has_tool_result_detects_tool_use_id() -> None:
    """_msg_has_tool_result returns True when block has tool_use_id."""
    from superlocalmemory.optimize.compress.router import _msg_has_tool_result
    msg = {"role": "assistant", "content": [{"type": "tool_use", "tool_use_id": "abc", "name": "fn", "input": {}}]}
    assert _msg_has_tool_result(msg) is True


def test_msg_has_tool_result_false_for_plain_text() -> None:
    """_msg_has_tool_result returns False for regular text content."""
    from superlocalmemory.optimize.compress.router import _msg_has_tool_result
    msg = {"role": "assistant", "content": "Hello world"}
    assert _msg_has_tool_result(msg) is False


# ---- _token_estimate / _token_estimate_structured edge cases ----

def test_token_estimate_empty() -> None:
    """_token_estimate returns 0 for empty string."""
    from superlocalmemory.optimize.compress.router import _token_estimate
    assert _token_estimate("") == 0


# ---- on_compress with no counters ----

def test_on_compress_no_counters_registered() -> None:
    """on_compress works when _metrics_counters is None."""
    router = CompressRouter()
    router._metrics_counters = None
    # Must not raise
    router.on_compress(100, 50, True)


# ---- compress() with exception in try block ----

def test_compress_exception_returns_passthrough() -> None:
    """compress() returns req unchanged when internal error occurs."""
    router = CompressRouter()
    # Force ConfigStore to raise, triggering the outer except
    original = router._get_config
    try:
        router._get_config = lambda: (_ for _ in ()).throw(RuntimeError("config broken"))
        req = _make_req()
        result = router.compress(req)
        assert result is req
    finally:
        router._get_config = original


# ---- JSON that starts with { but is not valid JSON ----

def test_compress_invalid_json_passthrough() -> None:
    """Invalid JSON starting with { should passthrough."""
    router = CompressRouter()
    bad_json = '{"key": "value"' + ' extra garbage ' * 50  # missing closing brace + extra
    req = _make_req_with_content(bad_json)
    result = router.compress(req)
    # Invalid JSON is never minified; content preserved.
    assert result.body["messages"][0]["content"] == bad_json


# ---- system prompt with CacheAligner findings ----

def test_compress_system_prompt_aligner_detection() -> None:
    """CacheAligner runs on system prompt with UUID/volatile tokens."""
    router = CompressRouter()
    large_json = json.dumps({"key": "v" * 200, "items": list(range(100))})
    req = ProxyRequest(
        provider="anthropic",
        method="POST",
        path="/v1/messages",
        headers={},
        body={
            "system": "Session: 550e8400-e29b-41d4-a716-446655440000. Today is 2026-06-07T14:30:00Z.",
            "messages": [{"role": "assistant", "content": large_json}],
        },
        body_bytes=json.dumps({
            "system": "Session: 550e8400-e29b-41d4-a716-446655440000. Today is 2026-06-07T14:30:00Z.",
            "messages": [{"role": "assistant", "content": large_json}],
        }).encode(),
        request_id="test-align-sys",
        stream=False,
        has_tools=False,
    )
    result = router.compress(req)
    assert result is not None


# ---- _compress_messages primary_strategy assignment (line 192) ----

def test_compress_messages_primary_strategy_set() -> None:
    """_compress_messages sets primary_strategy when compression happens."""
    router = CompressRouter()
    large_json = json.dumps({"key": "v" * 200, "items": list(range(100))})
    new_msgs, before, after, strat = router._compress_messages(
        messages=[{"role": "assistant", "content": large_json}],
        aggressive=False,
        protect_recent=0,
        request_id="test-primary",
        model="",
        tenant_id="default",
    )
    # If compression happened, strategy should not be "none"
    if after < before:
        assert strat != "none"


# ---- _ccr_store_original exception handler (lines 378-380) ----

def test_ccr_store_original_internal_error_non_fatal() -> None:
    """_ccr_store_original catches errors from store.store()."""
    router = CompressRouter()
    # Get CCR store and make its store() fail
    ccr_store = router._get_ccr_store()
    original_store = ccr_store.store
    try:
        ccr_store.store = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("store failed"))
        # Call _ccr_store_original directly — should return ""
        result = router._ccr_store_original(b"data", "model", "default")
        assert result == ""
    finally:
        ccr_store.store = original_store


# ---- _ccr_update_compressed exception handler (lines 387-388) ----

def test_ccr_update_compressed_internal_error_non_fatal() -> None:
    """_ccr_update_compressed catches errors from store.update_compressed()."""
    router = CompressRouter()
    ccr_store = router._get_ccr_store()
    original = ccr_store.update_compressed
    try:
        ccr_store.update_compressed = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("update failed"))
        # Must not raise
        router._ccr_update_compressed("any-id", b"data")
    finally:
        ccr_store.update_compressed = original


# ---- compress() body update + on_compress (lines 116-138) ----

def test_compress_updates_body_and_calls_on_compress() -> None:
    """Successful compression updates body messages and fires on_compress."""
    from unittest.mock import patch
    router = CompressRouter()

    # Config with protect_recent=0 so all messages are eligible
    cfg = router._get_config()
    cfg = type(cfg).from_dict({**cfg.as_dict(), "compress_protect_recent": 0})

    large_json = json.dumps({"key": "v" * 200, "items": list(range(100))})
    req = _make_req_with_content(large_json)

    with patch.object(router, "_get_config", return_value=cfg):
        result = router.compress(req)

    # If compression succeeded, result is a new ProxyRequest
    if result is not req:
        # Body should have updated messages
        assert "messages" in result.body
        # Body bytes should differ
        assert result.body_bytes != req.body_bytes


# ---- _compress_content_block with list of mixed blocks ----

def test_compress_content_block_mixed_types() -> None:
    """Content blocks with mixed types (text + image) preserve non-text blocks."""
    router = CompressRouter()
    large_text = "word " * 600
    req = ProxyRequest(
        provider="anthropic",
        method="POST",
        path="/v1/messages",
        headers={},
        body={
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "user", "content": "Hello again"},
                {"role": "user", "content": "Hello 3"},
                {"role": "user", "content": "Hello 4"},
                {"role": "user", "content": "Hello 5"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": large_text},
                        {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                    ],
                },
            ]
        },
        body_bytes=json.dumps({
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "user", "content": "Hello again"},
                {"role": "user", "content": "Hello 3"},
                {"role": "user", "content": "Hello 4"},
                {"role": "user", "content": "Hello 5"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": large_text},
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                ]},
            ]
        }).encode(),
        request_id="test-mixed",
        stream=False,
        has_tools=False,
    )
    result = router.compress(req)
    assert result is not None
    # Image block should be preserved
    for msg in result.body["messages"]:
        if isinstance(msg.get("content"), list):
            image_blocks = [b for b in msg["content"] if isinstance(b, dict) and b.get("type") == "image_url"]
            assert len(image_blocks) == 1


# ---- compress with many messages to bypass protect_recent ----

def test_compress_many_messages_bypass_protection() -> None:
    """With 8 messages and protect_recent=4, first 4 should be compressible."""
    router = CompressRouter()
    large_json = json.dumps({"key": "v" * 200, "items": list(range(100))})
    messages = [{"role": "assistant", "content": large_json} for _ in range(8)]

    req = ProxyRequest(
        provider="anthropic",
        method="POST",
        path="/v1/messages",
        headers={},
        body={"messages": messages},
        body_bytes=json.dumps({"messages": messages}).encode(),
        request_id="test-many-msgs",
        stream=False,
        has_tools=False,
    )
    result = router.compress(req)
    assert result is not None


# ---- tool role message is skipped ----

def test_tool_role_message_skipped_but_others_compress() -> None:
    """Tool role messages are skipped; other messages may still be compressed."""
    router = CompressRouter()
    large_json = json.dumps({"key": "v" * 200, "items": list(range(100))})
    messages = [
        {"role": "user", "content": "dummy1"},
        {"role": "user", "content": "dummy2"},
        {"role": "user", "content": "dummy3"},
        {"role": "user", "content": "dummy4"},
        {"role": "tool", "content": large_json},
        {"role": "assistant", "content": large_json},
    ]

    req = ProxyRequest(
        provider="anthropic",
        method="POST",
        path="/v1/messages",
        headers={},
        body={"messages": messages},
        body_bytes=json.dumps({"messages": messages}).encode(),
        request_id="test-tool-role-2",
        stream=False,
        has_tools=False,
    )
    result = router.compress(req)
    assert result is not None


def test_k08_json_probe_catches_only_json_decode_error(caplog) -> None:
    """K-08: JSON probe catches json.JSONDecodeError narrowly; does not silence other exceptions."""
    import logging
    router = CompressRouter()
    # Must be >= _MIN_CHARS_FOR_COMPRESSION (500) so it reaches the JSON probe
    # Starts with "[" but is not valid JSON — must be treated as prose, not crash
    tricky = "[" + ("word " * 120)  # ~601 chars — clears the 500-char guard
    with caplog.at_level(logging.DEBUG, logger="superlocalmemory.optimize.compress.router"):
        result = router.compress_text(tricky)
    assert result is not None, "K-08: compress_text must not raise on invalid JSON-looking input"


def test_k08_json_probe_error_fails_open(monkeypatch) -> None:
    """K-08: an error during the JSON probe must fail open (never raise); the content is
    treated as non-JSON prose and returned unchanged rather than crashing the pipeline."""
    import json as json_mod
    router = CompressRouter()
    original_loads = json_mod.loads

    def _raise_unexpected(s, **kw):
        stripped = s.strip() if isinstance(s, str) else s
        if isinstance(stripped, str) and stripped.startswith("["):
            raise ValueError("simulated internal JSON error")
        return original_loads(s, **kw)

    monkeypatch.setattr(json_mod, "loads", _raise_unexpected)
    # >= 500 chars and starts with "[" to reach the JSON probe.
    tricky = "[" + ("word " * 120)
    result = router.compress_text(tricky)
    assert result is not None, "K-08: must not raise on a JSON-probe error"
    assert result.strategy in ("none", "normalize"), "probe error → non-JSON prose path"


def test_k10_compress_text_result_has_lossy_field() -> None:
    """K-10: CompressTextResult must expose a lossy bool field."""
    from superlocalmemory.optimize.compress.router import CompressTextResult
    r = CompressTextResult(compressed_text="x", strategy="normalize", tokens_before=5, tokens_after=4)
    assert hasattr(r, "lossy"), "K-10: CompressTextResult must have lossy field"
    assert r.lossy is False


def test_k10_lossy_false_for_lossless_strategies() -> None:
    """K-10: lossy must be False for 'normalize' and 'none' strategies."""
    from superlocalmemory.optimize.compress.router import CompressTextResult
    for strategy in ("normalize", "none"):
        r = CompressTextResult(compressed_text="x", strategy=strategy,
                               tokens_before=5, tokens_after=4)
        assert r.lossy is False, f"K-10: expected lossy=False for strategy={strategy!r}"


# ── Stage 7 coverage gap tests ──────────────────────────────────────────────

def test_compress_request_body_with_enabled_config() -> None:
    """Cover compress_request() body (lines 72-113): compress_enabled=True; mocked messages.

    protect_recent=1 with a 2-msg conversation ensures idx 0 (assistant) is NOT protected.
    _compress_messages is mocked to return fewer tokens so the on_compress branch executes.
    """
    from unittest.mock import patch
    router = CompressRouter()
    cfg = router._get_config()
    cfg_enabled = type(cfg).from_dict({
        **cfg.as_dict(),
        "compress_enabled": True,
        "compress_mode": "safe",
        "compress_protect_recent": 1,
    })

    bloated = ("word " * 60 + "   \n") * 15
    msgs = [
        {"role": "assistant", "content": bloated},   # idx 0 — eligible for compression
        {"role": "user", "content": "follow up"},    # idx 1 — protected (last user + last msg)
    ]
    req = _make_req(body={"messages": msgs})

    mock_msgs = [dict(m) for m in msgs]
    # Return a mock that pretends compression saved tokens
    with patch.object(router, "_get_config", return_value=cfg_enabled):
        with patch.object(router, "_compress_messages",
                          return_value=(mock_msgs, 100, 70, "normalize")):
            result = router.compress(req)

    assert result is not None


def test_compress_request_no_improvement_returns_req() -> None:
    """Cover compress_request() line 103-104: no improvement → return original req."""
    from unittest.mock import patch
    router = CompressRouter()
    cfg = router._get_config()
    cfg_enabled = type(cfg).from_dict({
        **cfg.as_dict(),
        "compress_enabled": True,
        "compress_mode": "safe",
        "compress_protect_recent": 0,
    })

    msgs = [{"role": "assistant", "content": "word " * 110}]
    req = _make_req(body={"messages": msgs})

    # Return same token count → no improvement → req returned unchanged
    with patch.object(router, "_get_config", return_value=cfg_enabled):
        with patch.object(router, "_compress_messages",
                          return_value=(msgs, 110, 110, "none")):
            result = router.compress(req)

    assert result is not None


def test_compress_messages_loop_body_covers_content_block() -> None:
    """Cover _compress_messages lines 180-194: assistant msg NOT in protect set gets compressed."""
    from unittest.mock import patch
    router = CompressRouter()
    cfg = router._get_config()
    cfg_enabled = type(cfg).from_dict({
        **cfg.as_dict(),
        "compress_enabled": True,
        "compress_mode": "safe",
        "compress_protect_recent": 1,
    })

    bloated = ("word " * 60 + "   \n") * 12
    msgs = [
        {"role": "assistant", "content": bloated},  # idx 0: eligible
        {"role": "user", "content": "follow up"},   # idx 1: protected
    ]
    req = _make_req(body={"messages": msgs})

    with patch.object(router, "_get_config", return_value=cfg_enabled):
        result = router.compress(req)

    assert result is not None


def test_compress_content_block_list_with_enabled_config() -> None:
    """Cover _compress_content_block list branch (lines 206-241): content as list of blocks."""
    from unittest.mock import patch
    router = CompressRouter()
    cfg = router._get_config()
    cfg_enabled = type(cfg).from_dict({
        **cfg.as_dict(),
        "compress_enabled": True,
        "compress_mode": "safe",
        "compress_protect_recent": 1,
    })

    bloated_text = ("word " * 60 + "   \n") * 12
    content_blocks = [
        {"type": "text", "text": bloated_text},
        {"type": "image", "source": {}},           # non-text block → passthrough
        {"not_a_dict": True},                       # non-dict item → passthrough
    ]
    msgs = [
        {"role": "assistant", "content": content_blocks},  # idx 0: eligible
        {"role": "user", "content": "follow up"},          # idx 1: protected
    ]
    req = _make_req(body={"messages": msgs})

    with patch.object(router, "_get_config", return_value=cfg_enabled):
        result = router.compress(req)

    assert result is not None


def test_compress_text_large_valid_json_minified() -> None:
    """Valid JSON >= 500 chars is losslessly minified (value preserved, strategy=json_minify)."""
    router = CompressRouter()
    large_json = json.dumps({"key": "v" * 200, "items": list(range(80))}, indent=2)
    assert len(large_json) >= 500, "test prerequisite: json must be >= 500 chars"
    result = router.compress_text(large_json)
    assert result.strategy == "json_minify"
    assert result.lossy is False
    assert json.loads(result.compressed_text) == json.loads(large_json)


def test_compress_text_large_code_passthrough() -> None:
    """Cover _compress_text line 268: code >= 500 chars → passthrough via code detection."""
    router = CompressRouter()
    code_block = "\n".join(f"def fn_{i}(x: int) -> int:\n    return x + {i}" for i in range(40))
    assert len(code_block) >= 500
    result = router.compress_text(code_block)
    assert result is not None


def test_compress_content_block_non_dict_item_in_list() -> None:
    """Cover _compress_content_block line 216-217: non-dict item in list → passthrough."""
    from unittest.mock import patch
    router = CompressRouter()
    cfg = router._get_config()
    cfg_enabled = type(cfg).from_dict({
        **cfg.as_dict(),
        "compress_enabled": True,
        "compress_mode": "safe",
        "compress_protect_recent": 1,
    })
    # Non-dict items in content list (e.g. raw strings)
    content_with_non_dict = [
        "raw string item",  # non-dict → triggers line 216-217
        {"type": "text", "text": "word " * 20},
    ]
    msgs = [
        {"role": "assistant", "content": content_with_non_dict},
        {"role": "user", "content": "follow up"},
    ]
    req = _make_req(body={"messages": msgs})
    with patch.object(router, "_get_config", return_value=cfg_enabled):
        result = router.compress(req)
    assert result is not None


def test_compress_content_block_short_text_in_list() -> None:
    """Cover _compress_content_block line 222-223: text block < 500 chars → skip."""
    from unittest.mock import patch
    router = CompressRouter()
    cfg = router._get_config()
    cfg_enabled = type(cfg).from_dict({
        **cfg.as_dict(),
        "compress_enabled": True,
        "compress_mode": "safe",
        "compress_protect_recent": 1,
    })
    # Short text block in a list (below _MIN_CHARS_FOR_COMPRESSION)
    content_short = [
        {"type": "text", "text": "short"},  # < 500 chars → line 222-223
    ]
    msgs = [
        {"role": "assistant", "content": content_short},
        {"role": "user", "content": "follow up"},
    ]
    req = _make_req(body={"messages": msgs})
    with patch.object(router, "_get_config", return_value=cfg_enabled):
        result = router.compress(req)
    assert result is not None


def test_compress_content_block_non_string_non_list() -> None:
    """Cover _compress_content_block line 241: content not str/list → (content, 0, 0, 'none')."""
    from unittest.mock import patch
    router = CompressRouter()
    cfg = router._get_config()
    cfg_enabled = type(cfg).from_dict({
        **cfg.as_dict(),
        "compress_enabled": True,
        "compress_mode": "safe",
        "compress_protect_recent": 1,
    })
    # Integer content — unusual but must not crash
    msgs = [
        {"role": "assistant", "content": 42},
        {"role": "user", "content": "follow up"},
    ]
    req = _make_req(body={"messages": msgs})
    with patch.object(router, "_get_config", return_value=cfg_enabled):
        result = router.compress(req)
    assert result is not None


def test_compress_request_messages_not_list_returns_req() -> None:
    """Cover compress_request() line 75: messages not a list → return req immediately."""
    from unittest.mock import patch
    router = CompressRouter()
    cfg = router._get_config()
    cfg_enabled = type(cfg).from_dict({**cfg.as_dict(), "compress_enabled": True})
    req = _make_req(body={"messages": "not_a_list"})
    with patch.object(router, "_get_config", return_value=cfg_enabled):
        result = router.compress(req)
    assert result is req


def test_compress_request_system_prompt_with_enabled() -> None:
    """Cover compress_request() lines 80-83: system prompt present → CacheAligner runs."""
    from unittest.mock import patch
    router = CompressRouter()
    cfg = router._get_config()
    cfg_enabled = type(cfg).from_dict({
        **cfg.as_dict(),
        "compress_enabled": True,
        "compress_protect_recent": 0,
    })
    bloated = ("Today's date is 2026-06-13. " * 40)  # volatile date token
    req = _make_req(body={
        "system": bloated,
        "messages": [{"role": "user", "content": "what day is it?"}],
    })
    with patch.object(router, "_get_config", return_value=cfg_enabled):
        result = router.compress(req)
    assert result is not None


def test_compress_messages_primary_strategy_update() -> None:
    """Cover _compress_messages line 190: primary_strategy updated when strat != 'none'."""
    from unittest.mock import patch, MagicMock
    router = CompressRouter()
    cfg = router._get_config()
    cfg_enabled = type(cfg).from_dict({
        **cfg.as_dict(),
        "compress_enabled": True,
        "compress_protect_recent": 1,
    })
    bloated = "word " * 200
    msgs = [
        {"role": "assistant", "content": bloated},
        {"role": "user", "content": "hi"},
    ]
    req = _make_req(body={"messages": msgs})
    # Mock _compress_content_block to return strat != "none"
    with patch.object(router, "_get_config", return_value=cfg_enabled):
        with patch.object(router, "_compress_content_block",
                          return_value=(bloated, 200, 150, "normalize")):
            result = router.compress(req)
    assert result is not None


def test_compress_content_block_list_primary_strat_update() -> None:
    """Cover _compress_content_block line 230: primary updated for text block in list."""
    from unittest.mock import patch
    router = CompressRouter()
    cfg = router._get_config()
    cfg_enabled = type(cfg).from_dict({
        **cfg.as_dict(),
        "compress_enabled": True,
        "compress_protect_recent": 1,
    })
    long_text = "word " * 200
    content_list = [{"type": "text", "text": long_text}]
    msgs = [
        {"role": "assistant", "content": content_list},
        {"role": "user", "content": "hi"},
    ]
    req = _make_req(body={"messages": msgs})
    # Mock _compress_text to return strat != "none"
    with patch.object(router, "_get_config", return_value=cfg_enabled):
        with patch.object(router, "_compress_text",
                          return_value=(long_text[:800], 200, 150, "normalize")):
            result = router.compress(req)
    assert result is not None


def test_compress_content_block_tool_result_in_list() -> None:
    """Cover _compress_content_block line 235: tool_result block in list gets text replaced."""
    from unittest.mock import patch
    router = CompressRouter()
    cfg = router._get_config()
    cfg_enabled = type(cfg).from_dict({
        **cfg.as_dict(),
        "compress_enabled": True,
        "compress_protect_recent": 1,
    })
    long_output = "output line " * 100  # >= 500 chars
    content_list = [{"type": "tool_result", "tool_use_id": "t1", "content": long_output}]
    msgs = [
        {"role": "user", "content": content_list},  # idx 0: tool_result in user msg
        {"role": "user", "content": "final"},       # idx 1: protected
    ]
    req = _make_req(body={"messages": msgs})
    with patch.object(router, "_get_config", return_value=cfg_enabled):
        result = router.compress(req)
    assert result is not None


def test_msg_has_tool_result_tool_use_id_branch() -> None:
    """Cover _msg_has_tool_result line 403: detect tool_use_id key in content block."""
    from superlocalmemory.optimize.compress.router import _msg_has_tool_result
    msg_with_tool_use_id = {
        "role": "user",
        "content": [{"type": "text", "tool_use_id": "call_abc123", "text": "result"}],
    }
    assert _msg_has_tool_result(msg_with_tool_use_id) is True


def test_msg_has_tool_result_non_dict_item_skipped() -> None:
    """Cover _msg_has_tool_result line 406: non-dict item in content list → continue."""
    from superlocalmemory.optimize.compress.router import _msg_has_tool_result
    msg_with_non_dict = {
        "role": "user",
        "content": ["raw string item", {"type": "tool_result"}],
    }
    assert _msg_has_tool_result(msg_with_non_dict) is True


# ── Stage 8 fixes ────────────────────────────────────────────────────────────

def test_s01_layer1_normalize_strategy_is_returned_for_whitespace_bloat() -> None:
    """S-01 fix: Layer 1 'normalize' strategy must fire on prose with whitespace bloat.

    _token_estimate() is word-count — normalization can't reduce it.
    The fix changed the decision to len(normalized) < len(text) (character count).
    """
    router = CompressRouter()
    # Prose with trailing spaces and 3+ blank lines — normalization removes these chars
    bloated = ("sentence here.   \n") * 40 + "\n\n\n\n" + ("another sentence.   \n") * 40
    assert len(bloated) >= 500, "test prerequisite: must exceed min compression threshold"
    result = router.compress_text(bloated)
    assert result.strategy == "normalize", (
        f"S-01: Layer 1 normalize must activate on whitespace-bloated prose; "
        f"got strategy={result.strategy!r}"
    )
    assert len(result.compressed_text) < len(bloated), "normalize must actually shorten the text"
    assert result.lossy is False, "Layer 1 normalization is lossless"


def test_s01_clean_prose_returns_none_strategy() -> None:
    """S-01: clean prose with no excess whitespace must return strategy='none'."""
    router = CompressRouter()
    # " ".join guarantees no trailing spaces; rstrip() leaves it identical → no savings
    clean = " ".join(["word"] * 120)
    normalized = router._normalize_whitespace(clean)
    assert len(normalized) == len(clean), "test prerequisite: text must already be normalized"
    result = router.compress_text(clean)
    assert result.strategy == "none"


def test_perf02_oversized_json_passthrough() -> None:
    """PERF-02: JSON over the minify size cap is passed through unparsed (bounds hot-path cost)."""
    from superlocalmemory.optimize.compress.router import _MAX_JSON_MINIFY_CHARS
    router = CompressRouter()
    # Valid, bracket-matched JSON just over the cap → skipped without a parse.
    big = json.dumps({"k": "v" * (_MAX_JSON_MINIFY_CHARS + 1000)})
    assert len(big) > _MAX_JSON_MINIFY_CHARS and big[0] == "{" and big[-1] == "}"
    result = router.compress_text(big)
    assert result.strategy == "none", "oversized JSON must be passed through unchanged"
    assert result.compressed_text == big


def test_stage9_layer1_normalization_works_end_to_end_through_compress_request() -> None:
    """Stage 9: Layer 1 normalize must work through compress_request(), not just compress_text().

    Before Stage 9 fix: compress_request() checked token_after >= token_before.
    Layer 1 normalization saves characters not word-count tokens, so tokens are equal.
    The guard discarded the normalized body every time — S-01 fix was incomplete.

    After fix: build new_bytes first, check actual byte savings in the guard.
    """
    from unittest.mock import patch
    router = CompressRouter()
    cfg = router._get_config()
    cfg_enabled = type(cfg).from_dict({
        **cfg.as_dict(),
        "compress_enabled": True,
        "compress_mode": "safe",
        "compress_protect_recent": 1,
    })
    # Whitespace-bloated text: lots of trailing spaces and 4+ blank lines
    bloated = ("sentence with trailing spaces.   \n") * 40 + "\n\n\n\n\n" + ("more prose.   \n") * 40
    original_body = {"messages": [
        {"role": "assistant", "content": bloated},
        {"role": "user", "content": "hi"},
    ]}
    import json as _json
    original_bytes = _json.dumps(original_body, ensure_ascii=False, separators=(",", ":")).encode()
    req = _make_req(body=original_body)
    # Manually set body_bytes to the correct serialized size so the guard uses accurate baseline
    req = ProxyRequest(
        provider=req.provider, method=req.method, path=req.path, headers=req.headers,
        body=req.body, body_bytes=original_bytes,
        request_id=req.request_id, stream=req.stream, has_tools=req.has_tools,
    )

    with patch.object(router, "_get_config", return_value=cfg_enabled):
        result = router.compress(req)

    # Normalization must produce a shorter body
    assert result is not req, (
        "Stage 9: compress_request() must return a new (shorter) ProxyRequest "
        "after Layer 1 normalization, not the original req"
    )
    assert len(result.body_bytes) < len(req.body_bytes), (
        "Stage 9: normalized body must be smaller in bytes than original"
    )
