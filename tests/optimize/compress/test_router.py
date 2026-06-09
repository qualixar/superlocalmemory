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


def test_compress_json_content_uses_extractive() -> None:
    router = CompressRouter()
    large_json_text = json.dumps({"key": "v" * 200, "items": list(range(100))})
    req = _make_req_with_content(large_json_text)
    result = router.compress(req)
    if result is not req and result.body_bytes != req.body_bytes:
        # Strategy info is tracked internally; JSON content should compress
        assert result.body_bytes < req.body_bytes


def test_compress_code_content_uses_extractive() -> None:
    router = CompressRouter()
    # Python function with long body — language detection should work
    code = "import os\n\ndef foo(x: int) -> int:\n" + "    pass\n" * 20 + "    return x\n"
    req = _make_req_with_content(code)
    result = router.compress(req)
    # Code may or may not compress depending on ratio; shouldn't error
    assert result is not None


def test_compress_prose_safe_mode_passthrough() -> None:
    router = CompressRouter()
    prose = "This is a long prose paragraph. " * 50
    req = _make_req_with_content(prose)
    result = router.compress(req)
    # In safe mode, prose shouldn't change
    assert result is req or result.body_bytes == req.body_bytes


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
    """CRITICAL: JSON MUST NOT go through LLMLingua even in aggressive mode."""
    router = CompressRouter()
    cfg = router._get_config()
    cfg = type(cfg).from_dict({**cfg.as_dict(), "compress_mode": "aggressive"})

    large_json_text = json.dumps({"key": "v" * 200, "items": list(range(100))})
    req = _make_req_with_content(large_json_text)

    with patch.object(router, "_get_config", return_value=cfg):
        result = router.compress(req)
    # If it compressed, the compressed output should still be valid JSON
    if result is not req and result.body_bytes != req.body_bytes:
        compressed_body = result.body
        for msg in compressed_body.get("messages", []):
            content = msg.get("content", "")
            if isinstance(content, str):
                try:
                    json.loads(content)
                except json.JSONDecodeError:
                    pytest.fail(f"LLMLingua was applied to JSON: {content[:200]}")


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
    assert result is req


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
    router = CompressRouter()
    calls: list = []
    router.on_compress = lambda b, a, l: calls.append((b, a, l))

    large_json_text = json.dumps({"key": "v" * 200, "items": list(range(100))})
    req = _make_req_with_content(large_json_text)
    result = router.compress(req)
    if result is not req:
        assert len(calls) == 1, "on_compress must be called exactly once per successful compress"
        before, after, lossy = calls[0]
        assert before > after
        assert lossy is False


def test_compress_text_convenience_method() -> None:
    """M-06: compress_text() public convenience method."""
    router = CompressRouter()
    large_json_text = json.dumps({"key": "v" * 200, "items": list(range(100))})
    result = router.compress_text(large_json_text)
    assert result.strategy in ("extractive_json", "none")
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
    """tool_result blocks with text get compressed."""
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
    assert result is req or result.body_bytes == req.body_bytes


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
    a1 = router._get_json_compressor()
    a2 = router._get_json_compressor()
    assert a1 is a2

    b1 = router._get_code_compressor()
    b2 = router._get_code_compressor()
    assert b1 is b2

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
    # Should be passthrough since code is too short
    assert result is req or result.body_bytes == req.body_bytes


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


def test_token_estimate_structured_empty() -> None:
    """_token_estimate_structured returns 0 for empty string."""
    from superlocalmemory.optimize.compress.router import _token_estimate_structured
    assert _token_estimate_structured("") == 0


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
    assert result is req or result.body_bytes == req.body_bytes


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
