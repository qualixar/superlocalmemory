# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for optimize shadow-capture mode (v3.6.10, plan §7).

Covers:
    - capture_enabled() env gating
    - ShadowCapture.record() JSONL append + 0600 perms + fail-open
    - extract_usage() per provider
    - _load_hooks() returns empty chain in capture mode (no cache/compress)
    - surfaces in capture mode: pure passthrough + corpus record (non-stream)
    - isolation: only optimize_capture.jsonl is written

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from superlocalmemory.optimize.config.schema import OptimizeConfig
from superlocalmemory.optimize.proxy import capture as capture_mod
from superlocalmemory.optimize.proxy.capture import (
    ShadowCapture,
    build_entry,
    capture_enabled,
    extract_usage,
)
from superlocalmemory.optimize.proxy.lifecycle import HookChain
from superlocalmemory.optimize.proxy.server import ProxyApp, _load_hooks, build_proxy_router


# ---------------------------------------------------------------------------
# capture_enabled() — env gating
# ---------------------------------------------------------------------------

class TestCaptureEnabled:
    def test_unset_is_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SLM_OPTIMIZE_CAPTURE", raising=False)
        assert capture_enabled() is False

    @pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on", " Yes "])
    def test_truthy_values_enable(self, monkeypatch: pytest.MonkeyPatch, val: str) -> None:
        monkeypatch.setenv("SLM_OPTIMIZE_CAPTURE", val)
        assert capture_enabled() is True

    @pytest.mark.parametrize("val", ["0", "false", "no", "off", "", "maybe"])
    def test_falsy_values_disable(self, monkeypatch: pytest.MonkeyPatch, val: str) -> None:
        monkeypatch.setenv("SLM_OPTIMIZE_CAPTURE", val)
        assert capture_enabled() is False


# ---------------------------------------------------------------------------
# extract_usage() — per provider
# ---------------------------------------------------------------------------

class TestExtractUsage:
    def test_anthropic(self) -> None:
        body = b'{"model":"claude-x","usage":{"input_tokens":10,"output_tokens":5}}'
        assert extract_usage("anthropic", body) == (10, 5, "claude-x")

    def test_openai(self) -> None:
        body = b'{"model":"gpt-x","usage":{"prompt_tokens":7,"completion_tokens":3}}'
        assert extract_usage("openai", body) == (7, 3, "gpt-x")

    def test_gemini(self) -> None:
        body = b'{"modelVersion":"gemini-2","usageMetadata":{"promptTokenCount":9,"candidatesTokenCount":4}}'
        assert extract_usage("gemini", body) == (9, 4, "gemini-2")

    def test_gemini_openai_compat_uses_openai_fields(self) -> None:
        body = b'{"model":"gemini-flash","usage":{"prompt_tokens":2,"completion_tokens":1}}'
        # any non-anthropic/non-gemini provider label hits the openai branch
        assert extract_usage("gemini-openai-compat", body) == (2, 1, "gemini-flash")

    def test_none_body(self) -> None:
        assert extract_usage("anthropic", None) == (0, 0, "")

    def test_unparseable_body(self) -> None:
        assert extract_usage("openai", b"{not json") == (0, 0, "")

    def test_non_dict_json(self) -> None:
        assert extract_usage("openai", b'["a","b"]') == (0, 0, "")

    def test_missing_usage_fields(self) -> None:
        assert extract_usage("anthropic", b'{"model":"m"}') == (0, 0, "m")

    def test_unknown_provider_returns_zero_tokens(self) -> None:
        # Audit fix: unknown provider must NOT be silently parsed as openai.
        body = b'{"model":"x","usage":{"prompt_tokens":99,"completion_tokens":88}}'
        itok, otok, mdl = extract_usage("azure-anthropic-compat", body)
        assert (itok, otok) == (0, 0)
        assert mdl == "x"


# ---------------------------------------------------------------------------
# ShadowCapture.record() — JSONL append, perms, fail-open
# ---------------------------------------------------------------------------

class TestShadowCaptureRecord:
    def test_appends_jsonl_line(self, tmp_path: Path) -> None:
        cap = ShadowCapture(path=tmp_path / "cap.jsonl")
        assert cap.record({"a": 1}) is True
        assert cap.record({"b": 2}) is True

        lines = (tmp_path / "cap.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"a": 1}
        assert json.loads(lines[1]) == {"b": 2}
        assert cap.count == 2

    def test_file_created_0600(self, tmp_path: Path) -> None:
        cap = ShadowCapture(path=tmp_path / "cap.jsonl")
        cap.record({"x": 1})
        mode = stat.S_IMODE(os.stat(tmp_path / "cap.jsonl").st_mode)
        assert mode == 0o600

    def test_creates_parent_dir(self, tmp_path: Path) -> None:
        cap = ShadowCapture(path=tmp_path / "nested" / "deep" / "cap.jsonl")
        assert cap.record({"x": 1}) is True
        assert (tmp_path / "nested" / "deep" / "cap.jsonl").exists()

    def test_non_serialisable_entry_fails_open(self, tmp_path: Path) -> None:
        cap = ShadowCapture(path=tmp_path / "cap.jsonl")
        # a set is not JSON-serialisable
        assert cap.record({"bad": {1, 2, 3}}) is False
        assert not (tmp_path / "cap.jsonl").exists()

    def test_write_error_fails_open(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cap = ShadowCapture(path=tmp_path / "cap.jsonl")

        def _boom(*a: Any, **k: Any):
            raise OSError("disk full")

        monkeypatch.setattr("builtins.open", _boom)
        # first write uses os.open path; force that too
        monkeypatch.setattr("os.open", _boom)
        assert cap.record({"x": 1}) is False  # must not raise

    def test_singleton_get_instance(self) -> None:
        ShadowCapture.reset_instance()
        a = ShadowCapture.get_instance()
        b = ShadowCapture.get_instance()
        assert a is b
        ShadowCapture.reset_instance()

    def test_perms_stay_0600_across_multiple_writes(self, tmp_path: Path) -> None:
        # Audit fix: every write uses os.open 0600 (no TOCTOU drop to umask).
        cap = ShadowCapture(path=tmp_path / "cap.jsonl")
        cap.record({"a": 1})
        cap.record({"b": 2})
        cap.record({"c": 3})
        mode = stat.S_IMODE(os.stat(tmp_path / "cap.jsonl").st_mode)
        assert mode == 0o600

    def test_symlink_at_path_is_refused(self, tmp_path: Path) -> None:
        # Audit fix LOW-2: O_NOFOLLOW refuses a pre-placed symlink (symlink-append).
        target = tmp_path / "secret.txt"
        target.write_text("original secret\n")
        link = tmp_path / "cap.jsonl"
        link.symlink_to(target)
        cap = ShadowCapture(path=link)
        ok = cap.record({"attack": "append"})
        assert ok is False, "write through a symlink must be refused (fail-open)"
        # The symlink target must be untouched.
        assert target.read_text() == "original secret\n"


# ---------------------------------------------------------------------------
# build_entry() — shape + truncation
# ---------------------------------------------------------------------------

class TestBuildEntry:
    def test_basic_shape(self) -> None:
        e = build_entry(
            provider="anthropic", model="claude-x",
            request_body=b'{"q":"hi"}', response_body=b'{"a":"yo"}',
            content_type="application/json",
            input_tokens=10, output_tokens=5, status_code=200, stream=False,
        )
        assert e["provider"] == "anthropic"
        assert e["model"] == "claude-x"
        assert e["input_tokens"] == 10
        assert e["output_tokens"] == 5
        assert e["request"] == '{"q":"hi"}'
        assert e["response"] == '{"a":"yo"}'
        assert e["request_truncated"] is False
        assert e["response_truncated"] is False

    def test_truncation_marks_flag(self) -> None:
        big = b"x" * (2 * 1024 * 1024)  # 2 MB > 1 MB cap
        e = build_entry(
            provider="openai", model="m",
            request_body=big, response_body=b"ok",
            content_type="application/json",
            input_tokens=0, output_tokens=0, status_code=200, stream=False,
        )
        assert e["request_truncated"] is True
        assert len(e["request"]) == 1 * 1024 * 1024
        assert e["response_truncated"] is False


# ---------------------------------------------------------------------------
# _load_hooks() — capture mode forces empty chain
# ---------------------------------------------------------------------------

class TestLoadHooksCaptureMode:
    def test_capture_mode_returns_empty_hooks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLM_OPTIMIZE_CAPTURE", "1")
        cfg = OptimizeConfig.from_dict(dict(
            enabled=True, proxy_enabled=True,
            cache_enabled=True, compress_enabled=True, ttl_seconds=300,
        ))
        hooks = _load_hooks(cfg)
        assert hooks.cache is None
        assert hooks.compress is None

    def test_non_capture_mode_loads_hooks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SLM_OPTIMIZE_CAPTURE", raising=False)
        cfg = OptimizeConfig.from_dict(dict(
            enabled=True, proxy_enabled=True,
            cache_enabled=False, compress_enabled=False, ttl_seconds=300,
        ))
        hooks = _load_hooks(cfg)
        # cache/compress disabled in config → still None, but not via capture gate
        assert hooks.cache is None
        assert hooks.compress is None


# ---------------------------------------------------------------------------
# Integration: surfaces in capture mode → passthrough + record
# ---------------------------------------------------------------------------

class _MockTransport(httpx.AsyncBaseTransport):
    def __init__(self, handler):
        self._handler = handler
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return await self._handler(request)


def _make_app(transport: _MockTransport) -> tuple[FastAPI, ProxyApp]:
    cfg = OptimizeConfig.from_dict(dict(
        enabled=True, proxy_enabled=True,
        cache_enabled=False, compress_enabled=False, ttl_seconds=300,
    ))
    proxy = ProxyApp(config=cfg)
    proxy.hooks = HookChain.empty()  # capture mode = empty hooks
    proxy.http_client = httpx.AsyncClient(transport=transport)
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")
    return app, proxy


@pytest.fixture
def capture_to_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Enable capture mode and redirect the corpus to a temp file."""
    monkeypatch.setenv("SLM_OPTIMIZE_CAPTURE", "1")
    ShadowCapture.reset_instance()
    cap = ShadowCapture(path=tmp_path / "optimize_capture.jsonl")
    monkeypatch.setattr(ShadowCapture, "_instance", cap)
    yield cap
    ShadowCapture.reset_instance()


@pytest.mark.asyncio
async def test_anthropic_capture_records_and_passthrough(capture_to_tmp) -> None:
    anthropic_resp = {
        "id": "msg_01", "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": "hello"}],
        "model": "claude-sonnet-4-6", "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 6},
    }

    async def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=anthropic_resp)

    app, proxy = _make_app(_MockTransport(_handler))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/messages",
            json={"model": "claude-sonnet-4-6", "max_tokens": 50,
                  "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-api-key": "sk-ant", "content-type": "application/json"},
        )
    await proxy.http_client.aclose()

    # passthrough: client got the upstream response unchanged
    assert resp.status_code == 200
    assert resp.json()["content"][0]["text"] == "hello"

    # recorded: one corpus line with correct usage + provider
    lines = capture_to_tmp.path.read_text().strip().split("\n")
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["provider"] == "anthropic"
    assert entry["model"] == "claude-sonnet-4-6"
    assert entry["input_tokens"] == 12
    assert entry["output_tokens"] == 6
    assert entry["stream"] is False
    assert json.loads(entry["request"])["messages"][0]["content"] == "hi"


@pytest.mark.asyncio
async def test_openai_capture_records(capture_to_tmp) -> None:
    openai_resp = {
        "id": "chatcmpl-1", "object": "chat.completion", "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "yo"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 8, "completion_tokens": 2},
    }

    async def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=openai_resp)

    app, proxy = _make_app(_MockTransport(_handler))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
            headers={"authorization": "Bearer sk-x", "content-type": "application/json"},
        )
    await proxy.http_client.aclose()

    assert resp.status_code == 200
    entry = json.loads(capture_to_tmp.path.read_text().strip())
    assert entry["provider"] == "openai"
    assert entry["model"] == "gpt-4o"
    assert entry["input_tokens"] == 8
    assert entry["output_tokens"] == 2


@pytest.mark.asyncio
async def test_gemini_native_capture_records(capture_to_tmp) -> None:
    gemini_resp = {
        "candidates": [{"content": {"parts": [{"text": "hi"}], "role": "model"},
                        "finishReason": "STOP"}],
        "modelVersion": "gemini-2.0-flash",
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3},
    }

    async def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=gemini_resp)

    app, proxy = _make_app(_MockTransport(_handler))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1beta/models/gemini-2.0-flash:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            headers={"x-goog-api-key": "k", "content-type": "application/json"},
        )
    await proxy.http_client.aclose()

    assert resp.status_code == 200
    entry = json.loads(capture_to_tmp.path.read_text().strip())
    assert entry["provider"] == "gemini"
    assert entry["model"] == "gemini-2.0-flash"
    assert entry["input_tokens"] == 5
    assert entry["output_tokens"] == 3


@pytest.mark.asyncio
async def test_capture_failopen_when_upstream_errors(capture_to_tmp) -> None:
    """Upstream error in capture mode → fail-open response, no crash."""
    async def _handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("upstream down", request=request)

    app, proxy = _make_app(_MockTransport(_handler))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/messages",
            json={"model": "claude-sonnet-4-6", "max_tokens": 10,
                  "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-api-key": "sk-ant", "content-type": "application/json"},
        )
    await proxy.http_client.aclose()
    # fail-open returns a 502 error envelope, not an exception
    assert resp.status_code == 502
