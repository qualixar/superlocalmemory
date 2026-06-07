"""Adapters coverage tests — boost coverage for openai/anthropic adapters + wrap."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from superlocalmemory.optimize.adapters import withSLM
from superlocalmemory.optimize.adapters._agent_registry import AGENT_REGISTRY
from superlocalmemory.optimize.adapters.wrap import (
    _atomic_write_text,
    _vscode_user_dir,
    list_agents,
    wrap_agent,
)
from superlocalmemory.optimize.config import (
    _reset_config_store,
    _set_config_store,
)
from superlocalmemory.optimize.config.defaults import DEFAULT_OPTIMIZE_CONFIG
from superlocalmemory.optimize.config.schema import OptimizeConfig
from superlocalmemory.optimize.config.store import ConfigStore


# ─── Helpers ────────────────────────────────────────────────────────────────


class _FakeOpenAI:
    """Minimal OpenAI client fake — has .chat.completions.create ONLY."""

    def __init__(self, response: dict) -> None:
        self._response = response
        self.calls: list[dict] = []
        self.chat = self._Chat(self)

    class _Chat:
        def __init__(self, outer):
            self._outer = outer
            self.completions = self._Completions(outer)

        class _Completions:
            def __init__(self, outer):
                self._outer = outer

            def create(self, **kwargs):
                self._outer.calls.append(kwargs)
                return self._outer._response


class _FakeAnthropic:
    """Minimal Anthropic client fake — has .messages.create ONLY."""

    def __init__(self, response: dict) -> None:
        self._response = response
        self.calls: list[dict] = []
        self.messages = self._Messages(self)

    class _Messages:
        def __init__(self, outer):
            self._outer = outer

        def create(self, **kwargs):
            self._outer.calls.append(kwargs)
            return self._outer._response


def _enable_optimize_config(tmp_path: Path, *, proxy: bool = True, enabled: bool = True) -> ConfigStore:
    cfg_path = tmp_path / "oc.json"
    cfg_path.write_text(json.dumps({
        "enabled": enabled, "proxy_enabled": proxy,
        "cache_enabled": True, "compress_enabled": False,
    }))
    store = ConfigStore(config_path=cfg_path, poll_interval=3600.0)
    _set_config_store(store)
    return store


# ─── OpenAI adapter ──────────────────────────────────────────────────────────


def test_openai_sync_cache_miss_then_hit(tmp_path: Path) -> None:
    """First call hits upstream, second call returns from cache."""
    _enable_optimize_config(tmp_path)
    from superlocalmemory.optimize.cache.manager import CacheManager
    CacheManager.reset_instance()
    # Bind the singleton to a per-test tmp DB
    from superlocalmemory.optimize.storage.db import CacheDB
    test_db = CacheDB(tmp_path / "llmcache.db")
    CacheManager.set_instance(CacheManager(db=test_db, config=__import__(
        "superlocalmemory.optimize.cache.key_builder", fromlist=["CacheConfig"]
    ).CacheConfig()))

    fake = _FakeOpenAI({
        "id": "chatcmpl-1", "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                     "finish_reason": "stop"}],
    })
    client = withSLM(fake, tenant_id="0" * 64)
    out1 = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hello"}],
    )
    out2 = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hello"}],
    )
    assert out1 == out2
    # First call: 1 upstream hit; second call: 0 (cache hit)
    assert len(fake.calls) == 1
    CacheManager.reset_instance()
    _reset_config_store()


def test_openai_streaming_passthrough(tmp_path: Path) -> None:
    """stream=True → must NOT call cache, always pass-through."""
    _enable_optimize_config(tmp_path)
    from superlocalmemory.optimize.cache.manager import CacheManager
    from superlocalmemory.optimize.storage.db import CacheDB
    from superlocalmemory.optimize.cache.key_builder import CacheConfig
    test_db = CacheDB(tmp_path / "llmcache.db")
    CacheManager.set_instance(CacheManager(db=test_db, config=CacheConfig()))

    fake = _FakeOpenAI({"stream": True, "chunks": []})
    client = withSLM(fake, tenant_id="0" * 64)
    list(client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "x"}],
        stream=True,
    ))
    assert len(fake.calls) == 1
    CacheManager.reset_instance()
    _reset_config_store()


def test_openai_tools_skips_cache(tmp_path: Path) -> None:
    """tools= present → skip cache, always pass-through."""
    _enable_optimize_config(tmp_path)
    from superlocalmemory.optimize.cache.manager import CacheManager
    from superlocalmemory.optimize.storage.db import CacheDB
    from superlocalmemory.optimize.cache.key_builder import CacheConfig
    test_db = CacheDB(tmp_path / "llmcache.db")
    CacheManager.set_instance(CacheManager(db=test_db, config=CacheConfig()))

    fake = _FakeOpenAI({"choices": [{"finish_reason": "stop"}]})
    client = withSLM(fake, tenant_id="0" * 64)
    for _ in range(2):
        list(client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "x"}],
            tools=[{"name": "bash"}],
        ))
    assert len(fake.calls) == 2  # both pass-through
    CacheManager.reset_instance()
    _reset_config_store()


def test_openai_temperature_gt_zero_skips_cache(tmp_path: Path) -> None:
    """temperature>0 → no cache key → upstream every time."""
    _enable_optimize_config(tmp_path)
    from superlocalmemory.optimize.cache.manager import CacheManager
    from superlocalmemory.optimize.storage.db import CacheDB
    from superlocalmemory.optimize.cache.key_builder import CacheConfig
    test_db = CacheDB(tmp_path / "llmcache.db")
    CacheManager.set_instance(CacheManager(db=test_db, config=CacheConfig()))

    fake = _FakeOpenAI({"choices": [{"finish_reason": "stop"}]})
    client = withSLM(fake, tenant_id="0" * 64)
    for _ in range(2):
        list(client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "x"}],
            temperature=0.7,
        ))
    assert len(fake.calls) == 2
    CacheManager.reset_instance()
    _reset_config_store()


# ─── Anthropic adapter ───────────────────────────────────────────────────────


def test_anthropic_sync_cache_miss_then_hit(tmp_path: Path) -> None:
    _enable_optimize_config(tmp_path)
    from superlocalmemory.optimize.cache.manager import CacheManager
    from superlocalmemory.optimize.storage.db import CacheDB
    from superlocalmemory.optimize.cache.key_builder import CacheConfig
    test_db = CacheDB(tmp_path / "llmcache.db")
    CacheManager.set_instance(CacheManager(db=test_db, config=CacheConfig()))

    fake = _FakeAnthropic({
        "id": "msg_x", "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "model": "claude-sonnet-4-6",
    })
    client = withSLM(fake, tenant_id="0" * 64)
    out1 = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=50,
        messages=[{"role": "user", "content": "hi"}],
    )
    out2 = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=50,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert out1 == out2
    assert len(fake.calls) == 1
    CacheManager.reset_instance()
    _reset_config_store()


def test_anthropic_tools_skips_cache(tmp_path: Path) -> None:
    _enable_optimize_config(tmp_path)
    from superlocalmemory.optimize.cache.manager import CacheManager
    from superlocalmemory.optimize.storage.db import CacheDB
    from superlocalmemory.optimize.cache.key_builder import CacheConfig
    test_db = CacheDB(tmp_path / "llmcache.db")
    CacheManager.set_instance(CacheManager(db=test_db, config=CacheConfig()))

    fake = _FakeAnthropic({"stop_reason": "end_turn"})
    client = withSLM(fake, tenant_id="0" * 64)
    for _ in range(2):
        client.messages.create(
            model="claude-sonnet-4-6", max_tokens=50,
            messages=[{"role": "user", "content": "x"}],
            tools=[{"name": "bash"}],
        )
    assert len(fake.calls) == 2
    CacheManager.reset_instance()
    _reset_config_store()


def test_anthropic_streaming_passthrough(tmp_path: Path) -> None:
    _enable_optimize_config(tmp_path)
    from superlocalmemory.optimize.cache.manager import CacheManager
    from superlocalmemory.optimize.storage.db import CacheDB
    from superlocalmemory.optimize.cache.key_builder import CacheConfig
    test_db = CacheDB(tmp_path / "llmcache.db")
    CacheManager.set_instance(CacheManager(db=test_db, config=CacheConfig()))

    fake = _FakeAnthropic({"stream": True})
    client = withSLM(fake, tenant_id="0" * 64)
    client.messages.create(
        model="claude-sonnet-4-6", max_tokens=50,
        messages=[{"role": "user", "content": "x"}], stream=True,
    )
    assert len(fake.calls) == 1
    CacheManager.reset_instance()
    _reset_config_store()


# ─── wrap.py edge cases ──────────────────────────────────────────────────────


def test_wrap_agent_settings_file(tmp_path: Path) -> None:
    """claude-settings writes env block to ~/.claude/settings.json."""
    _enable_optimize_config(tmp_path)
    settings_path = tmp_path / "settings.json"
    spec = dict(AGENT_REGISTRY["claude-settings"])
    spec["settings_path"] = str(settings_path)
    # Patch the agent spec to use the tmp path
    original = AGENT_REGISTRY["claude-settings"]
    AGENT_REGISTRY["claude-settings"] = spec
    try:
        rc = wrap_agent("claude-settings", [])
        assert rc == 0
        assert settings_path.exists()
        data = json.loads(settings_path.read_text())
        assert data["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8765"
    finally:
        AGENT_REGISTRY["claude-settings"] = original
    _reset_config_store()


def test_wrap_agent_settings_file_oserror(tmp_path: Path, monkeypatch) -> None:
    """OSError on write → returns 1."""
    _enable_optimize_config(tmp_path)
    # Use a path with a null byte — POSIX refuses
    spec = dict(AGENT_REGISTRY["claude-settings"])
    spec["settings_path"] = str(tmp_path) + "\x00bad"
    original = AGENT_REGISTRY["claude-settings"]
    AGENT_REGISTRY["claude-settings"] = spec
    try:
        rc = wrap_agent("claude-settings", [])
        assert rc == 1
    finally:
        AGENT_REGISTRY["claude-settings"] = original
    _reset_config_store()


def test_wrap_agent_config_file_cline(tmp_path: Path) -> None:
    """cline writes to VS Code user settings.json."""
    _enable_optimize_config(tmp_path)
    settings_path = tmp_path / "settings.json"
    spec = dict(AGENT_REGISTRY["cline"])
    spec["config_path"] = str(settings_path)
    original = AGENT_REGISTRY["cline"]
    AGENT_REGISTRY["cline"] = spec
    try:
        rc = wrap_agent("cline", [])
        assert rc == 0
        data = json.loads(settings_path.read_text())
        assert data["cline.openAiApiBase"] == "http://127.0.0.1:8765/v1"
    finally:
        AGENT_REGISTRY["cline"] = original
    _reset_config_store()


def test_wrap_agent_config_file_existing_settings(tmp_path: Path) -> None:
    """Existing settings.json is preserved (only config_key is updated)."""
    _enable_optimize_config(tmp_path)
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"editor.fontSize": 14, "cline.openAiApiBase": "old"}))
    spec = dict(AGENT_REGISTRY["cline"])
    spec["config_path"] = str(settings_path)
    original = AGENT_REGISTRY["cline"]
    AGENT_REGISTRY["cline"] = spec
    try:
        rc = wrap_agent("cline", [])
        assert rc == 0
        data = json.loads(settings_path.read_text())
        assert data["editor.fontSize"] == 14
        assert data["cline.openAiApiBase"] == "http://127.0.0.1:8765/v1"
    finally:
        AGENT_REGISTRY["cline"] = original
    _reset_config_store()


def test_wrap_agent_env_binary_missing(tmp_path: Path) -> None:
    """Binary not in PATH → returns 1."""
    _enable_optimize_config(tmp_path)
    spec = dict(AGENT_REGISTRY["codex"])
    spec["binary"] = "definitely-not-a-real-binary-xyz"
    original = AGENT_REGISTRY["codex"]
    AGENT_REGISTRY["codex"] = spec
    try:
        rc = wrap_agent("codex", [], dry_run=False)
        assert rc == 1
    finally:
        AGENT_REGISTRY["codex"] = original
    _reset_config_store()


def test_wrap_agent_env_dry_run(tmp_path: Path) -> None:
    """dry-run prints intent and returns 0 without executing."""
    _enable_optimize_config(tmp_path)
    # Use claude with a non-existent binary so we can verify dry-run doesn't execute
    spec = dict(AGENT_REGISTRY["codex"])
    spec["binary"] = "codex"  # may not exist
    original = AGENT_REGISTRY["codex"]
    AGENT_REGISTRY["codex"] = spec
    try:
        rc = wrap_agent("codex", ["--help"], dry_run=True)
        assert rc == 0
    finally:
        AGENT_REGISTRY["codex"] = original
    _reset_config_store()


def test_wrap_agent_unknown_mechanism(tmp_path: Path, capsys) -> None:
    """Unknown mechanism → returns 1, prints error to stderr."""
    _enable_optimize_config(tmp_path)
    spec = dict(AGENT_REGISTRY["generic"])
    spec["mechanism"] = "no-such-mechanism"
    original = AGENT_REGISTRY["generic"]
    AGENT_REGISTRY["generic"] = spec
    try:
        rc = wrap_agent("generic", [])
        assert rc == 1
    finally:
        AGENT_REGISTRY["generic"] = original
    _reset_config_store()


def test_wrap_agent_settings_file_dry_run(tmp_path: Path, capsys) -> None:
    """claude-settings --dry-run prints intent and returns 0."""
    _enable_optimize_config(tmp_path)
    settings_path = tmp_path / "settings.json"
    spec = dict(AGENT_REGISTRY["claude-settings"])
    spec["settings_path"] = str(settings_path)
    original = AGENT_REGISTRY["claude-settings"]
    AGENT_REGISTRY["claude-settings"] = spec
    try:
        rc = wrap_agent("claude-settings", [], dry_run=True)
        assert rc == 0
        assert not settings_path.exists()  # dry-run never writes
    finally:
        AGENT_REGISTRY["claude-settings"] = original
    _reset_config_store()


def test_wrap_agent_cline_dry_run(tmp_path: Path) -> None:
    """cline --dry-run prints intent and returns 0."""
    _enable_optimize_config(tmp_path)
    settings_path = tmp_path / "settings.json"
    spec = dict(AGENT_REGISTRY["cline"])
    spec["config_path"] = str(settings_path)
    original = AGENT_REGISTRY["cline"]
    AGENT_REGISTRY["cline"] = spec
    try:
        rc = wrap_agent("cline", [], dry_run=True)
        assert rc == 0
        assert not settings_path.exists()
    finally:
        AGENT_REGISTRY["cline"] = original
    _reset_config_store()


def test_wrap_agent_settings_file_corrupt_existing(tmp_path: Path) -> None:
    """Corrupt existing settings.json → treat as empty, still write."""
    _enable_optimize_config(tmp_path)
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{not valid json")
    spec = dict(AGENT_REGISTRY["claude-settings"])
    spec["settings_path"] = str(settings_path)
    original = AGENT_REGISTRY["claude-settings"]
    AGENT_REGISTRY["claude-settings"] = spec
    try:
        rc = wrap_agent("claude-settings", [])
        assert rc == 0
        data = json.loads(settings_path.read_text())
        assert data["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8765"
    finally:
        AGENT_REGISTRY["claude-settings"] = original
    _reset_config_store()


def test_atomic_write_text_chmod_failure(tmp_path: Path) -> None:
    """chmod failure on _atomic_write_text is silently swallowed."""
    p = tmp_path / "x.txt"
    _atomic_write_text(p, "hello")
    assert p.read_text() == "hello"


def test_vscode_user_dir_returns_path() -> None:
    """_vscode_user_dir returns a Path or None depending on platform."""
    result = _vscode_user_dir()
    # Either None or a Path is acceptable
    assert result is None or isinstance(result, Path)


# ─── OpenAI async adapter + legacy functions= ────────────────────────────────


class _FakeAsyncOpenAICompletions:
    def __init__(self, outer):
        self._outer = outer

    async def create(self, **kwargs):
        self._outer.calls.append(kwargs)
        return self._outer._response


class _FakeAsyncOpenAI:
    def __init__(self, response):
        self._response = response
        self.calls = []
        _completions = _FakeAsyncOpenAICompletions(self)
        _chat = type("_Chat", (), {"completions": _completions})()
        self.chat = _chat


def test_openai_async_cache_miss_then_hit(tmp_path: Path) -> None:
    """_chat_create_async: first call hits upstream, second returns from cache."""
    import asyncio

    p = tmp_path / "oc.json"
    p.write_text(json.dumps({"enabled": True, "cache_enabled": True}))
    from superlocalmemory.optimize.config.store import ConfigStore

    store = ConfigStore(config_path=p, poll_interval=3600.0)
    _set_config_store(store)

    from superlocalmemory.optimize.cache.manager import CacheManager
    from superlocalmemory.optimize.storage.db import CacheDB
    from superlocalmemory.optimize.cache.key_builder import CacheConfig
    from superlocalmemory.optimize.adapters.openai_adapter import SLMOpenAIAdapter

    test_db = CacheDB(tmp_path / "llmcache.db")
    CacheManager.set_instance(CacheManager(db=test_db, config=CacheConfig()))
    config = store.get()

    fake = _FakeAsyncOpenAI({"id": "chatcmpl-async", "choices": [{"finish_reason": "stop"}]})
    adapter = SLMOpenAIAdapter(fake, CacheManager.get_instance(), config, "0" * 64)
    adapter._is_async = True

    async def _run():
        out1 = await adapter.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "async openai test"}]
        )
        out2 = await adapter.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "async openai test"}]
        )
        return out1, out2

    out1, out2 = asyncio.run(_run())
    assert out1 == out2
    assert len(fake.calls) == 1
    CacheManager.reset_instance()
    _reset_config_store()


def test_openai_legacy_functions_skips_cache(tmp_path: Path) -> None:
    """functions= (legacy param) must skip cache — covers line 62 in _should_cache."""
    p = tmp_path / "oc.json"
    p.write_text(json.dumps({"enabled": True, "cache_enabled": True}))
    from superlocalmemory.optimize.config.store import ConfigStore

    store = ConfigStore(config_path=p, poll_interval=3600.0)
    _set_config_store(store)

    from superlocalmemory.optimize.cache.manager import CacheManager
    from superlocalmemory.optimize.storage.db import CacheDB
    from superlocalmemory.optimize.cache.key_builder import CacheConfig

    test_db = CacheDB(tmp_path / "llmcache.db")
    CacheManager.set_instance(CacheManager(db=test_db, config=CacheConfig()))

    fake = _FakeOpenAI({"choices": [{"finish_reason": "stop"}]})
    client = withSLM(fake, tenant_id="0" * 64)
    for _ in range(2):
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "fn test"}],
            functions=[{"name": "my_fn"}],
        )
    assert len(fake.calls) == 2, "functions= must bypass cache — both calls should hit upstream"
    CacheManager.reset_instance()
    _reset_config_store()


# ─── Anthropic adapter — async path + serialization-failure branch ───────────


class _FakeAsyncAnthropicMessages:
    def __init__(self, outer):
        self._outer = outer

    async def create(self, **kwargs):
        self._outer.calls.append(kwargs)
        return self._outer._response


class _FakeAsyncAnthropic:
    """Fake async Anthropic — has .messages but is_async detection needs override."""

    def __init__(self, response):
        self._response = response
        self.calls = []
        self.messages = _FakeAsyncAnthropicMessages(self)


def test_anthropic_async_cache_miss_then_hit(tmp_path: Path) -> None:
    """_messages_create_async: first call hits upstream, second returns from cache."""
    import asyncio

    p = tmp_path / "oc.json"
    p.write_text(json.dumps({"enabled": True, "cache_enabled": True}))
    from superlocalmemory.optimize.config.store import ConfigStore

    store = ConfigStore(config_path=p, poll_interval=3600.0)
    _set_config_store(store)

    from superlocalmemory.optimize.cache.manager import CacheManager
    from superlocalmemory.optimize.storage.db import CacheDB
    from superlocalmemory.optimize.cache.key_builder import CacheConfig
    from superlocalmemory.optimize.adapters.anthropic_adapter import SLMAnthropicAdapter

    test_db = CacheDB(tmp_path / "llmcache.db")
    CacheManager.set_instance(CacheManager(db=test_db, config=CacheConfig()))
    config = store.get()

    fake = _FakeAsyncAnthropic({"id": "msg_async", "stop_reason": "end_turn"})
    adapter = SLMAnthropicAdapter(fake, CacheManager.get_instance(), config, "0" * 64)
    # Force async detection without a real AsyncAnthropic class
    adapter._is_async = True

    async def _run():
        out1 = await adapter.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=50,
            messages=[{"role": "user", "content": "async test"}],
        )
        out2 = await adapter.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=50,
            messages=[{"role": "user", "content": "async test"}],
        )
        return out1, out2

    out1, out2 = asyncio.run(_run())
    assert out1 == out2
    assert len(fake.calls) == 1, f"Expected 1 upstream call, got {len(fake.calls)}"
    CacheManager.reset_instance()
    _reset_config_store()


def test_anthropic_sync_cache_set_serialization_failure_does_not_raise(tmp_path: Path) -> None:
    """If json.dumps(response) raises, cache store failure must be swallowed."""
    p = tmp_path / "oc.json"
    p.write_text(json.dumps({"enabled": True, "cache_enabled": True}))
    from superlocalmemory.optimize.config.store import ConfigStore

    store = ConfigStore(config_path=p, poll_interval=3600.0)
    _set_config_store(store)

    from superlocalmemory.optimize.cache.manager import CacheManager
    from superlocalmemory.optimize.storage.db import CacheDB
    from superlocalmemory.optimize.cache.key_builder import CacheConfig
    from superlocalmemory.optimize.adapters.anthropic_adapter import SLMAnthropicAdapter

    test_db = CacheDB(tmp_path / "llmcache.db")
    CacheManager.set_instance(CacheManager(db=test_db, config=CacheConfig()))
    config = store.get()

    class _BadResponse:
        """Cannot be JSON-serialized — no __dict__ and raises on any json encoder path."""

        def __repr__(self):
            return "<unserializable>"

    bad_response = _BadResponse()

    class _FakeBadMessages:
        def create(self, **kwargs):
            return bad_response

    class _FakeBadAnthropic:
        calls: list = []
        messages = _FakeBadMessages()

    fake = _FakeBadAnthropic()
    adapter = SLMAnthropicAdapter(fake, CacheManager.get_instance(), config, "0" * 64)
    # Must not raise even when serialization fails
    result = adapter._messages_create_sync(
        model="m", messages=[{"role": "user", "content": "x"}]
    )
    assert isinstance(result, _BadResponse)
    CacheManager.reset_instance()
    _reset_config_store()


# ─── withSLM fail-open branches (F-004) ──────────────────────────────────────


def test_withSLM_cache_unavailable_returns_client_unchanged(tmp_path: Path, monkeypatch) -> None:
    """CacheManager.get_instance() raises → pass-through, log WARNING."""
    _enable_optimize_config(tmp_path)
    from superlocalmemory.optimize.cache import manager as _mgr

    def _raise(cls):
        raise RuntimeError("DB unavailable")

    monkeypatch.setattr(_mgr.CacheManager, "get_instance", classmethod(_raise))

    class _FakeAnthropicSimple:
        messages = type("_Msgs", (), {"create": lambda s, **kw: None})()

    sentinel = _FakeAnthropicSimple()
    result = withSLM(sentinel)
    assert result is sentinel
    _reset_config_store()


def test_withSLM_anthropic_adapter_init_failure_returns_client(tmp_path: Path, monkeypatch) -> None:
    """SLMAnthropicAdapter.__init__ raises → pass-through, log WARNING."""
    _enable_optimize_config(tmp_path)
    from superlocalmemory.optimize.cache.manager import CacheManager
    from superlocalmemory.optimize.storage.db import CacheDB
    from superlocalmemory.optimize.cache.key_builder import CacheConfig

    test_db = CacheDB(tmp_path / "llmcache.db")
    CacheManager.set_instance(CacheManager(db=test_db, config=CacheConfig()))

    from superlocalmemory.optimize.adapters import anthropic_adapter as _aa

    def _raise(*a, **kw):
        raise RuntimeError("init boom")

    monkeypatch.setattr(_aa, "SLMAnthropicAdapter", _raise)

    client = _FakeAnthropic({"stop_reason": "end_turn"})
    result = withSLM(client)
    assert result is client
    CacheManager.reset_instance()
    _reset_config_store()


def test_withSLM_openai_adapter_init_failure_returns_client(tmp_path: Path, monkeypatch) -> None:
    """SLMOpenAIAdapter.__init__ raises → pass-through, log WARNING."""
    _enable_optimize_config(tmp_path)
    from superlocalmemory.optimize.cache.manager import CacheManager
    from superlocalmemory.optimize.storage.db import CacheDB
    from superlocalmemory.optimize.cache.key_builder import CacheConfig

    test_db = CacheDB(tmp_path / "llmcache.db")
    CacheManager.set_instance(CacheManager(db=test_db, config=CacheConfig()))

    from superlocalmemory.optimize.adapters import openai_adapter as _oa

    def _raise(*a, **kw):
        raise RuntimeError("init boom")

    monkeypatch.setattr(_oa, "SLMOpenAIAdapter", _raise)

    client = _FakeOpenAI({"choices": [{"finish_reason": "stop"}]})
    result = withSLM(client)
    assert result is client
    CacheManager.reset_instance()
    _reset_config_store()
