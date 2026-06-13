# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""TDD — RED phase: 15 test cases for Surface B MCP optimize tools.

LLD-01 v2 §9: write tests first, implement to make them green.
95% coverage floor / 100% target.

NEVER TOUCH: memory.db, conftest.py, test_router.py.
"""

from __future__ import annotations

import re
import uuid
from unittest.mock import MagicMock

import pytest

_UUID4_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
)


# ─── Helpers ─────────────────────────────────────────────────────────────────


class _MockServer:
    """Minimal mock that captures @server.tool() decorated functions."""

    def __init__(self):
        self._tools: dict = {}

    def tool(self, *args, **kwargs):
        def decorator(fn):
            self._tools[fn.__name__] = fn
            return fn
        return decorator


@pytest.fixture()
def tools():
    from superlocalmemory.mcp.tools_optimize import register_optimize_tools
    srv = _MockServer()
    register_optimize_tools(srv)
    return srv._tools


@pytest.fixture(autouse=True)
def _reset_kv_counters():
    """Reset module-level KV counters before each test (isolated stats)."""
    try:
        import superlocalmemory.mcp.tools_optimize as mod
        mod._kv_hits = 0
        mod._kv_misses = 0
    except ImportError:
        pass
    yield


# ─── Test 1: slm_compress lossless path ──────────────────────────────────────


async def test_compress_lossless_no_ccr(tools, monkeypatch):
    """ok:True, strategy in {normalize,none}, ccr_id is None when lossy=False."""
    from superlocalmemory.optimize.compress.router import CompressTextResult

    mock_router = MagicMock()
    mock_router.compress_text.return_value = CompressTextResult(
        compressed_text="short",
        strategy="none",
        tokens_before=10,
        tokens_after=8,
        lossy=False,
    )
    mock_cr = MagicMock()
    mock_cr.get_instance.return_value = mock_router
    monkeypatch.setattr("superlocalmemory.mcp.tools_optimize.CompressRouter", mock_cr)

    result = await tools["slm_compress"](content="some text to compress")

    assert result["ok"] is True
    assert result["strategy"] in {"normalize", "none"}
    assert result["ccr_id"] is None
    assert result["lossy"] is False
    assert "tokens_before" in result
    assert "tokens_after" in result


# ─── Test 2: mode="normalize" always lossless, uses real _normalize_whitespace ─


async def test_compress_normalize_mode_lossless(tools):
    """mode=normalize → lossless, calls _normalize_whitespace directly (no engine).

    _normalize_whitespace: collapses 3+ consecutive newlines to 2, strips trailing
    spaces per line. It does NOT collapse inline spaces.
    """
    result = await tools["slm_compress"](
        content="line1   \nline2  \n\n\nline3",  # trailing spaces + 3-newline run
        mode="normalize",
    )
    assert result["ok"] is True
    assert result["strategy"] == "normalize"
    assert result["lossy"] is False
    assert result["ccr_id"] is None
    assert "\n\n\n" not in result["compressed"]  # 3+ newlines collapsed


# ─── Test 3: lossy path → ccr_id is UUID4 ────────────────────────────────────


async def test_compress_lossy_stores_ccr(tools, monkeypatch):
    """lossy=True + reversible=True → ccr_id is a valid UUID4."""
    from superlocalmemory.optimize.compress.router import CompressTextResult

    fake_id = str(uuid.uuid4())
    mock_router = MagicMock()
    mock_router.compress_text.return_value = CompressTextResult(
        compressed_text="short",
        strategy="llmlingua2_prose",
        tokens_before=100,
        tokens_after=40,
        lossy=True,
    )
    mock_cr = MagicMock()
    mock_cr.get_instance.return_value = mock_router
    mock_ccr = MagicMock()
    mock_ccr.get_instance.return_value = MagicMock(store=MagicMock(return_value=fake_id))

    monkeypatch.setattr("superlocalmemory.mcp.tools_optimize.CompressRouter", mock_cr)
    monkeypatch.setattr("superlocalmemory.mcp.tools_optimize.CCRStore", mock_ccr)

    result = await tools["slm_compress"](content="A" * 1000, mode="auto", reversible=True)

    assert result["ok"] is True
    assert result["lossy"] is True
    assert result["ccr_id"] is not None
    assert _UUID4_RE.match(result["ccr_id"]), f"Not UUID4: {result['ccr_id']}"


# ─── Test 4: CCR round-trip byte-identical ────────────────────────────────────


async def test_ccr_round_trip_byte_identical(tools, monkeypatch):
    """compress(reversible=True, lossy) → retrieve(ccr_id) → original bytes."""
    from superlocalmemory.optimize.compress.router import CompressTextResult

    original = "Original content: 日本語テスト — must survive byte-identical."
    fake_id = str(uuid.uuid4())

    mock_router = MagicMock()
    mock_router.compress_text.return_value = CompressTextResult(
        compressed_text="compressed",
        strategy="llmlingua2_prose",
        tokens_before=50,
        tokens_after=20,
        lossy=True,
    )
    ccr_store_mock = MagicMock()
    ccr_store_mock.store.return_value = fake_id
    ccr_store_mock.retrieve.return_value = original.encode("utf-8")

    mock_cr = MagicMock()
    mock_cr.get_instance.return_value = mock_router
    mock_ccr = MagicMock()
    mock_ccr.get_instance.return_value = ccr_store_mock

    monkeypatch.setattr("superlocalmemory.mcp.tools_optimize.CompressRouter", mock_cr)
    monkeypatch.setattr("superlocalmemory.mcp.tools_optimize.CCRStore", mock_ccr)

    c = await tools["slm_compress"](content=original, mode="auto", reversible=True)
    assert c["ccr_id"] == fake_id

    r = await tools["slm_retrieve"](ccr_id=fake_id)
    assert r["ok"] is True
    assert r["content"] == original
    assert r["size_bytes"] == len(original.encode("utf-8"))


# ─── Test 5: slm_retrieve bad id (not UUID4) ─────────────────────────────────


async def test_retrieve_bad_id_not_uuid4(tools):
    """Non-UUID4 ccr_id → ok:False, size_bytes:0, no raise."""
    result = await tools["slm_retrieve"](ccr_id="not-a-uuid")
    assert result["ok"] is False
    assert result["size_bytes"] == 0
    assert result["content"] is None


# ─── Test 6: slm_retrieve missing id (UUID4 format, not found) ───────────────


async def test_retrieve_missing_id(tools, monkeypatch):
    """Valid UUID4 but not in store → ok:False."""
    ccr_store_mock = MagicMock()
    ccr_store_mock.retrieve.return_value = None
    mock_ccr = MagicMock()
    mock_ccr.get_instance.return_value = ccr_store_mock
    monkeypatch.setattr("superlocalmemory.mcp.tools_optimize.CCRStore", mock_ccr)

    result = await tools["slm_retrieve"](ccr_id=str(uuid.uuid4()))
    assert result["ok"] is False
    assert result["content"] is None
    assert result["size_bytes"] == 0


# ─── Test 7: slm_retrieve includes size_bytes matching original byte length ───


async def test_retrieve_size_bytes_matches_original(tools, monkeypatch):
    """size_bytes in retrieve response equals len(original_bytes)."""
    original = "Exactly this text."
    fake_id = str(uuid.uuid4())

    ccr_store_mock = MagicMock()
    ccr_store_mock.retrieve.return_value = original.encode("utf-8")
    mock_ccr = MagicMock()
    mock_ccr.get_instance.return_value = ccr_store_mock
    monkeypatch.setattr("superlocalmemory.mcp.tools_optimize.CCRStore", mock_ccr)

    result = await tools["slm_retrieve"](ccr_id=fake_id)
    assert result["ok"] is True
    assert result["size_bytes"] == len(original.encode("utf-8"))


# ─── Test 8: KV set/get hit + miss + tenant isolation ────────────────────────


async def test_kv_set_get_hit_miss_and_tenant_isolation(tools, monkeypatch):
    """KV hit, miss, tenant isolation: agent_a key unreachable as agent_b."""
    store: dict = {}

    def _set(cache_key, tenant_id, value_bytes, *, model, ttl_expires, tags):
        store[(cache_key, tenant_id)] = value_bytes

    def _get_value(cache_key, tenant_id):
        return store.get((cache_key, tenant_id))

    mock_db = MagicMock()
    mock_db.set.side_effect = _set
    mock_db.get_value.side_effect = _get_value
    monkeypatch.setattr("superlocalmemory.mcp.tools_optimize.CacheDB", type("FakeCDB", (), {"get_default": staticmethod(lambda: mock_db)}))

    monkeypatch.setenv("SLM_AGENT_ID", "agent_a")
    set_r = await tools["slm_cache_set"](key="my_key", value="agent_a_value")
    assert set_r["ok"] is True
    assert set_r["stored"] is True

    get_hit = await tools["slm_cache_get"](key="my_key")
    assert get_hit["ok"] is True
    assert get_hit["hit"] is True
    assert get_hit["value"] == "agent_a_value"

    monkeypatch.setenv("SLM_AGENT_ID", "agent_b")
    get_miss = await tools["slm_cache_get"](key="my_key")
    assert get_miss["ok"] is True
    assert get_miss["hit"] is False
    assert get_miss["value"] is None


# ─── Test 9: slm_cache_set value >1MB → rejected ─────────────────────────────


async def test_cache_set_oversized_value(tools, monkeypatch):
    """Value > 1MB → ok:False, stored:False (CWE-400 guard), DB not touched."""
    mock_db = MagicMock()
    monkeypatch.setattr("superlocalmemory.mcp.tools_optimize.CacheDB", type("FakeCDB", (), {"get_default": staticmethod(lambda: mock_db)}))

    result = await tools["slm_cache_set"](key="big", value="x" * 1_000_001)
    assert result["ok"] is False
    assert result["stored"] is False
    mock_db.set.assert_not_called()


# ─── Test 10: slm_cache_set key >512 chars → rejected ─────────────────────────


async def test_cache_set_key_too_long(tools, monkeypatch):
    """Key > 512 chars → ok:False, stored:False, DB not touched."""
    mock_db = MagicMock()
    monkeypatch.setattr("superlocalmemory.mcp.tools_optimize.CacheDB", type("FakeCDB", (), {"get_default": staticmethod(lambda: mock_db)}))

    result = await tools["slm_cache_set"](key="k" * 513, value="value")
    assert result["ok"] is False
    assert result["stored"] is False
    mock_db.set.assert_not_called()


# ─── Test 11: slm_optimize_stats fresh → ok:True, all int fields ─────────────


async def test_stats_fresh_all_int_fields_present(tools, monkeypatch):
    """Fresh stats: ok:True, all integer fields present, no exception."""
    from superlocalmemory.optimize.storage.db import MetricsSnapshot

    mock_db = MagicMock()
    mock_db.metrics_load.return_value = MetricsSnapshot()
    monkeypatch.setattr("superlocalmemory.mcp.tools_optimize.CacheDB", type("FakeCDB", (), {"get_default": staticmethod(lambda: mock_db)}))

    result = await tools["slm_optimize_stats"]()

    assert result["ok"] is True
    for field in (
        "compress_runs",
        "tokens_saved_compress",
        "cache_proxy_hits",
        "cache_proxy_misses",
        "cache_kv_hits",
        "cache_kv_misses",
    ):
        assert field in result, f"Missing field: {field}"
        assert isinstance(result[field], int), f"{field} must be int, got {type(result[field])}"


# ─── Test 12: slm_optimize_stats after KV hit → cache_kv_hits >= 1 ───────────


async def test_stats_after_kv_hit_increments_counter(tools, monkeypatch):
    """After a cache hit, slm_optimize_stats reports cache_kv_hits >= 1."""
    from superlocalmemory.optimize.storage.db import MetricsSnapshot

    store: dict = {}

    def _set(cache_key, tenant_id, value_bytes, *, model, ttl_expires, tags):
        store[(cache_key, tenant_id)] = value_bytes

    def _get_value(cache_key, tenant_id):
        return store.get((cache_key, tenant_id))

    mock_db = MagicMock()
    mock_db.set.side_effect = _set
    mock_db.get_value.side_effect = _get_value
    mock_db.metrics_load.return_value = MetricsSnapshot()
    monkeypatch.setattr("superlocalmemory.mcp.tools_optimize.CacheDB", type("FakeCDB", (), {"get_default": staticmethod(lambda: mock_db)}))

    await tools["slm_cache_set"](key="stat_key", value="stat_val")
    await tools["slm_cache_get"](key="stat_key")

    result = await tools["slm_optimize_stats"]()
    assert result["ok"] is True
    assert result["cache_kv_hits"] >= 1


# ─── Test 13: oversize content → ccr_id:None, note set, no crash ─────────────


async def test_compress_oversize_content_no_crash(tools):
    """Content > 1MB → result returned, ccr_id:None, note mentions 1MB, no raise."""
    big_content = "A" * 1_000_001
    result = await tools["slm_compress"](
        content=big_content,
        mode="normalize",
        reversible=True,
    )
    assert isinstance(result, dict)
    assert result["ccr_id"] is None
    assert result["note"] is not None
    note_lower = result["note"].lower()
    assert "1mb" in note_lower or "1 mb" in note_lower, (
        f"Note should mention 1MB cap, got: {result['note']}"
    )


# ─── Test 14: fail-open when CompressRouter.get_instance raises ───────────────


async def test_compress_fail_open_on_engine_crash(tools, monkeypatch):
    """Engine raises → ok:False, original returned, no exception propagated."""
    def _explode():
        raise RuntimeError("engine is completely dead")

    mock_cr = MagicMock()
    mock_cr.get_instance.side_effect = _explode
    monkeypatch.setattr("superlocalmemory.mcp.tools_optimize.CompressRouter", mock_cr)

    content = "important content that must survive engine failure"
    result = await tools["slm_compress"](content=content)

    assert result["ok"] is False
    assert result["compressed"] == content
    assert result["note"] is not None
    assert "internal error" in result["note"].lower()


# ─── Test 15: integration — _FilteredServer, 5 tools, no headroom_* ──────────


async def test_integration_filtered_server_five_tools_no_headroom():
    """Register on _FilteredServer with optimize names allowed.

    Asserts: exactly 5 tools registered, names match _OPTIMIZE_TOOL_NAMES,
    zero headroom_* tools — fulfills AUDIT L-03.
    """
    from superlocalmemory.mcp.tools_optimize import register_optimize_tools, _OPTIMIZE_TOOL_NAMES

    inner = _MockServer()

    class _FakeFilteredServer:
        def __init__(self, real, allowed):
            self._real = real
            self._allowed = allowed

        def tool(self, *args, **kwargs):
            def decorator(fn):
                if fn.__name__ in self._allowed:
                    return self._real.tool(*args, **kwargs)(fn)
                return fn
            return decorator

    filtered = _FakeFilteredServer(inner, frozenset(_OPTIMIZE_TOOL_NAMES))
    register_optimize_tools(filtered)

    registered = set(inner._tools.keys())

    assert len(registered) == 5, (
        f"Expected 5 optimize tools, got {len(registered)}: {registered}"
    )
    assert registered == set(_OPTIMIZE_TOOL_NAMES), (
        f"Name mismatch. Expected {_OPTIMIZE_TOOL_NAMES}, got {registered}"
    )
    headroom = {n for n in registered if n.startswith("headroom")}
    assert not headroom, f"headroom_* tools must not be registered: {headroom}"
