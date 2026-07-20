"""MCP pool adapter — bridges WorkerPool dicts to RecallResponse-shaped objects.

After 5c, MCP tool modules construct AutoRecall / AutoCapture with the pool
adapter as ``recall_fn`` / ``store_fn`` instead of handing them the engine.
This lets the MCP process run with a LIGHT engine (no embedder) while still
serving session_init / observe / session_context correctly.
"""
from __future__ import annotations

import pytest
from types import SimpleNamespace


class _FakePool:
    """Stand-in for WorkerPool.shared() used across these tests."""
    def __init__(self):
        self.recall_calls = []
        self.store_calls = []

    def recall(
        self, query: str, limit: int = 10, session_id: str = "",
        fast: bool = False,
    ):
        self.recall_calls.append((query, limit, session_id, fast))
        return {
            "ok": True,
            "query": query,
            "query_type": "general",
            "result_count": 2,
            "retrieval_time_ms": 12.3,
            "results": [
                {
                    "fact_id": "f1", "memory_id": "m1",
                    "content": "first memory content",
                    "score": 0.91, "confidence": 0.8,
                    "trust_score": 0.5, "channel_scores": {"bm25": 0.7},
                },
                {
                    "fact_id": "f2", "memory_id": "m2",
                    "content": "second memory content",
                    "score": 0.77, "confidence": 0.7,
                    "trust_score": 0.4, "channel_scores": {},
                },
            ],
        }

    def store(self, content: str, metadata: dict | None = None):
        self.store_calls.append((content, metadata))
        return {"ok": True, "fact_ids": ["stored-1"], "count": 1}


class TestPoolAdapter:
    def test_pool_recall_reshapes_dict(self, monkeypatch):
        from superlocalmemory.mcp import _pool_adapter
        fake = _FakePool()
        monkeypatch.setattr(_pool_adapter, "_pool", lambda: fake)

        resp = _pool_adapter.pool_recall("hello", limit=5)

        assert hasattr(resp, "results")
        assert len(resp.results) == 2
        assert resp.results[0].fact.fact_id == "f1"
        assert resp.results[0].fact.content == "first memory content"
        assert resp.results[0].score == 0.91
        assert fake.recall_calls == [("hello", 5, "", False)]

    def test_pool_recall_forwards_session_id_and_fast(self, monkeypatch):
        from superlocalmemory.mcp import _pool_adapter
        fake = _FakePool()
        monkeypatch.setattr(_pool_adapter, "_pool", lambda: fake)

        _pool_adapter.pool_recall("hello", limit=5, session_id="s-1", fast=True)

        assert fake.recall_calls == [("hello", 5, "s-1", True)]

    def test_pool_store_returns_fact_ids(self, monkeypatch):
        from superlocalmemory.mcp import _pool_adapter
        fake = _FakePool()
        monkeypatch.setattr(_pool_adapter, "_pool", lambda: fake)

        ids = _pool_adapter.pool_store("content body", {"k": "v"})

        assert ids == ["stored-1"]
        assert fake.store_calls[0][0] == "content body"

    def test_pool_recall_handles_empty_results(self, monkeypatch):
        from superlocalmemory.mcp import _pool_adapter

        class _Empty:
            def recall(self, query, limit=10, session_id="", fast=False):
                return {"ok": True, "results": []}

        monkeypatch.setattr(_pool_adapter, "_pool", lambda: _Empty())
        resp = _pool_adapter.pool_recall("nothing")
        assert resp.results == []

    def test_pool_recall_raises_on_ok_false(self, monkeypatch):
        """Worker death returns {"ok": False} — must raise PoolError,
        not silently return empty results."""
        from superlocalmemory.mcp import _pool_adapter
        from superlocalmemory.mcp._pool_adapter import PoolError

        class _Dead:
            def recall(self, query, limit=10, session_id="", fast=False):
                return {"ok": False, "error": "worker died"}

        monkeypatch.setattr(_pool_adapter, "_pool", lambda: _Dead())
        with pytest.raises(PoolError, match="worker died"):
            _pool_adapter.pool_recall("anything")

    def test_pool_store_raises_on_ok_false(self, monkeypatch):
        from superlocalmemory.mcp import _pool_adapter
        from superlocalmemory.mcp._pool_adapter import PoolError

        class _Dead:
            def store(self, content, metadata=None):
                return {"ok": False, "error": "write failed"}

        monkeypatch.setattr(_pool_adapter, "_pool", lambda: _Dead())
        with pytest.raises(PoolError, match="write failed"):
            _pool_adapter.pool_store("test content")

    def test_pool_recall_returns_typed_dataclass(self, monkeypatch):
        from superlocalmemory.mcp import _pool_adapter
        from superlocalmemory.mcp._pool_adapter import (
            PoolRecallResponse, PoolRecallItem, PoolFact,
        )
        fake = _FakePool()
        monkeypatch.setattr(_pool_adapter, "_pool", lambda: fake)
        resp = _pool_adapter.pool_recall("typed check")
        assert isinstance(resp, PoolRecallResponse)
        assert isinstance(resp.results[0], PoolRecallItem)
        assert isinstance(resp.results[0].fact, PoolFact)

    def test_pool_recall_integrates_with_auto_recall(self, monkeypatch):
        """End-to-end: AutoRecall with recall_fn=pool_recall returns context."""
        from superlocalmemory.mcp import _pool_adapter
        from superlocalmemory.hooks.auto_recall import AutoRecall

        monkeypatch.setattr(_pool_adapter, "_pool", lambda: _FakePool())
        auto = AutoRecall(recall_fn=_pool_adapter.pool_recall)
        ctx = auto.get_session_context(query="anything")
        assert "first memory content" in ctx
        assert "second memory content" in ctx


class TestToolsActiveUsesPool:
    """The MCP session_init tool must construct AutoRecall with pool_recall,
    not with the engine, so it works against a LIGHT engine.
    """
    def test_session_init_uses_pool_adapter_not_engine_recall(self, monkeypatch):
        import asyncio
        from superlocalmemory.mcp import tools_active, _pool_adapter

        fake_pool = _FakePool()
        monkeypatch.setattr(_pool_adapter, "_pool", lambda: fake_pool)

        # Minimal engine stand-in — DB-only. Any engine.recall call would
        # fail this test (AttributeError), because our stand-in lacks it.
        class _LightEngine:
            profile_id = "p"
            mode = "a"
            config = SimpleNamespace(injection=None)
            _config = SimpleNamespace(mode=SimpleNamespace(value="a"))
            _adaptive_learner = None
            db = None

            # Represent the LIGHT contract: attempts to recall raise.
            def recall(self, *a, **kw):
                raise AssertionError(
                    "tools_active must not call engine.recall; route via pool_recall"
                )

        # Capture FastMCP tool registrations
        registered: dict = {}

        class _Server:
            def tool(self):
                def _wrap(fn):
                    registered[fn.__name__] = fn
                    return fn
                return _wrap

        tools_active.register_active_tools(_Server(), lambda: _LightEngine())
        result = asyncio.run(registered["session_init"](project_path="/tmp/p"))

        assert result["success"] is True
        # v3.4.52: full recall (fast=False). Ollama is kept warm
        # via keep_alive=-1 + eager pre-warm at daemon boot, so no cold-start
        # penalty. FTS5 fallback only triggers when daemon is completely down.
        assert fake_pool.recall_calls == [
            ("project context /tmp/p", 10, "", False),
        ], "session_init should use full recall (fast=False)"

    def test_observe_uses_pool_adapter_not_engine_store(self, monkeypatch):
        import asyncio
        from superlocalmemory.mcp import tools_active, _pool_adapter

        fake_pool = _FakePool()
        monkeypatch.setattr(_pool_adapter, "_pool", lambda: fake_pool)

        class _LightEngine:
            profile_id = "p"
            _config = SimpleNamespace(mode=SimpleNamespace(value="a"))

            def store(self, *a, **kw):
                raise AssertionError(
                    "tools_active must not call engine.store; route via pool_store"
                )

        registered: dict = {}

        class _Server:
            def tool(self):
                def _wrap(fn):
                    registered[fn.__name__] = fn
                    return fn
                return _wrap

        tools_active.register_active_tools(_Server(), lambda: _LightEngine())
        # A decision-looking sentence so AutoCapture fires.
        content = (
            "We decided to use Postgres because the write pattern is "
            "transactional and we prefer strong consistency."
        )
        result = asyncio.run(registered["observe"](content=content))

        # At least we didn't blow up. Capture may or may not fire depending
        # on rules engine config — what matters is no engine.store was hit
        # and if pool_store was used, it received our content.
        assert result is not None
        if result.get("captured"):
            assert fake_pool.store_calls, \
                "observe captured but did not go through pool_store"
