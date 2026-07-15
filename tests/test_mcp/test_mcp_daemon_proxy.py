"""Daemon HTTP proxy + pool-error semantics.

MCP processes must not spawn their own worker subprocess when a daemon
is running. The proxy forwards recall/store to the daemon over HTTP,
keeping ONNX in exactly one process. Worker death (or any ok=False
envelope) surfaces as PoolError, not silent empty results.
"""
from __future__ import annotations

import json

import pytest

from superlocalmemory.mcp._daemon_proxy import DaemonPoolProxy
from superlocalmemory.mcp._pool_adapter import (
    PoolError, pool_recall, pool_store,
)


class TestPoolErrorSurfacing:
    def test_pool_recall_raises_on_ok_false(self, monkeypatch):
        class _Dead:
            def recall(self, query, limit=10, session_id="", fast=False):
                return {"ok": False, "error": "worker died"}
        from superlocalmemory.mcp import _pool_adapter
        monkeypatch.setattr(_pool_adapter, "_pool", lambda: _Dead())
        with pytest.raises(PoolError) as exc:
            pool_recall("any")
        assert "worker died" in str(exc.value)

    def test_pool_store_raises_on_ok_false(self, monkeypatch):
        class _Dead:
            def store(self, content, metadata=None):
                return {"ok": False, "error": "daemon down"}
        from superlocalmemory.mcp import _pool_adapter
        monkeypatch.setattr(_pool_adapter, "_pool", lambda: _Dead())
        with pytest.raises(PoolError) as exc:
            pool_store("content")
        assert "daemon down" in str(exc.value)

    def test_pool_recall_success_does_not_raise(self, monkeypatch):
        class _Ok:
            def recall(self, query, limit=10, session_id="", fast=False):
                return {"ok": True, "results": [], "query_type": "x"}
        from superlocalmemory.mcp import _pool_adapter
        monkeypatch.setattr(_pool_adapter, "_pool", lambda: _Ok())
        resp = pool_recall("any")
        assert resp.results == []
        assert resp.query_type == "x"


class TestDaemonPoolProxy:
    def test_recall_forwards_http_request(self, monkeypatch):
        captured = {}

        def _fake_urlopen(req, timeout=30):
            captured["url"] = getattr(req, "full_url", req)
            return _FakeResp(json.dumps({
                "ok": True, "results": [{"fact_id": "f1", "content": "hi",
                                          "score": 0.8}],
                "query_type": "semantic",
            }).encode())

        import superlocalmemory.mcp._daemon_proxy as mod
        monkeypatch.setattr(mod.urllib.request, "urlopen", _fake_urlopen)

        proxy = DaemonPoolProxy(port=9999)
        out = proxy.recall("what did we ship", limit=3, session_id="s-1")
        assert out["ok"] is True
        assert "q=what+did+we+ship" in captured["url"] \
            or "q=what%20did%20we%20ship" in captured["url"]
        assert "limit=3" in captured["url"]
        assert "session_id=s-1" in captured["url"]

    def test_recall_forwards_fast_flag(self, monkeypatch):
        captured = {}

        def _fake_urlopen(req, timeout=30):
            captured["url"] = getattr(req, "full_url", req)
            return _FakeResp(json.dumps({
                "ok": True, "results": [], "query_type": "semantic",
            }).encode())

        import superlocalmemory.mcp._daemon_proxy as mod
        monkeypatch.setattr(mod.urllib.request, "urlopen", _fake_urlopen)

        proxy = DaemonPoolProxy(port=9999)
        out = proxy.recall("fast path", fast=True)
        assert out["ok"] is True
        assert "fast=true" in captured["url"]

    def test_store_forwards_http_post(self, monkeypatch):
        captured = {}

        def _fake_urlopen(req, timeout=30):
            captured["url"] = req.full_url
            captured["body"] = req.data
            return _FakeResp(json.dumps({
                "ok": True, "fact_ids": ["f1", "f2"], "count": 2,
            }).encode())

        import superlocalmemory.mcp._daemon_proxy as mod
        monkeypatch.setattr(mod.urllib.request, "urlopen", _fake_urlopen)

        proxy = DaemonPoolProxy(port=9999)
        out = proxy.store("hello", metadata={"tags": "tag1"})
        assert out["fact_ids"] == ["f1", "f2"]
        assert captured["url"].endswith("/remember")
        body = json.loads(captured["body"].decode())
        assert body["content"] == "hello"
        assert body["tags"] == "tag1"

    def test_store_delegates_to_owned_daemon_client(self, monkeypatch):
        captured = {}

        def _owned_request(method, path, body=None):
            captured.update(method=method, path=path, body=body)
            return {"ok": True, "fact_ids": ["owned-fact"], "count": 1}

        import superlocalmemory.mcp._daemon_proxy as mod
        monkeypatch.setattr(
            "superlocalmemory.cli.daemon.daemon_request",
            _owned_request,
        )
        monkeypatch.setattr(
            mod.urllib.request,
            "urlopen",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("raw daemon write bypassed identity client")
            ),
        )

        out = DaemonPoolProxy(port=9999).store(
            "identity-bound content",
            metadata={"tags": "audit", "agent_id": "caller-label"},
        )

        assert out["fact_ids"] == ["owned-fact"]
        assert captured == {
            "method": "POST",
            "path": "/remember",
            "body": {
                "content": "identity-bound content",
                "tags": "audit",
                "metadata": {"tags": "audit", "agent_id": "caller-label"},
                "session_id": "",
                "idempotency_key": None,
            },
        }

    def test_recall_returns_ok_false_on_http_error(self, monkeypatch):
        def _fake_urlopen(req, timeout=30):
            raise ConnectionRefusedError("daemon closed")

        import superlocalmemory.mcp._daemon_proxy as mod
        monkeypatch.setattr(mod.urllib.request, "urlopen", _fake_urlopen)

        proxy = DaemonPoolProxy(port=9999)
        out = proxy.recall("x")
        assert out["ok"] is False
        assert "daemon closed" in out["error"]

    def test_store_returns_ok_false_on_http_error(self, monkeypatch):
        def _fake_urlopen(req, timeout=30):
            raise TimeoutError("slow")

        import superlocalmemory.mcp._daemon_proxy as mod
        monkeypatch.setattr(mod.urllib.request, "urlopen", _fake_urlopen)

        proxy = DaemonPoolProxy(port=9999)
        out = proxy.store("x")
        assert out["ok"] is False


class TestChoosePool:
    def test_prefers_daemon_proxy_when_running(self, monkeypatch):
        import superlocalmemory.mcp._daemon_proxy as mod
        monkeypatch.setattr(
            "superlocalmemory.cli.daemon.is_daemon_running",
            lambda: True,
        )
        monkeypatch.setattr(
            "superlocalmemory.cli.daemon._get_port",
            lambda: 9999,
        )
        pool = mod.choose_pool()
        assert isinstance(pool, DaemonPoolProxy)
        assert pool._port == 9999

    def test_falls_back_to_worker_pool_when_daemon_absent(self, monkeypatch):
        import superlocalmemory.mcp._daemon_proxy as mod
        from superlocalmemory.core.worker_pool import WorkerPool
        monkeypatch.setattr(WorkerPool, "_instance", None)
        monkeypatch.setattr(
            "superlocalmemory.cli.daemon.is_daemon_running",
            lambda: False,
        )
        pool = mod.choose_pool()
        assert not isinstance(pool, DaemonPoolProxy)

    def test_falls_back_on_probe_exception(self, monkeypatch):
        import superlocalmemory.mcp._daemon_proxy as mod
        from superlocalmemory.core.worker_pool import WorkerPool
        monkeypatch.setattr(WorkerPool, "_instance", None)

        def _boom():
            raise RuntimeError("psutil exploded")
        monkeypatch.setattr(
            "superlocalmemory.cli.daemon.is_daemon_running",
            _boom,
        )
        pool = mod.choose_pool()
        assert not isinstance(pool, DaemonPoolProxy)


class _FakeResp:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass
