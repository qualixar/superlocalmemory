"""Daemon HTTP proxy + pool-error semantics.

MCP processes must not spawn their own worker subprocess when a daemon
is running. The proxy forwards recall/store to the daemon over HTTP,
keeping ONNX in exactly one process. Worker death (or any ok=False
envelope) surfaces as PoolError, not silent empty results.
"""
from __future__ import annotations

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

        def _owned_request(method, path, body=None, **kwargs):
            captured.update(method=method, path=path, body=body, kwargs=kwargs)
            return {
                "ok": True, "results": [{"fact_id": "f1", "content": "hi",
                                          "score": 0.8}],
                "query_type": "semantic",
            }

        monkeypatch.setattr(
            "superlocalmemory.cli.daemon.daemon_request", _owned_request,
        )

        proxy = DaemonPoolProxy(port=9999)
        out = proxy.recall("what did we ship", limit=3, session_id="s-1")
        assert out["ok"] is True
        assert captured["method"] == "GET"
        assert "q=what+did+we+ship" in captured["path"] \
            or "q=what%20did%20we%20ship" in captured["path"]
        assert "limit=3" in captured["path"]
        assert "session_id=s-1" in captured["path"]
        assert captured["kwargs"] == {"timeout_seconds": 30.0}

    def test_recall_forwards_fast_flag(self, monkeypatch):
        captured = {}

        def _owned_request(method, path, body=None, **kwargs):
            captured.update(method=method, path=path)
            return {
                "ok": True, "results": [], "query_type": "semantic",
            }

        monkeypatch.setattr(
            "superlocalmemory.cli.daemon.daemon_request", _owned_request,
        )

        proxy = DaemonPoolProxy(port=9999)
        out = proxy.recall("fast path", fast=True)
        assert out["ok"] is True
        assert "fast=true" in captured["path"]

    def test_store_forwards_http_post(self, monkeypatch):
        captured = {}

        def _owned_request(method, path, body=None):
            captured.update(method=method, path=path, body=body)
            return {"ok": True, "fact_ids": ["f1", "f2"], "count": 2}

        monkeypatch.setattr(
            "superlocalmemory.cli.daemon.daemon_request",
            _owned_request,
        )

        proxy = DaemonPoolProxy(port=9999)
        out = proxy.store("hello", metadata={"tags": "tag1"})
        assert out["fact_ids"] == ["f1", "f2"]
        assert captured["method"] == "POST"
        assert captured["path"] == "/remember"
        body = captured["body"]
        assert body["content"] == "hello"
        assert body["tags"] == "tag1"

    def test_store_delegates_to_owned_daemon_client(self, monkeypatch):
        captured = {}

        def _owned_request(method, path, body=None):
            captured.update(method=method, path=path, body=body)
            return {"ok": True, "fact_ids": ["owned-fact"], "count": 1}

        monkeypatch.setattr(
            "superlocalmemory.cli.daemon.daemon_request",
            _owned_request,
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
        def _owned_request(*args, **kwargs):
            raise ConnectionRefusedError("daemon closed")

        monkeypatch.setattr(
            "superlocalmemory.cli.daemon.daemon_request", _owned_request,
        )

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


# ---------------------------------------------------------------------------
# SEC-H-01: switch_profile daemon-branch local re-validation.
#
# The daemon-branch of the switch_profile MCP tool must not blindly trust
# the daemon's HTTP acknowledgement. It must (a) confirm the daemon actually
# acknowledged the REQUESTED profile_id and (b) locally re-validate that the
# profile exists in this process's own engine._db handle before syncing
# engine.profile_id / engine._config.active_profile. Only a fully confirmed
# profile_id may ever be applied to local state.
# ---------------------------------------------------------------------------

class _RecordingServer:
    """Minimal @server.tool() capture, matching the tools_core convention."""

    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *args, **kwargs):
        def register(fn):
            self.tools[fn.__name__] = fn
            return fn
        return register


def _switch_profile_tool(engine):
    from superlocalmemory.mcp.tools_core import register_core_tools
    from unittest.mock import MagicMock

    server = _RecordingServer()
    register_core_tools(server, MagicMock(return_value=engine))
    return server.tools["switch_profile"]


def _daemon_engine(starting_profile: str = "owner-profile"):
    from unittest.mock import MagicMock

    engine = MagicMock()
    engine.profile_id = starting_profile
    engine._config.active_profile = starting_profile
    # run_pre must not raise for these happy/guard-path tests (contrast
    # with test_core_control_mutation_policy.py which deliberately makes
    # it raise to test the earlier policy-rejection path).
    engine._hooks.run_pre.return_value = None
    return engine


class TestSwitchProfileDaemonLocalConsistency:
    def test_daemon_acknowledged_switch_syncs_to_confirmed_profile(
        self, monkeypatch
    ):
        """(a) A daemon-acknowledged switch, confirmed to exist locally,
        syncs engine.profile_id to the ACKNOWLEDGED profile."""
        import asyncio

        engine = _daemon_engine("owner-profile")
        # Local DB confirms the target profile exists.
        engine._db.execute.return_value = [(1,)]

        monkeypatch.setattr(
            "superlocalmemory.cli.daemon.is_daemon_running",
            lambda: True,
        )
        monkeypatch.setattr(
            "superlocalmemory.cli.daemon.daemon_request",
            lambda method, path: {
                "success": True,
                "active_profile": "team-profile",
                "generation": 7,
            },
        )

        tool = _switch_profile_tool(engine)
        result = asyncio.run(tool("team-profile"))

        assert result["success"] is True
        assert result["current_profile"] == "team-profile"
        assert engine.profile_id == "team-profile"
        assert engine._config.active_profile == "team-profile"

    def test_daemon_response_missing_acknowledgement_does_not_sync_state(
        self, monkeypatch
    ):
        """(b) A malformed daemon response (no active_profile field) must
        NOT silently sync local state to the unvalidated request value."""
        import asyncio

        engine = _daemon_engine("owner-profile")
        engine._db.execute.return_value = [(1,)]

        monkeypatch.setattr(
            "superlocalmemory.cli.daemon.is_daemon_running",
            lambda: True,
        )
        monkeypatch.setattr(
            "superlocalmemory.cli.daemon.daemon_request",
            lambda method, path: {"success": True},  # no active_profile
        )

        tool = _switch_profile_tool(engine)
        result = asyncio.run(tool("team-profile"))

        assert result["success"] is False
        assert engine.profile_id == "owner-profile"
        assert engine._config.active_profile == "owner-profile"

    def test_daemon_response_mismatched_acknowledgement_does_not_sync_state(
        self, monkeypatch
    ):
        """(b) A daemon response acknowledging a DIFFERENT profile than the
        one requested must not sync local state to either value."""
        import asyncio

        engine = _daemon_engine("owner-profile")
        engine._db.execute.return_value = [(1,)]

        monkeypatch.setattr(
            "superlocalmemory.cli.daemon.is_daemon_running",
            lambda: True,
        )
        monkeypatch.setattr(
            "superlocalmemory.cli.daemon.daemon_request",
            lambda method, path: {
                "success": True,
                "active_profile": "some-other-profile",
                "generation": 3,
            },
        )

        tool = _switch_profile_tool(engine)
        result = asyncio.run(tool("team-profile"))

        assert result["success"] is False
        assert engine.profile_id == "owner-profile"
        assert engine._config.active_profile == "owner-profile"

    def test_daemon_confirmed_profile_missing_locally_does_not_sync_state(
        self, monkeypatch
    ):
        """(b) Even when the daemon correctly acknowledges the requested
        profile, if this process's OWN local DB has no such profile row,
        the tool must not blindly overwrite local state — it errors."""
        import asyncio

        engine = _daemon_engine("owner-profile")
        # Local DB does NOT have this profile (e.g. stale/divergent handle).
        engine._db.execute.return_value = []

        monkeypatch.setattr(
            "superlocalmemory.cli.daemon.is_daemon_running",
            lambda: True,
        )
        monkeypatch.setattr(
            "superlocalmemory.cli.daemon.daemon_request",
            lambda method, path: {
                "success": True,
                "active_profile": "team-profile",
                "generation": 5,
            },
        )

        tool = _switch_profile_tool(engine)
        result = asyncio.run(tool("team-profile"))

        assert result["success"] is False
        assert engine.profile_id == "owner-profile"
        assert engine._config.active_profile == "owner-profile"
