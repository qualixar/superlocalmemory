# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Per-request profile routing at the daemon layer (spec section 3/5).

A non-empty ``profile_id`` on POST /remember (body) and GET /recall (query)
is pure routing: the request is served against THAT profile without touching
the ProfileRuntime active pointer or its generation. An unknown profile is
rejected with 404 + ``{"success": false, "error": {"code":
"unknown_profile"}}`` and never implicitly created. An empty/absent
profile_id keeps the legacy path byte-identical, including the stale-client
guard slot, which is unreachable for routed requests.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from superlocalmemory.server.unified_daemon import create_app
from superlocalmemory.storage.migrations import (
    M018_ingestion_operations,
    M032_write_coordinator_admission,
    M033_projection_transactions,
    M034_obligation_integrity,
    M042_correction_case_ledger,
)


@contextmanager
def _daemon(engine, profiles=("a", "b")):
    """TestClient daemon with pre-created profiles, per tests/test_server convention.

    Mirrors ``test_canonical_remember_route._client``: the daemon-owned
    canonical writer is injected because TestClient does not enter lifespan.
    """
    from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime

    with engine._db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
        M032_write_coordinator_admission.apply(conn)
        M033_projection_transactions.apply(conn)
        M034_obligation_integrity.apply(conn)
        M042_correction_case_ledger.apply(conn)
        for profile_id in profiles:
            conn.execute(
                "INSERT OR IGNORE INTO profiles (profile_id, name) "
                "VALUES (?, ?)",
                (profile_id, f"Profile {profile_id}"),
            )
        conn.commit()
    app = create_app()
    app.state.engine = engine
    runtime = CanonicalRememberRuntime.for_engine(engine)
    runtime.start()
    app.state.canonical_remember_runtime = runtime
    client = TestClient(app)
    client.headers["X-SLM-Daemon-Capability"] = (
        app.state.daemon_descriptor.capability
    )
    client.headers["X-SLM-Target-Instance"] = (
        app.state.daemon_descriptor.instance_id
    )
    try:
        yield client, app
    finally:
        runtime.stop()


@pytest.fixture
def daemon(engine_with_mock_deps):
    with _daemon(engine_with_mock_deps) as pair:
        yield pair


def _facts_in(engine, profile_id: str, needle: str = "") -> list[str]:
    rows = engine._db.execute(
        "SELECT fact_id FROM atomic_facts WHERE profile_id = ? AND content LIKE ?",
        (profile_id, f"%{needle}%"),
    )
    return [row["fact_id"] for row in rows]


def _profile_row_count(engine) -> int:
    rows = engine._db.execute("SELECT COUNT(*) AS c FROM profiles")
    return int(dict(rows[0])["c"])


class TestRouting:
    def test_remember_routes_to_explicit_profile(self, daemon) -> None:
        client, app = daemon
        engine = app.state.engine
        response = client.post(
            "/remember",
            json={
                "content": (
                    "Doris owns the release branch schedule and records "
                    "every platform freeze window."
                ),
                "profile_id": "b",
                "idempotency_key": "route-remember-b-1",
            },
        )

        assert response.status_code == 200, response.text
        assert _facts_in(engine, "b", "Doris"), (
            "the routed fact must land in profile b"
        )
        assert _facts_in(engine, "a", "Doris") == []

    def test_recall_routes_to_explicit_profile(self, daemon) -> None:
        client, _ = daemon
        client.post(
            "/remember",
            json={
                "content": (
                    "Doris owns the release branch schedule and records "
                    "every platform freeze window."
                ),
                "profile_id": "b",
                "idempotency_key": "route-recall-b-1",
            },
        )
        client.post(
            "/remember",
            json={
                "content": (
                    "Zebra coordinates the zonal inventory audit and keeps "
                    "the northern warehouse ledger."
                ),
                "profile_id": "a",
                "idempotency_key": "route-recall-a-1",
            },
        )

        hit = client.get(
            "/recall", params={"q": "Doris release branch schedule", "profile_id": "b"},
        )
        assert hit.status_code == 200, hit.text
        assert hit.json()["results"], "recall must hit the routed profile"

        isolated = client.get(
            "/recall", params={"q": "Doris release branch schedule", "profile_id": "a"},
        )
        assert isolated.status_code == 200, isolated.text
        assert isolated.json()["results"] == []

        reverse = client.get(
            "/recall", params={"q": "Zebra inventory audit", "profile_id": "b"},
        )
        assert reverse.json()["results"] == []

    def test_global_pointer_untouched(self, daemon) -> None:
        client, _ = daemon
        before = client.get("/status").json()
        write = client.post(
            "/remember",
            json={
                "content": (
                    "Quartz buffers the quarterly readiness review for the "
                    "on-call rotation."
                ),
                "profile_id": "b",
                "idempotency_key": "route-pointer-b-1",
            },
        )
        lookup = client.get(
            "/recall", params={"q": "Quartz readiness review", "profile_id": "b"},
        )
        after = client.get("/status").json()

        # The routed calls must have actually succeeded, or the pointer
        # comparison below would pass vacuously.
        assert write.status_code == 200, write.text
        assert lookup.status_code == 200, lookup.text
        assert after["profile"] == before["profile"]
        assert after["profile_generation"] == before["profile_generation"]

    def test_unknown_profile_rejected(self, daemon) -> None:
        client, app = daemon
        engine = app.state.engine
        profiles_before = _profile_row_count(engine)

        remembered = client.post(
            "/remember",
            json={
                "content": (
                    "Ghost owns no profile and must not be silently created."
                ),
                "profile_id": "ghost",
                "idempotency_key": "route-unknown-1",
            },
        )
        recalled = client.get(
            "/recall", params={"q": "Ghost profile", "profile_id": "ghost"},
        )

        assert remembered.status_code == 404, remembered.text
        body = remembered.json()
        assert body["success"] is False
        assert body["error"]["code"] == "unknown_profile"
        assert body["error"]["profile_id"] == "ghost"

        assert recalled.status_code == 404, recalled.text
        recall_body = recalled.json()
        assert recall_body["success"] is False
        assert recall_body["error"]["code"] == "unknown_profile"

        # No implicit creation, no engine touch.
        assert _profile_row_count(engine) == profiles_before
        assert engine._db.execute(
            "SELECT COUNT(*) AS c FROM atomic_facts WHERE profile_id = 'ghost'"
        )[0]["c"] == 0

    def test_empty_profile_id_is_legacy_path(self, daemon) -> None:
        client, app = daemon
        engine = app.state.engine
        active = client.get("/status").json()["profile"]

        response = client.post(
            "/remember",
            json={
                "content": (
                    "Legacy writes keep landing in the active profile "
                    "exactly as before."
                ),
                "idempotency_key": "route-legacy-1",
            },
        )

        assert response.status_code == 200, response.text
        assert _facts_in(engine, active, "Legacy writes"), (
            "the legacy path must write to the active profile"
        )
        assert _facts_in(engine, "b", "Legacy writes") == []

    def test_active_profile_explicit_is_not_error(self, daemon) -> None:
        client, _ = daemon
        active = client.get("/status").json()["profile"]

        response = client.post(
            "/remember",
            json={
                "content": (
                    "Naming the active profile explicitly is routing, not "
                    "a stale-client conflict."
                ),
                "profile_id": active,
                "idempotency_key": "route-active-1",
            },
        )

        assert response.status_code == 200, response.text
        assert response.json()["fact_ids"]

    def test_routed_responses_report_the_routed_profile(self, daemon) -> None:
        """The success envelopes echo the profile that served the request.

        A routed b-profile answer reporting ``"profile": "default"`` (the
        active snapshot) would tell the caller their memory came from a
        profile that never saw it. profile_generation stays from the
        snapshot: it describes global switch state, which per-request
        routing never moves.
        """
        client, _ = daemon
        status = client.get("/status").json()
        active = status["profile"]

        remembered = client.post(
            "/remember",
            json={
                "content": (
                    "Hazel tracks the harbor crane maintenance windows for "
                    "the coastal crew."
                ),
                "profile_id": "b",
                "idempotency_key": "route-echo-b-1",
            },
        )
        assert remembered.status_code == 200, remembered.text
        assert remembered.json()["profile"] == "b"

        recalled = client.get(
            "/recall",
            params={"q": "Hazel crane maintenance", "profile_id": "b"},
        )
        assert recalled.status_code == 200, recalled.text
        assert recalled.json()["profile"] == "b"
        assert recalled.json()["profile_generation"] == status["profile_generation"]

        legacy = client.post(
            "/remember",
            json={
                "content": (
                    "Ivory keeps the inland depot roster on the legacy path."
                ),
                "idempotency_key": "route-echo-legacy-1",
            },
        )
        assert legacy.status_code == 200, legacy.text
        assert legacy.json()["profile"] == active

        legacy_recall = client.get(
            "/recall", params={"q": "Ivory depot roster"},
        )
        assert legacy_recall.status_code == 200, legacy_recall.text
        assert legacy_recall.json()["profile"] == active

    def test_failed_rebind_rolls_back_routed_writers_and_limits(
        self, daemon, monkeypatch,
    ) -> None:
        """A failed rebind restores the whole binding, not just the writer.

        The routed-handler cache and the writer limits swapped in before
        replay_pending() must not outlive a rebind that never completed.
        """
        from types import SimpleNamespace

        client, app = daemon
        engine = app.state.engine
        runtime = app.state.canonical_remember_runtime

        warmed = client.post(
            "/remember",
            json={
                "content": (
                    "Juniper anchors the joint readiness ledger before the "
                    "rebind attempt."
                ),
                "profile_id": "b",
                "idempotency_key": "route-rebind-warm-1",
            },
        )
        assert warmed.status_code == 200, warmed.text
        assert "b" in runtime._routed_writers, (
            "a routed write must have cached a handler for profile b"
        )
        bound_profile = runtime._profile_id
        bound_limits = (
            runtime._max_verbatim_chars, runtime._max_ingest_bytes,
        )
        bound_generation = runtime._generation

        def _fail_replay():
            raise RuntimeError("replay failed after rebind")

        monkeypatch.setattr(runtime, "replay_pending", _fail_replay)
        rebinding = SimpleNamespace(
            _db=engine._db,
            _profile_id="a",
            _config=SimpleNamespace(
                store=SimpleNamespace(
                    max_verbatim_chars=99, max_ingest_bytes=99,
                ),
            ),
        )

        with pytest.raises(RuntimeError, match="replay failed"):
            runtime.rebind_engine(rebinding)

        assert runtime._profile_id == bound_profile
        assert (
            runtime._max_verbatim_chars, runtime._max_ingest_bytes,
        ) == bound_limits
        assert runtime._routed_writers == {}
        assert runtime._generation == bound_generation


# ---------------------------------------------------------------------------
# Task 4: the MCP tool surface (spec section 4)
#
# remember/recall accept an optional ``profile_id`` and thread it to the
# daemon's per-request routing (Task 3). Empty keeps the legacy call
# byte-identical: the parameter must not appear in the daemon request at
# all when it was not set, and it must be optional in the MCP schema.
# ---------------------------------------------------------------------------

class _ToolCaptureServer:
    """Minimal @server.tool() capture, matching the tests/test_mcp convention."""

    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *args, **kwargs):
        def register(fn):
            self.tools[fn.__name__] = fn
            return fn
        return register


def _core_tools() -> dict[str, object]:
    from unittest.mock import MagicMock

    from superlocalmemory.mcp.tools_core import register_core_tools

    srv = _ToolCaptureServer()
    register_core_tools(srv, MagicMock())
    return srv.tools


def _ok_pool() -> "MagicMock":
    from unittest.mock import MagicMock

    pool = MagicMock()
    pool.store.return_value = {
        "ok": True, "fact_ids": ["mcp-fact"], "count": 1,
        "operation_id": "op-mcp", "pending_id": None,
        "materialization_state": "complete",
    }
    pool.recall.return_value = {
        "ok": True, "results": [{"fact_id": "mcp-fact", "content": "mcp fact",
                                 "score": 0.9}],
        "result_count": 1, "query_type": "sandbox",
    }
    return pool


class TestMcpSurface:
    def test_remember_tool_accepts_and_routes_profile_id(self, monkeypatch) -> None:
        """remember(profile_id="b") puts the routing anchor in the daemon body."""
        import asyncio

        import superlocalmemory.cli.daemon as _d

        captured: dict = {}

        def _request(method, path, body=None, **kwargs):
            captured.update(method=method, path=path, body=body)
            return {"ok": True, "fact_ids": ["mcp-fact"], "count": 1,
                    "status": "stored"}

        monkeypatch.setattr(_d, "is_daemon_running", lambda *a, **k: True)
        monkeypatch.setattr(_d, "daemon_request", _request)

        remember = _core_tools()["remember"]
        result = asyncio.run(remember("mcp fact", profile_id="b"))

        assert result["success"] is True, result
        assert captured["method"] == "POST"
        assert captured["path"] == "/remember"
        assert captured["body"]["profile_id"] == "b"

    def test_remember_tool_offline_fallback_threads_profile_id(
        self, monkeypatch,
    ) -> None:
        """The pool.store fallback carries the anchor in worker metadata."""
        import asyncio

        import superlocalmemory.cli.daemon as _d

        monkeypatch.setattr(_d, "is_daemon_running", lambda *a, **k: False)
        pool = _ok_pool()

        remember = _core_tools()["remember"]
        with patch(
            "superlocalmemory.mcp._daemon_proxy.choose_pool", return_value=pool,
        ):
            result = asyncio.run(remember("mcp fact", profile_id="b"))

        assert result["success"] is True, result
        pool.store.assert_called_once()
        assert pool.store.call_args.args[1]["profile_id"] == "b"

    def test_remember_tool_legacy_call_has_no_profile_anchor(
        self, monkeypatch,
    ) -> None:
        """No profile_id → the daemon body is the legacy shape, key absent."""
        import asyncio

        import superlocalmemory.cli.daemon as _d

        captured: dict = {}

        def _request(method, path, body=None, **kwargs):
            captured.update(method=method, path=path, body=body)
            return {"ok": True, "fact_ids": ["mcp-fact"], "count": 1,
                    "status": "stored"}

        monkeypatch.setattr(_d, "is_daemon_running", lambda *a, **k: True)
        monkeypatch.setattr(_d, "daemon_request", _request)

        remember = _core_tools()["remember"]
        result = asyncio.run(remember("mcp fact"))

        assert result["success"] is True, result
        assert "profile_id" not in captured["body"]

    def test_recall_tool_accepts_and_routes_profile_id(self, monkeypatch) -> None:
        """recall(profile_id="b") threads the anchor to pool.recall."""
        import asyncio

        import superlocalmemory.cli.daemon as _d

        monkeypatch.setattr(_d, "is_daemon_running", lambda *a, **k: False)
        pool = _ok_pool()

        recall = _core_tools()["recall"]
        with patch(
            "superlocalmemory.mcp._daemon_proxy.choose_pool", return_value=pool,
        ):
            result = asyncio.run(recall("mcp fact", profile_id="b"))

        assert result["success"] is True, result
        assert result["results"], "recall must surface the sandbox hit"
        assert pool.recall.call_args.kwargs["profile_id"] == "b"

    def test_recall_tool_legacy_call_has_no_profile_anchor(
        self, monkeypatch,
    ) -> None:
        """No profile_id → pool.recall is called without the parameter."""
        import asyncio

        import superlocalmemory.cli.daemon as _d

        monkeypatch.setattr(_d, "is_daemon_running", lambda *a, **k: False)
        pool = _ok_pool()

        recall = _core_tools()["recall"]
        with patch(
            "superlocalmemory.mcp._daemon_proxy.choose_pool", return_value=pool,
        ):
            result = asyncio.run(recall("mcp fact"))

        assert result["success"] is True, result
        assert "profile_id" not in pool.recall.call_args.kwargs

    def test_daemon_pool_proxy_recall_sends_profile_id_param(
        self, monkeypatch,
    ) -> None:
        """DaemonPoolProxy serializes the anchor into the GET /recall query.

        This is the brief's "verify the proxy passes it through" check made
        executable: without the parameter the MCP tool's recall cannot reach
        the daemon's per-request routing at all.
        """
        from superlocalmemory.mcp._daemon_proxy import DaemonPoolProxy

        captured: dict = {}

        def _request(method, path, body=None, **kwargs):
            captured.update(method=method, path=path)
            return {"ok": True, "results": [], "query_type": "sandbox"}

        monkeypatch.setattr(
            "superlocalmemory.cli.daemon.daemon_request", _request,
        )

        proxy = DaemonPoolProxy(port=9999)
        assert proxy.recall("mcp fact", profile_id="b")["ok"] is True
        assert captured["method"] == "GET"
        assert "profile_id=b" in captured["path"]

        # Legacy shape: unset anchor never appears on the wire.
        assert proxy.recall("mcp fact")["ok"] is True
        assert "profile_id=" not in captured["path"]

    def test_tool_schema_allows_new_optional_param(self) -> None:
        """The schema layer exposes profile_id as optional on both tools."""
        from unittest.mock import MagicMock

        from superlocalmemory.mcp.http_transport import SLMFastMCP
        from superlocalmemory.mcp.tools_core import register_core_tools

        srv = SLMFastMCP("schema probe")
        register_core_tools(srv, MagicMock())
        tools = {t.name: t for t in srv._tool_manager.list_tools()}

        for name in ("remember", "recall"):
            schema = tools[name].parameters
            assert "profile_id" in schema["properties"], (
                f"{name} must expose profile_id"
            )
            required = schema.get("required", [])
            assert required.count("profile_id") == 0, (
                f"{name}.profile_id must be optional"
            )


# ---------------------------------------------------------------------------
# Final-review I-1: routed writes get inline enrichment against THEIR profile
#
# The daemon's post-write enrichment used the engine's ACTIVE profile for its
# tenant-scoped lookups, so a routed write's fact_ids resolved to None: every
# routed response said searchable_by="wording" no matter what the embedder
# could have done. The write target must be threaded through
# _enrich_and_release → engine.enrich_new_facts_now.
# ---------------------------------------------------------------------------

class TestRoutedInlineEnrichment:
    @staticmethod
    def _spy_enrich(engine, monkeypatch):
        """Capture how the daemon calls engine.enrich_new_facts_now."""
        import superlocalmemory.core.engine as _engine_mod

        real = _engine_mod.MemoryEngine.enrich_new_facts_now
        captured: dict = {}

        def _spy(fact_ids, **kwargs):
            captured["fact_ids"] = list(fact_ids)
            captured["kwargs"] = kwargs
            # Set on the instance, so no implicit self arrives here.
            return real(engine, fact_ids, **kwargs)

        monkeypatch.setattr(engine, "enrich_new_facts_now", _spy)
        return captured

    @staticmethod
    def _warm_mock_embedder(engine, monkeypatch):
        """Let the mocked-deps engine actually embed inline.

        The fixture's mock embedder reads cold-and-remote, so the warm guard
        declines by design; patching the guard is the sanctioned seam that
        makes an enriched>0 outcome observable (same helper as the engine
        test file).
        """
        monkeypatch.setattr(
            engine, "_warm_guard_embed",
            lambda text, *, timeout_s=None: ([0.01] * 768, [0.0] * 768, [1.0] * 768),
        )

    def test_routed_remember_enriches_against_the_routed_profile(
        self, daemon, monkeypatch,
    ) -> None:
        client, app = daemon
        engine = app.state.engine
        active = client.get("/status").json()["profile"]
        assert active != "b"
        self._warm_mock_embedder(engine, monkeypatch)
        captured = self._spy_enrich(engine, monkeypatch)

        response = client.post(
            "/remember",
            json={
                "content": (
                    "Routed Rowan keeps the relay baton rota for the night "
                    "shift and files the handover notes."
                ),
                "profile_id": "b",
                "idempotency_key": "route-enrich-b-1",
            },
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["profile"] == "b"
        assert body["fact_ids"]

        # The daemon threaded the write target into the enrichment call —
        # before the fix this was always the active profile and the routed
        # facts resolved to None inside the tenant-scoped lookup.
        assert captured["kwargs"].get("profile_id") == "b", captured
        assert captured["fact_ids"] == body["fact_ids"]

        # And the routed rows were genuinely enriched: the receipt reports
        # meaning-searchable instead of the permanent "wording" the routed
        # path used to return.
        assert body["enriched_now"] == body["count"], body
        assert body["searchable_by"] == "meaning", body

    def test_legacy_remember_enrichment_uses_the_active_profile(
        self, daemon, monkeypatch,
    ) -> None:
        client, app = daemon
        engine = app.state.engine
        active = client.get("/status").json()["profile"]
        self._warm_mock_embedder(engine, monkeypatch)
        captured = self._spy_enrich(engine, monkeypatch)

        response = client.post(
            "/remember",
            json={
                "content": (
                    "Legacy Lachlan enriches exactly as before the feature, "
                    "against the active profile."
                ),
                "idempotency_key": "route-enrich-legacy-1",
            },
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["profile"] == active
        assert captured["kwargs"].get("profile_id") == active, captured
        assert body["enriched_now"] == body["count"], body
        assert body["searchable_by"] == "meaning", body

    def test_routed_requests_leave_one_routing_log_line(
        self, daemon, caplog,
    ) -> None:
        """M-2: routed requests are distinguishable in the daemon log.

        One info line per routed request (method, path, profile), nothing on
        the legacy path — the routing is otherwise invisible: the response
        envelope names the profile, but the global pointer never moves.
        """
        import logging

        client, _ = daemon
        # The daemon module names its logger without the ".server." segment.
        with caplog.at_level(
            logging.INFO, logger="superlocalmemory.unified_daemon",
        ):
            routed_write = client.post(
                "/remember",
                json={
                    "content": (
                        "LogLine Lyra proves routed writes are visible in "
                        "the daemon log."
                    ),
                    "profile_id": "b",
                    "idempotency_key": "route-logline-b-1",
                },
            )
            routed_read = client.get(
                "/recall", params={"q": "LogLine Lyra", "profile_id": "b"},
            )
            legacy_write = client.post(
                "/remember",
                json={
                    "content": (
                        "LogLine Lena stays silent on the legacy path."
                    ),
                    "idempotency_key": "route-logline-legacy-1",
                },
            )
            legacy_read = client.get(
                "/recall", params={"q": "LogLine Lena"},
            )

        assert routed_write.status_code == 200, routed_write.text
        assert routed_read.status_code == 200, routed_read.text
        assert legacy_write.status_code == 200, legacy_write.text
        assert legacy_read.status_code == 200, legacy_read.text

        messages = [r.getMessage() for r in caplog.records]
        assert any(
            "per-request profile routing: POST /remember profile=b" in m
            for m in messages
        ), f"no routed-write log line in {messages}"
        assert any(
            "per-request profile routing: GET /recall profile=b" in m
            for m in messages
        ), f"no routed-read log line in {messages}"
        # Legacy requests stay silent: exactly two routing lines, both for b.
        routing_lines = [
            m for m in messages if "per-request profile routing" in m
        ]
        assert len(routing_lines) == 2, routing_lines
        assert all(m.endswith("profile=b") for m in routing_lines), routing_lines


class TestAuditHardening:
    """Regression tests for the 4.1.14 independent-audit findings."""

    def test_whitespace_profile_id_is_legacy_no_409(self, daemon) -> None:
        client, app = daemon
        engine = app.state.engine
        active = client.get("/status").json()["profile"]

        response = client.post(
            "/remember",
            json={
                "content": "Whitespace anchor must behave as legacy input.",
                "profile_id": "   ",
                "idempotency_key": "audit-ws-legacy-1",
            },
        )

        assert response.status_code == 200, response.text
        assert _facts_in(engine, active, "Whitespace anchor") != []
        assert response.json()["profile"] == active

    def test_empty_recall_echoes_served_profile(self, daemon) -> None:
        client, _ = daemon
        status = client.get("/status").json()

        response = client.get("/recall", params={"profile_id": "b"})

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["profile"] == "b", body
        assert body["profile_generation"] == status["profile_generation"]

    def test_keyword_fallback_echoes_served_profile(
        self, engine_with_mock_deps,
    ) -> None:
        from superlocalmemory.server.unified_daemon import (
            _recall_keyword_fallback,
        )

        body = _recall_keyword_fallback(
            engine_with_mock_deps, "Harbor crane", 5,
            profile_id="b", profile="b", profile_generation=7,
        )

        assert body["profile"] == "b", body
        assert body["profile_generation"] == 7, body
        assert body["retrieval_mode"] == "degraded_lexical"

    def test_deleted_profile_fails_closed_after_cache(self, daemon) -> None:
        client, app = daemon
        engine = app.state.engine
        runtime = app.state.canonical_remember_runtime

        warmed = client.post(
            "/remember",
            json={
                "content": "CacheWarm caches a handler for profile b first.",
                "profile_id": "b",
                "idempotency_key": "audit-cache-warm-1",
            },
        )
        assert warmed.status_code == 200, warmed.text
        assert "b" in runtime._routed_writers

        engine._db.execute("DELETE FROM atomic_facts WHERE profile_id = 'b'")
        engine._db.execute("DELETE FROM memories WHERE profile_id = 'b'")
        engine._db.execute(
            "DELETE FROM ingestion_operations WHERE profile_id = 'b'"
        )
        engine._db.execute("DELETE FROM profiles WHERE profile_id = 'b'")

        with runtime._binding_lock:
            with pytest.raises(ValueError, match="unknown profile"):
                runtime._routed_writer_locked("b")
        assert "b" not in runtime._routed_writers

    def test_late_unknown_profile_stays_404_not_503(
        self, daemon, monkeypatch,
    ) -> None:
        """4.1.14 audit: a profile deleted between the existence gate and
        admission must stay a 404 unknown_profile, never a 503 retryable —
        the coordinator wraps failures, so the HTTP boundary unwraps the
        cause chain for the distinctive type."""
        import superlocalmemory.server.unified_daemon as _ud

        client, app = daemon
        engine = app.state.engine

        engine._db.execute("DELETE FROM atomic_facts WHERE profile_id = 'b'")
        engine._db.execute("DELETE FROM memories WHERE profile_id = 'b'")
        engine._db.execute(
            "DELETE FROM ingestion_operations WHERE profile_id = 'b'"
        )
        engine._db.execute("DELETE FROM profiles WHERE profile_id = 'b'")
        monkeypatch.setattr(
            _ud, "_daemon_profile_exists", lambda engine, pid: True,
        )

        response = client.post(
            "/remember",
            json={
                "content": "LateDelete probes the check-to-admit race window.",
                "profile_id": "b",
                "idempotency_key": "audit-late-unknown-1",
            },
        )

        assert response.status_code == 404, response.text
        body = response.json()
        assert body["success"] is False
        assert body["error"]["code"] == "unknown_profile"

    def test_routed_writer_cache_is_bounded(
        self, daemon, engine_with_mock_deps, monkeypatch,
    ) -> None:
        from superlocalmemory.core import remember_runtime as _rr

        client, app = daemon
        runtime = app.state.canonical_remember_runtime
        monkeypatch.setattr(_rr, "_ROUTED_WRITERS_CAP", 3)
        for name in ("p1", "p2", "p3", "p4"):
            engine_with_mock_deps._db.execute(
                "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
                (name, name),
            )
        with runtime._binding_lock:
            for name in ("p1", "p2", "p3", "p4"):
                runtime._routed_writer_locked(name)
            assert len(runtime._routed_writers) <= 3
            assert set(runtime._routed_writers) <= {"p2", "p3", "p4"}
