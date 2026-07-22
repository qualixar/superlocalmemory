"""V3.7.7 regression contracts for daemon-aware profile switching."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient


class _ToolCollector:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *args, **kwargs):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn

        return decorator


def _add_profile(engine, profile_id: str) -> None:
    engine._db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
        (profile_id, profile_id),
    )
    from superlocalmemory.server.routes.helpers import ensure_profile_in_json

    ensure_profile_in_json(profile_id)


def _daemon_headers(app) -> dict[str, str]:
    descriptor = app.state.daemon_descriptor
    return {
        "X-SLM-Daemon-Capability": descriptor.capability,
        "X-SLM-Target-Instance": descriptor.instance_id,
    }


def _wait_for(predicate, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition did not become true before timeout")


def test_runtime_transition_is_linearizable() -> None:
    """Operations arriving during a switch wait for the new generation."""
    from superlocalmemory.server.profile_runtime import ProfileRuntime

    runtime = ProfileRuntime("alpha")
    old_admitted = threading.Event()
    release_old = threading.Event()
    new_admitted = threading.Event()
    seen: dict[str, str] = {}

    def _old_operation() -> None:
        with runtime.operation() as snapshot:
            seen["old"] = snapshot.profile_id
            old_admitted.set()
            assert release_old.wait(2)

    def _switch() -> None:
        runtime.transition("beta", lambda previous, target: None)

    def _new_operation() -> None:
        with runtime.operation() as snapshot:
            seen["new"] = snapshot.profile_id
            new_admitted.set()

    old_thread = threading.Thread(target=_old_operation)
    switch_thread = threading.Thread(target=_switch)
    new_thread = threading.Thread(target=_new_operation)
    old_thread.start()
    assert old_admitted.wait(2)
    switch_thread.start()
    _wait_for(lambda: runtime.transitioning)
    new_thread.start()
    assert not new_admitted.wait(0.05)

    release_old.set()
    old_thread.join(2)
    switch_thread.join(2)
    new_thread.join(2)

    assert seen == {"old": "alpha", "new": "beta"}
    assert runtime.snapshot.profile_id == "beta"
    assert runtime.snapshot.generation == 1


def test_runtime_transition_failure_rolls_back_and_unblocks_waiters() -> None:
    """A failed durable/rebind callback retains the former generation."""
    from superlocalmemory.server.profile_runtime import ProfileRuntime

    runtime = ProfileRuntime("alpha")

    def _fail(previous, target) -> None:
        raise OSError("disk full")

    try:
        runtime.transition("beta", _fail)
    except OSError as exc:
        assert str(exc) == "disk full"
    else:  # pragma: no cover
        raise AssertionError("transition failure was swallowed")

    with runtime.operation() as snapshot:
        assert snapshot.profile_id == "alpha"
        assert snapshot.generation == 0
    assert runtime.transitioning is False


def test_profile_commit_rolls_engine_back_when_persistence_fails(
    monkeypatch,
) -> None:
    """Runtime binding and generation stay old if durable state cannot commit."""
    from superlocalmemory.server import profile_runtime as runtime_module

    config = SimpleNamespace(active_profile="alpha")
    database = MagicMock()
    database.execute.return_value = [{"exists": 1}]
    engine = SimpleNamespace(
        profile_id="alpha",
        _config=config,
        _db=database,
    )
    state = SimpleNamespace(engine=engine, config=config)
    runtime = runtime_module.ProfileRuntime("alpha")

    def _disk_failure(profile_id: str):
        raise OSError("disk full")

    monkeypatch.setattr(
        runtime_module,
        "persist_active_profile",
        _disk_failure,
    )

    try:
        runtime.transition(
            "beta",
            lambda previous, target: runtime_module.commit_daemon_profile_switch(
                state, previous, target,
            ),
        )
    except OSError as exc:
        assert str(exc) == "disk full"
    else:  # pragma: no cover
        raise AssertionError("persistence failure was swallowed")

    assert engine.profile_id == "alpha"
    assert config.active_profile == "alpha"
    assert runtime.snapshot.profile_id == "alpha"
    assert runtime.snapshot.generation == 0


def test_cancelled_http_waiter_cannot_leak_an_operation_lease() -> None:
    """Cancellation while a transition drains must not block later switches."""
    from superlocalmemory.server.profile_runtime import (
        ProfileRuntime,
        ProfileRuntimeMiddleware,
    )

    runtime = ProfileRuntime("alpha")
    commit_entered = threading.Event()
    release_commit = threading.Event()

    def _switch() -> None:
        runtime.transition(
            "beta",
            lambda previous, target: (
                commit_entered.set(), release_commit.wait(2)
            ),
        )

    switch_thread = threading.Thread(target=_switch)
    switch_thread.start()
    assert commit_entered.wait(2)

    async def _exercise() -> None:
        async def _inner(scope, receive, send) -> None:
            raise AssertionError("cancelled request must not reach the app")

        async def _receive() -> dict:
            return {"type": "http.disconnect"}

        async def _send(message: dict) -> None:
            return None

        middleware = ProfileRuntimeMiddleware(
            _inner,
            app_state=SimpleNamespace(profile_runtime=runtime),
        )
        task = asyncio.create_task(middleware(
            {"type": "http", "path": "/status", "method": "GET"},
            _receive,
            _send,
        ))
        await asyncio.sleep(0.02)
        task.cancel()
        release_commit.set()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:  # pragma: no cover
            raise AssertionError("request cancellation was swallowed")

    asyncio.run(_exercise())
    switch_thread.join(2)
    assert not switch_thread.is_alive()

    # A leaked operation lease would leave this transition waiting forever.
    final_switch = threading.Thread(
        target=runtime.transition,
        args=("gamma", lambda previous, target: None),
    )
    final_switch.start()
    final_switch.join(2)
    assert not final_switch.is_alive()
    assert runtime.snapshot.profile_id == "gamma"


def test_queue_recall_adapter_holds_runtime_lease() -> None:
    """Hook-queue recalls already admitted must drain before a switch commits."""
    from superlocalmemory.server.profile_runtime import ProfileRuntime
    from superlocalmemory.server.unified_daemon import EngineRecallAdapter

    runtime = ProfileRuntime("alpha")
    recall_entered = threading.Event()
    release_recall = threading.Event()
    switch_committed = threading.Event()
    engine = MagicMock()
    engine.recall.side_effect = lambda *args, **kwargs: (
        recall_entered.set(), release_recall.wait(2), MagicMock(results=[])
    )[-1]

    recall_thread = threading.Thread(
        target=EngineRecallAdapter(engine, runtime).recall,
        args=("probe",),
    )
    switch_thread = threading.Thread(
        target=runtime.transition,
        args=("beta", lambda previous, target: switch_committed.set()),
    )
    recall_thread.start()
    assert recall_entered.wait(2)
    switch_thread.start()
    _wait_for(lambda: runtime.transitioning)
    assert not switch_committed.wait(0.05)

    release_recall.set()
    recall_thread.join(2)
    switch_thread.join(2)
    assert switch_committed.is_set()
    assert runtime.snapshot.profile_id == "beta"


def test_recall_health_tick_holds_runtime_lease() -> None:
    """The daemon health probe cannot observe an engine mid-rebind."""
    from superlocalmemory.server.profile_runtime import ProfileRuntime
    from superlocalmemory.server.recall_health import RecallHealth, run_health_tick

    runtime = ProfileRuntime("alpha")
    recall_entered = threading.Event()
    release_recall = threading.Event()
    switch_committed = threading.Event()
    engine = MagicMock()
    engine.recall.side_effect = lambda *args, **kwargs: (
        recall_entered.set(), release_recall.wait(2), MagicMock(results=[])
    )[-1]

    health_thread = threading.Thread(
        target=run_health_tick,
        args=(engine, RecallHealth()),
        kwargs={"runtime": runtime},
    )
    switch_thread = threading.Thread(
        target=runtime.transition,
        args=("beta", lambda previous, target: switch_committed.set()),
    )
    health_thread.start()
    assert recall_entered.wait(2)
    switch_thread.start()
    _wait_for(lambda: runtime.transitioning)
    assert not switch_committed.wait(0.05)

    release_recall.set()
    health_thread.join(2)
    switch_thread.join(2)
    assert switch_committed.is_set()
    assert runtime.snapshot.profile_id == "beta"


def test_runtime_reconfigure_drains_profile_operations() -> None:
    """Mode/provider rebuilds share the same no-half-switched barrier."""
    from superlocalmemory.server.profile_runtime import ProfileRuntime

    runtime = ProfileRuntime("alpha")
    operation_entered = threading.Event()
    release_operation = threading.Event()
    reconfigured = threading.Event()

    def _operation() -> None:
        with runtime.operation():
            operation_entered.set()
            assert release_operation.wait(2)

    operation_thread = threading.Thread(target=_operation)
    reconfigure_thread = threading.Thread(
        target=runtime.reconfigure,
        args=(lambda snapshot: reconfigured.set(),),
    )
    operation_thread.start()
    assert operation_entered.wait(2)
    reconfigure_thread.start()
    _wait_for(lambda: runtime.transitioning)
    assert not reconfigured.wait(0.05)
    release_operation.set()
    operation_thread.join(2)
    reconfigure_thread.join(2)
    assert reconfigured.is_set()
    assert runtime.snapshot.profile_id == "alpha"


def test_hot_reconfigure_rebinds_all_long_lived_engine_references(
    monkeypatch,
) -> None:
    """A dashboard config save publishes one engine to every daemon consumer."""
    from superlocalmemory.core import engine as engine_module
    from superlocalmemory.server import recall_health, unified_daemon

    new_engine = MagicMock(profile_id="alpha")
    fake_config = MagicMock(active_profile="alpha")
    fake_engine_type = MagicMock(return_value=new_engine)
    monkeypatch.setattr(engine_module, "MemoryEngine", fake_engine_type)
    monkeypatch.setattr(unified_daemon, "_configure_scale_backends", MagicMock())
    health_stop = threading.Event()
    monkeypatch.setattr(
        recall_health,
        "start_recall_health_monitor",
        lambda engine, runtime=None: (MagicMock(), health_stop, MagicMock()),
    )
    old_engine = MagicMock(profile_id="alpha")
    old_health_stop = threading.Event()
    adapter = MagicMock()
    state = SimpleNamespace(
        engine=old_engine,
        config=MagicMock(active_profile="alpha"),
        profile_runtime=MagicMock(),
        engine_recall_adapter=adapter,
        recall_health_stop=old_health_stop,
    )
    application = SimpleNamespace(state=state)

    unified_daemon._hot_reconfigure_engine(
        application, fake_config, mode_change=True,
    )

    new_engine.initialize.assert_called_once_with()
    fake_config.save.assert_called_once_with(mode_change=True)
    assert state.engine is new_engine
    assert state.config is fake_config
    adapter.set_engine.assert_called_once_with(new_engine)
    assert old_health_stop.is_set()
    assert state.recall_health_stop is health_stop
    old_engine.close.assert_called_once_with()


def test_reconfigure_cannot_restore_a_stale_profile_candidate() -> None:
    """A settings request loaded before a switch keeps runtime profile truth."""
    from superlocalmemory.server.profile_runtime import (
        ProfileRuntime,
        reconfigure_daemon_engine,
    )

    candidate = SimpleNamespace(active_profile="alpha")
    applied: list[str] = []
    state = SimpleNamespace(
        profile_runtime=ProfileRuntime("beta", generation=3),
        reconfigure_engine=lambda config, mode_change=False: applied.append(
            config.active_profile
        ),
    )

    reconfigure_daemon_engine(state, candidate, mode_change=False)

    assert candidate.active_profile == "beta"
    assert applied == ["beta"]


def test_public_api_switch_rebinds_resident_engine(engine_with_mock_deps) -> None:
    """The dashboard/API path must change daemon runtime, not only files."""
    from superlocalmemory.server.unified_daemon import create_app

    engine = engine_with_mock_deps
    engine.profile_id = "alpha"
    engine._config.active_profile = "alpha"
    _add_profile(engine, "alpha")
    _add_profile(engine, "beta")

    app = create_app()
    app.state.engine = engine
    app.state.config = engine._config
    response = TestClient(app).post("/api/profiles/beta/switch")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["active_profile"] == "beta"
    assert payload["generation"] >= 1
    assert engine.profile_id == "beta"
    assert app.state.config.active_profile == "beta"


def test_daemon_profile_switch_isolates_write_status_and_recall(
    engine_with_mock_deps,
) -> None:
    """Alpha and beta remain isolated through one resident daemon engine."""
    from superlocalmemory.server.unified_daemon import create_app

    engine = engine_with_mock_deps
    engine.profile_id = "alpha"
    engine._config.active_profile = "alpha"
    _add_profile(engine, "alpha")
    _add_profile(engine, "beta")
    app = create_app()
    app.state.engine = engine
    app.state.config = engine._config
    client = TestClient(app)
    headers = _daemon_headers(app)
    engine_identity = id(engine)

    alpha_write = client.post(
        "/remember?wait=true",
        json={"content": "alpha-isolation-token-77 belongs only to alpha."},
        headers=headers,
    )
    assert alpha_write.status_code == 200, alpha_write.text
    alpha_status = client.get("/status").json()
    assert alpha_status["profile"] == "alpha"
    assert alpha_status["fact_count"] >= 1

    switched = client.post("/api/profiles/beta/switch").json()
    assert switched["active_profile"] == "beta"
    beta_empty = client.get("/status").json()
    assert beta_empty["profile"] == "beta"
    assert beta_empty["fact_count"] == 0

    beta_write = client.post(
        "/remember?wait=true",
        json={"content": "beta-isolation-token-88 belongs only to beta."},
        headers=headers,
    )
    assert beta_write.status_code == 200, beta_write.text
    beta_status = client.get("/status").json()
    assert beta_status["profile"] == "beta"
    assert beta_status["fact_count"] >= 1

    beta_recall = client.get(
        "/recall?q=beta-isolation-token-88&limit=10&fast=true",
    ).json()
    beta_text = " ".join(item["content"] for item in beta_recall["results"])
    assert "beta-isolation-token-88" in beta_text
    assert "alpha-isolation-token-77" not in beta_text

    switched_back = client.post("/api/profiles/alpha/switch").json()
    assert switched_back["active_profile"] == "alpha"
    alpha_recall = client.get(
        "/recall?q=alpha-isolation-token-77&limit=10&fast=true",
    ).json()
    alpha_text = " ".join(item["content"] for item in alpha_recall["results"])
    assert "alpha-isolation-token-77" in alpha_text
    assert "beta-isolation-token-88" not in alpha_text
    assert id(app.state.engine) == engine_identity


def test_default_recall_scope_stays_personal_after_profile_switch(
    engine_with_mock_deps,
) -> None:
    """Shared/global memories require explicit recall opt-in after a switch."""
    from superlocalmemory.server.unified_daemon import create_app

    engine = engine_with_mock_deps
    engine.profile_id = "alpha"
    engine._config.active_profile = "alpha"
    _add_profile(engine, "alpha")
    _add_profile(engine, "beta")
    app = create_app()
    app.state.engine = engine
    app.state.config = engine._config
    client = TestClient(app)
    headers = _daemon_headers(app)
    token = "global-scope-opt-in-token-991"

    stored = client.post(
        "/remember?wait=true",
        json={"content": f"{token} is globally visible only by opt-in.", "scope": "global"},
        headers=headers,
    )
    assert stored.status_code == 200, stored.text
    switched = client.post("/api/profiles/beta/switch", headers=headers)
    assert switched.status_code == 200, switched.text

    default_results = client.get(
        "/recall", params={"q": token, "fast": "true"}, headers=headers,
    ).json()["results"]
    opted_in_results = client.get(
        "/recall",
        params={"q": token, "fast": "true", "include_global": "true"},
        headers=headers,
    ).json()["results"]

    assert all(token not in result.get("content", "") for result in default_results)
    assert any(token in result.get("content", "") for result in opted_in_results)


def test_write_arriving_during_switch_waits_and_lands_in_new_profile(
    engine_with_mock_deps,
    monkeypatch,
) -> None:
    """The transition barrier prevents the production cross-profile race."""
    from superlocalmemory.server.routes import profiles as profile_routes
    from superlocalmemory.server.unified_daemon import create_app

    engine = engine_with_mock_deps
    engine.profile_id = "alpha"
    engine._config.active_profile = "alpha"
    _add_profile(engine, "alpha")
    _add_profile(engine, "beta")
    app = create_app()
    app.state.engine = engine
    app.state.config = engine._config
    headers = _daemon_headers(app)
    commit_entered = threading.Event()
    allow_commit = threading.Event()
    write_finished = threading.Event()
    original_commit = profile_routes.commit_daemon_profile_switch

    def _blocked_commit(app_state, previous, target) -> None:
        commit_entered.set()
        assert allow_commit.wait(2)
        original_commit(app_state, previous, target)

    monkeypatch.setattr(
        profile_routes,
        "commit_daemon_profile_switch",
        _blocked_commit,
    )
    results: dict[str, object] = {}

    def _switch() -> None:
        results["switch"] = TestClient(app).post(
            "/api/profiles/beta/switch"
        )

    def _write() -> None:
        results["write"] = TestClient(app).post(
            "/remember?wait=true",
            json={"content": "transition-race-token-99 must land in beta."},
            headers=headers,
        )
        write_finished.set()

    switch_thread = threading.Thread(target=_switch)
    write_thread = threading.Thread(target=_write)
    switch_thread.start()
    assert commit_entered.wait(2)
    write_thread.start()
    assert not write_finished.wait(0.05)
    allow_commit.set()
    switch_thread.join(5)
    write_thread.join(5)

    assert results["switch"].status_code == 200
    assert results["write"].status_code == 200
    rows = engine._db.execute(
        "SELECT DISTINCT profile_id FROM atomic_facts WHERE content LIKE ?",
        ("%transition-race-token-99%",),
    )
    assert {dict(row)["profile_id"] for row in rows} == {"beta"}


def test_cli_switch_requires_owned_daemon_acknowledgement(
    monkeypatch,
    capsys,
) -> None:
    """A resident daemon is runtime truth; CLI must wait for its ACK."""
    from superlocalmemory.cli import daemon
    from superlocalmemory.cli.commands import cmd_profile

    cmd_profile(Namespace(action="create", name="beta", json=True))
    capsys.readouterr()

    calls: list[tuple[str, str, dict | None]] = []
    monkeypatch.setattr(daemon, "is_daemon_running", lambda: True)

    def _request(method: str, path: str, body: dict | None = None, **kwargs):
        calls.append((method, path, body))
        return {"success": True, "active_profile": "beta", "generation": 2}

    monkeypatch.setattr(daemon, "daemon_request", _request)
    cmd_profile(Namespace(action="switch", name="beta", json=True))

    envelope = json.loads(capsys.readouterr().out)
    assert calls == [("POST", "/api/profiles/beta/switch", None)]
    assert envelope["data"]["profile"] == "beta"
    assert envelope["data"]["generation"] == 2


def test_mcp_switch_uses_canonical_daemon_transition(monkeypatch) -> None:
    """MCP must not mutate only its process-local LIGHT engine."""
    from superlocalmemory.cli import daemon
    from superlocalmemory.mcp import tools_core

    engine = MagicMock()
    engine.profile_id = "alpha"
    authorization = MagicMock()
    monkeypatch.setattr(
        tools_core,
        "authorize_mcp_mutation",
        lambda *args, **kwargs: authorization,
    )
    monkeypatch.setattr(daemon, "is_daemon_running", lambda: True)
    calls: list[tuple[str, str, dict | None]] = []

    def _request(method: str, path: str, body: dict | None = None, **kwargs):
        calls.append((method, path, body))
        return {
            "success": True,
            "previous_profile": "alpha",
            "active_profile": "beta",
            "generation": 4,
        }

    monkeypatch.setattr(daemon, "daemon_request", _request)
    collector = _ToolCollector()
    tools_core.register_core_tools(collector, lambda: engine)

    result = asyncio.run(collector.tools["switch_profile"]("beta"))

    assert result["success"] is True
    assert result["current_profile"] == "beta"
    assert result["generation"] == 4
    assert calls == [("POST", "/api/profiles/beta/switch", None)]
    authorization.complete.assert_called_once_with()


def test_mcp_profile_defaults_resolve_from_daemon_runtime(monkeypatch) -> None:
    """A second MCP process cannot keep reading its startup profile."""
    from superlocalmemory.cli import daemon
    from superlocalmemory.mcp import tools_core

    engine = MagicMock()
    engine.profile_id = "alpha"
    engine._db.search_facts_fts.return_value = []
    monkeypatch.setattr(daemon, "is_daemon_running", lambda: True)
    monkeypatch.setattr(
        daemon,
        "daemon_request",
        lambda method, path, *args, **kwargs: {
            "profile": "beta", "profile_generation": 9,
        },
    )
    collector = _ToolCollector()
    tools_core.register_core_tools(collector, lambda: engine)

    result = asyncio.run(collector.tools["search"]("runtime-token"))

    assert result["success"] is True
    engine._db.search_facts_fts.assert_called_once_with(
        "runtime-token", "beta", limit=10,
    )


def test_mcp_mutations_use_profile_leased_daemon_routes(monkeypatch) -> None:
    """MCP delete/update must not use a stale profile-bound worker process."""
    from superlocalmemory.cli import daemon
    from superlocalmemory.mcp import tools_core

    engine = MagicMock(profile_id="alpha")
    monkeypatch.setattr(daemon, "is_daemon_running", lambda: True)
    calls: list[tuple[str, str, dict | None]] = []

    def _request(method: str, path: str, body: dict | None = None, **kwargs):
        calls.append((method, path, body))
        if method == "DELETE":
            return {"success": True, "deleted": "fact-1"}
        return {"success": True, "fact_id": "fact-1", "content": "new"}

    monkeypatch.setattr(daemon, "daemon_request", _request)
    collector = _ToolCollector()
    tools_core.register_core_tools(collector, lambda: engine)

    deleted = asyncio.run(collector.tools["delete_memory"]("fact-1", "agent-a"))
    updated = asyncio.run(
        collector.tools["update_memory"]("fact-1", "new", "agent-a")
    )

    assert deleted["success"] is True
    assert updated["success"] is True
    assert calls == [
        ("DELETE", "/api/memories/fact-1", None),
        ("PATCH", "/api/memories/fact-1", {"content": "new"}),
    ]


def test_switch_profile_is_available_in_default_mcp_surface() -> None:
    """Profile control is not hidden behind the all-tools opt-in."""
    from superlocalmemory.mcp.server import _ESSENTIAL_TOOLS

    assert "switch_profile" in _ESSENTIAL_TOOLS


def test_scope_config_api_preserves_personal_default_and_hot_applies(
    monkeypatch,
) -> None:
    """UI scope controls use the acknowledged daemon reconfiguration path."""
    from superlocalmemory.core.config import ScopeConfig
    from superlocalmemory.server.routes import v3_api

    config = MagicMock()
    config.scope = ScopeConfig()
    applied: list[tuple[object, bool]] = []

    async def _apply(request, candidate, *, mode_change: bool) -> None:
        applied.append((candidate, mode_change))

    request = MagicMock()
    # SEC-H-01: set_scope_config now calls require_manage(request). This unit
    # test drives the handler directly with a mock request, so make app.state
    # reflect a no-RBAC install (the machine owner is authorized) rather than a
    # truthy auto-mock engine that would produce a malformed principal.
    request.app.state.rbac = None
    request.json = MagicMock(
        side_effect=lambda: asyncio.sleep(0, result={
            "default_scope": "personal",
            "recall_include_shared": True,
            "recall_include_global": False,
        })
    )
    monkeypatch.setattr(
        "superlocalmemory.core.config.SLMConfig.load", lambda: config,
    )
    monkeypatch.setattr(v3_api, "_apply_runtime_config", _apply)

    result = asyncio.run(v3_api.set_scope_config(request))

    assert result["success"] is True
    assert result["default_scope"] == "personal"
    assert result["recall_include_shared"] is True
    assert result["recall_include_global"] is False
    assert applied == [(config, False)]


def test_failed_daemon_switch_does_not_report_cli_success(
    monkeypatch,
    capsys,
) -> None:
    """CLI must not persist or claim a profile the daemon rejected."""
    from superlocalmemory.cli import daemon
    from superlocalmemory.cli.commands import cmd_profile

    cmd_profile(Namespace(action="create", name="beta", json=True))
    capsys.readouterr()
    monkeypatch.setattr(daemon, "is_daemon_running", lambda: True)
    monkeypatch.setattr(daemon, "daemon_request", lambda *args, **kwargs: None)

    try:
        cmd_profile(Namespace(action="switch", name="beta", json=True))
    except SystemExit as exc:
        assert exc.code == 1
    else:  # pragma: no cover - documents the release-blocking failure
        raise AssertionError("CLI reported success without daemon acknowledgement")

    envelope = json.loads(capsys.readouterr().out)
    assert envelope["error"]["code"] == "PROFILE_SWITCH_FAILED"


def test_offline_cli_switch_keeps_compatible_persistent_fallback(
    monkeypatch,
    capsys,
) -> None:
    """Without a daemon, the existing durable CLI behavior remains valid."""
    from superlocalmemory.cli import daemon
    from superlocalmemory.cli.commands import cmd_profile
    from superlocalmemory.core.config import SLMConfig

    cmd_profile(Namespace(action="create", name="beta", json=True))
    capsys.readouterr()
    monkeypatch.setattr(daemon, "is_daemon_running", lambda: False)
    cmd_profile(Namespace(action="switch", name="beta", json=True))

    envelope = json.loads(capsys.readouterr().out)
    assert envelope["data"]["profile"] == "beta"
    assert envelope["data"]["runtime"] == "offline"
    assert SLMConfig.load().active_profile == "beta"


def test_cli_status_prefers_resident_daemon_runtime(
    monkeypatch,
    capsys,
) -> None:
    """Status must report the runtime profile used by daemon reads/writes."""
    from superlocalmemory.cli import daemon
    from superlocalmemory.cli.commands import cmd_status

    monkeypatch.setattr(daemon, "is_daemon_running", lambda: True)
    monkeypatch.setattr(
        daemon,
        "daemon_request",
        lambda method, path, *args, **kwargs: {
            "mode": "a",
            "provider": "none",
            "profile": "beta",
            "base_dir": "/isolated",
            "db_path": "/isolated/memory.db",
            "db_size_mb": 1.25,
            "fact_count": 1,
            "entity_count": 2,
            "edge_count": 3,
            "profile_generation": 7,
        },
    )

    cmd_status(Namespace(json=True))

    envelope = json.loads(capsys.readouterr().out)
    assert envelope["data"]["profile"] == "beta"
    assert envelope["data"]["fact_count"] == 1
    assert envelope["data"]["profile_generation"] == 7


def test_mcp_status_prefers_resident_daemon_runtime(monkeypatch) -> None:
    """Every MCP process must report the daemon's active generation."""
    from superlocalmemory.cli import daemon
    from superlocalmemory.mcp import tools_core

    local_engine = MagicMock()
    local_engine.profile_id = "alpha"
    monkeypatch.setattr(daemon, "is_daemon_running", lambda: True)
    monkeypatch.setattr(
        daemon,
        "daemon_request",
        lambda method, path, *args, **kwargs: {
            "mode": "a",
            "provider": "none",
            "profile": "beta",
            "base_dir": "/isolated",
            "db_path": "/isolated/memory.db",
            "db_size_mb": 1.25,
            "fact_count": 1,
            "entity_count": 2,
            "edge_count": 3,
            "profile_generation": 7,
        },
    )
    collector = _ToolCollector()
    tools_core.register_core_tools(collector, lambda: local_engine)

    result = asyncio.run(collector.tools["get_status"]())

    assert result["profile"] == "beta"
    assert result["profile_generation"] == 7
    assert result["fact_count"] == 1
