"""Release readiness must differ from process liveness."""

from __future__ import annotations

import asyncio


def _health_route(app):
    return next(route for route in app.routes if getattr(route, "path", None) == "/health")


def _ready_writer():
    return type("ReadyWriter", (), {"ready": True})()


def test_app_uses_only_lifespan_startup_hooks(tmp_path, monkeypatch) -> None:
    from superlocalmemory.server import unified_daemon

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    app = unified_daemon.create_app()

    assert app.router.on_startup == []


def test_health_is_live_but_not_ready_after_required_migration_failure(
    tmp_path,
    monkeypatch,
) -> None:
    from superlocalmemory.server import unified_daemon

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    app = unified_daemon.create_app()
    app.state.engine = object()
    app.state.canonical_remember_runtime = _ready_writer()
    app.state.migration_result = {
        "applied": [],
        "skipped": [],
        "failed": ["M018"],
        "details": {"M018": "database is locked"},
    }

    payload = asyncio.run(_health_route(app).endpoint())

    assert payload["status"] == "ok"
    assert payload["ready"] is False
    assert payload["readiness"]["migration_failures"] == ["M018"]


def test_health_ready_requires_engine_migrations_writer_and_retrieval(
    tmp_path,
    monkeypatch,
) -> None:
    from superlocalmemory.server import unified_daemon

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(unified_daemon, "_embedding_warm", True)
    app = unified_daemon.create_app()
    app.state.engine = object()
    app.state.canonical_remember_runtime = _ready_writer()
    app.state.migration_result = {
        "applied": ["M018"],
        "skipped": [],
        "failed": [],
        "details": {},
    }

    payload = asyncio.run(_health_route(app).endpoint())

    assert payload["ready"] is True
    assert payload["readiness"] == {
        "engine": True,
        "migrations": True,
        "writer": True,
        "embedding": True,
        "recall_health": True,
        "retrieval": True,
        "migration_failures": [],
    }
    assert payload["state"] == "ready"
    assert payload["runtime_state"] == "serving_full"


def test_health_is_live_but_warming_until_embedding_is_usable(
    tmp_path,
    monkeypatch,
) -> None:
    from superlocalmemory.server import unified_daemon

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(unified_daemon, "_embedding_warm", False)
    app = unified_daemon.create_app()
    app.state.engine = object()
    app.state.canonical_remember_runtime = _ready_writer()
    app.state.migration_result = {
        "applied": ["M018"],
        "skipped": [],
        "failed": [],
        "details": {},
    }

    payload = asyncio.run(_health_route(app).endpoint())

    assert payload["status"] == "ok"
    assert payload["ready"] is False
    assert payload["state"] == "ready"
    assert payload["runtime_state"] == "warming"
    assert payload["readiness"]["retrieval"] is False


def test_health_is_not_ready_when_the_canonical_writer_is_absent(
    tmp_path,
    monkeypatch,
) -> None:
    """A live reader must not advertise a writable daemon as ready."""
    from superlocalmemory.server import unified_daemon

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(unified_daemon, "_embedding_warm", True)
    app = unified_daemon.create_app()
    app.state.engine = object()
    app.state.migration_result = {
        "applied": ["M018"],
        "skipped": [],
        "failed": [],
        "details": {},
    }

    payload = asyncio.run(_health_route(app).endpoint())

    assert payload["ready"] is False
    assert payload["state"] == "starting"
    assert payload["runtime_state"] == "not_ready"
    assert payload["readiness"]["writer"] is False


def test_lifespan_failure_cleanup_releases_a_started_canonical_writer(
    tmp_path,
    monkeypatch,
) -> None:
    """A later startup failure cannot retain the process-wide writer lease."""
    from superlocalmemory.server import unified_daemon

    class StartedWriter:
        def __init__(self) -> None:
            self.stop_calls = 0

        def stop(self) -> None:
            self.stop_calls += 1

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    app = unified_daemon.create_app()
    writer = StartedWriter()
    app.state.canonical_remember_runtime = writer

    unified_daemon._release_canonical_remember_runtime(app)

    assert writer.stop_calls == 1
    assert app.state.canonical_remember_runtime is None


def test_lifespan_cleanup_keeps_a_draining_canonical_writer_reachable(
    tmp_path,
    monkeypatch,
) -> None:
    """Cleanup cannot close the engine after a bounded writer-stop timeout."""
    from superlocalmemory.server import unified_daemon

    class DrainingWriter:
        def stop(self) -> None:
            raise RuntimeError("writer is still draining")

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    app = unified_daemon.create_app()
    writer = DrainingWriter()
    app.state.canonical_remember_runtime = writer

    stopped = unified_daemon._release_canonical_remember_runtime(app)

    assert stopped is False
    assert app.state.canonical_remember_runtime is writer


def test_configured_daemon_port_honours_the_isolated_port(monkeypatch) -> None:
    """Operator-facing readiness text must name the port the process binds."""
    from superlocalmemory.server import unified_daemon

    monkeypatch.setenv("SLM_DAEMON_PORT", "18765")

    assert unified_daemon._configured_daemon_port() == 18765


def test_daemon_lifespan_does_not_block_on_reranker_warmup() -> None:
    """A usable daemon must publish routes while local models warm in background."""
    import inspect

    from superlocalmemory.server import unified_daemon

    source = inspect.getsource(unified_daemon.lifespan)
    assert "reranker.warmup_sync(timeout=120)" not in source


def test_daemon_reserves_listener_before_engine_or_migration_work() -> None:
    """A duplicate daemon must fail before touching the shared data root."""
    import inspect

    from superlocalmemory.server import unified_daemon

    source = inspect.getsource(unified_daemon.start_server)
    assert source.index("listener.bind") < source.index("_publish_process_descriptor")
    assert source.index("listener.bind") < source.index("_start_memory_watchdog")
    assert "server.run(sockets=[listener])" in source
