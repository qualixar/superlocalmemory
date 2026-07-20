"""Release readiness must differ from process liveness."""

from __future__ import annotations

import asyncio


def _health_route(app):
    return next(
        route for route in app.routes if getattr(route, "path", None) == "/health"
    )


def test_app_uses_only_lifespan_startup_hooks(tmp_path, monkeypatch) -> None:
    from superlocalmemory.server import unified_daemon

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    app = unified_daemon.create_app()

    assert app.router.on_startup == []


def test_health_is_live_but_not_ready_after_required_migration_failure(
    tmp_path, monkeypatch,
) -> None:
    from superlocalmemory.server import unified_daemon

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    app = unified_daemon.create_app()
    app.state.engine = object()
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


def test_health_ready_requires_engine_and_clean_migrations(tmp_path, monkeypatch) -> None:
    from superlocalmemory.server import unified_daemon

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(unified_daemon, "_embedding_warm", True)
    app = unified_daemon.create_app()
    app.state.engine = object()
    app.state.migration_result = {
        "applied": ["M018"], "skipped": [], "failed": [], "details": {},
    }

    payload = asyncio.run(_health_route(app).endpoint())

    assert payload["ready"] is True
    assert payload["readiness"] == {
        "engine": True,
        "migrations": True,
        "retrieval": True,
        "migration_failures": [],
    }


def test_health_is_live_but_warming_until_embedding_is_usable(
    tmp_path, monkeypatch,
) -> None:
    from superlocalmemory.server import unified_daemon

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(unified_daemon, "_embedding_warm", False)
    app = unified_daemon.create_app()
    app.state.engine = object()
    app.state.migration_result = {
        "applied": ["M018"], "skipped": [], "failed": [], "details": {},
    }

    payload = asyncio.run(_health_route(app).endpoint())

    assert payload["status"] == "ok"
    assert payload["ready"] is False
    assert payload["state"] == "warming"
    assert payload["readiness"]["retrieval"] is False


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
