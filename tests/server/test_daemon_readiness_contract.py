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
        "migration_failures": [],
    }
