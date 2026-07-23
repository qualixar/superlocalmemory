# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Route contracts that keep blocking dashboard work off the event loop."""

from __future__ import annotations

import inspect

from fastapi import APIRouter, FastAPI
from fastapi.routing import APIRoute

from superlocalmemory.server.routes import config_api, learning, ratelimit
from superlocalmemory.server.unified_daemon import create_app


def _iter_runtime_routes(
    routes,
    router_ancestry: frozenset[int] = frozenset(),
):
    """Walk both flattened FastAPI routes and lazy included-router wrappers."""
    for route in routes:
        yield route
        original_router = getattr(route, "original_router", None)
        if (
            original_router is not None
            and id(original_router) not in router_ancestry
        ):
            yield from _iter_runtime_routes(
                original_router.routes,
                router_ancestry | {id(original_router)},
            )


def test_patterns_get_has_one_canonical_app_route() -> None:
    app = create_app()
    matches = [
        route
        for route in _iter_runtime_routes(app.routes)
        if isinstance(route, APIRoute)
        and route.path == "/api/patterns"
        and "GET" in route.methods
    ]

    assert len(matches) == 1
    assert matches[0].endpoint.__module__ == learning.__name__


def test_runtime_route_walk_counts_duplicate_inclusion_paths() -> None:
    router = APIRouter()

    @router.get("/duplicate-witness")
    def duplicate_witness() -> dict:
        return {}

    app = FastAPI()
    app.include_router(router)
    app.include_router(router)

    matches = [
        route
        for route in _iter_runtime_routes(app.routes)
        if isinstance(route, APIRoute)
        and route.path == "/duplicate-witness"
        and "GET" in route.methods
    ]
    assert len(matches) == 2


def test_patterns_canonical_schema_is_stable_when_learning_is_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(learning, "BEHAVIORAL_AVAILABLE", False)
    monkeypatch.setattr(learning, "LEARNING_AVAILABLE", False)

    result = learning.get_patterns()

    assert result["available"] is False
    assert result["patterns"] == {
        "preference": [],
        "style": [],
        "terminology": [],
        "workflow": [],
    }
    assert result["signal_stats"] == {}
    assert result["total_patterns"] == 0
    assert result["pattern_types"] == []
    assert result["confidence_stats"] == {
        "avg": 0.0,
        "min": 0.0,
        "max": 0.0,
    }


def test_config_file_handlers_are_sync_for_fastapi_threadpool() -> None:
    handlers = (
        config_api.get_storage_config,
        config_api.put_storage_config,
        config_api.get_daemon_config,
        config_api.put_daemon_config,
        config_api.get_mesh_config,
        config_api.put_mesh_config,
        config_api.get_trust_config,
        config_api.put_trust_config,
        config_api.get_forgetting_config,
        config_api.put_forgetting_config,
    )

    assert not any(inspect.iscoroutinefunction(handler) for handler in handlers)


def test_blocking_learning_handlers_are_sync_for_fastapi_threadpool() -> None:
    handlers = (
        learning.record_feedback,
        learning.record_dwell,
        learning.feedback_stats,
        learning.delete_pattern,
        learning.get_patterns,
        learning.learning_backup,
        learning.learning_reset,
        learning.learning_migrate_legacy,
    )

    assert not any(inspect.iscoroutinefunction(handler) for handler in handlers)
    assert inspect.iscoroutinefunction(learning.learning_retrain)


def test_ratelimit_persistence_handler_is_sync_for_fastapi_threadpool() -> None:
    assert not inspect.iscoroutinefunction(ratelimit.put_ratelimit)
