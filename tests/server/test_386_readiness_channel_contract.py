# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A daemon may be live while degraded, but it may not advertise full readiness."""

from __future__ import annotations

import asyncio


def _health_route(app):
    return next(route for route in app.routes if getattr(route, "path", None) == "/health")


def test_386_full_readiness_requires_healthy_required_retrieval_channels(tmp_path, monkeypatch) -> None:
    """Warm embeddings alone cannot override a failed semantic/recall probe."""
    from superlocalmemory.server import unified_daemon

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(unified_daemon, "_embedding_warm", True)
    monkeypatch.setattr(
        "superlocalmemory.server.recall_health.get_recall_health",
        lambda: {
            "recall_healthy": False,
            "consecutive_failures": 3,
            "last_error": "semantic channel dead",
        },
    )
    app = unified_daemon.create_app()
    app.state.engine = object()
    app.state.canonical_remember_runtime = type(
        "ReadyWriter",
        (),
        {"ready": True},
    )()
    app.state.migration_result = {
        "applied": ["M018"], "skipped": [], "failed": [], "details": {},
    }

    payload = asyncio.run(_health_route(app).endpoint())

    assert payload["ready"] is False
    assert payload["state"] == "ready"
    assert payload["runtime_state"] == "serving_degraded"
    assert payload["readiness"]["recall_health"] is False
