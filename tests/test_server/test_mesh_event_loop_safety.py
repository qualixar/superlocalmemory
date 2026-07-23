# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Regression contract for SQLite-backed Mesh routes."""

from __future__ import annotations

import inspect

from superlocalmemory.server.routes import mesh


def test_sqlite_backed_mesh_routes_run_in_fastapi_threadpool() -> None:
    """SQLite/network busy waits must never block the daemon event loop."""
    handlers = (
        mesh.register,
        mesh.deregister,
        mesh.peers,
        mesh.heartbeat,
        mesh.summary,
        mesh.send,
        mesh.inbox,
        mesh.mark_read,
        mesh.pending,
        mesh.state_all,
        mesh.state_set,
        mesh.state_get,
        mesh.lock,
        mesh.events,
        mesh.status,
    )
    assert all(not inspect.iscoroutinefunction(handler) for handler in handlers)
