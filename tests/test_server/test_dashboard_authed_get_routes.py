# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
# Part of SuperLocalMemory V3

"""Regression: authenticated dashboard GET routes must trust the local owner.

ROOT CAUSE (audit cycle-3, 2026-07-22)
--------------------------------------
Cycle-2 hardening added an auth check to GET routes that are *not* covered
by the mutation middleware (they read/download sensitive data):

    GET /api/backup/list
    GET /api/compliance/audit
    GET /api/compliance/gdpr/export   (a.href download)
    GET /api/export                   (window.location download)

The check used ``require_write_actor``, which demands an explicit credential
header (daemon capability, X-Install-Token, or X-SLM-API-Key). But the
same-origin dashboard cannot satisfy it on these calls:

  * the global fetch patch (ui/js/core.js) attaches X-Install-Token only to
    *mutating* methods, so plain GET fetches (backup/list, compliance/audit)
    went out token-less;
  * the export routes are triggered by a top-level navigation
    (``window.location.href`` / ``a.href``), which cannot carry ANY custom
    header at all.

Result: the machine owner's own dashboard got 403 on its backup pane and every
export/download was permanently broken.

FIX
---
These routes use ``require_http_mutation_actor`` — the same loopback-trusted
boundary every dashboard mutation already uses. The local owner (loopback /
in-process test client) is trusted without a header; a non-loopback caller must
present a credential; an uncredentialed remote caller still fails closed. A
backup filename list is no more sensitive than the memory content already served
to the same loopback owner.

These tests assert the trusted local caller is NOT forbidden (403) on each
remaining GET route — the exact regression. They intentionally do not assert 200,
because module-availability early-returns (BACKUP/COMPLIANCE not importable in a
minimal env) legitimately short-circuit before the handler body; a 403 is the
only status that would signal the auth regression has returned.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from superlocalmemory.server.profile_runtime import bind_profile_runtime
from superlocalmemory.server.unified_daemon import create_app

# Every route below is reached by the dashboard WITHOUT a credential header.
_TRUSTED_GET_ROUTES = [
    "/api/backup/list",
    "/api/compliance/audit?limit=5",
    "/api/compliance/gdpr/export",
    "/api/export?format=json",
]


@pytest.fixture
def trusted_client(engine_with_mock_deps):
    """Daemon app bound to the seeded engine; TestClient == trusted loopback."""
    engine = engine_with_mock_deps
    engine.profile_id = "default"
    engine._config.active_profile = "default"
    engine._db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
        ("default", "default"),
    )
    if hasattr(engine._db, "commit"):
        engine._db.commit()

    app = create_app()
    app.state.engine = engine
    app.state.config = engine._config
    bind_profile_runtime(app.state, engine, engine._config)
    return TestClient(app)


@pytest.mark.parametrize("route", _TRUSTED_GET_ROUTES)
def test_dashboard_get_route_not_forbidden_for_local_owner(trusted_client, route):
    """The local owner's token-less GET must not 403 (cycle-2 regression)."""
    resp = trusted_client.get(route)
    assert resp.status_code != 403, (
        f"{route} returned 403 to the trusted local owner — a header-required "
        f"gate (require_write_actor) was re-introduced on a header-less "
        f"dashboard GET. Use require_http_mutation_actor instead."
    )
