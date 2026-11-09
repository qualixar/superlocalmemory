# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Dashboard routes for cloud backup encryption.

Status is key-free and needs READ. Showing the recovery key needs MANAGE and
is a POST, so it is never prefetched, cached by URL, or left in history.
"""

# ruff: noqa: F811 (pytest fixtures are imported and redeclared as parameters)

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from superlocalmemory.infra import backup_keys as bk
from superlocalmemory.server.routes import backup_encryption as routes

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "test_infra"))
from _backup_key_env import key_env  # noqa: E402,F401

_SRC = Path(__file__).resolve().parents[2] / "src" / "superlocalmemory"


def _request() -> Request:
    return Request({
        "type": "http", "method": "POST", "path": "/", "headers": [],
        "client": ("127.0.0.1", 1), "server": ("localhost", 8765),
        "scheme": "http", "app": object(),
    })


@pytest.fixture
def guards(monkeypatch) -> list[str]:
    seen: list[str] = []
    monkeypatch.setattr(routes, "_require_read", lambda request: seen.append("read"))
    monkeypatch.setattr(routes, "_require_manage", lambda request: seen.append("manage"))
    return seen


def test_handlers_are_sync_for_the_threadpool() -> None:
    for handler in (routes.encryption_status_route, routes.show_recovery_key_route,
                    routes.acknowledge_notice_route):
        assert not inspect.iscoroutinefunction(handler)


def test_status_needs_read_and_never_contains_the_key(key_env, guards) -> None:
    key, _ = bk.ensure_backup_key(origin="backup", legacy_uploads=True)
    status = routes.encryption_status_route(_request())
    assert guards == ["read"]
    assert status["recovery_key_pending"] is True
    assert bk.format_recovery_key(key) not in json.dumps(status)


def test_showing_the_key_needs_manage(key_env, guards) -> None:
    key, _ = bk.ensure_backup_key(origin="backup")
    shown = routes.show_recovery_key_route(_request())
    assert guards == ["manage"]
    assert bk.parse_recovery_key(shown["recovery_key"]) == key
    assert bk.encryption_status()["recovery_key_pending"] is False


def test_showing_the_key_when_there_is_none_is_a_clear_404(key_env, guards) -> None:
    with pytest.raises(HTTPException) as exc:
        routes.show_recovery_key_route(_request())
    assert exc.value.status_code == 404
    assert "No backup encryption key" in exc.value.detail


def test_acknowledging_the_legacy_notice_needs_manage(key_env, guards) -> None:
    bk.ensure_backup_key(origin="backup", legacy_uploads=True)
    routes.acknowledge_notice_route(_request())
    assert guards == ["manage"]
    assert bk.encryption_status()["legacy_plaintext_uploads"] is False


def test_recovery_key_route_is_post_only() -> None:
    methods = {
        route.path: route.methods for route in routes.router.routes  # type: ignore[attr-defined]
    }
    assert methods["/api/backup/recovery-key"] == {"POST"}
    assert methods["/api/backup/encryption"] == {"GET"}


@pytest.mark.parametrize("app_module", ["server/unified_daemon.py", "server/api.py", "server/ui.py"])
def test_every_dashboard_app_mounts_the_router(app_module: str) -> None:
    source = (_SRC / app_module).read_text(encoding="utf-8")
    assert "routes.backup_encryption import router" in source
