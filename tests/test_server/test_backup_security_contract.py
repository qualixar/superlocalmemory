"""Security and truthfulness contracts for the backup control plane."""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from superlocalmemory.server.routes import backup


def test_blocking_backup_handlers_run_in_fastapi_threadpool() -> None:
    handlers = (
        backup.backup_status,
        backup.backup_create,
        backup.backup_configure,
        backup.backup_list,
        backup.list_destinations,
        backup.connect_github_route,
        backup.configure_gdrive_client,
        backup.connect_gdrive_route,
        backup.disconnect_destination,
        backup.sync_cloud,
        backup.export_backup,
        backup.google_oauth_start,
        backup.google_oauth_callback,
        backup.github_oauth_start,
        backup.github_oauth_callback,
    )
    assert not any(inspect.iscoroutinefunction(handler) for handler in handlers)


def _request(
    *,
    client: str = "127.0.0.1",
    user_agent: str = "slm-test",
    headers: list[tuple[bytes, bytes]] | None = None,
) -> Request:
    request_headers = [(b"user-agent", user_agent.encode())]
    request_headers.extend(headers or [])
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": request_headers,
            "client": (client, 12345),
            "server": ("localhost", 8765),
            "scheme": "http",
            "app": object(),
        }
    )


def test_oauth_state_is_random_bound_one_time_and_replay_safe() -> None:
    request = _request()
    first = backup._issue_oauth_state("google", request)
    second = backup._issue_oauth_state("google", request)

    assert first != second
    assert len(first) >= 32
    assert backup._consume_oauth_state(first, "google", request) is True
    assert backup._consume_oauth_state(first, "google", request) is False


def test_oauth_state_rejects_missing_hostile_provider_and_context() -> None:
    request = _request()
    assert backup._consume_oauth_state("", "google", request) is False
    assert backup._consume_oauth_state("attacker-controlled", "google", request) is False

    provider_state = backup._issue_oauth_state("google", request)
    assert backup._consume_oauth_state(provider_state, "github", request) is False

    context_state = backup._issue_oauth_state("github", request)
    assert (
        backup._consume_oauth_state(
            context_state,
            "github",
            _request(client="127.0.0.2"),
        )
        is False
    )


def test_oauth_state_rejects_expired_state() -> None:
    request = _request()
    state = backup._issue_oauth_state("google", request)
    with backup._OAUTH_STATE_LOCK:
        backup._OAUTH_STATES[state]["expires_at"] = 0
    assert backup._consume_oauth_state(state, "google", request) is False


def test_oauth_callbacks_reject_missing_and_replayed_state_before_exchange() -> None:
    request = _request()
    missing = backup.google_oauth_callback(
        request, code="hostile-code", state="",
    )
    assert missing.status_code == 400

    state = backup._issue_oauth_state("github", request)
    assert backup._consume_oauth_state(state, "github", request) is True
    replay = backup.github_oauth_callback(
        request, code="replayed-code", state=state,
    )
    assert replay.status_code == 400


def test_remote_owner_cannot_initiate_oauth_without_user_session(monkeypatch) -> None:
    monkeypatch.setattr(
        backup,
        "_require_manage",
        lambda request: {"kind": "owner"},
    )
    with pytest.raises(HTTPException) as exc:
        backup._require_oauth_start(_request(client="192.0.2.10"))
    assert exc.value.status_code == 403


def test_cross_site_oauth_start_is_rejected_before_manage(monkeypatch) -> None:
    managed: list[bool] = []
    monkeypatch.setattr(
        backup,
        "_require_manage",
        lambda request: managed.append(True) or {"kind": "owner"},
    )
    request = _request(headers=[(b"sec-fetch-site", b"cross-site")])
    with pytest.raises(HTTPException) as exc:
        backup._require_oauth_start(request)
    assert exc.value.status_code == 403
    assert managed == []


def test_other_local_port_cannot_start_oauth_as_personal_owner(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        backup,
        "_require_manage",
        lambda request: {"kind": "owner"},
    )
    request = _request(
        headers=[
            (b"origin", b"http://localhost:8417"),
            (b"sec-fetch-site", b"same-site"),
        ],
    )
    with pytest.raises(HTTPException) as exc:
        backup._require_oauth_start(request)
    assert exc.value.status_code == 403


def test_exact_daemon_origin_can_start_oauth_as_personal_owner(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        backup,
        "_require_manage",
        lambda request: {"kind": "owner"},
    )
    request = _request(
        headers=[
            (b"origin", b"http://localhost:8765"),
            (b"sec-fetch-site", b"same-origin"),
        ],
    )
    backup._require_oauth_start(request)


def test_cookie_authenticated_admin_cannot_start_oauth_from_other_local_port(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        backup,
        "_require_manage",
        lambda request: {"kind": "user", "user_id": "admin"},
    )
    request = _request(
        headers=[
            (b"origin", b"http://localhost:8417"),
            (b"sec-fetch-site", b"same-site"),
        ],
    )
    with pytest.raises(HTTPException) as exc:
        backup._require_oauth_start(request)
    assert exc.value.status_code == 403


def test_explicit_header_does_not_bypass_oauth_origin_boundary(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        backup,
        "_require_manage",
        lambda request: {"kind": "user", "user_id": "admin"},
    )
    request = _request(
        headers=[
            (b"origin", b"http://localhost:8417"),
            (b"sec-fetch-site", b"same-site"),
            (b"x-slm-user-session", b"explicit-session-token"),
        ],
    )
    with pytest.raises(HTTPException) as exc:
        backup._require_oauth_start(request)
    assert exc.value.status_code == 403


def test_status_and_destinations_call_read_guard(monkeypatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(backup, "_require_read", lambda request: seen.append("read"))
    monkeypatch.setattr(backup, "BACKUP_AVAILABLE", False)
    monkeypatch.setattr(backup, "CLOUD_AVAILABLE", False)
    request = _request()

    backup.backup_status(request)
    backup.list_destinations(request)

    assert seen == ["read", "read"]


def test_mutations_call_manage_guard(monkeypatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(backup, "_require_manage", lambda request: seen.append("manage"))
    monkeypatch.setattr(backup, "BACKUP_AVAILABLE", False)
    monkeypatch.setattr(backup, "CLOUD_AVAILABLE", False)
    request = _request()

    backup.backup_create(request)
    backup.backup_configure(
        request,
        backup.BackupConfigRequest(enabled=False),
    )
    for operation in (
        lambda: backup.connect_github_route(
            request, backup.GitHubConnectRequest(pat="test-token"),
        ),
        lambda: backup.configure_gdrive_client(
            request,
            backup.GDriveClientConfig(client_id="client", client_secret="secret"),
        ),
        lambda: backup.connect_gdrive_route(
            request, backup.GDriveConnectRequest(auth_code="test-code"),
        ),
        lambda: backup.disconnect_destination(request, "missing"),
        lambda: backup.sync_cloud(request),
    ):
        with pytest.raises(HTTPException) as exc:
            operation()
        assert exc.value.status_code == 501

    assert seen == ["manage"] * 7


def test_export_is_post_only_and_removes_transport_archive(
    monkeypatch,
    tmp_path,
) -> None:
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    snapshot = backup_dir / "snapshot.db"
    snapshot.write_bytes(b"sqlite-snapshot")

    class FakeManager:
        def create_backup(self, label: str):
            assert label == "export"
            return snapshot.name

    monkeypatch.setattr(backup, "_require_manage", lambda request: {"kind": "owner"})
    monkeypatch.setattr(backup, "_get_backup_manager", FakeManager)
    monkeypatch.setattr(backup, "MEMORY_DIR", tmp_path)
    monkeypatch.setattr(backup, "BACKUP_AVAILABLE", True)

    route = next(
        item for item in backup.router.routes
        if getattr(item, "path", "") == "/api/backup/export"
    )
    assert route.methods == {"POST"}

    response = backup.export_backup(_request())
    transport = response.path
    assert transport.endswith(".db.gz")
    assert snapshot.exists()
    assert Path(transport).exists()
    asyncio.run(response.background())
    assert not Path(transport).exists()
