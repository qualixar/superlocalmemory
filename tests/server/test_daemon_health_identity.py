# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The daemon health contract must expose verifiable process identity."""

from __future__ import annotations

import asyncio
import signal
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import HTTPException, Request

from superlocalmemory.infra.daemon_identity import (
    DAEMON_PROTOCOL,
    DAEMON_SERVICE,
    capability_fingerprint,
    namespace_id_for,
)


def test_health_preserves_compatibility_and_adds_owned_identity(
    tmp_path: Path, monkeypatch,
) -> None:
    from superlocalmemory.server import unified_daemon

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SLM_DAEMON_PORT", "43127")
    monkeypatch.setenv("SLM_DAEMON_INSTANCE_ID", "health-instance")
    monkeypatch.setenv("SLM_DAEMON_CAPABILITY", "health-capability")
    monkeypatch.setattr(unified_daemon, "_ACTIVE_DAEMON_DESCRIPTOR", None)

    app = unified_daemon.create_app()
    route = next(route for route in app.routes if getattr(route, "path", None) == "/health")
    payload = asyncio.run(route.endpoint())

    assert payload["status"] == "ok"
    assert payload["engine"] in {"initialized", "unavailable"}
    assert "embedding_warm" in payload
    assert "recall_health" in payload
    assert payload["service"] == DAEMON_SERVICE
    assert payload["daemon_protocol"] == DAEMON_PROTOCOL
    assert payload["namespace_id"] == namespace_id_for(tmp_path)
    assert payload["instance_id"] == "health-instance"
    assert payload["capability_fingerprint"] == capability_fingerprint(
        "health-capability"
    )
    assert payload["port"] == 43127


def _request(*, capability: str = "", instance_id: str = "") -> Request:
    headers = []
    if capability:
        headers.append((b"x-slm-daemon-capability", capability.encode()))
    if instance_id:
        headers.append((b"x-slm-target-instance", instance_id.encode()))
    return Request({"type": "http", "method": "POST", "path": "/stop", "headers": headers})


def test_stop_rejects_missing_or_wrong_process_capability(
    tmp_path: Path, monkeypatch,
) -> None:
    from superlocalmemory.server import unified_daemon

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SLM_DAEMON_PORT", "43131")
    monkeypatch.setenv("SLM_DAEMON_INSTANCE_ID", "stop-instance")
    monkeypatch.setenv("SLM_DAEMON_CAPABILITY", "stop-capability")
    monkeypatch.setattr(unified_daemon, "_ACTIVE_DAEMON_DESCRIPTOR", None)
    app = unified_daemon.create_app()
    route = next(route for route in app.routes if getattr(route, "path", None) == "/stop")

    with pytest.raises(HTTPException) as missing:
        asyncio.run(route.endpoint(_request()))
    assert missing.value.status_code == 403

    with pytest.raises(HTTPException) as wrong_instance:
        asyncio.run(route.endpoint(_request(
            capability="stop-capability", instance_id="replacement",
        )))
    assert wrong_instance.value.status_code == 409


def test_stop_accepts_only_the_owned_process_instance(
    tmp_path: Path, monkeypatch,
) -> None:
    from superlocalmemory.server import unified_daemon

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SLM_DAEMON_PORT", "43132")
    monkeypatch.setenv("SLM_DAEMON_INSTANCE_ID", "stop-instance")
    monkeypatch.setenv("SLM_DAEMON_CAPABILITY", "stop-capability")
    monkeypatch.setattr(unified_daemon, "_ACTIVE_DAEMON_DESCRIPTOR", None)
    app = unified_daemon.create_app()
    route = next(route for route in app.routes if getattr(route, "path", None) == "/stop")

    with (
        patch.object(unified_daemon._observe_buffer, "flush_sync"),
        patch("os.kill") as kill,
    ):
        payload = asyncio.run(route.endpoint(_request(
            capability="stop-capability", instance_id="stop-instance",
        )))

    assert payload == {"status": "stopping"}
    kill.assert_called_once_with(unified_daemon.os.getpid(), signal.SIGTERM)


def test_legacy_redirect_targets_the_actual_runtime_port() -> None:
    import inspect

    from superlocalmemory.server import unified_daemon

    source = inspect.getsource(unified_daemon.lifespan)
    assert "_start_legacy_redirect(identity.port, _LEGACY_PORT)" in source


def test_status_reports_the_actual_runtime_port(tmp_path: Path, monkeypatch) -> None:
    from superlocalmemory.server import unified_daemon

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SLM_DAEMON_PORT", "43134")
    monkeypatch.setenv("SLM_DAEMON_INSTANCE_ID", "status-instance")
    monkeypatch.setenv("SLM_DAEMON_CAPABILITY", "status-capability")
    monkeypatch.setattr(unified_daemon, "_ACTIVE_DAEMON_DESCRIPTOR", None)
    app = unified_daemon.create_app()
    route = next(
        route for route in app.routes if getattr(route, "path", None) == "/status"
    )

    payload = asyncio.run(route.endpoint())
    assert payload["port"] == 43134
