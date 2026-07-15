# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Shared HTTP write-identity boundary contracts."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException


class _Request:
    def __init__(self, headers: dict[str, str], client_host: str = "127.0.0.1"):
        self.headers = headers
        self.client = SimpleNamespace(host=client_host)


def test_exact_daemon_capability_derives_process_actor() -> None:
    from superlocalmemory.server.write_identity import require_daemon_actor

    descriptor = SimpleNamespace(
        capability="private-capability",
        instance_id="instance-1",
        capability_fingerprint="fingerprint-1",
    )
    request = _Request({
        "X-SLM-Daemon-Capability": "private-capability",
        "X-SLM-Target-Instance": "instance-1",
    })

    assert require_daemon_actor(request, descriptor) == (
        "daemon-capability:fingerprint-1"
    )


def test_install_token_derives_local_actor(monkeypatch) -> None:
    from superlocalmemory.server.write_identity import require_write_actor

    monkeypatch.setattr(
        "superlocalmemory.core.security_primitives.verify_install_token",
        lambda token: token == "install-token",
    )
    monkeypatch.setattr(
        "superlocalmemory.core.engine_ingestion.local_trusted_actor_id",
        lambda kind: f"trusted:{kind}",
    )

    actor = require_write_actor(
        _Request({"X-Install-Token": "install-token"}),
        descriptor=None,
        actor_kind="dashboard",
    )

    assert actor == "trusted:dashboard"


def test_missing_write_credential_fails_closed() -> None:
    from superlocalmemory.server.write_identity import require_write_actor

    with pytest.raises(HTTPException) as exc_info:
        require_write_actor(_Request({}), descriptor=None)

    assert exc_info.value.status_code == 403


def test_configured_api_key_derives_authenticated_api_actor(monkeypatch) -> None:
    from superlocalmemory.server.write_identity import require_write_actor

    monkeypatch.setattr(
        "superlocalmemory.infra.auth_middleware.verify_api_key",
        lambda token: token == "external-api-key",
    )

    actor = require_write_actor(
        _Request({"X-SLM-API-Key": "external-api-key"}),
        descriptor=None,
        actor_kind="http-api",
    )

    assert actor.startswith("api-key:http-api:")
    assert "external-api-key" not in actor


def test_local_http_mutation_gets_capability_derived_process_actor(
    monkeypatch,
) -> None:
    from superlocalmemory.server.write_identity import require_http_mutation_actor

    monkeypatch.setattr(
        "superlocalmemory.core.engine_ingestion.local_trusted_actor_id",
        lambda kind: f"trusted:{kind}",
    )
    actor = require_http_mutation_actor(
        _Request({}, client_host="127.0.0.1"),
        descriptor=None,
        actor_kind="http-route",
    )
    assert actor == "trusted:http-route"


def test_remote_http_mutation_without_credential_fails_closed() -> None:
    from superlocalmemory.server.write_identity import require_http_mutation_actor

    with pytest.raises(HTTPException) as exc_info:
        require_http_mutation_actor(
            _Request({}, client_host="192.0.2.44"),
            descriptor=None,
            actor_kind="http-route",
        )
    assert exc_info.value.status_code == 403
