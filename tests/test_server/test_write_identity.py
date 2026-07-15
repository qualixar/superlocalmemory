# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Shared HTTP write-identity boundary contracts."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException


class _Request:
    def __init__(self, headers: dict[str, str]):
        self.headers = headers


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
