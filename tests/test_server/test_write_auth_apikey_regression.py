# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Regression: a configured api_key file must not 401 already-authorized writes.

v3.7.6 (#71 / #73 / #74). Once ``<data_root>/api_key`` exists, the daemon used
to run a second, stricter gate (``check_api_key``) that only understood
``X-SLM-API-Key`` — 401ing write paths that the richer mutation-actor gate had
already authorized:

  * #71  MCP ``remember`` write-through, authenticated by the daemon capability.
  * #73/#74  the dashboard's install-token writes / config tests.

The mutation-actor gate remains the authoritative boundary: the daemon
capability, the install token, and a matching ``X-SLM-API-Key`` all authorize a
write, while a wrong/absent credential is still rejected.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from superlocalmemory.core.security_primitives import ensure_install_token
from superlocalmemory.infra.auth_middleware import API_KEY_FILE
from superlocalmemory.server.unified_daemon import create_app
from superlocalmemory.storage.migrations import M018_ingestion_operations

_API_KEY = "secret-key-123"


def _write_api_key_file() -> None:
    """Create the opt-in api_key file inside the test's isolated data root."""
    Path(API_KEY_FILE).write_text(_API_KEY + "\n", encoding="utf-8")


def _app(engine):
    with engine._db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
    app = create_app()
    app.state.engine = engine
    return app


def _body(key: str) -> dict:
    return {
        "content": (
            "An authenticated local reliability decision is recorded through "
            "the canonical ingestion command."
        ),
        "idempotency_key": key,
    }


def test_capability_write_through_succeeds_with_api_key_file(
    engine_with_mock_deps,
) -> None:
    """#71: a daemon-capability write must not be 401'd by the api_key gate."""
    _write_api_key_file()
    app = _app(engine_with_mock_deps)
    client = TestClient(app)
    resp = client.post(
        "/remember?wait=true",
        json=_body("capability-with-apikey-file"),
        headers={
            "X-SLM-Daemon-Capability": app.state.daemon_descriptor.capability,
            "X-SLM-Target-Instance": app.state.daemon_descriptor.instance_id,
            # Deliberately NO X-SLM-API-Key — the capability is the credential.
        },
    )
    assert resp.status_code == 200, resp.text


def test_install_token_write_succeeds_with_api_key_file(
    engine_with_mock_deps,
) -> None:
    """#73/#74: install-token dashboard writes must survive an api_key file."""
    _write_api_key_file()
    app = _app(engine_with_mock_deps)
    client = TestClient(app)
    resp = client.post(
        "/remember?wait=true",
        json=_body("install-token-with-apikey-file"),
        headers={"X-Install-Token": ensure_install_token()},
    )
    assert resp.status_code == 200, resp.text


def test_matching_api_key_write_succeeds(engine_with_mock_deps) -> None:
    """The api_key path itself still authorizes a write when it matches."""
    _write_api_key_file()
    app = _app(engine_with_mock_deps)
    client = TestClient(app)
    resp = client.post(
        "/remember?wait=true",
        json=_body("correct-apikey"),
        headers={"X-SLM-API-Key": _API_KEY},
    )
    assert resp.status_code == 200, resp.text


def test_wrong_api_key_credential_is_still_rejected(
    engine_with_mock_deps,
) -> None:
    """Security intact: presenting only a wrong credential fails closed."""
    _write_api_key_file()
    app = _app(engine_with_mock_deps)
    client = TestClient(app)
    resp = client.post(
        "/remember?wait=true",
        json=_body("wrong-apikey"),
        headers={"X-SLM-API-Key": "not-the-configured-key"},
    )
    assert resp.status_code == 403, resp.text
