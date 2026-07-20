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

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from superlocalmemory.core.security_primitives import ensure_install_token
from superlocalmemory.infra.auth_middleware import (
    API_KEY_FILE,
    SLM_REQUIRE_API_KEY_LOOPBACK_ENV,
)
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


@pytest.fixture(autouse=True)
def _clear_strict_loopback_env():
    """Isolate SLM_REQUIRE_API_KEY_LOOPBACK across tests in this module."""
    prior = os.environ.pop(SLM_REQUIRE_API_KEY_LOOPBACK_ENV, None)
    yield
    if prior is None:
        os.environ.pop(SLM_REQUIRE_API_KEY_LOOPBACK_ENV, None)
    else:
        os.environ[SLM_REQUIRE_API_KEY_LOOPBACK_ENV] = prior


def _ingest_body(dedup_key: str) -> dict:
    """/ingest goes through authenticated_request_actor, which trusts the
    middleware's request.state.authenticated_actor -- the uncredentialed
    loopback-trust branch of require_http_mutation_actor -- unlike /remember,
    which calls require_write_actor directly and always demands an explicit
    credential. /ingest is the correct probe for the loopback-trust boundary
    this env flag gates."""
    return {
        "content": (
            "An authenticated local reliability decision is recorded through "
            "the canonical ingestion adapter."
        ),
        "source_type": "test-adapter",
        "dedup_key": dedup_key,
    }


def test_loopback_write_with_no_credential_succeeds_by_default(
    engine_with_mock_deps,
) -> None:
    """F1 baseline: with the flag OFF (default), an api_key file does not
    force uncredentialed loopback writes to present a credential -- the
    v3.7.6 local-first fix (#71/#73/#74) stays the default behavior."""
    _write_api_key_file()
    app = _app(engine_with_mock_deps)
    client = TestClient(app)
    resp = client.post(
        "/ingest",
        json=_ingest_body("loopback-no-credential-flag-off"),
    )
    assert resp.status_code == 200, resp.text


def test_loopback_write_with_no_credential_rejected_when_strict_flag_set(
    engine_with_mock_deps,
) -> None:
    """F1/F4: SLM_REQUIRE_API_KEY_LOOPBACK=1 + an api_key file restores the
    strict shared-host posture -- an uncredentialed loopback write is
    rejected."""
    _write_api_key_file()
    os.environ[SLM_REQUIRE_API_KEY_LOOPBACK_ENV] = "1"
    app = _app(engine_with_mock_deps)
    client = TestClient(app)
    resp = client.post(
        "/ingest",
        json=_ingest_body("loopback-no-credential-flag-on"),
    )
    assert resp.status_code in (401, 403), resp.text


def test_loopback_write_with_matching_api_key_succeeds_under_strict_flag(
    engine_with_mock_deps,
) -> None:
    """F1/F4: the same strict-flag scenario succeeds once the caller presents
    a matching X-SLM-API-Key -- the flag adds a requirement, it does not
    remove the existing api_key escape hatch."""
    _write_api_key_file()
    os.environ[SLM_REQUIRE_API_KEY_LOOPBACK_ENV] = "1"
    app = _app(engine_with_mock_deps)
    client = TestClient(app)
    resp = client.post(
        "/ingest",
        json=_ingest_body("loopback-matching-key-flag-on"),
        headers={"X-SLM-API-Key": _API_KEY},
    )
    assert resp.status_code == 200, resp.text
