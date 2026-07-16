# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""HTTP remember must route through the durable canonical ingestion command."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from superlocalmemory.core.ingestion_command import IngestionState
from superlocalmemory.server.unified_daemon import create_app
from superlocalmemory.storage.migrations import M018_ingestion_operations


def _client(engine) -> TestClient:
    with engine._db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
    app = create_app()
    app.state.engine = engine
    client = TestClient(app)
    client.headers["X-SLM-Daemon-Capability"] = (
        app.state.daemon_descriptor.capability
    )
    client.headers["X-SLM-Target-Instance"] = (
        app.state.daemon_descriptor.instance_id
    )
    return client


def test_remember_rejects_missing_or_wrong_daemon_capability(
    engine_with_mock_deps,
) -> None:
    """A caller cannot borrow the daemon's trusted actor identity."""
    with engine_with_mock_deps._db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
    app = create_app()
    app.state.engine = engine_with_mock_deps
    client = TestClient(app)
    body = {
        "content": (
            "Mallory claims the daemon identity without presenting the "
            "private local capability."
        ),
        "idempotency_key": "untrusted-caller-1",
    }

    missing = client.post("/remember", json=body)
    wrong = client.post(
        "/remember",
        json=body,
        headers={"X-SLM-Daemon-Capability": "caller-selected-admin"},
    )

    assert missing.status_code == 403
    assert wrong.status_code == 403
    assert engine_with_mock_deps._db.execute(
        "SELECT * FROM ingestion_operations"
    ) == []


def test_dashboard_remember_accepts_verified_install_token(
    engine_with_mock_deps,
) -> None:
    from superlocalmemory.core.security_primitives import ensure_install_token

    with engine_with_mock_deps._db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
    app = create_app()
    app.state.engine = engine_with_mock_deps
    client = TestClient(app)

    response = client.post(
        "/remember?wait=true",
        json={
            "content": (
                "The dashboard records an authenticated local reliability "
                "decision through the canonical ingestion command."
            ),
            "idempotency_key": "dashboard-install-token-1",
        },
        headers={"X-Install-Token": ensure_install_token()},
    )

    assert response.status_code == 200, response.text
    operation = dict(engine_with_mock_deps._db.execute(
        "SELECT trusted_actor_id FROM ingestion_operations"
    )[0])
    assert operation["trusted_actor_id"].startswith(
        "local-capability:dashboard:"
    )


def test_async_remember_returns_durable_operation_and_is_idempotent(
    engine_with_mock_deps,
) -> None:
    client = _client(engine_with_mock_deps)
    body = {
        "content": (
            "Alice owns the incident review process and publishes every "
            "corrective action to the platform team."
        ),
        "idempotency_key": "http-session-4:turn-9",
        "metadata": {"agent_id": "caller-selected-admin"},
    }

    first = client.post("/remember", json=body)
    second = client.post("/remember", json=body)

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    first_payload = first.json()
    second_payload = second.json()
    assert first_payload["operation_id"] == second_payload["operation_id"]
    assert first_payload["materialization_state"] == "queryable"
    assert first_payload["pending_id"] == first_payload["operation_id"]

    operations = engine_with_mock_deps._db.execute(
        "SELECT * FROM ingestion_operations"
    )
    assert len(operations) == 1
    operation = dict(operations[0])
    assert operation["trusted_actor_id"].startswith("daemon-capability:")
    assert operation["trusted_actor_id"] != "caller-selected-admin"
    assert len(engine_with_mock_deps._db.execute("SELECT * FROM memories")) == 1


def test_wait_remember_completes_same_canonical_operation(
    engine_with_mock_deps,
) -> None:
    client = _client(engine_with_mock_deps)

    response = client.post(
        "/remember?wait=true",
        json={
            "content": (
                "Bob leads the database reliability review and records the "
                "approved recovery decision for every production incident."
            ),
            "idempotency_key": "http-sync-1",
            "session_id": "session-sync",
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["materialization_state"] == "complete"
    assert payload["fact_ids"]
    operation = engine_with_mock_deps._db.execute(
        "SELECT state, session_id FROM ingestion_operations "
        "WHERE operation_id=?",
        (payload["operation_id"],),
    )
    assert dict(operation[0]) == {
        "state": "complete",
        "session_id": "session-sync",
    }


def test_wait_remember_keeps_durable_fact_queryable_when_enrichment_retries(
    engine_with_mock_deps,
) -> None:
    """A cold optional embedding path must not turn an admitted fact into 500."""
    client = _client(engine_with_mock_deps)
    receipt = SimpleNamespace(
        operation_id="cold-embedding-op",
        state=IngestionState.QUERYABLE,
        fact_ids=("fact-queryable",),
    )
    deferred = SimpleNamespace(
        operation_id="cold-embedding-op",
        state=IngestionState.FAILED,
        fact_ids=("fact-queryable",),
        last_error="incomplete derivation stages: embeddings",
    )
    command = SimpleNamespace(
        submit=lambda _request: receipt,
        materialize=lambda _operation_id: deferred,
    )

    with patch(
        "superlocalmemory.core.engine_ingestion.build_engine_ingestion_command",
        return_value=command,
    ):
        response = client.post(
            "/remember?wait=true",
            json={
                "content": "A durable fact remains available while local embeddings warm.",
                "idempotency_key": "cold-embedding-route-1",
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "queryable"
    assert payload["materialization_state"] == "failed"
    assert payload["fact_ids"] == ["fact-queryable"]
    assert "retry" in payload["note"]
