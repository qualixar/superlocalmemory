# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""HTTP remember must route through the durable canonical ingestion command."""

from __future__ import annotations

from contextlib import contextmanager

from fastapi.testclient import TestClient

from superlocalmemory.server.unified_daemon import create_app
from superlocalmemory.storage.migrations import (
    M018_ingestion_operations,
    M032_write_coordinator_admission,
)


@contextmanager
def _client(engine):
    """Inject the daemon-owned writer; TestClient does not enter lifespan."""
    from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime

    with engine._db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
        M032_write_coordinator_admission.apply(conn)
    app = create_app()
    app.state.engine = engine
    runtime = CanonicalRememberRuntime.for_engine(engine)
    runtime.start()
    app.state.canonical_remember_runtime = runtime
    client = TestClient(app)
    client.headers["X-SLM-Daemon-Capability"] = (
        app.state.daemon_descriptor.capability
    )
    client.headers["X-SLM-Target-Instance"] = (
        app.state.daemon_descriptor.instance_id
    )
    try:
        yield client
    finally:
        runtime.stop()


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

    with _client(engine_with_mock_deps) as client:
        client.headers.pop("X-SLM-Daemon-Capability")
        client.headers.pop("X-SLM-Target-Instance")
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
    body = {
        "content": (
            "Alice owns the incident review process and publishes every "
            "corrective action to the platform team."
        ),
        "idempotency_key": "http-session-4:turn-9",
        "metadata": {"agent_id": "caller-selected-admin"},
    }
    with _client(engine_with_mock_deps) as client:
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
    with _client(engine_with_mock_deps) as client:
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
    assert payload["materialization_state"] == "queryable"
    assert payload["fact_ids"]
    assert payload["wait_ignored"] is True
    operation = engine_with_mock_deps._db.execute(
        "SELECT state, session_id FROM ingestion_operations "
        "WHERE operation_id=?",
        (payload["operation_id"],),
    )
    assert dict(operation[0]) == {
        "state": "queryable",
        "session_id": "session-sync",
    }


def test_wait_remember_never_runs_inline_materialization(engine_with_mock_deps) -> None:
    """The compatibility query parameter cannot make a model call inline."""
    with _client(engine_with_mock_deps) as client:
        response = client.post(
            "/remember?wait=true",
            json={
                "content": "A slow enrichment stays outside the request transaction.",
                "idempotency_key": "bounded-wait-route-1",
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "queryable"
    assert payload["materialization_state"] == "queryable"
    assert payload["wait_ignored"] is True


def test_trust_rejection_occurs_before_journal_or_canonical_write(
    engine_with_mock_deps,
    monkeypatch,
) -> None:
    """A denied actor leaves neither replay work nor memory evidence behind."""
    def reject(_operation, _payload) -> None:
        raise PermissionError("trust policy rejected this actor")

    monkeypatch.setattr(engine_with_mock_deps._hooks, "run_pre", reject)
    with _client(engine_with_mock_deps) as client:
        runtime = client.app.state.canonical_remember_runtime
        response = client.post(
            "/remember",
            json={
                "content": "Trust policy denial must not create durable work.",
                "idempotency_key": "trust-rejected-route-1",
            },
        )
        assert runtime.journal.count() == 0

    assert response.status_code == 403
    assert engine_with_mock_deps._db.execute("SELECT * FROM ingestion_operations") == []
    assert engine_with_mock_deps._db.execute("SELECT * FROM atomic_facts") == []


def test_deterministic_rejection_leaves_no_journal_or_canonical_write(
    engine_with_mock_deps,
) -> None:
    """Low-information input is rejected before durable admission preparation."""
    with _client(engine_with_mock_deps) as client:
        runtime = client.app.state.canonical_remember_runtime
        response = client.post(
            "/remember",
            json={"content": "x", "idempotency_key": "low-quality-route-1"},
        )
        assert runtime.journal.count() == 0

    assert response.status_code == 422
    assert engine_with_mock_deps._db.execute("SELECT * FROM ingestion_operations") == []
    assert engine_with_mock_deps._db.execute("SELECT * FROM atomic_facts") == []


def test_dashboard_delete_and_update_use_canonical_mutation_receipts(
    engine_with_mock_deps,
) -> None:
    """Dashboard mutations share the writer and honor an HTTP retry key."""
    with _client(engine_with_mock_deps) as client:
        stored = client.post(
            "/remember",
            json={
                "content": "Dashboard mutation receipts must remain durable.",
                "idempotency_key": "dashboard-mutation-source",
            },
        )
        fact_id = stored.json()["fact_ids"][0]
        update = client.patch(
            f"/api/memories/{fact_id}",
            json={"content": "Dashboard mutation receipts remain durable after edit."},
            headers={"X-Idempotency-Key": "dashboard-update-retry"},
        )
        first_delete = client.delete(
            f"/api/memories/{fact_id}",
            headers={"X-Idempotency-Key": "dashboard-delete-retry"},
        )
        second_delete = client.delete(
            f"/api/memories/{fact_id}",
            headers={"X-Idempotency-Key": "dashboard-delete-retry"},
        )

    assert update.status_code == 200, update.text
    assert first_delete.status_code == 200, first_delete.text
    assert second_delete.status_code == 200, second_delete.text
    assert engine_with_mock_deps._db.execute(
        "SELECT fact_id FROM atomic_facts WHERE fact_id = ?", (fact_id,)
    ) == []
    kinds = engine_with_mock_deps._db.execute(
        "SELECT command_kind FROM write_commits "
        "WHERE command_kind IN (?, ?) ORDER BY command_kind",
        ("update_fact", "delete_fact"),
    )
    assert [row["command_kind"] for row in kinds] == ["delete_fact", "update_fact"]


def test_dashboard_archive_merge_and_scope_use_canonical_mutation_commands(
    engine_with_mock_deps,
) -> None:
    """Bounded dashboard lifecycle mutations never open a route-owned writer."""
    with _client(engine_with_mock_deps) as client:
        first = client.post(
            "/remember",
            json={
                "content": "The merge loser is isolated to the default profile.",
                "idempotency_key": "dashboard-merge-first",
            },
        ).json()["fact_ids"][0]
        kept = client.post(
            "/remember",
            json={
                "content": "The merge winner remains isolated to the default profile.",
                "idempotency_key": "dashboard-merge-second",
            },
        ).json()["fact_ids"][0]
        scoped = client.patch(
            f"/api/memories/{kept}/scope",
            json={"scope": "shared", "shared_with": ["team-alpha"]},
        )
        merged = client.post(f"/api/memories/{first}/merge", json={"into": kept})
        archived = client.post(f"/api/memories/{kept}/forget")

    assert scoped.status_code == 200, scoped.text
    assert merged.status_code == 200, merged.text
    assert archived.status_code == 200, archived.text
    state = engine_with_mock_deps._db.execute(
        "SELECT scope, shared_with, archive_status FROM atomic_facts WHERE fact_id = ?",
        (kept,),
    )
    assert state[0]["scope"] == "shared"
    assert state[0]["archive_status"] == "archived"
    kinds = engine_with_mock_deps._db.execute(
        "SELECT command_kind FROM write_commits "
        "WHERE command_kind IN (?, ?, ?) ORDER BY command_kind",
        ("archive_fact", "merge_fact", "set_fact_scope"),
    )
    assert [row["command_kind"] for row in kinds] == [
        "archive_fact",
        "merge_fact",
        "set_fact_scope",
    ]


def test_dashboard_mutation_rejects_invalid_or_drifted_idempotency_key(
    engine_with_mock_deps,
) -> None:
    """The retry boundary is bounded and never replays a different payload."""
    with _client(engine_with_mock_deps) as client:
        fact_id = client.post(
            "/remember",
            json={
                "content": "Dashboard mutation conflicts are explicit.",
                "idempotency_key": "dashboard-conflict-source",
            },
        ).json()["fact_ids"][0]
        invalid = client.patch(
            f"/api/memories/{fact_id}",
            json={"content": "Invalid key is rejected."},
            headers={"X-Idempotency-Key": "contains spaces"},
        )
        first = client.patch(
            f"/api/memories/{fact_id}",
            json={"content": "The first dashboard mutation wins."},
            headers={"X-Idempotency-Key": "dashboard-drift-key"},
        )
        conflict = client.patch(
            f"/api/memories/{fact_id}",
            json={"content": "The retry payload is not silently accepted."},
            headers={"X-Idempotency-Key": "dashboard-drift-key"},
        )

    assert invalid.status_code == 422
    assert first.status_code == 200
    assert conflict.status_code == 409
