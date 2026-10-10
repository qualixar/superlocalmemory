# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""HTTP remember must route through the durable canonical ingestion command."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager

from fastapi.testclient import TestClient

from superlocalmemory.server.unified_daemon import create_app
from superlocalmemory.storage.migrations import (
    M018_ingestion_operations,
    M032_write_coordinator_admission,
    M033_projection_transactions,
    M034_obligation_integrity,
    M042_correction_case_ledger,
)


@contextmanager
def _client(engine):
    """Inject the daemon-owned writer; TestClient does not enter lifespan."""
    from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime

    with engine._db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
        M032_write_coordinator_admission.apply(conn)
        M033_projection_transactions.apply(conn)
        M034_obligation_integrity.apply(conn)
        M042_correction_case_ledger.apply(conn)
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
    assert payload["wait_ignored"] is False, (
        "wait now buys a larger best-effort enrichment budget; it is no "
        "longer discarded"
    )
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
    assert payload["wait_ignored"] is False, (
        "wait now buys a larger best-effort enrichment budget; it is no "
        "longer discarded"
    )


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
        deletable = client.post(
            "/remember",
            json={
                "content": "A distinct dashboard delete witness remains removable.",
                "idempotency_key": "dashboard-delete-source",
            },
        ).json()["fact_ids"][0]
        update = client.patch(
            f"/api/memories/{fact_id}",
            json={"content": "Dashboard mutation receipts remain durable after edit."},
            headers={"X-Idempotency-Key": "dashboard-update-retry"},
        )
        first_delete = client.delete(
            f"/api/memories/{deletable}",
            headers={"X-Idempotency-Key": "dashboard-delete-retry"},
        )
        second_delete = client.delete(
            f"/api/memories/{deletable}",
            headers={"X-Idempotency-Key": "dashboard-delete-retry"},
        )

    assert update.status_code == 202, update.text
    assert first_delete.status_code == 200, first_delete.text
    assert second_delete.status_code == 200, second_delete.text
    assert engine_with_mock_deps._db.execute(
        "SELECT fact_id FROM atomic_facts WHERE fact_id = ?", (deletable,)
    ) == []
    kinds = engine_with_mock_deps._db.execute(
        "SELECT command_kind FROM write_commits "
        "WHERE command_kind IN (?, ?) ORDER BY command_kind",
        ("propose_correction", "delete_fact"),
    )
    assert [row["command_kind"] for row in kinds] == ["delete_fact", "propose_correction"]


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
    assert first.status_code == 202
    assert conflict.status_code == 409


def test_authenticated_http_correction_lifecycle_is_review_gated_and_immutable(
    engine_with_mock_deps,
) -> None:
    """HTTP proposes first; an authenticated reviewer alone applies truth."""
    with _client(engine_with_mock_deps) as client:
        predecessor = client.post(
            "/remember",
            json={
                "content": "The original synthetic release decision remains traceable.",
                "idempotency_key": "http-correction-source",
            },
        ).json()["fact_ids"][0]
        proposed = client.patch(
            f"/api/memories/{predecessor}",
            json={"content": "The reviewed synthetic release decision supersedes the prior one."},
            headers={"X-Idempotency-Key": "http-correction-propose"},
        )
        assert proposed.status_code == 202, proposed.text
        proposal = proposed.json()
        case = proposal["correction_case"]
        successor = proposal["successor_fact_id"]

        listed = client.get("/api/corrections")
        fetched = client.get(f"/api/corrections/{case['case_id']}")
        protected_delete = client.delete(f"/api/memories/{predecessor}")
        applied = client.post(
            f"/api/corrections/{case['case_id']}/apply",
            json={"expected_version": case["version"]},
            headers={"X-Idempotency-Key": "http-correction-apply"},
        )
        protected_successor_delete = client.delete(f"/api/memories/{successor}")

    assert listed.status_code == 200, listed.text
    assert fetched.status_code == 200, fetched.text
    assert protected_delete.status_code == 409, protected_delete.text
    assert applied.status_code == 200, applied.text
    assert protected_successor_delete.status_code == 409, protected_successor_delete.text
    assert listed.json()["corrections"][0]["status"] == "proposed"
    assert "content" not in fetched.json()["correction"]
    assert applied.json()["correction_case"]["status"] == "applied"
    assert engine_with_mock_deps._db.get_fact(predecessor).content == (
        "The original synthetic release decision remains traceable."
    )
    assert engine_with_mock_deps._db.get_fact(successor).content == (
        "The reviewed synthetic release decision supersedes the prior one."
    )
    assert engine_with_mock_deps._db.get_invalidated_fact_ids(
        [predecessor], "default"
    ) == {predecessor}
    assert engine_with_mock_deps._db.get_nonapplied_correction_successor_ids(
        [successor], "default"
    ) == set()


def test_authenticated_mcp_correction_lifecycle_uses_the_same_resident_daemon(
    engine_with_mock_deps,
    monkeypatch,
) -> None:
    """MCP is a transport adapter, never a second correction writer."""
    from superlocalmemory.mcp.tools_core import register_core_tools

    class _McpServer:
        def __init__(self) -> None:
            self.tools: dict[str, object] = {}

        def tool(self, *args, **kwargs):
            def register(function):
                self.tools[function.__name__] = function
                return function

            return register

    with _client(engine_with_mock_deps) as client:
        predecessor = client.post(
            "/remember",
            json={
                "content": "The MCP lifecycle source remains independently traceable.",
                "idempotency_key": "mcp-correction-source",
            },
        ).json()["fact_ids"][0]

        def daemon_request(method: str, path: str, payload=None, **_flags):
            response = client.request(method, path, json=payload)
            content_type = response.headers.get("content-type", "")
            return response.json() if content_type.startswith("application/json") else None

        monkeypatch.setattr("superlocalmemory.cli.daemon.is_daemon_running", lambda: True)
        monkeypatch.setattr("superlocalmemory.cli.daemon.daemon_request", daemon_request)
        server = _McpServer()
        register_core_tools(server, lambda: engine_with_mock_deps)

        proposed = asyncio.run(
            server.tools["update_memory"](
                predecessor,
                "The MCP lifecycle successor is review-gated before it becomes current.",
                "codex",
            )
        )
        case_id = proposed["correction_case"]["case_id"]
        listed = asyncio.run(server.tools["list_corrections"]())
        applied = asyncio.run(server.tools["review_correction"](case_id, "apply", 0))

    assert proposed["success"] is True
    assert proposed["review_required"] is True
    assert listed["success"] is True
    assert [item["case_id"] for item in listed["corrections"]] == [case_id]
    assert applied["success"] is True
    assert applied["correction_case"]["status"] == "applied"


def test_remember_under_a_held_write_lock_is_202_accepted_then_saved_once(
    engine_with_mock_deps,
) -> None:
    """A busy writer answers 202 'accepted, not yet searchable' — never 503.

    The write lock is held by a second connection (what a long enrichment or
    maintenance transaction does under load), so the canonical commit cannot
    land inside the caller deadline. The memory must still be saved exactly
    once, and the answer must not claim it is searchable before it is.
    """
    import sqlite3
    import time

    body = {
        "content": (
            "Priya rotates the harbour pilot schedule every second Tuesday "
            "while the writer is under load."
        ),
        "idempotency_key": "held-lock-route-1",
    }
    db_path = engine_with_mock_deps._db.db_path
    with _client(engine_with_mock_deps) as client:
        runtime = client.app.state.canonical_remember_runtime
        # Time the writer call the route makes, not the in-process client round
        # trip: the client adds thread hand-offs whose cost depends on how busy
        # the test machine is (a full suite beside other runs measured 1.59 s
        # end to end for a 1.2 s admission wait). The route's own budget is
        # what the 1.5 s contract is about.
        writer_calls: list[float] = []
        real_remember = runtime.remember

        def timed_remember(*args, **kwargs):
            started_call = time.monotonic()
            try:
                return real_remember(*args, **kwargs)
            finally:
                writer_calls.append(time.monotonic() - started_call)

        runtime.remember = timed_remember
        holder = sqlite3.connect(str(db_path), timeout=5, isolation_level=None)
        holder.execute("BEGIN IMMEDIATE")
        try:
            started = time.monotonic()
            accepted = client.post("/remember", json=body)
            elapsed = time.monotonic() - started
        finally:
            holder.execute("ROLLBACK")
            holder.close()
        assert runtime.wait_for_deferred(timeout=15.0)
        final = client.post("/remember", json=body)
        runtime.remember = real_remember

    assert accepted.status_code == 202, accepted.text
    payload = accepted.json()
    assert payload["ok"] is True
    assert payload["status"] == "accepted"
    assert payload["durable"] is True
    assert payload["queryable"] is False
    assert payload["fact_ids"] == []
    assert payload["idempotency_key"] == "held-lock-route-1"
    # The 1.5 s remember ceiling is a property of the route's own constants:
    # it answers "accepted" when the admission wait runs out, and that wait
    # is set inside the ceiling with room for the response. Pinned exactly.
    from superlocalmemory.server import unified_daemon as daemon

    admission_s = daemon._REMEMBER_ADMISSION_DEADLINE_MS / 1000.0
    journal_s = daemon._REMEMBER_JOURNAL_DEADLINE_MS / 1000.0
    ceiling_s = daemon._REMEMBER_TOTAL_CEILING_SECONDS
    assert admission_s < ceiling_s < journal_s
    # The product's own ceiling (remember <= 1.5 s) is the bound, not the
    # longer journal deadline: a route that sat out the journal deadline
    # instead of the admission timer takes at least journal_s, so this still
    # catches that regression. Loosening the assertion to journal_s would
    # silently accept an acknowledgement anywhere up to 2.0 s, which breaks
    # the remember <= 1.5 s product contract this test exists to pin.
    assert writer_calls and writer_calls[0] < ceiling_s, (
        f"the route waited {writer_calls[0]:.3f}s for the writer")
    # A gross hang (the route sitting out the journal deadline and more) still
    # fails end to end.
    assert elapsed < journal_s + 1.0, f"acknowledgement took {elapsed:.3f}s"
    assert final.status_code == 200, final.text
    assert final.json()["status"] == "queryable"
    assert len(final.json()["fact_ids"]) >= 1
    assert len(engine_with_mock_deps._db.execute(
        "SELECT * FROM ingestion_operations"
    )) == 1


def test_saturated_journal_is_an_honest_503_with_retry_after(
    engine_with_mock_deps, monkeypatch,
) -> None:
    """A full save queue says so, says nothing was saved, and says when to retry.

    It must not blame the disk: nothing is wrong with storage, the daemon is
    simply receiving more saves at once than its journal queues.
    """
    from superlocalmemory.storage.journal_writer import AdmissionJournalOverloaded

    body = {
        "content": "Ines files the ferry manifest before the evening sailing.",
        "idempotency_key": "saturated-journal-route-1",
    }
    with _client(engine_with_mock_deps) as client:
        journal = client.app.state.canonical_remember_runtime.journal

        def overloaded(*_args, **_kwargs):
            raise AdmissionJournalOverloaded(
                "too many saves are waiting", retry_after_seconds=1,
            )

        monkeypatch.setattr(journal._writer, "submit", overloaded)
        refused = client.post("/remember", json=body)
        monkeypatch.undo()
        retried = client.post("/remember", json=body)

    assert refused.status_code == 503, refused.text
    assert refused.headers.get("retry-after") == "1"
    detail = refused.json()["detail"]
    assert "not saved" in detail
    assert "disk" not in detail
    assert "idempotency_key" in detail
    assert retried.status_code == 200, retried.text
    assert len(engine_with_mock_deps._db.execute(
        "SELECT * FROM ingestion_operations"
    )) == 1


def test_status_counts_saves_set_aside_as_unreadable(engine_with_mock_deps) -> None:
    """An operator can see that a save was set aside, without reading logs."""
    from superlocalmemory.cli.commands import _admission_status_text
    from superlocalmemory.storage.admission_journal import Actor, RememberRequest

    with _client(engine_with_mock_deps) as client:
        runtime = client.app.state.canonical_remember_runtime
        clean = client.get("/status").json()
        entry = runtime.journal.prepare(
            RememberRequest(
                content="A save whose bytes this machine can no longer read.",
                profile_id=engine_with_mock_deps._profile_id,
                source_type="http",
                idempotency_key="status-unreadable-1",
            ),
            Actor("a", frozenset({engine_with_mock_deps._profile_id}),
                  frozenset({"personal"})),
        )
        runtime.journal.quarantine(entry.journal_id)
        flagged = client.get("/status").json()

    assert clean["unreadable_saves"] == 0
    assert clean["saves_waiting"] == 0
    assert flagged["unreadable_saves"] == 1
    assert _admission_status_text(clean) == ""
    assert "Saves set aside: 1" in _admission_status_text(flagged)
