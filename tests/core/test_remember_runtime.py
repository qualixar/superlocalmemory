"""Daemon-owned runtime contracts for canonical remember admission."""

from __future__ import annotations

import sqlite3

_FUNCTIONAL_DEADLINE_MS = 1_500


def _install_write_commits(path) -> None:
    from superlocalmemory.storage.migrations import (
        M018_ingestion_operations,
        M032_write_coordinator_admission,
        M033_projection_transactions,
        M034_obligation_integrity,
    )

    conn = sqlite3.connect(path)
    try:
        M018_ingestion_operations.apply(conn)
        M032_write_coordinator_admission.apply(conn)
        # Phase-3: M033 is now mandatory (fail-closed schema guard); apply here
        # so all runtime tests use a fully-migrated database.
        M033_projection_transactions.apply(conn)
        M034_obligation_integrity.apply(conn)
        conn.commit()
    finally:
        conn.close()


def test_runtime_replays_prepared_entry_through_one_typed_command(tmp_path) -> None:
    """A restart completes the journal entry without a second projection."""
    from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime
    from superlocalmemory.storage.admission_journal import Actor, RememberRequest
    from superlocalmemory.storage.database import DatabaseManager

    db_path = tmp_path / "memory.db"
    _install_write_commits(db_path)
    db = DatabaseManager(db_path)
    from superlocalmemory.storage import schema
    db.initialize(schema)
    calls: list[str] = []

    def writer(request, operation_id):
        calls.append(operation_id)
        db.execute(
            "INSERT INTO runtime_probe(operation_id, content) VALUES (?, ?)",
            (operation_id, request.content),
        )
        return ["fact-1"]

    runtime = CanonicalRememberRuntime(
        db=db,
        profile_id="default",
        writer=writer,
        journal_path=tmp_path / "admission_journal.db",
    )
    db.execute("CREATE TABLE runtime_probe(operation_id TEXT, content TEXT)")
    actor = Actor("actor", frozenset({"default"}), frozenset({"personal"}))
    request = RememberRequest(
        content="canonical restart witness",
        profile_id="default",
        source_type="http",
        idempotency_key="runtime-replay-1",
        trusted_actor_id="actor",
    )
    prepared = runtime.journal.prepare(request, actor)

    runtime.start()
    try:
        assert runtime.journal.get(prepared.journal_id).state == "committed"
        receipt = runtime.remember(
            request, actor, deadline_ms=_FUNCTIONAL_DEADLINE_MS
        ).payload
        assert receipt["status"] == "queryable"
        assert receipt["fact_ids"] == ["fact-1"]
        assert calls == [receipt["operation_id"]]
    finally:
        runtime.stop()


def test_terminal_admission_rejection_cannot_poison_restart(tmp_path) -> None:
    """A deterministic post-journal rejection is terminal, not startup work."""
    import pytest

    from superlocalmemory.core.remember_admission import AdmissionRejected
    from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.admission_journal import Actor, RememberRequest
    from superlocalmemory.storage.database import DatabaseManager

    db_path = tmp_path / "memory.db"
    journal_path = tmp_path / "admission_journal.db"
    _install_write_commits(db_path)
    db = DatabaseManager(db_path)
    db.initialize(schema)
    request = RememberRequest(
        content="A configuration race can reject this after journal preparation.",
        profile_id="default",
        source_type="http",
        idempotency_key="terminal-rejection-restart",
        trusted_actor_id="actor",
    )
    actor = Actor("actor", frozenset({"default"}), frozenset({"personal"}))

    first = CanonicalRememberRuntime(
        db=db,
        profile_id="default",
        writer=lambda _request, _operation_id: [],
        journal_path=journal_path,
    )
    first.start()
    try:
        with pytest.raises(AdmissionRejected) as caught:
            first.remember(
                request, actor, deadline_ms=_FUNCTIONAL_DEADLINE_MS
            )
        assert caught.value.retryable is False
        entry = first.journal.get_by_idempotency_key(
            request.profile_id, request.idempotency_key,
        )
        assert entry is not None
        assert entry.state == "rejected"
        assert entry.error_code == "COMMAND_REJECTED"
    finally:
        first.stop()

    restarted = CanonicalRememberRuntime(
        db=db,
        profile_id="default",
        writer=lambda _request, _operation_id: [],
        journal_path=journal_path,
    )
    restarted.start()
    try:
        assert restarted.ready
        assert restarted.replay_pending() == 0
        with pytest.raises(AdmissionRejected):
            restarted.remember(
                request, actor, deadline_ms=_FUNCTIONAL_DEADLINE_MS
            )
    finally:
        restarted.stop()


def test_runtime_uses_bounded_admission_and_never_invokes_materializer(tmp_path) -> None:
    """The immediate transaction contains only queryable projection work."""
    from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime
    from superlocalmemory.storage.admission_journal import Actor, RememberRequest
    from superlocalmemory.storage.database import DatabaseManager

    db_path = tmp_path / "memory.db"
    _install_write_commits(db_path)
    db = DatabaseManager(db_path)
    from superlocalmemory.storage import schema
    db.initialize(schema)
    materializer_called = False

    def writer(_request, _operation_id):
        return ["fact-2"]

    runtime = CanonicalRememberRuntime(
        db=db,
        profile_id="default",
        writer=writer,
        journal_path=tmp_path / "admission_journal.db",
        materialize=lambda _operation: materializer_called,
    )
    runtime.start()
    try:
        actor = Actor("actor", frozenset({"default"}), frozenset({"personal"}))
        receipt = runtime.remember(
            RememberRequest(
                content="bounded canonical path",
                profile_id="default",
                source_type="http",
                idempotency_key="runtime-nowait-1",
                trusted_actor_id="actor",
            ),
            actor,
            deadline_ms=_FUNCTIONAL_DEADLINE_MS,
        )
        assert receipt.payload["status"] == "queryable"
        assert materializer_called is False
    finally:
        runtime.stop()


def test_replay_after_canonical_commit_marks_journal_without_rerunning_handler(tmp_path) -> None:
    """A crash between the SQLite commit and journal acknowledgement is safe."""
    from superlocalmemory.core.remember_admission import RememberAdmissionCommand
    from superlocalmemory.core.remember_runtime import (
        CanonicalRememberRuntime,
        _CoordinatorAdapter,
    )
    from superlocalmemory.storage.admission_journal import Actor, RememberRequest
    from superlocalmemory.storage.database import DatabaseManager

    db_path = tmp_path / "memory.db"
    _install_write_commits(db_path)
    db = DatabaseManager(db_path)
    calls: list[str] = []

    def writer(_request, operation_id):
        calls.append(operation_id)
        return ["fact-crash"]

    runtime = CanonicalRememberRuntime(
        db=db,
        profile_id="default",
        writer=writer,
        journal_path=tmp_path / "admission_journal.db",
    )
    runtime.start()
    try:
        actor = Actor("actor", frozenset({"default"}), frozenset({"personal"}))
        request = RememberRequest(
            content="canonical commit before journal acknowledgement",
            profile_id="default",
            source_type="http",
            idempotency_key="runtime-crash-window-1",
            trusted_actor_id="actor",
        )
        prepared = runtime.journal.prepare(request, actor)
        receipt = _CoordinatorAdapter(runtime.coordinator).submit(
            RememberAdmissionCommand.from_prepared(prepared, request),
            wait_ms=500,
        )["receipt"]

        assert runtime.journal.get(prepared.journal_id).state == "prepared"
        assert runtime.replay_pending() == 1
        recovered = runtime.journal.get(prepared.journal_id).original_receipt
        assert recovered is not None
        assert recovered["operation_id"] == receipt["operation_id"]
        assert recovered["commit_sequence"] == receipt["commit_sequence"]
        assert calls == [receipt["operation_id"]]
    finally:
        runtime.stop()


def test_replay_rejects_same_key_with_different_committed_request_hash(tmp_path) -> None:
    """Recovery must not acknowledge different content with an old receipt."""
    import pytest

    from superlocalmemory.core.remember_admission import AdmissionRejected
    from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.admission_journal import Actor, RememberRequest
    from superlocalmemory.storage.database import DatabaseManager

    db_path = tmp_path / "memory.db"
    journal_path = tmp_path / "admission_journal.db"
    _install_write_commits(db_path)
    db = DatabaseManager(db_path)
    db.initialize(schema)
    actor = Actor("actor", frozenset({"default"}), frozenset({"personal"}))
    committed_request = RememberRequest(
        content="The canonical receipt belongs to this immutable request.",
        profile_id="default",
        source_type="http",
        idempotency_key="runtime-hash-conflict-1",
        trusted_actor_id="actor",
    )
    first = CanonicalRememberRuntime(
        db=db,
        profile_id="default",
        writer=lambda _request, _operation_id: ["fact-original"],
        journal_path=journal_path,
    )
    first.start()
    try:
        first.remember(
            committed_request, actor, deadline_ms=_FUNCTIONAL_DEADLINE_MS
        )
    finally:
        first.stop()

    conflicting_journal_path = tmp_path / "conflicting-journal" / "admission_journal.db"
    conflicting_request = RememberRequest(
        content="Different content must never inherit the old receipt.",
        profile_id="default",
        source_type="http",
        idempotency_key=committed_request.idempotency_key,
        trusted_actor_id="actor",
    )
    restarted = CanonicalRememberRuntime(
        db=db,
        profile_id="default",
        writer=lambda *_args: pytest.fail("conflicting request must not dispatch"),
        journal_path=conflicting_journal_path,
    )
    prepared = restarted.journal.prepare(conflicting_request, actor)

    restarted.start()
    try:
        entry = restarted.journal.get(prepared.journal_id)
        assert entry.state == "rejected"
        assert entry.error_code == "IDEMPOTENCY_CONFLICT"
        assert entry.original_receipt is None
        with pytest.raises(AdmissionRejected, match="IDEMPOTENCY_CONFLICT"):
            restarted.remember(
                conflicting_request,
                actor,
                deadline_ms=_FUNCTIONAL_DEADLINE_MS,
            )
    finally:
        restarted.stop()


def test_runtime_rebinds_profile_without_replacing_writer_ownership(tmp_path) -> None:
    """A drained profile transition keeps one coordinator and targets the new profile."""
    from types import SimpleNamespace

    from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.admission_journal import Actor, RememberRequest
    from superlocalmemory.storage.database import DatabaseManager

    db_path = tmp_path / "memory.db"
    _install_write_commits(db_path)
    db = DatabaseManager(db_path)
    db.initialize(schema)
    db.execute(
        "INSERT OR IGNORE INTO profiles(profile_id, name) VALUES (?, ?)",
        ("beta", "beta"),
    )
    runtime = CanonicalRememberRuntime(
        db=db,
        profile_id="default",
        writer=lambda _request, _operation_id: ["unused"],
        journal_path=tmp_path / "admission_journal.db",
    )
    runtime.start()
    owner_id = runtime.coordinator.owner_id
    try:
        runtime.rebind_engine(SimpleNamespace(_db=db, _profile_id="beta"))
        receipt = runtime.remember(
            RememberRequest(
                content="Beta profile owns this canonical transition witness.",
                profile_id="beta",
                source_type="http",
                idempotency_key="runtime-profile-rebind-1",
                trusted_actor_id="actor",
            ),
            Actor("actor", frozenset({"beta"}), frozenset({"personal"})),
            deadline_ms=_FUNCTIONAL_DEADLINE_MS,
        )
        assert receipt.payload["status"] == "queryable"
        assert runtime.coordinator.owner_id == owner_id
        facts = db.search_facts_fts("canonical transition witness", "beta")
        assert [fact.fact_id for fact in facts] == list(receipt.payload["fact_ids"])
    finally:
        runtime.stop()


def test_runtime_defers_other_profile_recovery_until_rebind(tmp_path) -> None:
    """A pending profile-B command cannot prevent profile-A daemon startup."""
    from types import SimpleNamespace

    from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.admission_journal import Actor, RememberRequest
    from superlocalmemory.storage.database import DatabaseManager

    db_path = tmp_path / "memory.db"
    _install_write_commits(db_path)
    db = DatabaseManager(db_path)
    db.initialize(schema)
    db.execute(
        "INSERT OR IGNORE INTO profiles(profile_id, name) VALUES (?, ?)",
        ("beta", "beta"),
    )
    runtime = CanonicalRememberRuntime(
        db=db,
        profile_id="default",
        writer=lambda _request, _operation_id: ["unused"],
        journal_path=tmp_path / "admission_journal.db",
    )
    pending = runtime.journal.prepare(
        RememberRequest(
            content="Beta recovery waits until the beta runtime is active.",
            profile_id="beta",
            source_type="http",
            idempotency_key="runtime-deferred-beta",
            trusted_actor_id="actor",
        ),
        Actor("actor", frozenset({"beta"}), frozenset({"personal"})),
    )

    runtime.start()
    try:
        assert runtime.journal.get(pending.journal_id).state == "prepared"
        runtime.rebind_engine(SimpleNamespace(_db=db, _profile_id="beta"))
        recovered = runtime.journal.get(pending.journal_id)
        assert recovered.state == "committed"
        assert recovered.original_receipt is not None
        fact_ids = recovered.original_receipt["fact_ids"]
        assert db.search_facts_fts("Beta recovery waits", "beta")[0].fact_id in fact_ids
    finally:
        runtime.stop()
