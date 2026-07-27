# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Canonical mutation commands share the daemon-owned writer with remember."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from superlocalmemory.storage.migrations import (
    M018_ingestion_operations,
    M032_write_coordinator_admission,
)


def test_delete_command_is_idempotent_and_preserves_immutable_receipt(
    engine_with_mock_deps,
) -> None:
    """A retry never re-runs a fact delete or rewrites its first receipt."""
    from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime
    from superlocalmemory.storage.admission_journal import Actor, RememberRequest

    with engine_with_mock_deps._db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
        M032_write_coordinator_admission.apply(conn)
    runtime = CanonicalRememberRuntime.for_engine(engine_with_mock_deps)
    runtime.start()
    try:
        actor = Actor("daemon:test", frozenset({"default"}), frozenset({"personal"}))
        remembered = runtime.remember(
            RememberRequest(
                content="The canonical delete retry witness remains deterministic.",
                profile_id="default",
                source_type="test",
                idempotency_key="canonical-mutation-source",
                trusted_actor_id="daemon:test",
            ),
            actor,
            deadline_ms=500,
        )
        fact_id = remembered.payload["fact_ids"][0]

        first = runtime.delete_fact(
            "default",
            fact_id,
            idempotency_key="canonical-delete-retry",
        )
        second = runtime.delete_fact(
            "default",
            fact_id,
            idempotency_key="canonical-delete-retry",
        )
    finally:
        runtime.stop()

    assert first == second
    assert first["ok"] is True
    assert engine_with_mock_deps._db.execute(
        "SELECT fact_id FROM atomic_facts WHERE fact_id = ?", (fact_id,)
    ) == []
    commits = engine_with_mock_deps._db.execute(
        "SELECT command_kind, receipt_json FROM write_commits "
        "WHERE command_kind = ?",
        ("delete_fact",),
    )
    assert len(commits) == 1
    assert commits[0]["command_kind"] == "delete_fact"


def test_concurrent_delete_retries_share_one_canonical_receipt(
    engine_with_mock_deps,
) -> None:
    """Simultaneous client retries serialize and acknowledge one delete commit."""
    from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime
    from superlocalmemory.storage.admission_journal import Actor, RememberRequest

    with engine_with_mock_deps._db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
        M032_write_coordinator_admission.apply(conn)
    runtime = CanonicalRememberRuntime.for_engine(engine_with_mock_deps)
    runtime.start()
    try:
        actor = Actor("daemon:test", frozenset({"default"}), frozenset({"personal"}))
        remembered = runtime.remember(
            RememberRequest(
                content="Concurrent delete retries must share their first receipt.",
                profile_id="default",
                source_type="test",
                idempotency_key="canonical-concurrent-source",
                trusted_actor_id="daemon:test",
            ),
            actor,
            deadline_ms=500,
        )
        fact_id = remembered.payload["fact_ids"][0]
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(
                lambda _: runtime.delete_fact(
                    "default", fact_id, idempotency_key="canonical-concurrent-delete",
                ),
                range(16),
            ))
    finally:
        runtime.stop()

    assert all(result == results[0] for result in results)
    assert results[0]["ok"] is True
    commits = engine_with_mock_deps._db.execute(
        "SELECT COUNT(*) AS count FROM write_commits WHERE command_kind = ?",
        ("delete_fact",),
    )
    assert commits[0]["count"] == 1


def test_mutation_retry_key_is_endpoint_scoped_and_rejects_payload_drift(
    engine_with_mock_deps,
) -> None:
    """One client key may cross endpoints, but not change one endpoint's input."""
    from superlocalmemory.core.remember_runtime import (
        CanonicalMutationConflict,
        CanonicalRememberRuntime,
    )
    from superlocalmemory.storage.admission_journal import Actor, RememberRequest

    with engine_with_mock_deps._db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
        M032_write_coordinator_admission.apply(conn)
    runtime = CanonicalRememberRuntime.for_engine(engine_with_mock_deps)
    runtime.start()
    try:
        actor = Actor("daemon:test", frozenset({"default"}), frozenset({"personal"}))
        fact_id = runtime.remember(
            RememberRequest(
                content="Mutation idempotency keys remain endpoint scoped.",
                profile_id="default",
                source_type="test",
                idempotency_key="mutation-key-scope-source",
                trusted_actor_id="daemon:test",
            ),
            actor,
            deadline_ms=500,
        ).payload["fact_ids"][0]

        first = runtime.update_fact(
            "default",
            fact_id,
            {"content": "The first immutable mutation payload wins."},
            idempotency_key="shared-client-key",
        )
        with pytest.raises(CanonicalMutationConflict):
            runtime.update_fact(
                "default",
                fact_id,
                {"content": "A retry cannot silently change its payload."},
                idempotency_key="shared-client-key",
            )
        deleted = runtime.delete_fact(
            "default",
            fact_id,
            idempotency_key="shared-client-key",
        )
    finally:
        runtime.stop()

    assert first["ok"] is True
    assert deleted["ok"] is True
    commits = engine_with_mock_deps._db.execute(
        "SELECT command_kind FROM write_commits "
        "WHERE command_kind IN ('update_fact', 'delete_fact') "
        "ORDER BY command_kind"
    )
    assert [row["command_kind"] for row in commits] == [
        "delete_fact",
        "update_fact",
    ]


def test_archive_and_merge_retries_replay_the_first_receipt(
    engine_with_mock_deps,
) -> None:
    """Server-generated IDs and timestamps must not make client retries conflict."""
    from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime
    from superlocalmemory.storage.admission_journal import Actor, RememberRequest

    with engine_with_mock_deps._db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
        M032_write_coordinator_admission.apply(conn)
    runtime = CanonicalRememberRuntime.for_engine(engine_with_mock_deps)
    runtime.start()
    try:
        actor = Actor("daemon:test", frozenset({"default"}), frozenset({"personal"}))

        def remember(content: str, key: str) -> str:
            return runtime.remember(
                RememberRequest(
                    content=content,
                    profile_id="default",
                    source_type="test",
                    idempotency_key=key,
                    trusted_actor_id="daemon:test",
                ),
                actor,
                deadline_ms=1_500,
            ).payload["fact_ids"][0]

        archive_id = remember(
            "The archive retry witness has deterministic command input.",
            "archive-retry-source",
        )
        merge_source = remember(
            "The merge retry source has deterministic command input.",
            "merge-retry-source",
        )
        merge_target = remember(
            "The merge retry target remains canonical.",
            "merge-retry-target",
        )

        first_archive = runtime.archive_fact(
            "default", archive_id, idempotency_key="archive-retry-key",
        )
        second_archive = runtime.archive_fact(
            "default", archive_id, idempotency_key="archive-retry-key",
        )
        first_merge = runtime.merge_fact(
            "default",
            merge_source,
            merge_target,
            idempotency_key="merge-retry-key",
        )
        second_merge = runtime.merge_fact(
            "default",
            merge_source,
            merge_target,
            idempotency_key="merge-retry-key",
        )
    finally:
        runtime.stop()

    assert first_archive == second_archive
    assert first_merge == second_merge
    assert first_archive["ok"] is True
    assert first_merge["ok"] is True


def test_mutation_receipts_are_metadata_only_and_allow_distinct_updates(
    engine_with_mock_deps,
) -> None:
    """The immutable ledger must not retain fact text or reject a later edit."""
    from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime
    from superlocalmemory.storage.admission_journal import Actor, RememberRequest

    with engine_with_mock_deps._db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
        M032_write_coordinator_admission.apply(conn)
    runtime = CanonicalRememberRuntime.for_engine(engine_with_mock_deps)
    runtime.start()
    original_secret = "SECRET-ORIGINAL-386 must disappear after deletion."
    first_edit = "SECRET-FIRST-EDIT-386 must not enter the receipt ledger."
    second_edit = "SECRET-SECOND-EDIT-386 must not enter the receipt ledger."
    try:
        actor = Actor("daemon:test", frozenset({"default"}), frozenset({"personal"}))
        fact_id = runtime.remember(
            RememberRequest(
                content=original_secret,
                profile_id="default",
                source_type="test",
                idempotency_key="receipt-redaction-source",
                trusted_actor_id="daemon:test",
            ),
            actor,
            deadline_ms=500,
        ).payload["fact_ids"][0]
        first = runtime.update_fact(
            "default",
            fact_id,
            {"content": first_edit},
            idempotency_key="receipt-redaction-update-one",
        )
        second = runtime.update_fact(
            "default",
            fact_id,
            {"content": second_edit},
            idempotency_key="receipt-redaction-update-two",
        )
        deleted = runtime.delete_fact(
            "default",
            fact_id,
            idempotency_key="receipt-redaction-delete",
        )
    finally:
        runtime.stop()

    assert first["ok"] is True
    assert second["ok"] is True
    assert first["operation_id"] != second["operation_id"]
    assert deleted["ok"] is True
    rows = engine_with_mock_deps._db.execute(
        "SELECT receipt_json FROM write_commits "
        "WHERE command_kind IN ('update_fact', 'delete_fact')"
    )
    receipts = [json.loads(row["receipt_json"]) for row in rows]
    serialized = json.dumps(receipts, sort_keys=True)
    assert original_secret not in serialized
    assert first_edit not in serialized
    assert second_edit not in serialized
    assert all("content" not in receipt for receipt in receipts)
    assert all("content_preview" not in receipt for receipt in receipts)
