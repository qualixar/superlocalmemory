# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Durability and idempotency tests for the 3.8.6 admission journal."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import pytest

from superlocalmemory.storage.admission_journal import (
    Actor,
    AdmissionAuthorizationError,
    AdmissionJournal,
    AdmissionPayloadError,
    IdempotencyConflict,
    RememberRequest,
    TerminalAdmissionError,
)


@dataclass(frozen=True)
class _TestCodec:
    """Deliberately non-production reversible codec for journal contract tests."""

    prefix: bytes = b"test-policy:"

    def encrypt(self, plaintext: bytes) -> bytes:
        return self.prefix + plaintext[::-1]

    def decrypt(self, ciphertext: bytes) -> bytes:
        assert ciphertext.startswith(self.prefix)
        return ciphertext[len(self.prefix) :][::-1]


@pytest.fixture
def actor() -> Actor:
    return Actor(
        principal_id="daemon-capability:test",
        allowed_profiles=frozenset({"default"}),
        allowed_scopes=frozenset({"personal"}),
    )


@pytest.fixture
def admission_request() -> RememberRequest:
    return RememberRequest(
        content="Mira approved the recovery admission contract.",
        profile_id="default",
        source_type="mcp",
        idempotency_key="admission:durable-replay",
        metadata={"source": "contract-test"},
    )


def test_same_idempotency_key_returns_original_receipt(tmp_path, actor, admission_request) -> None:
    journal = AdmissionJournal(tmp_path / "admission_journal.db", codec=_TestCodec())
    first = journal.prepare(admission_request, actor)
    receipt = {
        "operation_id": "operation-1",
        "fact_ids": ["fact-1"],
        "state": "queryable",
        "commit_sequence": 3,
    }
    journal.mark_committed(first.journal_id, receipt)

    duplicate = journal.prepare(admission_request, actor)

    assert duplicate.journal_id == first.journal_id
    assert duplicate.original_receipt == receipt
    assert journal.count() == 1


def test_changed_payload_with_same_key_conflicts(tmp_path, actor, admission_request) -> None:
    journal = AdmissionJournal(tmp_path / "admission_journal.db", codec=_TestCodec())
    journal.prepare(admission_request, actor)

    with pytest.raises(IdempotencyConflict, match="different immutable request"):
        journal.prepare(
            RememberRequest(
                content="Different evidence must never inherit a receipt.",
                profile_id=admission_request.profile_id,
                source_type=admission_request.source_type,
                idempotency_key=admission_request.idempotency_key,
            ),
            actor,
        )

    assert journal.count() == 1


def test_same_key_is_independent_between_authorized_profiles(tmp_path) -> None:
    """A client retry token is namespaced by its target profile."""
    journal = AdmissionJournal(tmp_path / "admission_journal.db", codec=_TestCodec())
    actor = Actor(
        principal_id="daemon-capability:test",
        allowed_profiles=frozenset({"default", "work"}),
        allowed_scopes=frozenset({"personal"}),
    )
    default = RememberRequest(
        content="Default profile keeps its own receipt.",
        profile_id="default",
        source_type="mcp",
        idempotency_key="profile-scoped-key",
    )
    work = RememberRequest(
        content="Work profile may reuse the client retry token.",
        profile_id="work",
        source_type="mcp",
        idempotency_key="profile-scoped-key",
    )

    first = journal.prepare(default, actor)
    second = journal.prepare(work, actor)

    assert first.journal_id != second.journal_id
    assert journal.get_by_idempotency_key("default", default.idempotency_key) == first
    assert journal.get_by_idempotency_key("work", work.idempotency_key) == second
    assert journal.count() == 2


def test_journal_upgrades_legacy_global_key_schema(tmp_path) -> None:
    """The standalone journal also upgrades provisional developer state safely."""
    path = tmp_path / "admission_journal.db"
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE admission_journal (
                journal_id TEXT PRIMARY KEY,
                idempotency_key TEXT NOT NULL UNIQUE,
                request_hash TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                command_json TEXT NOT NULL,
                state TEXT NOT NULL,
                canonical_operation_id TEXT,
                canonical_commit_sequence INTEGER,
                error_code TEXT,
                receipt_json TEXT,
                created_at_ms INTEGER NOT NULL,
                updated_at_ms INTEGER NOT NULL
            );
            CREATE INDEX idx_admission_replay
                ON admission_journal(state, updated_at_ms);
            """
        )
        conn.commit()
    finally:
        conn.close()

    journal = AdmissionJournal(path, codec=_TestCodec())
    actor = Actor(
        principal_id="daemon-capability:test",
        allowed_profiles=frozenset({"default", "work"}),
        allowed_scopes=frozenset({"personal"}),
    )
    for profile_id in ("default", "work"):
        journal.prepare(
            RememberRequest(
                content=f"{profile_id} profile upgrade witness.",
                profile_id=profile_id,
                source_type="mcp",
                idempotency_key="legacy-upgrade-key",
            ),
            actor,
        )

    assert journal.count() == 2


def test_recovery_lookup_is_scoped_by_profile_and_key(tmp_path) -> None:
    """Crash recovery cannot attach one profile's receipt to another profile."""
    journal = AdmissionJournal(tmp_path / "admission_journal.db", codec=_TestCodec())
    actor = Actor(
        principal_id="daemon-capability:test",
        allowed_profiles=frozenset({"default", "work"}),
        allowed_scopes=frozenset({"personal"}),
    )
    for profile_id in ("default", "work"):
        journal.prepare(
            RememberRequest(
                content=f"{profile_id} recovery boundary witness.",
                profile_id=profile_id,
                source_type="mcp",
                idempotency_key="recovery-profile-key",
            ),
            actor,
        )
    lookups: list[tuple[str, str]] = []

    recovered = journal.replay_pending(
        lambda entry: (
            lookups.append((entry.profile_id, entry.idempotency_key))
            or {"operation_id": f"operation:{entry.profile_id}", "fact_ids": []}
        ),
        lambda *_args: pytest.fail("profile-scoped receipt must prevent dispatch"),
    )

    assert recovered == 2
    assert sorted(lookups) == [
        ("default", "recovery-profile-key"),
        ("work", "recovery-profile-key"),
    ]


def test_replay_after_crash_before_canonical_commit_is_durable_and_once(
    tmp_path, actor, admission_request
) -> None:
    journal_path = tmp_path / "admission_journal.db"
    journal = AdmissionJournal(journal_path, codec=_TestCodec())
    prepared = journal.prepare(admission_request, actor)
    journal.mark_dispatched(prepared.journal_id)

    reopened = AdmissionJournal(journal_path, codec=_TestCodec())
    dispatched: list[str] = []

    replayed = reopened.replay_pending(
        find_canonical_receipt=lambda _entry: None,
        dispatch=lambda entry, decoded: (
            dispatched.append(decoded.content)
            or {
                "operation_id": f"operation-{entry.journal_id}",
                "fact_ids": ["fact-1"],
                "state": "queryable",
                "commit_sequence": 1,
            }
        ),
    )

    assert replayed == 1
    assert dispatched == [admission_request.content]
    assert reopened.get(prepared.journal_id).state == "committed"
    assert (
        reopened.replay_pending(
            lambda _entry: None,
            lambda *_args: pytest.fail("replayed twice"),
        )
        == 0
    )


def test_replay_after_canonical_commit_keeps_original_receipt_without_dispatch(
    tmp_path, actor, admission_request
) -> None:
    journal = AdmissionJournal(tmp_path / "admission_journal.db", codec=_TestCodec())
    prepared = journal.prepare(admission_request, actor)
    expected = {
        "operation_id": "already-committed",
        "fact_ids": ["fact-9"],
        "state": "queryable",
        "commit_sequence": 9,
    }

    assert (
        journal.replay_pending(
            find_canonical_receipt=lambda entry: (
                expected
                if (entry.profile_id, entry.idempotency_key)
                == (admission_request.profile_id, admission_request.idempotency_key)
                else None
            ),
            dispatch=lambda *_args: pytest.fail("canonical duplicate must not dispatch"),
        )
        == 1
    )
    assert journal.prepare(admission_request, actor).original_receipt == expected
    assert journal.get(prepared.journal_id).state == "committed"


def test_replay_marks_terminal_dispatch_rejection_and_continues(
    tmp_path, actor, admission_request
) -> None:
    journal = AdmissionJournal(tmp_path / "admission_journal.db", codec=_TestCodec())
    prepared = journal.prepare(admission_request, actor)

    replayed = journal.replay_pending(
        find_canonical_receipt=lambda _entry: None,
        dispatch=lambda _entry, _request: (_ for _ in ()).throw(
            TerminalAdmissionError("DETERMINISTIC_POLICY_REJECTED")
        ),
    )

    assert replayed == 1
    rejected = journal.get(prepared.journal_id)
    assert rejected.state == "rejected"
    assert rejected.error_code == "DETERMINISTIC_POLICY_REJECTED"
    assert journal.replay_pending(
        lambda _entry: None,
        lambda *_args: pytest.fail("terminal entry was replayed"),
    ) == 0


def test_journal_does_not_persist_plain_memory_content(tmp_path, actor, admission_request) -> None:
    journal = AdmissionJournal(tmp_path / "admission_journal.db", codec=_TestCodec())
    journal.prepare(admission_request, actor)

    assert admission_request.content not in journal_path_text(tmp_path / "admission_journal.db")


def test_illegal_terminal_transition_is_rejected_atomically(
    tmp_path, actor, admission_request
) -> None:
    journal = AdmissionJournal(tmp_path / "admission_journal.db", codec=_TestCodec())
    prepared = journal.prepare(admission_request, actor)
    journal.mark_committed(prepared.journal_id, {"operation_id": "op", "fact_ids": []})

    with pytest.raises(ValueError, match="already committed"):
        journal.mark_rejected(prepared.journal_id, "COMMAND_REJECTED")

    assert journal.get(prepared.journal_id).state == "committed"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"content": " "}, "content"),
        ({"source_type": " "}, "source_type"),
        ({"idempotency_key": "contains a space"}, "idempotency_key"),
        ({"scope": "tenant"}, "unsupported scope"),
        ({"metadata": ["not-an-object"]}, "metadata"),
    ],
)
def test_request_validation_rejects_unsafe_admission_input(kwargs, message) -> None:
    base = {
        "content": "safe content",
        "profile_id": "default",
        "source_type": "mcp",
        "idempotency_key": "validation:key",
    }
    base.update(kwargs)

    with pytest.raises(AdmissionPayloadError, match=message):
        RememberRequest(**base)


def test_prepare_requires_authorized_matching_actor(tmp_path, admission_request) -> None:
    journal = AdmissionJournal(tmp_path / "admission_journal.db", codec=_TestCodec())
    forbidden = Actor("daemon:test", frozenset(), frozenset({"personal"}))
    mismatched = Actor(
        "daemon:other",
        frozenset({"default"}),
        frozenset({"personal"}),
    )

    with pytest.raises(AdmissionAuthorizationError, match="not authorized"):
        journal.prepare(admission_request, forbidden)
    with pytest.raises(AdmissionAuthorizationError, match="does not match"):
        journal.prepare(
            RememberRequest(
                **{
                    **admission_request.canonical_payload(),
                    "shared_with": (),
                    "trusted_actor_id": "daemon:test",
                }
            ),
            mismatched,
        )


def test_prepare_rejects_empty_or_undecryptable_ciphertext(
    tmp_path, actor, admission_request
) -> None:
    class EmptyCodec:
        def encrypt(self, _plaintext: bytes) -> bytes:
            return b""

        def decrypt(self, _ciphertext: bytes) -> bytes:
            return b"{}"

    with pytest.raises(AdmissionPayloadError, match="no ciphertext"):
        AdmissionJournal(tmp_path / "empty.db", codec=EmptyCodec()).prepare(
            admission_request, actor
        )

    journal = AdmissionJournal(tmp_path / "corrupt.db", codec=_TestCodec())
    prepared = journal.prepare(admission_request, actor)
    with journal._connection() as conn:  # test-only corruption of opaque payload
        conn.execute(
            'UPDATE admission_journal SET command_json=\'{"ciphertext_b64": "not-base64!"}\''
        )
        conn.commit()
    with pytest.raises(AdmissionPayloadError, match="cannot be decrypted"):
        journal.request_for(prepared)


def test_receipt_and_transition_validation_are_bounded(tmp_path, actor, admission_request) -> None:
    journal = AdmissionJournal(tmp_path / "admission_journal.db", codec=_TestCodec())
    prepared = journal.prepare(admission_request, actor)

    with pytest.raises(ValueError, match="raw memory content"):
        journal.mark_committed(prepared.journal_id, {"content": "must not persist"})
    with pytest.raises(ValueError, match="raw memory content"):
        journal.mark_committed(
            prepared.journal_id,
            {"metadata": {"Content": "case variants must not persist"}},
        )
    with pytest.raises(ValueError, match="error_code"):
        journal.mark_rejected(prepared.journal_id, "")
    with pytest.raises(ValueError, match="operation_id"):
        journal.mark_committed(prepared.journal_id, {"operation_id": 1})
    with pytest.raises(ValueError, match="commit_sequence"):
        journal.mark_committed(prepared.journal_id, {"commit_sequence": "one"})

    rejected = journal.mark_rejected(prepared.journal_id, "COMMAND_REJECTED")
    assert rejected.state == "rejected"
    assert journal.mark_rejected(prepared.journal_id, "COMMAND_REJECTED").state == "rejected"


def journal_path_text(path) -> str:
    """Inspect the SQLite bytes only to prove plaintext content was excluded."""
    return path.read_bytes().decode("latin-1")
