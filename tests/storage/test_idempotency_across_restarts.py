# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A save retried with the same key after a daemon restart is the same save.

4.1.21 bound an idempotency key to ``daemon-capability:<fingerprint>``, which
changes on every daemon start, so every retry after a restart was refused as a
different request. The key is now bound to the stable caller; a different
caller still cannot reuse it, and rows journaled before the change still match.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, replace

import pytest

from superlocalmemory.core.ingestion_command import (
    IdempotencyConflict as IngestionConflict,
)
from superlocalmemory.core.ingestion_command import (
    IngestionCommand,
    IngestionOperationRepository,
    IngestionRequest,
)
from superlocalmemory.storage import schema
from superlocalmemory.storage.admission_journal import (
    Actor,
    AdmissionJournal,
    IdempotencyConflict,
    RememberRequest,
    _canonical_bytes,
)
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.idempotency_identity import (
    DAEMON_CLIENT_PRINCIPAL,
    idempotency_principal,
    same_principal,
)
from superlocalmemory.storage.migrations import M018_ingestion_operations

BOOT_ONE = "daemon-capability:" + "1" * 64
BOOT_TWO = "daemon-capability:" + "2" * 64
API_KEY = "api-key:dashboard:" + "a" * 64
DASHBOARD = "local-capability:dashboard:uid:501:ns:" + "b" * 64
RECEIPT = {"operation_id": "op-1", "fact_ids": ["fact-1"], "state": "queryable",
           "commit_sequence": 1}


@dataclass(frozen=True)
class _Codec:
    """Reversible, deliberately non-production codec for journal contracts."""

    def encrypt(self, plaintext: bytes) -> bytes:
        return b"t:" + plaintext[::-1]

    def decrypt(self, ciphertext: bytes) -> bytes:
        assert ciphertext.startswith(b"t:")
        return ciphertext[2:][::-1]


def _actor(principal: str) -> Actor:
    return Actor(principal_id=principal, allowed_profiles=frozenset({"default"}),
                 allowed_scopes=frozenset({"personal"}))


def _request(principal: str, **changes) -> RememberRequest:
    base = RememberRequest(
        content="Fixture record: the synthetic retry code is R-1.",
        profile_id="default", source_type="http",
        idempotency_key="retry-after-restart-1", metadata={"tags": "fixture"},
        trusted_actor_id=principal,
    )
    return replace(base, **changes) if changes else base


@pytest.fixture
def journal(tmp_path):
    journal = AdmissionJournal(tmp_path / "admission_journal.db", codec=_Codec())
    yield journal
    journal.close()


def _committed(journal: AdmissionJournal, principal: str):
    first = journal.prepare(_request(principal), _actor(principal))
    journal.mark_committed(first.journal_id, RECEIPT)
    return first


def test_a_retry_from_the_next_daemon_start_returns_the_original_receipt(journal) -> None:
    first = _committed(journal, BOOT_ONE)

    again = journal.prepare(_request(BOOT_TWO), _actor(BOOT_TWO))

    assert again.journal_id == first.journal_id
    assert again.original_receipt == RECEIPT
    assert journal.count() == 1


@pytest.mark.parametrize("other", [API_KEY, DASHBOARD])
def test_another_caller_still_cannot_reuse_the_key(journal, other) -> None:
    _committed(journal, BOOT_ONE)

    with pytest.raises(IdempotencyConflict):
        journal.prepare(_request(other), _actor(other))
    assert journal.count() == 1


def test_the_daemon_client_cannot_reuse_a_key_another_caller_holds(journal) -> None:
    _committed(journal, API_KEY)

    with pytest.raises(IdempotencyConflict):
        journal.prepare(_request(BOOT_TWO), _actor(BOOT_TWO))


def test_a_different_request_with_the_key_is_still_refused_after_a_restart(journal) -> None:
    _committed(journal, BOOT_ONE)

    with pytest.raises(IdempotencyConflict):
        journal.prepare(_request(BOOT_TWO, content="Other words, same key."),
                        _actor(BOOT_TWO))


def test_stable_callers_hash_exactly_as_before(journal) -> None:
    # Byte-identical hashes for every actor that was already stable, so their
    # retries across the upgrade match with no read-back at all.
    entry = journal.prepare(_request(API_KEY), _actor(API_KEY))
    legacy = hashlib.sha256(_canonical_bytes(_request(API_KEY).canonical_payload())).hexdigest()
    assert entry.request_hash == legacy


def _rewrite_hash_as_4121(journal: AdmissionJournal, journal_id: str, principal: str) -> None:
    legacy = hashlib.sha256(_canonical_bytes(_request(principal).canonical_payload())).hexdigest()
    conn = sqlite3.connect(journal.path)
    try:
        conn.execute("UPDATE admission_journal SET request_hash=? WHERE journal_id=?",
                     (legacy, journal_id))
        conn.commit()
    finally:
        conn.close()
    journal.close()  # drop pooled readers so the next prepare sees the row


def test_a_row_journaled_by_4121_still_matches_its_retry(journal) -> None:
    first = _committed(journal, BOOT_ONE)
    _rewrite_hash_as_4121(journal, first.journal_id, BOOT_ONE)

    again = journal.prepare(_request(BOOT_TWO), _actor(BOOT_TWO))

    assert again.journal_id == first.journal_id
    assert again.original_receipt == RECEIPT


def test_a_4121_row_whose_command_cannot_be_read_is_never_a_match(journal) -> None:
    first = _committed(journal, BOOT_ONE)
    _rewrite_hash_as_4121(journal, first.journal_id, BOOT_ONE)
    conn = sqlite3.connect(journal.path)
    try:
        conn.execute("UPDATE admission_journal SET command_json='{\"ciphertext_b64\":\"!!\"}' "
                     "WHERE journal_id=?", (first.journal_id,))
        conn.commit()
    finally:
        conn.close()
    journal.close()

    with pytest.raises(IdempotencyConflict):
        journal.prepare(_request(BOOT_TWO), _actor(BOOT_TWO))


def test_the_principal_mapping() -> None:
    assert idempotency_principal(BOOT_ONE) == DAEMON_CLIENT_PRINCIPAL
    assert same_principal(BOOT_ONE, BOOT_TWO)
    assert not same_principal(BOOT_ONE, API_KEY)
    assert not same_principal(API_KEY, "api-key:dashboard:" + "c" * 64)
    assert idempotency_principal(DASHBOARD) == DASHBOARD
    assert idempotency_principal("") == ""


@pytest.fixture
def db(tmp_path):
    manager = DatabaseManager(tmp_path / "memory.db")
    manager.initialize(schema)
    with manager.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
    return manager


def _ingestion(principal: str) -> IngestionRequest:
    return IngestionRequest(
        content="Fixture record: the synthetic retry code is R-2.",
        profile_id="default", source_type="http", idempotency_key="ingest-retry-1",
        metadata={}, trusted_actor_id=principal,
    )


def test_the_ingestion_record_accepts_the_next_daemon_start(db) -> None:
    command = IngestionCommand(IngestionOperationRepository(db),
                               write_queryable=lambda *_: ["fact-q"],
                               materialize=lambda *_: ["fact-f"])
    first = command.submit(_ingestion(BOOT_ONE))

    again = command.submit(_ingestion(BOOT_TWO))

    assert again.operation_id == first.operation_id
    with pytest.raises(IngestionConflict):
        command.submit(_ingestion(API_KEY))
