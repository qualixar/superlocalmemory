# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""F7 + F9 regressions: lease-heartbeat hardening in _run_with_lease_heartbeat.

F7 scenario: callback() raises an exception AND the heartbeat has already set
lost=True (lease stolen by a competing materializer). The LeaseLost signal is
masked because the original exception propagates from the try/finally block.
The caller sees IngestionState.FAILED and may spawn a duplicate materializer.
Fix: check lost.is_set() inside the finally block so LeaseLost wins.

F9 scenario: renew_enriching_lease() raises an unexpected exception
(e.g. AttributeError when the repository's DB is disconnected).  The inner
except sqlite3.Error does not catch it, so the heartbeat thread dies silently
without calling lost.set().  The materializer continues past lease expiry with
no signal.  Fix: wrap the heartbeat loop in a broad try/except that calls
lost.set() before the thread exits.
"""

from __future__ import annotations

import sqlite3
import threading
import time

import pytest

from superlocalmemory.core.ingestion_command import (
    IngestionCommand,
    IngestionOperationRepository,
    IngestionRequest,
    IngestionState,
    LeaseLost,
)
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.migrations import M018_ingestion_operations


@pytest.fixture
def db(tmp_path):
    manager = DatabaseManager(tmp_path / "memory.db")
    manager.initialize(schema)
    with manager.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
    return manager


@pytest.fixture
def base_request() -> IngestionRequest:
    return IngestionRequest(
        content="lease hardening test content",
        profile_id="work",
        source_type="mcp",
        idempotency_key="lease-hardening:f7f9",
        metadata={},
        scope="personal",
        shared_with=(),
        trusted_actor_id="capability:test",
        session_id="session-f7f9",
        session_date="2026-07-23",
        speaker="Varun",
        role="user",
    )


# ---------------------------------------------------------------------------
# F7: callback raises + lease stolen → LeaseLost must win
# ---------------------------------------------------------------------------

def test_f7_lease_lost_wins_when_callback_also_raises(
    db, base_request, monkeypatch
) -> None:
    """LeaseLost must be raised even when the materializer callback also raises.

    Before fix: RuntimeError propagates → caught by _materialize_locked as FAILED.
    After fix:  LeaseLost raised in finally → re-raised → LeaseLost propagates.
    """
    lease_stolen = threading.Event()

    def renew_steals(*_a, **_kw) -> bool:
        lease_stolen.set()
        return False  # lease not renewed → lost

    def materializer_that_crashes(operation):
        # Wait until the heartbeat steals the lease, then raise
        assert lease_stolen.wait(timeout=3), "heartbeat did not steal lease in 3s"
        time.sleep(0.05)  # give heartbeat time to call lost.set()
        raise RuntimeError("materializer crashed after lease stolen")

    command = IngestionCommand(
        IngestionOperationRepository(db),
        write_queryable=lambda *_: ["fact-fast"],
        materialize=materializer_that_crashes,
        lease_seconds=1,
    )
    receipt = command.submit(base_request)
    monkeypatch.setattr(command.repository, "renew_enriching_lease", renew_steals)

    with pytest.raises(LeaseLost):
        command.materialize(receipt.operation_id)


def test_f7_normal_lease_loss_without_callback_raise_still_works(
    db, base_request, monkeypatch
) -> None:
    """Existing normal LeaseLost path (callback succeeds) must remain intact."""
    renewal_attempted = threading.Event()

    def renew_steals(*_a, **_kw) -> bool:
        renewal_attempted.set()
        return False

    def materializer_waits(operation):
        assert renewal_attempted.wait(timeout=3)
        return ["fact-fast"]

    command = IngestionCommand(
        IngestionOperationRepository(db),
        write_queryable=lambda *_: ["fact-fast"],
        materialize=materializer_waits,
        lease_seconds=1,
    )
    receipt = command.submit(base_request)
    monkeypatch.setattr(command.repository, "renew_enriching_lease", renew_steals)

    with pytest.raises(LeaseLost):
        command.materialize(receipt.operation_id)


# ---------------------------------------------------------------------------
# F9: heartbeat thread dies on non-sqlite3 exception → lost must be set
# ---------------------------------------------------------------------------

def test_f9_lost_set_when_heartbeat_raises_non_sqlite_error(
    db, base_request, monkeypatch
) -> None:
    """lost.set() must be called when the heartbeat thread raises AttributeError.

    Before fix: thread dies silently, lost stays False, materializer returns OK.
    After fix:  thread calls lost.set() in except, main thread raises LeaseLost.
    """
    heartbeat_ran = threading.Event()

    def renew_raises_attr_error(*_a, **_kw):
        heartbeat_ran.set()
        raise AttributeError("repository.db is None — unexpected disconnect")

    def materializer_that_succeeds(operation):
        # Wait until the heartbeat has run and died
        assert heartbeat_ran.wait(timeout=3), "heartbeat did not run in 3s"
        time.sleep(0.1)  # ensure AttributeError propagated and lost.set() called
        return ["fact-fast"]

    command = IngestionCommand(
        IngestionOperationRepository(db),
        write_queryable=lambda *_: ["fact-fast"],
        materialize=materializer_that_succeeds,
        lease_seconds=1,
    )
    receipt = command.submit(base_request)
    monkeypatch.setattr(command.repository, "renew_enriching_lease", renew_raises_attr_error)

    with pytest.raises(LeaseLost):
        command.materialize(receipt.operation_id)
