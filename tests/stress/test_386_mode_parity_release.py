# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""3.8.6 release contract: one canonical writer across Modes A, B, and C.

This is deliberately a provider-free harness.  Mode configuration is real,
but no Ollama or cloud process is constructed: immediate admission must be
identical regardless of a mode's eventual enrichment provider.
"""

from __future__ import annotations

import multiprocessing
import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

_ACTOR = "release-mode-parity-daemon"
_WRITE_ACTIONS = {
    sqlite3.SQLITE_ALTER_TABLE,
    sqlite3.SQLITE_CREATE_INDEX,
    sqlite3.SQLITE_CREATE_TABLE,
    sqlite3.SQLITE_DELETE,
    sqlite3.SQLITE_DROP_INDEX,
    sqlite3.SQLITE_DROP_TABLE,
    sqlite3.SQLITE_INSERT,
    sqlite3.SQLITE_UPDATE,
}


def _bootstrap_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Create one real coordinator/runtime against a fixture-only data root."""
    from superlocalmemory.core.engine_ingestion import build_immediate_admission_handler
    from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage.migrations import (
        M018_ingestion_operations,
        M032_write_coordinator_admission,
        M033_projection_transactions,  # required: _record_projection_obligations fail-closed
        M034_obligation_integrity,      # required: obligation FK + index
    )

    data_dir = tmp_path / "isolated-slm-data"
    data_dir.mkdir()
    monkeypatch.setenv("SLM_DATA_DIR", str(data_dir))
    db = DatabaseManager(data_dir / "memory.db")
    db.initialize(schema)
    with db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
        M032_write_coordinator_admission.apply(conn)
        M033_projection_transactions.apply(conn)
        M034_obligation_integrity.apply(conn)
    for profile_id in ("mode-a", "mode-b", "mode-c"):
        db.execute(
            "INSERT INTO profiles(profile_id, name) VALUES (?, ?)",
            (profile_id, profile_id),
        )
    runtime = CanonicalRememberRuntime(
        db=db,
        profile_id="mode-a",
        writer=build_immediate_admission_handler(db, profile_id="mode-a"),
        journal_path=data_dir / "admission_journal.db",
        owner_id="mode-parity-release",
    )
    runtime.start()
    return data_dir, db, runtime


def _actor(profile_id: str):
    from superlocalmemory.storage.admission_journal import Actor

    return Actor(_ACTOR, frozenset({profile_id}), frozenset({"personal"}))


def _remember(profile_id: str, mode_name: str):
    from superlocalmemory.storage.admission_journal import RememberRequest

    token = f"releaseparity{mode_name}"
    return RememberRequest(
        content=(
            f"{token} witness: profile {profile_id} preserves immediate remember "
            "and pure FTS recall through a canonical writer."
        ),
        profile_id=profile_id,
        source_type="release-mode-parity",
        idempotency_key=f"release-mode-parity:{mode_name}",
        trusted_actor_id=_ACTOR,
        session_id="release-mode-parity",
        session_date="2026-07-27",
    )


def _pure_fts_recall(db_path: Path, profile_id: str, token: str) -> list[str]:
    """Run a recall-shaped FTS query under an authorizer that rejects writes."""
    from superlocalmemory.storage.read_connection import ReadConnectionFactory

    writes: list[int] = []
    with ReadConnectionFactory(db_path, timeout_ms=250).snapshot() as conn:
        # Construct the FTS5 virtual table before installing the diagnostic
        # authorizer. Some Linux SQLite builds perform internal schema work
        # during the first vtable access and report only "vtable constructor
        # failed" when an authorizer is already present. The connection is
        # physically mode=ro and query_only throughout this warm-up.
        conn.execute("SELECT fact_id FROM atomic_facts_fts LIMIT 0").fetchall()
        conn.set_authorizer(
            lambda action, _arg1, _arg2, _db, _source: (
                writes.append(action) or sqlite3.SQLITE_DENY
                if action in _WRITE_ACTIONS else sqlite3.SQLITE_OK
            ),
        )
        assert conn.execute("PRAGMA query_only").fetchone()[0] == 1
        rows = conn.execute(
            "SELECT facts.fact_id FROM atomic_facts_fts AS fts "
            "JOIN atomic_facts AS facts ON facts.fact_id = fts.fact_id "
            "WHERE fts.atomic_facts_fts MATCH ? AND facts.profile_id = ? "
            "ORDER BY fts.rank",
            (f'"{token}"', profile_id),
        ).fetchall()
    assert writes == []
    return [str(row[0]) for row in rows]


def test_release_386_mode_switches_share_one_writer_and_preserve_recall_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A/B/C rebind without recreating the daemon writer or crossing profiles."""
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.storage.models import Mode

    data_dir, db, runtime = _bootstrap_runtime(tmp_path, monkeypatch)
    initial_worker = runtime.coordinator._worker
    assert initial_worker is not None and initial_worker.is_alive()
    initial_children = multiprocessing.active_children()
    facts: dict[str, str] = {}
    modes = ((Mode.A, "mode-a"), (Mode.B, "mode-b"), (Mode.C, "mode-c"))
    try:
        for mode, profile_id in modes:
            config = SLMConfig.for_mode(mode, base_dir=data_dir)
            # The runtime must not instantiate the configured Ollama/cloud
            # provider merely to admit immediate SQLite evidence.
            runtime.rebind_engine(SimpleNamespace(
                _db=db,
                _profile_id=profile_id,
                _config=config,
            ))
            assert runtime.coordinator._worker is initial_worker
            assert runtime.coordinator._worker.is_alive()
            assert runtime._profile_id == profile_id
            assert config.mode is mode

            receipt = runtime.remember(
                _remember(profile_id, mode.value),
                _actor(profile_id),
                deadline_ms=2_000,
            ).payload
            assert receipt["status"] == "queryable"
            facts[profile_id] = str(receipt["fact_ids"][0])

            recalled = _pure_fts_recall(
                db.db_path, profile_id, f"releaseparity{mode.value}",
            )
            assert recalled == [facts[profile_id]]

        for mode, profile_id in modes:
            own = _pure_fts_recall(
                db.db_path, profile_id, f"releaseparity{mode.value}",
            )
            assert own == [facts[profile_id]]
            for other_mode, other_profile in modes:
                if other_profile != profile_id:
                    assert _pure_fts_recall(
                        db.db_path, profile_id, f"releaseparity{other_mode.value}",
                    ) == []

        assert len(multiprocessing.active_children()) == len(initial_children)
        active_writer_threads = [
            thread for thread in threading.enumerate()
            if thread.name.startswith("slm-write-coordinator-")
        ]
        assert active_writer_threads == [initial_worker]
    finally:
        runtime.stop()

    assert not initial_worker.is_alive()
    assert not [
        thread for thread in threading.enumerate()
        if thread.name.startswith("slm-write-coordinator-")
    ]
