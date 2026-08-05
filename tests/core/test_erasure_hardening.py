# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# DB factory helpers
# ---------------------------------------------------------------------------

def _fresh_db(tmp_path: Path):
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage.migrations import (
        M033_projection_transactions,
        M035_erasure_receipts,
    )

    db_path = tmp_path / "memory.db"
    conn = sqlite3.connect(db_path)
    try:
        M033_projection_transactions.apply(conn)
        M035_erasure_receipts.apply(conn)
        conn.commit()
    finally:
        conn.close()

    db = DatabaseManager(db_path)
    db.initialize(schema)
    db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('p1', 'p1')")
    return db, db_path


def _gdpr_db(tmp_path: Path):
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager

    db_path = tmp_path / "gdpr.db"
    db = DatabaseManager(db_path)
    db.initialize(schema)
    db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('p1', 'p1')")
    return db


# ---------------------------------------------------------------------------
# H1 — raw_vector_present detects map-only residue
# ---------------------------------------------------------------------------

def test_raw_vector_present_detects_map_residue(tmp_path: Path) -> None:
    """H1: raw_vector_present returns True when vector_row_map has entry but
    fact_embeddings (sqlite-vec) row was removed — catches orphan map records."""
    from superlocalmemory.retrieval.vector_store import VectorStore, VectorStoreConfig

    db_path = tmp_path / "vs.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE vector_row_map "
            "(fact_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL, vec_rowid INTEGER NOT NULL)"
        )
        conn.execute("INSERT INTO vector_row_map VALUES ('f1', 'p1', 42)")
        conn.commit()
    finally:
        conn.close()

    store = VectorStore(db_path, VectorStoreConfig())
    # This test targets the map-only fallback residue detection (the path used
    # when sqlite-vec is unavailable): an orphan vector_row_map entry with no
    # backing fact_embeddings row must be reported present so GC can catch it.
    # Force the fallback deterministically so the test holds on hosts whose
    # Python HAS sqlite-vec (where the real vec0 path would instead require an
    # actual embedding row). The sqlite-vec-present semantics are covered by the
    # store's own vec0 tests.
    store._available = False
    assert store.raw_vector_present("f1") is True
    assert store.raw_vector_present("f99") is False


def test_raw_vector_present_false_when_map_absent(tmp_path: Path) -> None:
    """H1: raw_vector_present returns False gracefully when vector_row_map table
    does not exist yet — no exception raised."""
    from superlocalmemory.retrieval.vector_store import VectorStore, VectorStoreConfig

    db_path = tmp_path / "vs_empty.db"
    db_path.touch()

    store = VectorStore(db_path, VectorStoreConfig())
    assert store.raw_vector_present("f1") is False


# ---------------------------------------------------------------------------
# H2 — tombstone UPSERT provenance repair
# ---------------------------------------------------------------------------

def test_tombstone_upsert_preserves_existing_memory_id(tmp_path: Path) -> None:
    """H2: ON CONFLICT DO UPDATE preserves the first non-null memory_id."""
    from superlocalmemory.core.transactions.erasure import write_tombstones

    db, _ = _fresh_db(tmp_path)
    write_tombstones(db, "p1", ("f1",), "erase-1", time.time(), "mem-A")
    write_tombstones(db, "p1", ("f1",), "erase-2", time.time(), "mem-B")

    with db.raw_connection() as conn:
        row = conn.execute(
            "SELECT memory_id FROM projection_tombstones WHERE profile_id='p1' AND fact_id='f1'"
        ).fetchone()
    assert row is not None
    assert row[0] == "mem-A"


def test_tombstone_upsert_fills_null_memory_id(tmp_path: Path) -> None:
    """H2: COALESCE fills in memory_id when the existing row stored NULL."""
    from superlocalmemory.core.transactions.erasure import write_tombstones

    db, _ = _fresh_db(tmp_path)
    write_tombstones(db, "p1", ("f1",), "erase-1", time.time(), None)
    write_tombstones(db, "p1", ("f1",), "erase-2", time.time(), "mem-A")

    with db.raw_connection() as conn:
        row = conn.execute(
            "SELECT memory_id FROM projection_tombstones WHERE profile_id='p1' AND fact_id='f1'"
        ).fetchone()
    assert row is not None
    assert row[0] == "mem-A"


def test_tombstone_conflict_logs_warning(tmp_path: Path, caplog) -> None:
    """H2: write_tombstones logs a WARNING when two different non-null memory_ids
    are written for the same (profile_id, fact_id) pair."""
    from superlocalmemory.core.transactions.erasure import write_tombstones

    db, _ = _fresh_db(tmp_path)
    write_tombstones(db, "p1", ("f1",), "erase-1", time.time(), "mem-A")

    with caplog.at_level(logging.WARNING, logger="superlocalmemory.core.transactions.erasure"):
        write_tombstones(db, "p1", ("f1",), "erase-2", time.time(), "mem-B")

    assert any("provenance conflict" in m for m in caplog.messages)


# ---------------------------------------------------------------------------
# H3 — _TombstoneReadError fail-closed
# ---------------------------------------------------------------------------

def test_tombstone_read_error_raised_on_db_failure() -> None:
    """H3: _fact_is_tombstoned raises _TombstoneReadError when DB execute fails."""
    from superlocalmemory.core.store_pipeline import (  # type: ignore[attr-defined]
        _fact_is_tombstoned,
        _TombstoneReadError,
    )

    class _BrokenDB:
        def execute(self, sql, params=None):
            raise sqlite3.OperationalError("simulated disk I/O error")

    with pytest.raises(_TombstoneReadError):
        _fact_is_tombstoned(_BrokenDB(), "p1", "f1")


def test_drop_resurrected_excludes_fact_on_read_error() -> None:
    """H3: _drop_resurrected_facts excludes the fact from survivors (fail-closed)
    when the tombstone check raises."""
    from superlocalmemory.core.store_pipeline import (  # type: ignore[attr-defined]
        _drop_resurrected_facts,
    )

    class _BrokenDB:
        def execute(self, sql, params=None):
            raise sqlite3.OperationalError("simulated disk I/O error")

        def delete_bm25_tokens_for_fact(self, *a, **k):
            pass

        def delete_fact(self, *a, **k):
            pass

    survivors = _drop_resurrected_facts(
        _BrokenDB(), "p1", ["f1"], None, None, None,
    )
    assert "f1" not in survivors


def test_drop_resurrected_logs_error_when_vector_residue_remains(
    tmp_path: Path, caplog
) -> None:
    """H3: logs an error when vector_row_map entry persists after resurrection undo."""
    from superlocalmemory.core.store_pipeline import (  # type: ignore[attr-defined]
        _drop_resurrected_facts,
    )

    db, _ = _fresh_db(tmp_path)

    with db.raw_connection() as conn:
        conn.execute(
            "INSERT INTO projection_tombstones (profile_id, fact_id, erasure_id, created_at) "
            "VALUES ('p1', 'f1', 'e1', ?)", (time.time(),)
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS vector_row_map "
            "(fact_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL, vec_rowid INTEGER NOT NULL)"
        )
        conn.execute("INSERT INTO vector_row_map VALUES ('f1', 'p1', 99)")
        conn.commit()

    with caplog.at_level(logging.ERROR, logger="superlocalmemory.core.store_pipeline"):
        survivors = _drop_resurrected_facts(db, "p1", ["f1"], None, None, None)

    assert "f1" not in survivors
    assert any("residue" in m for m in caplog.messages)


# ---------------------------------------------------------------------------
# H4 — ANN purge falls back to DB fact_ids when vector store unavailable
# ---------------------------------------------------------------------------

def test_purge_vector_ann_calls_ann_remove_via_db_fallback(tmp_path: Path) -> None:
    """H4: ANN.remove is called for DB-sourced fact_ids when vector store is
    not available — ensures ANN is not silently skipped."""
    from superlocalmemory.compliance.gdpr import GDPRCompliance
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager

    db_path = tmp_path / "mem.db"
    db = DatabaseManager(db_path)
    db.initialize(schema)
    db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('p1', 'p1')")
    db.execute("INSERT INTO memories (memory_id, profile_id, content) VALUES ('m1', 'p1', 'x')")
    db.execute(
        "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content) "
        "VALUES ('fid1', 'm1', 'p1', 'test fact')"
    )

    ann_calls: list[str] = []
    mock_ann = MagicMock()
    mock_ann.remove.side_effect = lambda fid: ann_calls.append(fid)

    mock_store = MagicMock()
    mock_store.available = False

    mock_engine = MagicMock()
    mock_engine._vector_store = mock_store
    mock_engine._ann_index = mock_ann

    compliance = GDPRCompliance(db, engine=mock_engine)
    compliance._purge_vector_and_ann("p1")

    assert "fid1" in ann_calls


def test_forget_profile_audit_completion_failure_surfaced(tmp_path: Path) -> None:
    """H4: audit_completion_failed key appears in counts when the completion
    AuditChain.log call raises after all deletions are done."""
    from superlocalmemory.compliance.gdpr import GDPRCompliance

    db = _gdpr_db(tmp_path)

    call_count = [0]

    def _counting_log(*a, **kw):
        call_count[0] += 1
        if call_count[0] >= 2:
            raise RuntimeError("chain write failed on completion")

    with patch("superlocalmemory.compliance.audit.AuditChain") as MockChain, \
         patch("superlocalmemory.infra.data_root.state_path", return_value=tmp_path / "audit.db"):
        mock_instance = MagicMock()
        mock_instance.log.side_effect = _counting_log
        MockChain.return_value = mock_instance

        compliance = GDPRCompliance(db, engine=None)
        counts = compliance.forget_profile("p1")

    assert counts.get("audit_completion_failed") == 1


# ---------------------------------------------------------------------------
# MED-6 — audit failure surfaced in returned dict
# ---------------------------------------------------------------------------

def test_forget_profile_audit_request_failure_in_counts(tmp_path: Path) -> None:
    """MED-6: audit_request_failed appears in forget_profile counts dict when
    AuditChain.log raises on the initial request audit."""
    from superlocalmemory.compliance.gdpr import GDPRCompliance

    db = _gdpr_db(tmp_path)

    with patch("superlocalmemory.compliance.audit.AuditChain") as MockChain, \
         patch("superlocalmemory.infra.data_root.state_path", return_value=tmp_path / "audit.db"):
        mock_instance = MagicMock()
        mock_instance.log.side_effect = RuntimeError("chain broken")
        MockChain.return_value = mock_instance

        compliance = GDPRCompliance(db, engine=None)
        counts = compliance.forget_profile("p1")

    assert counts.get("audit_request_failed") == 1


def test_forget_entity_audit_failure_in_result(tmp_path: Path) -> None:
    """MED-6: audit_request_failed appears in forget_entity result dict when
    AuditChain.log raises, even when entity is not found."""
    from superlocalmemory.compliance.gdpr import GDPRCompliance

    db = _gdpr_db(tmp_path)

    with patch("superlocalmemory.compliance.audit.AuditChain") as MockChain, \
         patch("superlocalmemory.infra.data_root.state_path", return_value=tmp_path / "audit.db"):
        mock_instance = MagicMock()
        mock_instance.log.side_effect = RuntimeError("chain broken")
        MockChain.return_value = mock_instance

        compliance = GDPRCompliance(db, engine=None)
        result = compliance.forget_entity("nonexistent-entity", "p1")

    assert result.get("audit_request_failed") == 1
    assert result.get("found") is False
