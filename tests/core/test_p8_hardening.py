"""P8 fail-closed hardening (crit/high/med findings from the Sol final audit).

Covers:
  F-66  PolicyDecision.annotations is immutable after construction.
  F-40  Audit-chain suffix truncation is detected via the external anchor.
  F-46  Tombstone provenance conflict fails the erasure closed.
  F-36  forget_profile aborts (deletes nothing) when the pre-deletion audit fails.
  F-35  forget_profile reports an explicit completeness flag.
  P-11  forget_profile is fail-closed on every purge/re-count error branch:
        a top-level vector-purge exception, a context-cache purge exception, and
        a residue re-count exception each force erasure_complete=0 (preprint audit).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch


def _gdpr_db(tmp_path: Path):
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager

    db = DatabaseManager(tmp_path / "gdpr.db")
    db.initialize(schema)
    db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('p1', 'p1')")
    db.execute(
        "CREATE TABLE IF NOT EXISTS projection_tombstones ("
        "profile_id TEXT NOT NULL, fact_id TEXT NOT NULL, erasure_id TEXT, "
        "memory_id TEXT, created_at REAL, PRIMARY KEY (profile_id, fact_id))"
    )
    return db


# --- F-66 --------------------------------------------------------------------

def test_policy_decision_annotations_immutable() -> None:
    from superlocalmemory.core.operation_policy_registry import PolicyDecision

    d = PolicyDecision(allowed=True, reason="allow", annotations={"audit": True})
    assert d.annotations["audit"] is True  # reads still work
    import pytest

    with pytest.raises(TypeError):
        d.annotations["audit"] = False  # mutation must fail


# --- F-40 --------------------------------------------------------------------

def test_audit_chain_truncation_detected(tmp_path: Path) -> None:
    from superlocalmemory.compliance.audit import AuditChain

    path = tmp_path / "audit_chain.db"
    chain = AuditChain(str(path))
    for i in range(3):
        chain.log("op", agent_id="a", profile_id="p", content_hash=f"h{i}")
    assert chain.verify_integrity() is True

    # Truncate the suffix by deleting the last row directly.
    conn = sqlite3.connect(str(path))
    conn.execute("DELETE FROM audit_chain WHERE id = (SELECT MAX(id) FROM audit_chain)")
    conn.commit()
    conn.close()

    # A fresh instance reads the persisted external anchor (count=3) and detects
    # that only 2 rows remain — truncation.
    assert AuditChain(str(path)).verify_integrity() is False


# --- F-46 --------------------------------------------------------------------

def test_tombstone_provenance_conflict_status(tmp_path: Path) -> None:
    from superlocalmemory.core.transactions.erasure import (
        TOMBSTONE_CONFLICT,
        TOMBSTONE_WRITTEN,
        write_tombstones_status,
    )

    db = _gdpr_db(tmp_path)
    # First write binds fact f1 -> memory_id A.
    s1 = write_tombstones_status(db, "p1", ("f1",), "erase-1", 1.0, memory_id="A")
    assert s1 == TOMBSTONE_WRITTEN
    # A conflicting memory_id for the same fact must fail closed.
    s2 = write_tombstones_status(db, "p1", ("f1",), "erase-2", 2.0, memory_id="B")
    assert s2 == TOMBSTONE_CONFLICT


# --- F-36 --------------------------------------------------------------------

def test_forget_profile_aborts_when_pre_audit_fails(tmp_path: Path) -> None:
    from superlocalmemory.compliance.gdpr import GDPRCompliance

    db = _gdpr_db(tmp_path)
    with patch("superlocalmemory.compliance.audit.AuditChain") as MockChain, \
         patch("superlocalmemory.infra.data_root.state_path",
               return_value=tmp_path / "audit.db"):
        inst = MagicMock()
        inst.log.side_effect = RuntimeError("chain broken")
        MockChain.return_value = inst

        counts = GDPRCompliance(db, engine=None).forget_profile("p1")

    assert counts.get("erasure_aborted") == 1
    # Fail-closed: the profile row must still exist (nothing was deleted).
    rows = db.execute("SELECT COUNT(*) AS c FROM profiles WHERE profile_id = 'p1'")
    assert int(dict(rows[0])["c"]) == 1


# --- F-35 --------------------------------------------------------------------

def test_forget_profile_reports_completeness(tmp_path: Path) -> None:
    from superlocalmemory.compliance.gdpr import GDPRCompliance

    db = _gdpr_db(tmp_path)
    # Mock the separate learning DB so its reset succeeds in the tmp env
    # (a real path failure would correctly mark the wipe incomplete).
    with patch("superlocalmemory.infra.data_root.state_path",
               return_value=tmp_path / "audit.db"), \
         patch("superlocalmemory.learning.database.LearningDatabase") as MockLDB:
        MockLDB.return_value = MagicMock()
        counts = GDPRCompliance(db, engine=None).forget_profile("p1")

    assert counts.get("erasure_aborted") is None  # audit succeeded -> not aborted
    assert counts.get("erasure_complete") == 1
    assert counts.get("residue_rows") == 0


# --- P-11: fail-closed on every purge / re-count error branch ----------------

class _RecountFailDB:
    """Delegates to a real DatabaseManager, but raises on the residue re-count
    COUNT(*) queries that run *after* VACUUM — simulating a table that cannot be
    re-counted. Earlier pre-deletion counts and the deletes themselves succeed."""

    def __init__(self, real) -> None:
        self._real = real
        self._vacuumed = False

    def __getattr__(self, name):  # delegate db_path, etc.
        return getattr(self._real, name)

    def execute(self, sql, *args, **kwargs):
        stripped = sql.strip().upper()
        if stripped.startswith("VACUUM"):
            self._vacuumed = True
            return self._real.execute(sql, *args, **kwargs)
        if self._vacuumed and stripped.startswith("SELECT COUNT(*)"):
            raise sqlite3.OperationalError("simulated residue re-count failure")
        return self._real.execute(sql, *args, **kwargs)


def _run_forget(db, tmp_path: Path):
    from superlocalmemory.compliance.gdpr import GDPRCompliance

    with patch("superlocalmemory.infra.data_root.state_path",
               return_value=tmp_path / "audit.db"), \
         patch("superlocalmemory.learning.database.LearningDatabase") as MockLDB:
        MockLDB.return_value = MagicMock()
        return GDPRCompliance(db, engine=None).forget_profile("p1")


def test_vector_purge_toplevel_failure_is_fail_closed(tmp_path: Path) -> None:
    from superlocalmemory.compliance.gdpr import GDPRCompliance

    db = _gdpr_db(tmp_path)
    with patch.object(GDPRCompliance, "_purge_vector_and_ann",
                      side_effect=RuntimeError("vector store down")), \
         patch("superlocalmemory.infra.data_root.state_path",
               return_value=tmp_path / "audit.db"), \
         patch("superlocalmemory.learning.database.LearningDatabase") as MockLDB:
        MockLDB.return_value = MagicMock()
        counts = GDPRCompliance(db, engine=None).forget_profile("p1")

    assert counts.get("vector_store_failures")            # explicit marker set
    assert counts.get("erasure_complete") == 0            # fail-closed


def test_context_cache_purge_failure_is_fail_closed(tmp_path: Path) -> None:
    db = _gdpr_db(tmp_path)
    with patch("superlocalmemory.core.context_cache.purge_profile_from_cache_db",
               side_effect=RuntimeError("cache db locked")):
        counts = _run_forget(db, tmp_path)

    assert counts.get("context_cache_failed") == 1
    assert counts.get("erasure_complete") == 0            # fail-closed


def test_residue_recount_failure_is_fail_closed(tmp_path: Path) -> None:
    db = _RecountFailDB(_gdpr_db(tmp_path))
    counts = _run_forget(db, tmp_path)

    assert counts.get("residue_recount_failed") == 1
    assert counts.get("erasure_complete") == 0            # fail-closed
