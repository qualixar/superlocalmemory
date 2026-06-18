"""Regression for issue #47: BackendOrchestrator used a non-existent
`self._db.conn` attribute, so the v3.4.5 migration (access_count_30d) silently
never ran and backend_status was never written — both swallowed by bare excepts.
Fixed by using DatabaseManager.execute() and the new raw_connection() helper.
"""
from pathlib import Path

import pytest

from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.core.backend_orchestrator import BackendOrchestrator


class _Cfg:
    def __init__(self, base):
        self.base_dir = base
        self.data_dir = base


def _db_with_core_tables(tmp_path: Path) -> DatabaseManager:
    db = DatabaseManager(tmp_path / "memory.db")
    with db.raw_connection() as conn:
        conn.execute("CREATE TABLE atomic_facts (id TEXT)")
        conn.execute("CREATE TABLE graph_edges (id TEXT)")
        conn.execute(
            "CREATE TABLE backend_status ("
            "backend_name TEXT PRIMARY KEY, status TEXT, record_count INTEGER, "
            "error_message TEXT, last_sync_at TEXT)"
        )
    return db


def test_raw_connection_commits_and_is_visible(tmp_path):
    db = DatabaseManager(tmp_path / "m.db")
    with db.raw_connection() as conn:
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.execute("INSERT INTO t VALUES (42)")
    # committed + closed; a fresh execute() sees it
    assert db.execute("SELECT x FROM t")[0][0] == 42


def test_raw_connection_rolls_back_on_error(tmp_path):
    db = DatabaseManager(tmp_path / "m.db")
    with db.raw_connection() as conn:
        conn.execute("CREATE TABLE t (x INTEGER)")
    with pytest.raises(ValueError):
        with db.raw_connection() as conn:
            conn.execute("INSERT INTO t VALUES (1)")
            raise ValueError("boom")
    # insert was rolled back
    assert db.execute("SELECT COUNT(*) FROM t")[0][0] == 0


def test_apply_schema_v345_actually_runs_migration(tmp_path):
    """#47: the access_count_30d column must now be added (was silently skipped)."""
    db = _db_with_core_tables(tmp_path)
    orch = BackendOrchestrator(_Cfg(tmp_path), db)
    orch._apply_schema_v345()
    cols = [r[1] for r in db.execute("PRAGMA table_info(atomic_facts)")]
    assert "access_count_30d" in cols


def test_update_status_writes_backend_status_row(tmp_path):
    """#47: _update_status must persist (was a no-op via the broken `.conn`)."""
    db = _db_with_core_tables(tmp_path)
    orch = BackendOrchestrator(_Cfg(tmp_path), db)
    orch._update_status("cozo", "active", 7)
    row = db.execute(
        "SELECT status, record_count FROM backend_status WHERE backend_name='cozo'"
    )
    assert row and row[0][0] == "active" and row[0][1] == 7


def test_database_manager_has_no_conn_attribute(tmp_path):
    """Guard: the bug was inventing a `.conn`; ensure callers don't reintroduce it."""
    db = DatabaseManager(tmp_path / "m.db")
    assert not hasattr(db, "conn")
