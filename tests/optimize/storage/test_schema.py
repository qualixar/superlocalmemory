"""LLD-00 §10.3 — schema tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def test_create_all_tables_idempotent(tmp_path: Path) -> None:
    from superlocalmemory.optimize.storage import schema
    conn = sqlite3.connect(str(tmp_path / "test.db"))
    schema.create_all_tables(conn)
    schema.create_all_tables(conn)  # second call must not raise
    conn.close()


def test_all_expected_tables_exist(tmp_path: Path) -> None:
    from superlocalmemory.optimize.storage import schema
    conn = sqlite3.connect(str(tmp_path / "test.db"))
    schema.create_all_tables(conn)
    conn.commit()
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    names = {r[0] for r in rows}
    conn.close()
    expected = set(schema.get_table_names())
    assert expected.issubset(names), f"missing tables: {expected - names}"


def test_no_memory_db_tables(tmp_path: Path) -> None:
    from superlocalmemory.optimize.storage import schema
    conn = sqlite3.connect(str(tmp_path / "test.db"))
    schema.create_all_tables(conn)
    conn.commit()
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    names = {r[0] for r in rows}
    conn.close()
    forbidden = {
        "memories", "atomic_facts", "profiles", "canonical_entities",
        "consolidation_log", "trust_scores", "bm25_tokens",
    }
    assert forbidden.isdisjoint(names), (
        f"Memory.db tables found in llmcache.db: {forbidden & names}"
    )


def test_metrics_row_seeded(tmp_path: Path) -> None:
    from superlocalmemory.optimize.storage import schema
    conn = sqlite3.connect(str(tmp_path / "test.db"))
    schema.create_all_tables(conn)
    conn.commit()
    row = conn.execute("SELECT id FROM llmcache_metrics WHERE id = 1").fetchone()
    conn.close()
    assert row is not None
    assert row[0] == 1


def test_drop_all_tables(tmp_path: Path) -> None:
    from superlocalmemory.optimize.storage import schema
    conn = sqlite3.connect(str(tmp_path / "test.db"))
    schema.create_all_tables(conn)
    conn.commit()
    schema.drop_all_tables(conn)
    conn.commit()
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    names = {r[0] for r in rows}
    conn.close()
    assert "llmcache_entries" not in names


# ---- assert_no_memory_db_tables ----

def test_assert_no_memory_db_tables_passes_on_clean_db(tmp_path: Path) -> None:
    """No forbidden tables present -> no exception."""
    from superlocalmemory.optimize.storage import schema
    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "clean.db"))
    conn.execute("CREATE TABLE foo (id INTEGER)")
    conn.commit()
    # Must not raise
    schema.assert_no_memory_db_tables(conn)
    conn.close()


def test_assert_no_memory_db_tables_raises_on_forbidden_table(tmp_path: Path) -> None:
    """Forbidden table present -> RuntimeError with table name."""
    from superlocalmemory.optimize.storage import schema
    import sqlite3
    import pytest
    conn = sqlite3.connect(str(tmp_path / "dirty.db"))
    conn.execute("CREATE TABLE atomic_facts (id TEXT)")
    conn.execute("CREATE TABLE profiles (id TEXT)")
    conn.commit()
    with pytest.raises(RuntimeError, match="ISOLATION VIOLATION"):
        schema.assert_no_memory_db_tables(conn)
    conn.close()
