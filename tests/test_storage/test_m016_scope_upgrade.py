"""Regression: M016 scope migration must survive the EXISTING-DB upgrade path.

The original PR put scope indexes in schema.create_all_tables(), which runs on
every boot. On an existing DB (tables predate the scope column) the boot-time
`CREATE INDEX ... ON memories(scope)` failed with 'no such column: scope' and
aborted schema setup BEFORE the deferred M016 migration could add the column —
bricking the upgrade for every existing user. The indexes now live in M016's
idempotent apply(), and schema.py keeps only the scope columns (fresh installs).
"""
import sqlite3

from superlocalmemory.storage import schema
from superlocalmemory.storage.migrations import M016_add_scope_support as M016


def _existing_pre_scope_db() -> sqlite3.Connection:
    """A DB whose memories table predates the scope column (an upgrading user)."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE memories (memory_id TEXT PRIMARY KEY, "
        "profile_id TEXT NOT NULL DEFAULT 'default', content TEXT, "
        "session_id TEXT NOT NULL DEFAULT '', created_at TEXT DEFAULT (datetime('now')))"
    )
    conn.execute("INSERT INTO memories (memory_id, content) VALUES ('m1','hi')")
    conn.commit()
    return conn


def test_create_all_tables_does_not_break_on_existing_pre_scope_db():
    """The boot-time schema setup must not fail on a column-less existing table."""
    conn = _existing_pre_scope_db()
    schema.create_all_tables(conn)  # must not raise 'no such column: scope'
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "atomic_facts" in tables  # setup completed past memories


def test_m016_apply_adds_scope_and_indexes_to_existing_tables():
    conn = _existing_pre_scope_db()
    schema.create_all_tables(conn)
    M016.apply(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)")}
    idx = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    assert "scope" in cols and "shared_with" in cols
    assert "idx_memories_scope" in idx and "idx_memories_profile_scope" in idx
    # existing row preserved + defaulted
    row = conn.execute("SELECT scope, shared_with FROM memories WHERE memory_id='m1'").fetchone()
    assert row == ("personal", None)


def test_m016_apply_is_idempotent():
    conn = _existing_pre_scope_db()
    schema.create_all_tables(conn)
    M016.apply(conn)
    M016.apply(conn)  # second run must not raise (duplicate column / index)
    assert M016.verify(conn) is True


def test_m016_apply_tolerates_missing_tables():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE atomic_facts (id TEXT, profile_id TEXT)")
    # only atomic_facts exists; the other 4 tables are absent
    M016.apply(conn)  # must not raise 'no such table: memories'
    cols = {r[1] for r in conn.execute("PRAGMA table_info(atomic_facts)")}
    assert "scope" in cols


def test_m016_verify_requires_column_and_index():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE atomic_facts (id TEXT, profile_id TEXT)")
    assert M016.verify(conn) is False          # no scope yet
    conn.execute("ALTER TABLE atomic_facts ADD COLUMN scope TEXT DEFAULT 'personal'")
    assert M016.verify(conn) is False          # column but no index -> still runs
    conn.execute("CREATE INDEX idx_atomic_facts_scope ON atomic_facts(scope)")
    assert M016.verify(conn) is True


def test_fresh_install_gets_scope_columns_in_schema():
    """schema.py still creates the scope COLUMNS (just not the indexes)."""
    conn = sqlite3.connect(":memory:")
    schema.create_all_tables(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(atomic_facts)")}
    assert "scope" in cols and "shared_with" in cols
