"""Tests for V2 to V3 Migration -- Task 18 of V3 build."""
import sqlite3
import pytest
from pathlib import Path
from superlocalmemory.storage.v2_migrator import V2Migrator


def _create_v2_db(path: Path, n_memories: int = 10):
    """Create a minimal V2-style database for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("""CREATE TABLE memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL,
        summary TEXT,
        project_path TEXT,
        tags TEXT,
        category TEXT,
        importance INTEGER DEFAULT 5,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        access_count INTEGER DEFAULT 0,
        lifecycle_state TEXT DEFAULT 'active',
        profile TEXT DEFAULT 'default'
    )""")
    conn.execute("""CREATE TABLE trust_scores (
        agent_id TEXT PRIMARY KEY,
        alpha REAL DEFAULT 1.0,
        beta_param REAL DEFAULT 1.0,
        total_signals INTEGER DEFAULT 0
    )""")
    for i in range(n_memories):
        conn.execute(
            "INSERT INTO memories (content, tags, profile) VALUES (?, ?, ?)",
            (f"Memory {i}: test content about topic {i % 5}", f"tag{i}", "default"),
        )
    conn.commit()
    conn.close()


@pytest.fixture
def migrator(tmp_path):
    # ``home`` selects the legacy V2 source only.  Pin the V3 destination
    # explicitly so migration tests are isolated from the process data root.
    return V2Migrator(
        home=tmp_path,
        v3_base=tmp_path / ".superlocalmemory",
    )


def test_detect_v2_when_exists(tmp_path, migrator):
    _create_v2_db(tmp_path / ".claude-memory" / "memory.db")
    assert migrator.detect_v2() == True


def test_detect_v2_when_missing(tmp_path, migrator):
    assert migrator.detect_v2() == False


def test_v2_stats(tmp_path, migrator):
    _create_v2_db(tmp_path / ".claude-memory" / "memory.db", n_memories=25)
    stats = migrator.get_v2_stats()
    assert stats["exists"] == True
    assert stats["memory_count"] == 25


def test_migrate_creates_v3_dir(tmp_path, migrator):
    _create_v2_db(tmp_path / ".claude-memory" / "memory.db")
    result = migrator.migrate()
    assert result["success"] == True
    assert (tmp_path / ".superlocalmemory").exists()


def test_migrate_creates_backup(tmp_path, migrator):
    _create_v2_db(tmp_path / ".claude-memory" / "memory.db")
    migrator.migrate()
    assert (tmp_path / ".superlocalmemory" / "memory-v2-backup.db").exists()


def test_migrate_extends_schema(tmp_path, migrator):
    _create_v2_db(tmp_path / ".claude-memory" / "memory.db")
    migrator.migrate()
    conn = sqlite3.connect(str(tmp_path / ".superlocalmemory" / "memory.db"))
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    conn.close()
    # V3 schema tables (created by schema.create_all_tables)
    assert "atomic_facts" in tables
    assert "canonical_entities" in tables
    assert "graph_edges" in tables
    assert "config" in tables


def test_migrate_preserves_data(tmp_path, migrator):
    _create_v2_db(tmp_path / ".claude-memory" / "memory.db", n_memories=50)
    migrator.migrate()
    conn = sqlite3.connect(str(tmp_path / ".superlocalmemory" / "memory.db"))
    count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    conn.close()
    assert count == 50


def test_is_already_migrated(tmp_path, migrator):
    _create_v2_db(tmp_path / ".claude-memory" / "memory.db")
    assert migrator.is_already_migrated() == False
    migrator.migrate()
    assert migrator.is_already_migrated() == True


def test_migrate_idempotent(tmp_path, migrator):
    _create_v2_db(tmp_path / ".claude-memory" / "memory.db")
    result1 = migrator.migrate()
    assert result1["success"] == True
    result2 = migrator.migrate()
    assert result2["success"] == True
    assert result2.get("message") == "Already migrated"


def test_rollback(tmp_path, migrator):
    _create_v2_db(tmp_path / ".claude-memory" / "memory.db", n_memories=10)
    migrator.migrate()
    result = migrator.rollback()
    assert result["success"] == True
    # Original .claude-memory should be restored (as dir, not symlink)
    assert (tmp_path / ".claude-memory").exists()


def test_migrate_no_v2(tmp_path, migrator):
    result = migrator.migrate()
    assert result["success"] == False
    assert "No V2" in result["error"]


def test_v2_stats_no_install(tmp_path, migrator):
    stats = migrator.get_v2_stats()
    assert stats["exists"] == False
