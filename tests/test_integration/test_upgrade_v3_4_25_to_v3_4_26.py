"""End-to-end upgrade path — v3.4.25 layout on disk meets v3.4.26 code.

Simulates what an existing user hits on ``pip install -U`` /
``npm install -g``:

  1. Data dir already contains the v3.4.25 shape (memory.db + friends).
  2. v3.4.26 code imports run.
  3. Upgrade banner fires exactly once.
  4. ``migrate_if_safe`` applies when no daemon is running; defers
     otherwise.
  5. memory.db content is byte-identical afterwards (additive migration
     contract).
  6. New recall_queue.db and ``.slm-v3.4.26-ready`` marker are created.
  7. Second invocation is silent (banner consumed, marker present).
"""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest


MEMORY_DB_MAGIC = b"SQLite format 3\x00"


def _fake_v3_4_25_memory_db(path: Path) -> None:
    """Create a plausible v3.4.25 memory.db — a real SQLite file with a
    couple of tables and rows that the migration must preserve."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY, content TEXT)")
        conn.execute("INSERT INTO atomic_facts VALUES ('f-pre-upgrade', 'sacred fact')")
        conn.commit()
    finally:
        conn.close()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def v3_4_25_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    _fake_v3_4_25_memory_db(tmp_path / "memory.db")
    # Pre-v3.4.26 user has no .version marker yet — the banner writes it.
    return tmp_path


class TestUpgradePath:
    def test_banner_fires_once_then_silent(self, v3_4_25_data_dir, capsys):
        from superlocalmemory.cli.version_banner import (
            check_and_emit_upgrade_banner,
        )
        first = check_and_emit_upgrade_banner("3.4.26")
        # v3.6.13: banner emits to STDERR (stdout reserved for MCP JSON-RPC).
        err1 = capsys.readouterr().err
        assert first is True
        assert "3.4.26" in err1

        second = check_and_emit_upgrade_banner("3.4.26")
        err2 = capsys.readouterr().err
        assert second is False
        assert err2 == ""

    def test_memory_db_bytes_unchanged_through_migration(
        self, v3_4_25_data_dir, monkeypatch,
    ):
        """The migration is contractually additive on memory.db —
        5,260+ live users depend on their facts surviving byte-identical.
        """
        import superlocalmemory.migrations.v3_4_25_to_v3_4_26 as mod
        monkeypatch.setattr(mod, "_daemon_running", lambda: False)

        before = _sha256(v3_4_25_data_dir / "memory.db")
        res = mod.migrate_if_safe(v3_4_25_data_dir)
        after = _sha256(v3_4_25_data_dir / "memory.db")

        assert res["status"] == "applied"
        assert before == after, "memory.db bytes changed — migration must be additive"

    def test_new_artifacts_created(self, v3_4_25_data_dir, monkeypatch):
        import superlocalmemory.migrations.v3_4_25_to_v3_4_26 as mod
        monkeypatch.setattr(mod, "_daemon_running", lambda: False)

        mod.migrate_if_safe(v3_4_25_data_dir)

        assert (v3_4_25_data_dir / "recall_queue.db").exists()
        assert (v3_4_25_data_dir / ".slm-v3.4.26-ready").exists()

    def test_rerunnable_after_apply(self, v3_4_25_data_dir, monkeypatch):
        import superlocalmemory.migrations.v3_4_25_to_v3_4_26 as mod
        monkeypatch.setattr(mod, "_daemon_running", lambda: False)

        first = mod.migrate_if_safe(v3_4_25_data_dir)
        second = mod.migrate_if_safe(v3_4_25_data_dir)

        assert first["status"] == "applied"
        assert second["status"] == "already_applied"

    def test_deferred_when_daemon_up_leaves_memory_db_intact(
        self, v3_4_25_data_dir, monkeypatch,
    ):
        import superlocalmemory.migrations.v3_4_25_to_v3_4_26 as mod
        monkeypatch.setattr(mod, "_daemon_running", lambda: True)

        before = _sha256(v3_4_25_data_dir / "memory.db")
        res = mod.migrate_if_safe(v3_4_25_data_dir)
        after = _sha256(v3_4_25_data_dir / "memory.db")

        assert res["status"] == "deferred"
        assert before == after
        # No new artifacts materialized while deferred.
        assert not (v3_4_25_data_dir / "recall_queue.db").exists()
        assert not (v3_4_25_data_dir / ".slm-v3.4.26-ready").exists()

    def test_facts_survive_migration_roundtrip(
        self, v3_4_25_data_dir, monkeypatch,
    ):
        """Read-back check: the pre-existing fact must still be queryable
        after the migration completes."""
        import superlocalmemory.migrations.v3_4_25_to_v3_4_26 as mod
        monkeypatch.setattr(mod, "_daemon_running", lambda: False)

        mod.migrate_if_safe(v3_4_25_data_dir)

        conn = sqlite3.connect(str(v3_4_25_data_dir / "memory.db"))
        try:
            rows = conn.execute(
                "SELECT content FROM atomic_facts WHERE fact_id = ?",
                ("f-pre-upgrade",),
            ).fetchall()
        finally:
            conn.close()
        assert rows == [("sacred fact",)]
