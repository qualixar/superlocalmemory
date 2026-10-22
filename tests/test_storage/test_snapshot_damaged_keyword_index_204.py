"""#204: a damaged keyword index must not block the pre-migration snapshot."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.storage.backup import SnapshotUnusableError, _backup_via_sqlite_api


def _store(path: Path) -> None:
    c = sqlite3.connect(path)
    c.executescript(
        "CREATE TABLE atomic_facts(fact_id TEXT, content TEXT);"
        "CREATE VIRTUAL TABLE atomic_facts_fts USING fts5("
        "content, content='atomic_facts', content_rowid='rowid');")
    c.executemany("INSERT INTO atomic_facts VALUES(?,?)",
                  [(str(i), f"fact {i} alpha beta gamma") for i in range(300)])
    c.execute("INSERT INTO atomic_facts_fts(atomic_facts_fts) VALUES('rebuild')")
    c.commit()
    c.close()


def _damage_index(path: Path) -> None:
    c = sqlite3.connect(path)
    c.execute("DELETE FROM atomic_facts_fts_data WHERE id > 10 AND rowid % 2 = 0")
    c.commit()
    # SQLite 3.54 reports this damage as an unreadable blob instead of a malformed
    # index; integrity_diagnosis accepts both.
    report = c.execute("PRAGMA quick_check").fetchone()[0]
    assert "malformed inverted index" in report or "fts5: corruption found" in report
    c.close()


def test_snapshot_rebuilds_a_damaged_keyword_index(tmp_path: Path) -> None:
    src, dest = tmp_path / "memory.db", tmp_path / "snap" / "memory.db"
    _store(src)
    _damage_index(src)
    _backup_via_sqlite_api(src, dest)
    snap = sqlite3.connect(dest)
    assert snap.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    assert snap.execute("SELECT count(*) FROM atomic_facts").fetchone()[0] == 300
    assert snap.execute(
        "SELECT count(*) FROM atomic_facts_fts WHERE atomic_facts_fts MATCH 'alpha'"
    ).fetchone()[0] == 300
    snap.close()
    # The live store is reported, never silently changed.
    live = sqlite3.connect(src)
    assert live.execute("PRAGMA quick_check").fetchone()[0] != "ok"
    live.close()


def test_other_damage_is_still_refused(tmp_path: Path, monkeypatch) -> None:
    import superlocalmemory.storage.backup as backup

    src, dest = tmp_path / "memory.db", tmp_path / "snap" / "memory.db"
    _store(src)
    monkeypatch.setattr(backup, "_quick_check",
                        lambda _p: ["*** in database main *** Page 7: btreeInitPage() returns error code 11"])
    with pytest.raises(SnapshotUnusableError):
        _backup_via_sqlite_api(src, dest)
    assert not dest.exists()
