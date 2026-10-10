"""On SQLite older than 3.42 (Ubuntu 22.04 ships 3.37.2) FTS5 has no
``secure-delete`` option and answers the attempt with a bare "SQL logic error".
That used to escape as a warning at every daemon start and every erasure
(GitHub #153). It is a known limit with a fallback (``slm db repair`` purges
deleted words later), so it is reported once, quietly, and never raised."""
import logging
import sqlite3

from superlocalmemory.storage import fts_residue


def _store() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE atomic_facts_fts USING fts5(content)")
    return conn


def test_an_old_sqlite_is_reported_unsupported_without_trying_or_warning(monkeypatch, caplog):
    monkeypatch.setattr(fts_residue.sqlite3, "sqlite_version_info", (3, 37, 2))
    monkeypatch.setattr(fts_residue, "_old_sqlite_reported", False)
    conn = _store()
    calls = []
    conn.set_trace_callback(calls.append)
    with caplog.at_level(logging.DEBUG, logger=fts_residue.__name__):
        first = fts_residue.ensure_secure_delete(conn)
        second = fts_residue.ensure_secure_delete(conn)
    assert first["atomic_facts_fts"] == "unsupported" == second["atomic_facts_fts"]
    assert not any("secure-delete" in sql for sql in calls)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len([r for r in caplog.records if "3.42" in r.getMessage()]) == 1


def test_a_current_sqlite_still_turns_it_on():
    if not fts_residue.secure_delete_supported():
        return  # too old, or in the range where it corrupts: covered in test_fts_secure_delete_versions
    assert fts_residue.ensure_secure_delete(_store())["atomic_facts_fts"] in {"enabled", "on"}
