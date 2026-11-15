# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W0: the migration runner across the 4.2.30 branch and main's hot fixes (blueprint §11).

Hot fixes on ``main`` take M085 and up while 4.2.30 holds M069-M084 (D-02). Two
real runs pin what the runner does:

* a store migrated by a pre-4.2.30 build that also completed a hot-fix
  migration M085 still gets every 4.2.30 migration: pending is decided by name,
  never by "the highest number seen";
* a store migrated by 4.2.30 is refused by a 4.2.19 build (ceiling 68) until
  ``prepare_downgrade`` stamps it down, which every 4.2.30 migration allows
  (``DOWNGRADE_FLOOR`` 51); after that the 4.2.19 catalogue runs on it with no
  failure.
"""

from __future__ import annotations

import sqlite3

from superlocalmemory.storage import _schema_version as sv
from superlocalmemory.storage import migration_runner as mr
from tests.test_core.test_source_fidelity_guard import _db as _base_db

TRUTH = ("M069_memory_slots", "M070_memory_leases", "M071_superseded_validity_is_closed",
         "M072_judge_receipts", "M073_valid_until_source")
CEILING_4219 = 68


def _log(path) -> dict[str, str]:
    with sqlite3.connect(path) as conn:
        return dict(conn.execute("SELECT name, status FROM migration_log"))


def _pre_430(monkeypatch):
    """The catalogue and ceiling of a build without the 4.2.30 migrations."""
    monkeypatch.setattr(mr, "DEFERRED_MIGRATIONS",
                        [m for m in mr.DEFERRED_MIGRATIONS if m.name not in TRUTH])
    monkeypatch.setattr(sv, "SUPPORTED_SCHEMA_VERSION", CEILING_4219)


def _migrate(tmp_path):
    memory = tmp_path / "memory.db"
    _base_db(memory)
    learning = tmp_path / "learning.db"
    first = mr.apply_all(learning, memory)
    deferred = mr.apply_deferred(learning, memory)
    return learning, memory, first, deferred


def test_a_hot_fix_migration_does_not_hide_the_430_migrations(tmp_path, monkeypatch) -> None:
    with monkeypatch.context() as m:
        _pre_430(m)
        learning, memory, first, deferred = _migrate(tmp_path)
        assert not first["failed"] and not deferred["failed"]
    assert not set(TRUTH) & set(_log(memory))
    with sqlite3.connect(memory) as conn:  # main's hot fix, already applied
        known = {"name": "M085_hotfix", "status": "complete", "applied_at": "2026-12-01T00:00:00Z"}
        row = {}
        for _cid, col, _type, notnull, default, _pk in conn.execute(
                "PRAGMA table_info(migration_log)"):
            if col in known:
                row[col] = known[col]
            elif notnull and default is None:
                row[col] = "0" * 64 if "sha" in col else "hotfix"
        conn.execute(f"INSERT INTO migration_log ({', '.join(row)}) "
                     f"VALUES ({', '.join('?' * len(row))})", tuple(row.values()))
    again = mr.apply_all(learning, memory)
    late = mr.apply_deferred(learning, memory)
    assert not again["failed"] and not late["failed"], (again, late)
    log = _log(memory)
    assert all(log.get(name) == "complete" for name in TRUTH), log
    assert log["M085_hotfix"] == "complete"


def test_an_older_build_is_refused_until_the_store_is_prepared_for_it(tmp_path,
                                                                    monkeypatch) -> None:
    import pytest

    from superlocalmemory.storage._downgrade import prepare_downgrade

    learning, memory, first, deferred = _migrate(tmp_path)
    assert not first["failed"] and not deferred["failed"]
    assert all(_log(memory).get(name) == "complete" for name in TRUTH)
    with monkeypatch.context() as m:
        _pre_430(m)
        with pytest.raises(sv.SchemaVersionError):  # refusing is the safe default
            sv.check_version_or_raise(memory)
    prepare_downgrade(target_schema=CEILING_4219, requested_by="test", data_root=tmp_path,
                      memory_db=memory, learning_db=learning)
    with monkeypatch.context() as m:
        _pre_430(m)
        sv.check_version_or_raise(learning)
        sv.check_version_or_raise(memory)
        older = mr.apply_all(learning, memory)
        older_late = mr.apply_deferred(learning, memory)
        assert not older["failed"] and not older_late["failed"], (older, older_late)
