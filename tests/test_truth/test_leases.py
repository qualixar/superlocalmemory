# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Leases: a memory saved with a source file knows when that file changed (D-38)."""

from __future__ import annotations

import json
import os
import sqlite3

import pytest

from superlocalmemory.truth import leases
from superlocalmemory.truth.invariants import open_read_only
from superlocalmemory.truth.slot_pass import LEASE_STATE, SlotPass
from tests.test_core.test_coverage_backfill import _db
from tests.test_truth.test_slot_pass import _fact, _flags


def _leased_fact(db, path) -> str:
    fact = _fact(db, "The daemon listens on port 8765", "2026-11-16T00:00:00Z")
    lease = leases.fingerprint(str(path)).as_meta()
    with db.raw_connection() as conn:
        conn.execute("UPDATE memories SET metadata_json = ? WHERE memory_id = "
                     "(SELECT memory_id FROM atomic_facts WHERE fact_id = ?)",
                     (json.dumps({leases.METADATA_KEY: lease}), fact))
    return fact


@pytest.fixture
def leased(tmp_path):
    db = _db(tmp_path / "memory.db")
    src = tmp_path / "config.yaml"
    src.write_text("port: 8765\n")
    fact = _leased_fact(db, src)
    SlotPass(db, _flags(**{"truth.slots": "shadow"}), state_dir=tmp_path).run()
    return db, fact, src, tmp_path


def _check(db, fact):
    conn = open_read_only(str(db.db_path))
    try:
        return leases.check_for_facts(conn, [fact]).get(fact)
    finally:
        conn.close()


def _bump(path, text=None):
    if text is not None:
        path.write_text(text)
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))


def test_the_slot_pass_gives_a_leased_memory_its_lease(leased) -> None:
    db, fact, src, tmp = leased
    row = db.execute("SELECT locator, state, fingerprint FROM memory_leases WHERE fact_id = ?",
                     (fact,))[0]
    assert row["locator"] == str(src.resolve()) and row["state"] == "fresh" and row["fingerprint"]
    assert json.loads((tmp / LEASE_STATE).read_text())["after_rowid"] > 0
    assert _check(db, fact) == {"state": "fresh", "file": str(src.resolve())}


def test_an_edited_file_shows_as_changed_on_the_next_recall(leased) -> None:
    db, fact, src, _tmp = leased
    _bump(src, "port: 9999\n")
    note = _check(db, fact)
    assert note["state"] == "changed" and note["next"] == leases.NEXT_ON_STALE


def test_a_touch_that_changes_nothing_is_not_a_change(leased) -> None:
    db, fact, src, _tmp = leased
    _bump(src)
    assert _check(db, fact)["state"] == "fresh"


def test_a_same_size_edit_is_caught_by_the_hash(leased) -> None:
    db, fact, src, _tmp = leased
    _bump(src, "port: 8766\n")
    assert _check(db, fact)["state"] == "changed"


def test_a_deleted_file_is_missing_and_the_memory_stays(leased) -> None:
    db, fact, src, _tmp = leased
    src.unlink()
    assert _check(db, fact)["state"] == "missing"
    assert db.execute("SELECT COUNT(*) AS n FROM atomic_facts WHERE fact_id = ?", (fact,))[0]["n"] == 1


def test_the_sweep_records_the_change_and_the_stale_list_shows_it(leased) -> None:
    db, fact, src, _tmp = leased
    _bump(src, "port: 9999\n")
    with db.raw_connection() as conn:
        counts = leases.sweep(conn)
        assert counts["changed"] == 1
        (item,) = leases.stale(conn, "default")
    assert item["fact_id"] == fact and item["state"] == "changed" and item["changed_at"]


def test_mark_still_true_takes_the_new_fingerprint(leased) -> None:
    db, fact, src, _tmp = leased
    _bump(src, "port: 9999\n")
    with db.raw_connection() as conn:
        leases.sweep(conn)
        assert leases.mark_still_true(conn, profile_id="default", fact_id=fact) == 1
        assert leases.stale(conn, "default") == []
    assert _check(db, fact)["state"] == "fresh"


def test_a_second_pass_does_not_rescan_or_duplicate(leased) -> None:
    db, fact, _src, tmp = leased
    SlotPass(db, _flags(**{"truth.slots": "shadow"}), state_dir=tmp).run()
    assert db.execute("SELECT COUNT(*) AS n FROM memory_leases")[0]["n"] == 1


@pytest.mark.parametrize("bad", ["", "relative/path.txt", "/no/such/file/anywhere", "/tmp\x00x"])
def test_a_bad_source_file_is_refused(bad) -> None:
    with pytest.raises(leases.SourceFileError):
        leases.fingerprint(bad)


def test_a_folder_is_not_a_source_file(tmp_path) -> None:
    with pytest.raises(leases.SourceFileError, match="regular file"):
        leases.fingerprint(str(tmp_path))


def test_a_store_without_the_lease_table_checks_nothing() -> None:
    conn = sqlite3.connect(":memory:")
    assert leases.check_for_facts(conn, ["f"]) == {}
    assert leases.due(conn) == [] and leases.stale(conn, "default") == []
    assert leases.materialize(conn, 7) == (0, 7)


def test_a_size_change_is_caught_by_stat_alone(leased, monkeypatch) -> None:
    db, fact, src, _tmp = leased
    src.write_text("port: 8765 and a longer line\n")

    def no_hash(*_a, **_k):
        raise AssertionError("a size change must not need a hash")

    monkeypatch.setattr(leases, "_hash", no_hash)
    assert _check(db, fact)["state"] == "changed"
