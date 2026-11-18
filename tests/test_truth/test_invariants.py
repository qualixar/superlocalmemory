# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm doctor --truth``: each invariant is reported when broken, and only then (R-27)."""

from __future__ import annotations

import sqlite3
from argparse import Namespace

import pytest

from superlocalmemory.truth import invariants as inv
from superlocalmemory.truth import slot_repair
from tests.test_truth.test_slot_pass import _reconcile, _resolve_explicit, _settled_by_a_person


def _report(db) -> dict[str, str]:
    with db.raw_connection() as conn:
        return {r.invariant: r.status for r in inv.check(conn)}


def _violated(db) -> set[str]:
    return {k for k, v in _report(db).items() if v == inv.VIOLATED}


@pytest.fixture
def slotted(store):
    db, old, new, tmp = store
    _resolve_explicit(db, old, "weekly", "2026-03-01")
    _resolve_explicit(db, new, "one big release", "2026-11-14T00:00:00+00:00")
    return db, old, new


def test_a_healthy_store_has_no_violations_and_names_what_it_did_not_check(slotted) -> None:
    db, _old, _new = slotted
    report = _report(db)
    assert _violated(db) == set()
    assert {k for k, v in report.items() if v == inv.OK} >= {
        "I-01", "I-02", "I-03", "I-04", "I-05", "I-06", "I-07", "I-08", "I-12", "I-16"}
    assert {k for k, v in report.items() if v == inv.NOT_CHECKED} == set(inv.NOT_CHECKED_HERE)


def test_i01_a_head_on_an_archived_fact_until_the_repair_runs(slotted) -> None:
    db, old, _new = slotted
    db.execute("UPDATE atomic_facts SET lifecycle = 'archived' WHERE fact_id = ?", (old,))
    assert "I-01" in _violated(db)
    with db.raw_connection() as conn:
        slot_repair.repair(conn)
    assert _violated(db) == set()


@pytest.mark.parametrize("inv_id,sql", [
    ("I-02", "UPDATE slot_revisions SET state = 'head'"),
    ("I-03", "UPDATE slot_disputes SET revision_a = revision_b"),
    ("I-04", "UPDATE slot_disputes SET state = 'dismissed'"),
    ("I-16", "UPDATE memory_slots SET state = 'current'"),
])
def test_each_broken_slot_invariant_is_reported(slotted, inv_id, sql) -> None:
    db, _old, _new = slotted
    db.execute(sql)
    assert inv_id in _violated(db)


def test_i05_and_i06_a_superseded_value_out_of_order_and_still_current(slotted) -> None:
    db, old, new = slotted
    from superlocalmemory.truth import slot_store

    with db.raw_connection() as conn:
        dispute = conn.execute("SELECT dispute_id FROM slot_disputes").fetchone()[0]
        slot_store.resolve_dispute(conn, dispute, "b", resolved_by="alice@cli")
    assert "I-06" in _violated(db)  # settled but not yet in the ledger
    _reconcile(db)
    assert _violated(db) == set()
    db.execute("UPDATE slot_revisions SET valid_from = '2026-12-01' WHERE fact_id = ?", (old,))
    assert "I-05" in _violated(db)


def test_i07_an_applied_slot_case_with_no_revision(store) -> None:
    db, old, new, _tmp = store
    _settled_by_a_person(db, old, new)
    _reconcile(db)
    db.execute("UPDATE slot_revisions SET case_id = NULL")
    assert "I-07" in _violated(db)


def test_i08_a_rollback_the_slot_has_not_followed(store) -> None:
    from superlocalmemory.truth.slot_resolver import rollback_slot_cases

    db, old, new, _tmp = store
    _settled_by_a_person(db, old, new)
    _reconcile(db)
    with db.raw_connection() as conn:
        rollback_slot_cases(conn)
    assert "I-08" in _violated(db)
    with db.raw_connection() as conn:
        slot_repair.repair(conn)
    assert "I-08" not in _violated(db)


@pytest.mark.parametrize("until,source", [("2026-01-01", "stated"), ("2026-12-01", None)])
def test_i12_an_end_before_its_start_or_with_no_source(slotted, until, source) -> None:
    db, old, _new = slotted
    db.execute("INSERT OR REPLACE INTO fact_temporal_validity (fact_id, profile_id, valid_from, "
               "valid_until, valid_until_source) VALUES (?, 'default', '2026-03-01', ?, ?)",
               (old, until, source))
    assert "I-12" in _violated(db)


def test_a_store_without_slots_has_nothing_to_check() -> None:
    conn = sqlite3.connect(":memory:")
    assert [r.status for r in inv.check(conn)] == [inv.NOT_APPLICABLE]


def test_the_store_is_opened_read_only(tmp_path) -> None:
    path = tmp_path / "s.db"
    sqlite3.connect(path).execute("CREATE TABLE t (x)")
    conn = inv.open_read_only(str(path))
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("INSERT INTO t VALUES (1)")


def test_slm_doctor_truth_exits_one_on_a_violation(slotted, capsys) -> None:
    from superlocalmemory.cli.commands import cmd_doctor

    db, old, _new = slotted
    path = db.db_path if hasattr(db, "db_path") else db._db_path
    cmd_doctor(Namespace(truth=True, db=str(path), json=False))
    out = capsys.readouterr().out
    assert "No violations" in out and "Second opinions" in out and "none yet" in out
    db.execute("UPDATE memory_slots SET state = 'current'")
    with pytest.raises(SystemExit) as stopped:
        cmd_doctor(Namespace(truth=True, db=str(path), json=False))
    assert stopped.value.code == 1 and "[FAIL] I-16" in capsys.readouterr().out
