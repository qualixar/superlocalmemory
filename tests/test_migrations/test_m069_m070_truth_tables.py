# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M069 slots and M070 leases: additive, idempotent, repairable, erasure-complete."""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.storage._migration_internals import _MODULES
from superlocalmemory.storage.migration_runner import DEFERRED_MIGRATIONS
from superlocalmemory.storage.migrations import M069_memory_slots as M069
from superlocalmemory.storage.migrations import M070_memory_leases as M070

_NOW = "2026-11-15T00:00:00+00:00"


def _conn(with_facts: bool = True) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    if with_facts:
        conn.execute("CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY)")
    return conn


@pytest.mark.parametrize("module", [M069, M070])
def test_registered_deferred_additive(module) -> None:
    assert _MODULES[module.NAME] is module
    assert module.NAME in {m.name for m in DEFERRED_MIGRATIONS}
    assert module.DOWNGRADE_FLOOR == 51 and module.DB_TARGET == "memory"


@pytest.mark.parametrize("module,trigger", [
    (M069, "trg_slot_fact_erased"), (M070, "trg_lease_fact_erased")])
def test_apply_twice_verify_repair(module, trigger) -> None:
    conn = _conn()
    assert not module.verify(conn)
    module.apply(conn)
    module.apply(conn)
    assert module.verify(conn)
    conn.execute(f"DROP TRIGGER {trigger}")
    assert not module.verify(conn)
    module.repair(conn)
    assert module.verify(conn)


@pytest.mark.parametrize("module", [M069, M070])
def test_a_store_without_facts_gets_tables_only(module) -> None:
    conn = _conn(with_facts=False)
    module.apply(conn)
    assert module.verify(conn)


def _slot_with_two_revisions(conn: sqlite3.Connection) -> None:
    conn.execute("INSERT INTO atomic_facts VALUES ('f1'), ('f2')")
    conn.execute(
        "INSERT INTO memory_slots (slot_id, profile_id, scope, subject_key, relation_key, "
        "key_source, head_revision_id, state, created_at, updated_at) "
        "VALUES ('s1', 'p', 'personal', 'e:slm', 'release_cadence', 'pattern', 'r2', "
        "'current', ?, ?)", (_NOW, _NOW))
    for rev, fact, state in (("r1", "f1", "superseded"), ("r2", "f2", "head")):
        conn.execute(
            "INSERT INTO slot_revisions (revision_id, slot_id, profile_id, fact_id, value_hash, "
            "state, decided_by, reason_code, recorded_at) VALUES (?, 's1', 'p', ?, 'h', ?, "
            "'deterministic', 'test', ?)", (rev, fact, state, _NOW))
    conn.execute(
        "INSERT INTO slot_disputes (dispute_id, slot_id, profile_id, revision_a, revision_b, "
        "reason_code, opened_by, opened_at) VALUES ('d1', 's1', 'p', 'r1', 'r2', 't', 'x', ?)",
        (_NOW,))


def test_erasing_the_head_fact_empties_the_slot_and_drops_its_disputes() -> None:
    conn = _conn()
    M069.apply(conn)
    _slot_with_two_revisions(conn)
    conn.execute("DELETE FROM atomic_facts WHERE fact_id = 'f2'")
    assert [r[0] for r in conn.execute("SELECT revision_id FROM slot_revisions")] == ["r1"]
    assert conn.execute("SELECT COUNT(*) FROM slot_disputes").fetchone()[0] == 0
    head, state, version = conn.execute(
        "SELECT head_revision_id, state, version FROM memory_slots").fetchone()
    assert (head, state, version) == (None, "empty", 1)


def test_only_one_open_dispute_per_pair() -> None:
    conn = _conn()
    M069.apply(conn)
    _slot_with_two_revisions(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO slot_disputes (dispute_id, slot_id, profile_id, revision_a, revision_b,"
            " reason_code, opened_by, opened_at) VALUES ('d2','s1','p','r1','r2','t','x',?)",
            (_NOW,))
    conn.execute("UPDATE slot_disputes SET state = 'resolved' WHERE dispute_id = 'd1'")
    conn.execute(
        "INSERT INTO slot_disputes (dispute_id, slot_id, profile_id, revision_a, revision_b,"
        " reason_code, opened_by, opened_at) VALUES ('d2','s1','p','r1','r2','t','x',?)", (_NOW,))


def test_erasing_a_fact_erases_its_leases() -> None:
    conn = _conn()
    M070.apply(conn)
    conn.execute("INSERT INTO atomic_facts VALUES ('f1'), ('f2')")
    for fact in ("f1", "f2"):
        conn.execute(
            "INSERT INTO memory_leases (fact_id, profile_id, source_kind, locator, fingerprint,"
            " leased_at, checked_at) VALUES (?, 'p', 'file', '/a', 'sha', ?, ?)",
            (fact, _NOW, _NOW))
    conn.execute("DELETE FROM atomic_facts WHERE fact_id = 'f1'")
    assert [r[0] for r in conn.execute("SELECT fact_id FROM memory_leases")] == ["f2"]


# -- B2 / R-14: erasure leaves a slot as if the erased fact had never been saved --------------

def _slot(conn, revisions, disputes=(), *, head, state):
    """revisions: (rev, fact, state[, superseded_by[, case_id]]); disputes: (id, a, b, state)."""
    conn.executemany("INSERT INTO atomic_facts VALUES (?)", [(r[1],) for r in revisions])
    conn.execute(
        "INSERT INTO memory_slots (slot_id, profile_id, scope, subject_key, relation_key, "
        "key_source, head_revision_id, state, version, created_at, updated_at) "
        "VALUES ('s1', 'p', 'personal', 'e:slm', 'release_cadence', 'pattern', ?, ?, 3, ?, ?)",
        (head, state, _NOW, _NOW))
    for rev, fact, rstate, *rest in revisions:
        by, case = (list(rest) + [None, None])[:2]
        conn.execute(
            "INSERT INTO slot_revisions (revision_id, slot_id, profile_id, fact_id, value_hash, "
            "state, superseded_by, decided_by, reason_code, case_id, recorded_at) "
            "VALUES (?, 's1', 'p', ?, 'h', ?, ?, 'deterministic', 'test', ?, ?)",
            (rev, fact, rstate, by, case, _NOW))
    for did, a, b, dstate in disputes:
        conn.execute(
            "INSERT INTO slot_disputes (dispute_id, slot_id, profile_id, revision_a, revision_b, "
            "state, reason_code, opened_by, opened_at) VALUES (?, 's1', 'p', ?, ?, ?, 't', 'x', ?)",
            (did, a, b, dstate, _NOW))


def _slot_row(conn):
    return conn.execute("SELECT head_revision_id, state, version, updated_at FROM memory_slots"
                        ).fetchone()


def test_erasing_the_challenger_leaves_a_current_slot_not_a_disputed_one() -> None:
    """Fable B2 reproduction: before, the slot stayed 'disputed' with no dispute."""
    conn = _conn()
    M069.apply(conn)
    _slot(conn, [("r1", "f1", "head"), ("r2", "f2", "disputed")], [("d1", "r1", "r2", "open")],
          head="r1", state="disputed")
    conn.execute("DELETE FROM atomic_facts WHERE fact_id = 'f2'")
    head, state, version, updated = _slot_row(conn)
    assert (head, state, version) == ("r1", "current", 4) and updated != _NOW
    assert conn.execute("SELECT COUNT(*) FROM slot_disputes").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM slot_pass_queue").fetchone()[0] == 0


def test_erasing_one_of_two_challengers_keeps_the_slot_disputed() -> None:
    conn = _conn()
    M069.apply(conn)
    _slot(conn, [("r1", "f1", "head"), ("r2", "f2", "disputed"), ("r3", "f3", "disputed")],
          [("d1", "r1", "r2", "open"), ("d2", "r1", "r3", "open")], head="r1", state="disputed")
    conn.execute("DELETE FROM atomic_facts WHERE fact_id = 'f2'")
    assert _slot_row(conn)[:2] == ("r1", "disputed")
    assert [r[0] for r in conn.execute("SELECT dispute_id FROM slot_disputes")] == ["d2"]


def test_erasing_the_head_rebuilds_the_slot_from_its_live_facts() -> None:
    conn = _conn()
    M069.apply(conn)
    _slot(conn, [("r0", "f0", "superseded", "r1"), ("r1", "f1", "head"),
                 ("r2", "f2", "disputed"), ("r3", "f3", "duplicate"), ("r4", "f4", "historical")],
          [("d1", "r1", "r2", "open"), ("d0", "r4", "r2", "dismissed")],
          head="r1", state="disputed")
    conn.execute("DELETE FROM atomic_facts WHERE fact_id = 'f1'")
    head, state, _v, _u = _slot_row(conn)
    assert (head, state) == (None, "empty")
    # Superseded history stays (the ledger retired its fact; erasure never restores it).
    assert conn.execute("SELECT revision_id, superseded_by FROM slot_revisions").fetchall() == [
        ("r0", None)]
    assert sorted(conn.execute("SELECT fact_id, reason FROM slot_pass_queue")) == [
        ("f2", "rebind"), ("f3", "rebind"), ("f4", "rebind")]
    assert conn.execute("SELECT COUNT(*) FROM slot_disputes").fetchone()[0] == 0  # closed too


def test_a_slot_left_with_no_revisions_is_deleted() -> None:
    conn = _conn()
    M069.apply(conn)
    _slot(conn, [("r1", "f1", "head")], head="r1", state="current")
    conn.execute("DELETE FROM atomic_facts WHERE fact_id = 'f1'")
    assert conn.execute("SELECT COUNT(*) FROM memory_slots").fetchone()[0] == 0


def test_a_case_id_whose_case_is_gone_is_cleared() -> None:
    conn = _conn()
    conn.execute("CREATE TABLE correction_cases (case_id TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO correction_cases VALUES ('kept')")
    M069.apply(conn)
    _slot(conn, [("r0", "f0", "superseded", "r1", "gone"), ("r9", "f9", "superseded", "r1", "kept"),
                 ("r1", "f1", "head")], head="r1", state="current")
    conn.execute("DELETE FROM atomic_facts WHERE fact_id = 'f1'")  # its case was closed first
    assert sorted(conn.execute("SELECT revision_id, case_id FROM slot_revisions")) == [
        ("r0", None), ("r9", "kept")]


def test_erasing_a_queued_fact_removes_it_from_the_queue() -> None:
    conn = _conn()
    M069.apply(conn)
    conn.execute("INSERT INTO atomic_facts VALUES ('f1')")
    conn.execute("INSERT INTO slot_pass_queue (fact_id, reason, next_at, queued_at) "
                 "VALUES ('f1', 'retry', ?, ?)", (_NOW, _NOW))
    conn.execute("DELETE FROM atomic_facts WHERE fact_id = 'f1'")
    assert conn.execute("SELECT COUNT(*) FROM slot_pass_queue").fetchone()[0] == 0


def test_a_copy_with_an_earlier_trigger_body_fails_verify_and_is_repaired() -> None:
    conn = _conn()
    M069.apply(conn)
    conn.execute("DROP TRIGGER trg_slot_revision_removed")
    conn.execute("CREATE TRIGGER trg_slot_revision_removed AFTER DELETE ON slot_revisions "
                 "BEGIN DELETE FROM slot_disputes WHERE revision_a = OLD.revision_id; END")
    assert not M069.verify(conn)
    M069.repair(conn)
    assert M069.verify(conn)
