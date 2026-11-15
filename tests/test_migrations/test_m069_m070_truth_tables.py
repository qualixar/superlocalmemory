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
