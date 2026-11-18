# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M082 and signed facts: admission, batches, tamper detection, folds, erase, what is reported."""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.storage._migration_internals import _MODULES
from superlocalmemory.storage.migration_runner import DEFERRED_MIGRATIONS
from superlocalmemory.storage.migrations import M082_fact_signatures as M082
from superlocalmemory.trust import signatures as sg
from superlocalmemory.trust import signing_key as sk
from tests.test_trust_signing.conftest import SIGNING, save


@pytest.fixture
def on(monkeypatch, signer):
    monkeypatch.setenv(SIGNING, "on")
    sk.install(signer)
    return signer


def _sigs(db):
    return {r["fact_id"]: dict(r) for r in db.execute("SELECT * FROM fact_signatures")}


def _conn(db):
    return db.raw_connection()


# -- the migration ------------------------------------------------------------------------------

def test_m082_is_deferred_additive_and_repairable() -> None:
    assert _MODULES[M082.NAME] is M082
    assert M082.NAME in {m.name for m in DEFERRED_MIGRATIONS}
    assert M082.DOWNGRADE_FLOOR == 51
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY)")
    assert not M082.verify(conn)
    M082.apply(conn)
    M082.apply(conn)
    assert M082.verify(conn)
    conn.execute("DROP TRIGGER trg_fact_signatures_fact_erased")
    assert not M082.verify(conn)
    M082.repair(conn)
    assert M082.verify(conn)
    conn.execute("DROP TABLE signing_progress")  # a store whose M082 ran before the mark existed
    assert not M082.verify(conn)
    M082.repair(conn)
    assert M082.verify(conn)


def test_the_profile_fold_moves_signatures() -> None:
    from superlocalmemory.storage.profile_fold import DECISIONS, MOVE

    assert DECISIONS["fact_signatures"][0] == MOVE


def test_the_table_keeps_no_text_and_refuses_an_empty_signature(db) -> None:
    columns = {r["name"] for r in db.execute("PRAGMA table_info(fact_signatures)")}
    assert columns == {"fact_id", "profile_id", "signed_profile_id", "sig", "key_id", "signed_at"}
    with pytest.raises(Exception):
        db.execute("INSERT INTO fact_signatures VALUES ('f','p','p','','k','t')")


# -- canonical form -----------------------------------------------------------------------------

def test_every_signed_field_changes_the_bytes_and_boundaries_cannot_move() -> None:
    base = sg.canonical("p", "f", "text", "2026-11-15 10:00:00")
    assert sg.canonical("p2", "f", "text", "2026-11-15 10:00:00") != base
    assert sg.canonical("p", "f2", "text", "2026-11-15 10:00:00") != base
    assert sg.canonical("p", "f", "texT", "2026-11-15 10:00:00") != base
    assert sg.canonical("p", "f", "text", "2026-11-15 10:00:01") != base
    assert sg.canonical("pf", "", "text", "t") != sg.canonical("p", "f", "text", "t")  # no run-together
    assert b"text" not in base  # the content is hashed, never stored in the signed bytes


# -- admission ----------------------------------------------------------------------------------

def test_a_fact_saved_while_the_daemon_holds_the_key_is_signed_at_admission(db, on) -> None:
    fact = save(db, "I keep tea in the blue tin.")
    row = _sigs(db)[fact.fact_id]
    assert row["key_id"] == on.key_id and row["profile_id"] == row["signed_profile_id"]
    report = sg.verify(_raw(db), on)
    assert report.signed == 1 and report.modified == 0


def _raw(db):
    return sqlite3.connect(str(db.db_path))


def _profile(db, name):
    db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)", (name, name))


def test_without_the_key_a_save_is_not_signed_and_costs_nothing(db, monkeypatch) -> None:
    monkeypatch.setenv(SIGNING, "on")  # flag on, but this process (a hook, a CLI) has no key
    fact = save(db, "Saved by a hook.")
    assert fact.fact_id not in _sigs(db)


def test_with_the_flag_off_nothing_is_signed_even_with_a_key(db, signer) -> None:
    sk.install(signer)
    fact = save(db, "Flag is off.")
    assert fact.fact_id not in _sigs(db)


def test_the_kill_switch_stops_signing(db, on, monkeypatch) -> None:
    from superlocalmemory.core.feature_flags import KILL_ENV

    monkeypatch.setenv(KILL_ENV, "on")
    assert save(db, "Killed.").fact_id not in _sigs(db)


def test_a_failure_while_signing_never_loses_the_save(db, on, monkeypatch) -> None:
    monkeypatch.setattr(sg, "canonical", lambda *a: (_ for _ in ()).throw(RuntimeError("x")))
    fact = save(db, "Still saved.")
    assert db.execute("SELECT 1 FROM atomic_facts WHERE fact_id = ?", (fact.fact_id,))


# -- verification -------------------------------------------------------------------------------

def test_an_edited_fact_is_reported_modified_by_id_never_by_text(db, on) -> None:
    fact = save(db, "The port is 8080.")
    other = save(db, "The editor is vim.")
    db.execute("UPDATE atomic_facts SET content = 'The port is 9999.' WHERE fact_id = ?",
               (fact.fact_id,))
    report = sg.verify(_raw(db), on, sample=None)
    assert report.modified == 1 and report.modified_ids == (fact.fact_id,) and not report.ok
    assert other.fact_id not in report.modified_ids
    assert "9999" not in str(report.as_dict())


def test_a_changed_created_at_or_profile_is_modified_too(db, on) -> None:
    _profile(db, "elsewhere")
    a = save(db, "Fact about the harbour.")
    b = save(db, "Fact about the lighthouse.")
    db.execute("UPDATE atomic_facts SET created_at = '1999-01-01 00:00:00' WHERE fact_id = ?",
               (a.fact_id,))
    db.execute("UPDATE atomic_facts SET profile_id = 'elsewhere' WHERE fact_id = ?", (b.fact_id,))
    report = sg.verify(_raw(db), on, sample=None)
    assert set(report.modified_ids) == {a.fact_id, b.fact_id}


def test_a_copy_without_the_key_cannot_forge_a_signature(db, on) -> None:
    fact = save(db, "Original text.")
    db.execute("UPDATE atomic_facts SET content = 'Forged text.' WHERE fact_id = ?", (fact.fact_id,))
    forger = sk.Signer(bytes(reversed(range(32))))  # someone with the file, not the keychain
    conn = _raw(db)
    row = conn.execute("SELECT profile_id, created_at FROM atomic_facts WHERE fact_id = ?",
                       (fact.fact_id,)).fetchone()
    conn.execute("UPDATE fact_signatures SET sig = ? WHERE fact_id = ?",
                 (forger.mac(sg.canonical(row[0], fact.fact_id, "Forged text.", row[1])),
                  fact.fact_id))
    conn.commit()
    assert sg.verify(conn, on, sample=None).modified == 1


def test_signatures_under_a_lost_key_are_reported_as_such_and_never_as_modified(db, on) -> None:
    save(db, "Signed under the old key.")
    new = sk.Signer(bytes(32))
    report = sg.verify(_raw(db), new, sample=None)
    assert report.other_key == 1 and report.modified == 0 and report.ok  # a lost key loses no data


def test_unsigned_facts_are_counted_not_failed(db, on) -> None:
    save(db, "Signed.")
    sk.install(None)
    save(db, "Saved with no key.")
    report = sg.verify(_raw(db), on, sample=None)
    assert (report.facts, report.signed, report.unsigned, report.modified) == (2, 1, 1, 0)


def test_without_a_key_counts_are_given_and_nothing_is_verified(db, on) -> None:
    save(db, "Signed.")
    report = sg.verify(_raw(db), None, key_status=sk.UNAVAILABLE_STORE)
    assert report.signed == 1 and report.checked == 0 and report.key_status == sk.UNAVAILABLE_STORE


def test_a_sample_checks_at_most_the_sample_size(db, on) -> None:
    for i in range(6):
        save(db, f"Distinct fact number {i} about topic {i * 7}.")
    report = sg.verify(_raw(db), on, sample=3)
    assert report.checked == 3 and not report.exhaustive
    assert sg.verify(_raw(db), on, sample=None).exhaustive
    assert sg.SAMPLE == 500  # routing-only: bounds a doctor run, makes no detection claim


def test_a_store_before_m082_reports_zero_signed(tmp_path) -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE atomic_facts (fact_id TEXT, profile_id TEXT, content TEXT, created_at TEXT)")
    conn.execute("INSERT INTO atomic_facts VALUES ('f','p','c','t')")
    report = sg.verify(conn, sk.Signer(bytes(32)))
    assert (report.facts, report.signed, report.unsigned) == (1, 0, 1)


# -- background batches -------------------------------------------------------------------------

def test_the_pass_signs_what_was_saved_without_a_key_in_bounded_batches(db, on) -> None:
    sk.install(None)
    ids = [save(db, f"Saved by a hook, item {i} of the list {i * 3}.").fact_id for i in range(5)]
    sk.install(on)
    with db.raw_connection() as conn:
        assert sg.sign_missing(conn, on, limit=2) == 2
    assert len(_sigs(db)) == 2
    assert sg.maintenance_step(db) == 3
    assert set(_sigs(db)) == set(ids)
    assert sg.maintenance_step(db) == 0  # done: resumable and idempotent


def test_the_pass_does_nothing_without_the_key_or_the_flag(db, signer, monkeypatch) -> None:
    save(db, "Unsigned.")
    assert sg.maintenance_step(db) == 0  # no key
    sk.install(signer)
    assert sg.maintenance_step(db) == 0  # flag off
    assert _sigs(db) == {}


def test_the_pass_never_rewrites_a_signature_of_a_changed_fact(db, on) -> None:
    fact = save(db, "The port is 8080.")
    before = _sigs(db)[fact.fact_id]["sig"]
    db.execute("UPDATE atomic_facts SET content = 'The port is 9999.' WHERE fact_id = ?",
               (fact.fact_id,))
    sg.maintenance_step(db)
    assert _sigs(db)[fact.fact_id]["sig"] == before
    assert sg.verify(_raw(db), on, sample=None).modified == 1


# -- erase --------------------------------------------------------------------------------------

def test_erasing_a_fact_erases_its_signature(db, on) -> None:
    fact = save(db, "To be erased.")
    assert fact.fact_id in _sigs(db)
    db.execute("DELETE FROM atomic_facts WHERE fact_id = ?", (fact.fact_id,))
    assert fact.fact_id not in _sigs(db)


# -- profile fold -------------------------------------------------------------------------------

def _fold(db, fact_id, target="other"):
    """What the fold's MOVE does to the two tables: profile_id changes, nothing else."""
    _profile(db, target)
    db.execute("UPDATE atomic_facts SET profile_id = ? WHERE fact_id = ?", (target, fact_id))
    db.execute("UPDATE fact_signatures SET profile_id = ? WHERE fact_id = ?", (target, fact_id))


def test_a_folded_fact_is_moved_not_modified_and_is_resigned_by_the_pass(db, on) -> None:
    fact = save(db, "Lives in the first profile.")
    _fold(db, fact.fact_id)
    report = sg.verify(_raw(db), on, sample=None)
    assert report.moved == 1 and report.modified == 0 and report.ok
    assert sg.maintenance_step(db) == 1
    row = _sigs(db)[fact.fact_id]
    assert row["profile_id"] == row["signed_profile_id"] == "other"
    after = sg.verify(_raw(db), on, sample=None)
    assert after.moved == 0 and after.modified == 0 and after.checked == 1


def test_a_fold_never_launders_a_bad_signature(db, on) -> None:
    fact = save(db, "Tampered and then folded.")
    db.execute("UPDATE atomic_facts SET content = 'Tampered text.' WHERE fact_id = ?", (fact.fact_id,))
    _fold(db, fact.fact_id)
    assert sg.maintenance_step(db) == 0
    assert sg.verify(_raw(db), on, sample=None).modified == 1


def test_a_profile_changed_in_only_one_table_is_modified(db, on) -> None:
    _profile(db, "x")
    fact = save(db, "Moved by hand in one table.")
    db.execute("UPDATE atomic_facts SET profile_id = 'x' WHERE fact_id = ?", (fact.fact_id,))
    assert sg.verify(_raw(db), on, sample=None).modified == 1
    assert sg.maintenance_step(db) == 0  # and the pass does not launder it
    assert sg.verify(_raw(db), on, sample=None).modified == 1


# -- re-signing is a person's decision ----------------------------------------------------------

def test_resign_signs_again_only_what_another_key_signed(db, on) -> None:
    keep = save(db, "Signed under the current key.")
    old = save(db, "Signed under a key that was lost.")
    conn = _raw(db)
    conn.execute("UPDATE fact_signatures SET key_id = 'lostkey000000000', sig = 'x' "
                 "WHERE fact_id = ?", (old.fact_id,))
    conn.commit()
    assert sg.resign(conn, on) == 1
    conn.commit()
    rows = {r[0]: r[1] for r in conn.execute("SELECT fact_id, key_id FROM fact_signatures")}
    assert rows == {keep.fact_id: on.key_id, old.fact_id: on.key_id}
    assert sg.verify(conn, on, sample=None).modified == 0


def test_a_repair_that_rewrites_text_drops_the_signature_so_it_is_signed_anew(db, on) -> None:
    fact = save(db, "Text a repair will restore.")
    sg.forget_signatures(db.execute, [fact.fact_id])
    assert fact.fact_id not in _sigs(db)
    sg.forget_signatures(lambda *_: (_ for _ in ()).throw(sqlite3.OperationalError("no table")),
                         ["x"])  # a store before M082 is skipped, not an error


# -- one fact, for `slm why` --------------------------------------------------------------------

def test_for_fact_says_in_words_what_it_knows(db, on) -> None:
    fact = save(db, "Why-able fact.")
    conn = _raw(db)
    assert sg.for_fact(conn, fact.fact_id, on)["state"] == sg.OK
    assert sg.for_fact(conn, fact.fact_id, None)["state"] == "signed_unchecked"
    assert sg.for_fact(conn, fact.fact_id, sk.Signer(bytes(32)))["state"] == sg.OTHER_KEY
    db.execute("UPDATE atomic_facts SET content = 'changed' WHERE fact_id = ?", (fact.fact_id,))
    assert sg.for_fact(_raw(db), fact.fact_id, on)["state"] == sg.MODIFIED
    assert sg.for_fact(conn, "nope", on)["state"] == sg.UNSIGNED
    bare = sqlite3.connect(":memory:")
    assert sg.for_fact(bare, "x", on)["state"] == "not_available"


def test_a_text_repair_through_the_real_restore_drops_the_signature(tmp_path) -> None:
    from superlocalmemory.storage.migrations import M082_fact_signatures
    from superlocalmemory.storage.store_repair import restore_blanked_text

    original = "Meet the harbour pilot at the north jetty at nine in the morning."
    blanked = "Meet the harbour pilot at the north jetty at [REDACTED:TIME:1]."
    conn = sqlite3.connect(tmp_path / "r.db", isolation_level=None)
    conn.executescript("""
        CREATE TABLE memories(memory_id TEXT PRIMARY KEY, content TEXT);
        CREATE TABLE atomic_facts(fact_id TEXT PRIMARY KEY, memory_id TEXT, content TEXT);
    """)
    M082_fact_signatures.apply(conn)
    conn.execute("INSERT INTO memories VALUES ('m1', ?)", (blanked,))
    conn.execute("INSERT INTO atomic_facts VALUES ('f1', 'm1', ?)", (blanked,))
    conn.execute("INSERT INTO fact_signatures VALUES ('f1','p','p','sig','k','t')")
    ok = restore_blanked_text(lambda sql, params=(): conn.execute(sql, params).fetchall(),
                              "m1", blanked, original, ("f1",))
    assert ok
    assert conn.execute("SELECT content FROM atomic_facts").fetchone()[0] == original
    assert conn.execute("SELECT COUNT(*) FROM fact_signatures").fetchone()[0] == 0
