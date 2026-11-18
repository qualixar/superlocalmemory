# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The Trust card and the Truth checks additions: origins, waiting, signatures, audit log, in words."""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.compliance.audit import AuditChain
from superlocalmemory.trust import overview, signatures
from superlocalmemory.trust import signing_key as sk
from superlocalmemory.truth.invariants import open_read_only
from tests.test_trust_signing.conftest import SIGNING, save


@pytest.fixture
def on(monkeypatch, signer):
    monkeypatch.setenv(SIGNING, "on")
    sk.install(signer)
    return signer


def _ro(db):
    return open_read_only(str(db.db_path))


def _build(db, tmp_path, signer):
    conn = _ro(db)
    try:
        return overview.build(conn, "default", slm_home=tmp_path / "slm-home",
                              audit_path=tmp_path / "audit_chain.db", signer=signer)
    finally:
        conn.close()


def _sign_all(db, signer):
    with db.raw_connection() as conn:
        signatures.sign_missing(conn, signer)
        conn.commit()


def test_unsigned_memories_are_counted_and_the_key_state_is_in_words(db, tmp_path) -> None:
    save(db, "One.")
    save(db, "Two.")
    got = _build(db, tmp_path, None)["signatures"]
    assert (got["facts"], got["signed"], got["unsigned"]) == (2, 0, 2)
    assert got["words"].startswith("No memory is signed yet.") and "Signing is off" in got["words"]


def test_signed_memories_that_still_match_are_checked_and_reported_intact(db, tmp_path, on) -> None:
    for t in ("One.", "Two.", "Three."):
        save(db, t)
    _sign_all(db, on)
    got = _build(db, tmp_path, on)["signatures"]
    assert got["signed"] == 3 and got["checked"] == 3 and got["modified"] == 0 and got["ok"] is True
    assert "none changed since signing" in got["words"]


def test_a_memory_changed_after_signing_is_reported_modified(db, tmp_path, on) -> None:
    fact = save(db, "Original text.")
    save(db, "Another.")
    _sign_all(db, on)
    db.execute("UPDATE atomic_facts SET content = 'Tampered text.' WHERE fact_id = ?", (fact.fact_id,))
    got = _build(db, tmp_path, on)["signatures"]
    assert got["modified"] == 1 and got["ok"] is False
    assert got["modified_ids"] == [fact.fact_id] and "no longer match" in got["words"]
    assert "Tampered" not in str(got)  # ids, never text


def test_without_the_key_signed_counts_are_given_and_nothing_is_claimed_checked(db, tmp_path, on) -> None:
    save(db, "One.")
    _sign_all(db, on)
    got = _build(db, tmp_path, None)["signatures"]
    assert got["signed"] == 1 and got["checked"] == 0 and got["ok"] is None
    assert "does not hold the key" in got["words"]


def test_a_failing_signature_check_is_not_reported_as_passed(db, tmp_path, monkeypatch, on) -> None:
    save(db, "One.")
    monkeypatch.setattr(signatures, "verify", lambda *a, **k: (_ for _ in ()).throw(sqlite3.Error("x")))
    got = _build(db, tmp_path, on)["signatures"]
    assert got["ok"] is None and "could not be checked" in got["words"]


def test_the_audit_log_is_intact_missing_or_broken_in_words(tmp_path, on) -> None:
    path = tmp_path / "audit_chain.db"
    assert overview.audit_state(path, on) == {
        "present": False, "ok": None, "words": "There is no audit log next to this store yet."}
    chain = AuditChain(path)
    chain.log("store", agent_id="a")
    chain.log("store", agent_id="b")
    ok = overview.audit_state(path, on)
    assert ok["present"] and ok["ok"] is True and ok["words"].startswith("The audit log is intact")
    assert "also checked with your key" in ok["words"]
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE audit_chain SET agent_id = 'evil' WHERE id = (SELECT MAX(id) FROM audit_chain)")
    bad = overview.audit_state(path, on)
    assert bad["ok"] is False and bad["words"].startswith("The audit log failed its check at entry")


def test_an_unreadable_audit_log_is_never_called_intact(tmp_path, on) -> None:
    path = tmp_path / "audit_chain.db"
    path.write_bytes(b"this is not a database at all" * 20)
    got = overview.audit_state(path, on)
    assert got["present"] and got["ok"] is False and "could not be read" in got["words"]


def test_the_overview_has_firewall_origins_secrets_and_both_checks(db, tmp_path, on) -> None:
    save(db, "One.")
    body = _build(db, tmp_path, on)
    assert set(body) == {"firewall", "secret_screen", "origins", "signatures", "audit_chain"}
    assert body["firewall"]["state"] in {"off", "shadow", "on"} and body["firewall"]["waiting"] == 0
    assert body["secret_screen"]["by_type"] == {} and body["audit_chain"]["present"] is False


def test_origins_are_named_in_words_most_trusted_first(db, tmp_path) -> None:
    from superlocalmemory.trust import fact_trust

    a, b = save(db, "Said by you."), save(db, "From a page.")
    for fact, taint in ((a, "user_direct"), (b, "web")):
        fact_trust.store(db.execute, fact.fact_id, "default", fact_trust.Trust(taint, taint, "test"))
    got = _build(db, tmp_path, None)["origins"]
    assert [(o["class"], o["label"], o["count"]) for o in got] == [
        ("user_direct", "you said this", 1), ("web", "a web page", 1)]
