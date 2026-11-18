# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""D-16: the audit chain's links are HMACs under the keychain key while the daemon holds it."""

from __future__ import annotations

import json
import sqlite3

import pytest

from superlocalmemory.compliance import audit_verify_cache as cache
from superlocalmemory.compliance.audit import AuditChain
from superlocalmemory.compliance.audit_keyed import KEYED_VERSION
from superlocalmemory.trust import signing_key as sk


@pytest.fixture
def path(tmp_path):
    return tmp_path / "audit_chain.db"


@pytest.fixture(autouse=True)
def _clean_cache():
    cache.forget()
    yield
    cache.forget()


def _rows(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM audit_chain ORDER BY id")]
    finally:
        conn.close()


def _poke(path, sql, *params):
    with sqlite3.connect(path) as conn:
        conn.execute(sql, params)


def test_without_a_key_links_are_plain_as_before(path) -> None:
    chain = AuditChain(path)
    chain.log("store", agent_id="a")
    rows = _rows(path)
    assert [r["hash_version"] for r in rows] == [2] and rows[0]["key_id"] == ""
    report = chain.verify_report()
    assert report["ok"] and report["keyed"] == 0 and report["anchors"] == 0


def test_with_the_key_the_first_keyed_link_is_preceded_by_an_anchor(path, signer) -> None:
    sk.install(signer)
    chain = AuditChain(path)
    chain.log("store", agent_id="a", metadata={"n": 1})
    rows = _rows(path)
    assert [(r["operation"], r["hash_version"]) for r in rows] == [
        ("key_anchor", KEYED_VERSION), ("store", KEYED_VERSION)]
    meta = json.loads(rows[0]["metadata"])
    assert meta == {"reason": "chain_started", "key_id": signer.key_id, "previous_key_id": ""}
    assert all(r["key_id"] == signer.key_id for r in rows)
    report = chain.verify_report()
    assert report["ok"] and (report["keyed"], report["anchors"], report["fully_covered"]) == (2, 1, 2)


def test_keyed_links_are_not_the_plain_hash_of_their_fields(path, signer) -> None:
    from superlocalmemory.compliance.audit import _compute_hash_v2

    sk.install(signer)
    AuditChain(path).log("store", agent_id="a")
    row = _rows(path)[1]
    plain = _compute_hash_v2(row["prev_hash"], row["operation"], row["agent_id"],
                             row["profile_id"], row["content_hash"], row["timestamp"],
                             row["metadata"])
    assert row["event_hash"] != plain  # someone who can rewrite the file cannot recompute it


def test_a_rewrite_made_without_the_key_is_caught(path, signer) -> None:
    sk.install(signer)
    chain = AuditChain(path)
    for i in range(3):
        chain.log("store", agent_id="a", metadata={"i": i})
    forger = sk.Signer(bytes(reversed(range(32))))
    rows = _rows(path)
    target = rows[2]
    from superlocalmemory.compliance.audit_keyed import keyed_hash

    # The forger rewrites an entry and recomputes its hash, and every later link, with their own key.
    prev = target["prev_hash"]
    for r in rows[2:]:
        meta = '{"i": 999}' if r["id"] == target["id"] else r["metadata"]
        h = keyed_hash(forger, prev, r["operation"], r["agent_id"], r["profile_id"],
                       r["content_hash"], r["timestamp"], meta)
        _poke(path, "UPDATE audit_chain SET metadata=?, prev_hash=?, event_hash=? WHERE id=?",
              meta, prev, h, r["id"])
        prev = h
    assert chain.verify_report()["ok"] is False


def test_a_plain_rewrite_of_a_keyed_entry_is_caught_or_at_least_counted(path, signer) -> None:
    from superlocalmemory.compliance.audit import _compute_hash_v2

    sk.install(signer)
    chain = AuditChain(path)
    chain.log("store", agent_id="a")
    chain.log("recall", agent_id="a")
    last = _rows(path)[-1]
    # Downgrade: relabel the entry as a plain version-2 link with a recomputed plain hash.
    h = _compute_hash_v2(last["prev_hash"], last["operation"], "evil", last["profile_id"],
                         last["content_hash"], last["timestamp"], last["metadata"])
    _poke(path, "UPDATE audit_chain SET agent_id='evil', hash_version=2, key_id='', "
                "event_hash=? WHERE id=?", h, last["id"])
    # The sidecar anchor names the old head: the replaced head is caught outright.
    assert chain.verify_report()["ok"] is False
    # An attacker who also rewrites the sidecar (it is a plain file) is not stopped by a plain
    # link, because a process without the key legitimately writes them; the report counts it.
    with open(str(path) + ".anchor", "w", encoding="utf-8") as fh:
        json.dump({"count": 3, "head": h}, fh)
    report = chain.verify_report()
    assert report["ok"] is True and report["plain_after_keyed"] == 1


def test_relabelling_a_tampered_plain_entry_as_keyed_is_caught(path, signer) -> None:
    sk.install(signer)
    chain = AuditChain(path)
    sk.install(None)
    chain.log("store", agent_id="a")
    sk.install(signer)
    chain.log("store", agent_id="b")
    _poke(path, "UPDATE audit_chain SET agent_id='evil', hash_version=3, key_id='someotherkey0000' "
                "WHERE id=1")
    assert chain.verify_report()["ok"] is False
    _poke(path, "UPDATE audit_chain SET key_id='' WHERE id=1")
    assert "names no key" in chain.verify_report()["reason"]


def test_plain_links_before_the_anchor_verify_by_their_plain_hash(path, signer) -> None:
    chain = AuditChain(path)
    chain.log("store", agent_id="a")  # written before signing was on
    sk.install(signer)
    chain.log("store", agent_id="b")
    rows = _rows(path)
    assert [(r["operation"], r["hash_version"]) for r in rows] == [
        ("store", 2), ("key_anchor", 3), ("store", 3)]
    assert json.loads(rows[1]["metadata"])["reason"] == "keying_started"
    report = chain.verify_report()
    assert report["ok"] and report["keyed"] == 2 and report["legacy"] == 0
    _poke(path, "UPDATE audit_chain SET agent_id='evil' WHERE id=1")
    assert chain.verify_report()["ok"] is False


def test_a_new_key_re_anchors_and_old_links_stay_in_order_but_cannot_be_checked(path) -> None:
    old, new = sk.Signer(bytes(range(32))), sk.Signer(bytes(32))
    sk.install(old)
    chain = AuditChain(path)
    chain.log("store", agent_id="a")
    sk.install(new)  # the key was lost and a new one made
    chain.log("store", agent_id="b")
    rows = _rows(path)
    anchors = [json.loads(r["metadata"]) for r in rows if r["operation"] == "key_anchor"]
    assert [a["reason"] for a in anchors] == ["chain_started", "key_changed"]
    assert anchors[1]["previous_key_id"] == old.key_id and anchors[1]["key_id"] == new.key_id
    report = chain.verify_report()
    assert report["ok"] and report["keyed"] == 2 and report["keyed_unverified"] == 2
    assert report["anchors"] == 2


def test_a_verifier_without_the_key_checks_order_and_says_it_could_not_check_the_rest(path, signer) -> None:
    sk.install(signer)
    chain = AuditChain(path)
    chain.log("store", agent_id="a")
    sk.install(None)
    report = chain.verify_report()
    assert report["ok"] and report["keyed"] == 0 and report["keyed_unverified"] == 2
    # a broken link is still found without the key
    _poke(path, "UPDATE audit_chain SET prev_hash='x' WHERE id=2")
    assert chain.verify_report()["ok"] is False


def test_a_process_without_the_key_writes_plain_and_the_report_counts_it(path, signer) -> None:
    sk.install(signer)
    chain = AuditChain(path)
    chain.log("store", agent_id="daemon")
    sk.install(None)  # a CLI command while the daemon is down
    chain.log("store", agent_id="cli")
    sk.install(signer)
    report = chain.verify_report()
    assert report["ok"] and report["plain_after_keyed"] == 1
    chain.log("store", agent_id="daemon")  # same key still active: no second anchor
    assert chain.verify_report()["anchors"] == 1


def test_an_old_chain_without_the_key_column_is_upgraded_in_place(path, signer) -> None:
    chain = AuditChain(path)
    chain.log("store", agent_id="a")
    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE audit_chain DROP COLUMN key_id")
    sk.install(signer)
    reopened = AuditChain(path)  # adds the column back
    reopened.log("store", agent_id="b")
    assert reopened.verify_report()["ok"]


def test_the_cache_resumes_across_a_key_anchor(path, signer) -> None:
    sk.install(signer)
    chain = AuditChain(path)
    chain.log("store", agent_id="a")
    first = cache.cached_report(chain)
    assert first["ok"] and first["keyed"] == 2
    chain.log("store", agent_id="b")
    second = cache.cached_report(chain)
    assert second["ok"] and second["keyed"] == 3 and second["entries"] == 3
    assert second == chain.verify_report()


def test_the_cache_still_finds_a_relabelled_new_entry(path, signer) -> None:
    sk.install(signer)
    chain = AuditChain(path)
    chain.log("store", agent_id="a")
    cache.cached_report(chain)
    sk.install(None)
    chain.log("store", agent_id="b")
    _poke(path, "UPDATE audit_chain SET agent_id='evil', hash_version=3, key_id='zzzz' "
                "WHERE id=3")
    assert cache.cached_report(chain)["ok"] is False


def test_logging_never_asks_the_keychain(path, signer, monkeypatch) -> None:
    sk.install(signer)

    def boom(*a, **k):
        raise AssertionError("the keychain must not be read when an event is logged")

    monkeypatch.setattr(sk, "load", boom)
    monkeypatch.setattr(sk.KeychainKeyStore, "read", boom)
    AuditChain(path).log("store", agent_id="a")


def test_the_dashboard_badge_counts_keyed_entries_as_covered(path, signer) -> None:
    from superlocalmemory.server.audit_view import integrity_summary, present_event

    sk.install(signer)
    chain = AuditChain(path)
    chain.log("store", agent_id="a")
    assert present_event(chain.query()[0])["fully_covered"] is True
    assert integrity_summary(chain)["verified"] is True


def test_a_read_only_chain_verifies_and_changes_nothing_even_when_old(path, signer) -> None:
    import os

    chain = AuditChain(path)
    chain.log("store", agent_id="a")
    chain.log("store", agent_id="b")
    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE audit_chain DROP COLUMN key_id")  # as written before 4.2.30
    del chain
    before = (os.stat(path).st_mtime_ns, os.stat(path).st_size, sorted(os.listdir(path.parent)))
    sk.install(signer)
    report = AuditChain(path, read_only=True).verify_report()
    after = (os.stat(path).st_mtime_ns, os.stat(path).st_size, sorted(os.listdir(path.parent)))
    assert report["ok"] and report["entries"] == 2 and before == after
    with sqlite3.connect(path) as conn:
        assert "key_id" not in {r[1] for r in conn.execute("PRAGMA table_info(audit_chain)")}


def test_an_anchor_that_declares_another_key_is_caught_even_without_the_key(path, signer) -> None:
    sk.install(signer)
    chain = AuditChain(path)
    chain.log("store", agent_id="a")
    sk.install(None)  # a verifier that cannot recompute the HMAC still checks the declaration
    assert chain.verify_report()["ok"] is True
    _poke(path, "UPDATE audit_chain SET metadata = ? WHERE id = 1",
          json.dumps({"reason": "chain_started", "key_id": "forged-key-id000", "previous_key_id": ""}))
    report = chain.verify_report()
    assert report["ok"] is False and "declares a different key" in report["reason"]
