# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""ProposalSets (WS-6): basis before any change (I-10), exact unit rollback, erase (I-11)."""

from __future__ import annotations

import json
import sqlite3

import pytest

from superlocalmemory.compiler import erasure, proposals, versions
from superlocalmemory.compiler.proposals import Basis, BasisRefused, StateRefused
from superlocalmemory.core.feature_flags import FeatureFlags
from superlocalmemory.storage.models import EntityProfile
from tests.test_compiler.conftest import ENTITY, PROFILE, raw, table_rows, versions_of

PERSON = Basis.person("alice", "cli")
#: Every table a proposal can change, as plain rows, before and after.
TOUCHED = ("atomic_facts", "fact_temporal_validity", "entity_profiles", "core_memory_blocks",
           "persona_summary", "community_summaries")


@pytest.fixture
def conn(store):
    db, _ids = store
    c = raw(db)
    yield c
    c.close()


def _snapshot(db) -> dict:
    return {t: table_rows(db, t) for t in TOUCHED}


def _case_states(db) -> list[str]:
    return [r["status"] for r in db.execute("SELECT status FROM correction_cases ORDER BY case_id")]


def _retire(ids, old=0, new=1) -> dict:
    return {"kind": "retire", "payload": {"predecessor": ids[old], "successor": ids[new]}}


def _profile_edit(db, ids) -> dict:
    db.store_entity_profile(EntityProfile(entity_id=ENTITY, profile_id=PROFILE,
                                          knowledge_summary="the original profile text",
                                          fact_ids=ids))
    entry = db.execute("SELECT profile_entry_id FROM entity_profiles")[0]["profile_entry_id"]
    return {"kind": "edit_derived", "payload": {
        "artifact": "entity_profile", "key": {"profile_id": PROFILE, "profile_entry_id": entry},
        "set": {"knowledge_summary": "a corrected profile text"}}}


def _certificate(db, *, tier="jev", question="SLOT_RELATION", verdict="replaces", voided=False,
                 profile=PROFILE, cert_id="cert-1") -> str:
    db.execute(
        "INSERT INTO judge_certificates (cert_id, profile_id, question_id, question_version, tier, "
        "model, verdict, population, regime, method, alpha, delta, n, errors, threshold, "
        "label_set_hash, issued_at, voided_at, void_reason) VALUES (?, ?, ?, '2', ?, 'm', ?, "
        "'slot_dispute', 'single', 'e-process', 0.05, 0.05, 200, 0, 0.85, 'h', "
        "'2026-11-01T00:00:00+00:00', ?, ?)",
        (cert_id, profile, question, tier, verdict,
         "2026-11-02T00:00:00+00:00" if voided else None, "labels" if voided else None))
    return cert_id


def _on() -> FeatureFlags:
    return FeatureFlags(file_values={"compiler.auto_apply": "on"}, environ={})


# -- exact unit rollback --------------------------------------------------------------------------

def test_rolling_back_a_retire_restores_every_touched_row_exactly(store, conn) -> None:
    db, ids = store
    before = _snapshot(db)
    set_id = proposals.create_set(conn, profile_id=PROFILE, origin="dream",
                                  proposals=[_retire(ids)])
    assert _snapshot(db) == before  # proposing changes nothing
    done = proposals.apply_set(conn, profile_id=PROFILE, set_id=set_id, basis=PERSON)
    assert done["state"] == "applied" and done["basis"] == {"kind": "decision", "ref": "alice@cli"}
    assert _snapshot(db) != before  # the old memory is retired
    assert _case_states(db) == ["applied"]
    proposals.rollback_set(conn, profile_id=PROFILE, set_id=set_id, basis=PERSON)
    assert _snapshot(db) == before
    assert _case_states(db) == ["rolled_back"]
    assert proposals.get_set(conn, PROFILE, set_id)["state"] == "rolled_back"


def test_a_mixed_set_rolls_back_as_one_unit(store, conn) -> None:
    db, ids = store
    edit = _profile_edit(db, ids)
    before = _snapshot(db)
    set_id = proposals.create_set(conn, profile_id=PROFILE, origin="dream", proposals=[
        {"kind": "merge", "payload": {"keep": ids[2], "retire": [ids[0], ids[1]]}}, edit])
    proposals.apply_set(conn, profile_id=PROFILE, set_id=set_id, basis=PERSON)
    live = db.execute("SELECT knowledge_summary FROM entity_profiles")[0]["knowledge_summary"]
    assert live == "a corrected profile text"
    assert _case_states(db) == ["applied", "applied"]
    proposals.rollback_set(conn, profile_id=PROFILE, set_id=set_id, basis=PERSON)
    assert _snapshot(db) == before
    assert {p["state"] for p in proposals.get_set(conn, PROFILE, set_id)["proposals"]} == {
        "rolled_back"}


def test_an_edit_rolls_back_to_the_exact_row_including_its_timestamps(store, conn) -> None:
    db, ids = store
    edit = _profile_edit(db, ids)
    before = table_rows(db, "entity_profiles")
    set_id = proposals.create_set(conn, profile_id=PROFILE, origin="person", proposals=[edit])
    proposals.apply_set(conn, profile_id=PROFILE, set_id=set_id, basis=PERSON)
    # the edit wrote a version of the row it replaced (versions first)
    assert [json.loads(v["row_json"])["knowledge_summary"]
            for v in versions_of(db, "entity_profile")] == ["the original profile text"]
    proposals.rollback_set(conn, profile_id=PROFILE, set_id=set_id, basis=PERSON)
    assert table_rows(db, "entity_profiles") == before


def test_an_edit_is_not_rolled_back_over_text_that_changed_again(store, conn) -> None:
    db, ids = store
    edit = _profile_edit(db, ids)
    set_id = proposals.create_set(conn, profile_id=PROFILE, origin="person", proposals=[edit])
    proposals.apply_set(conn, profile_id=PROFILE, set_id=set_id, basis=PERSON)
    db.execute("UPDATE entity_profiles SET knowledge_summary = 'rebuilt later'")
    with pytest.raises(proposals.ProposalError, match="changed again"):
        proposals.rollback_set(conn, profile_id=PROFILE, set_id=set_id, basis=PERSON)
    assert db.execute("SELECT knowledge_summary FROM entity_profiles")[0][0] == "rebuilt later"
    assert proposals.get_set(conn, PROFILE, set_id)["state"] == "applied"


def test_a_set_that_fails_halfway_changes_nothing(store, conn) -> None:
    db, ids = store
    edit = _profile_edit(db, ids)
    set_id = proposals.create_set(conn, profile_id=PROFILE, origin="dream", proposals=[
        _retire(ids), edit])
    db.execute("DELETE FROM entity_profiles")  # the text the second proposal edits is gone
    before = _snapshot(db)
    with pytest.raises(proposals.ProposalError):
        proposals.apply_set(conn, profile_id=PROFILE, set_id=set_id, basis=PERSON)
    assert _snapshot(db) == before and _case_states(db) == []
    assert proposals.get_set(conn, PROFILE, set_id)["state"] == "proposed"


# -- I-10: a basis before any change ---------------------------------------------------------------

def test_applying_without_a_basis_is_refused_and_changes_nothing(store, conn) -> None:
    db, ids = store
    set_id = proposals.create_set(conn, profile_id=PROFILE, origin="dream",
                                  proposals=[_retire(ids)])
    before = _snapshot(db)
    with pytest.raises(BasisRefused, match="certificate id, a rule-certificate id or a recorded decision"):
        proposals.apply_set(conn, profile_id=PROFILE, set_id=set_id, basis=None)
    assert _snapshot(db) == before and _case_states(db) == []
    assert proposals.get_set(conn, PROFILE, set_id)["state"] == "proposed"


@pytest.mark.parametrize("basis", [Basis.person("", "cli"), Basis.person("alice", "person"),
                                   Basis.person("alice", "agent"), Basis("person", "x"), Basis("whim", "x"),
                                   Basis.certificate("")])
def test_a_basis_that_names_no_person_or_certificate_is_refused(store, conn, basis) -> None:
    _db, ids = store
    set_id = proposals.create_set(conn, profile_id=PROFILE, origin="dream",
                                  proposals=[_retire(ids)])
    with pytest.raises(BasisRefused):
        proposals.apply_set(conn, profile_id=PROFILE, set_id=set_id, basis=basis,
                            flags=_on())


def test_the_store_itself_refuses_an_applied_set_with_no_basis(store, conn) -> None:
    _db, ids = store
    set_id = proposals.create_set(conn, profile_id=PROFILE, origin="dream",
                                  proposals=[_retire(ids)])
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE proposal_sets SET state = 'applied' WHERE set_id = ?", (set_id,))


def test_a_certificate_acts_only_while_auto_apply_is_on(store, conn) -> None:
    db, ids = store
    cert = _certificate(db)
    set_id = proposals.create_set(conn, profile_id=PROFILE, origin="dream",
                                  proposals=[_retire(ids)])
    off = FeatureFlags(file_values={}, environ={})
    with pytest.raises(BasisRefused, match="compiler.auto_apply"):
        proposals.apply_set(conn, profile_id=PROFILE, set_id=set_id,
                            basis=Basis.certificate(cert), flags=off)
    killed = FeatureFlags(file_values={"compiler.auto_apply": "on"}, environ={}, killed=True)
    with pytest.raises(BasisRefused, match="compiler.auto_apply"):
        proposals.apply_set(conn, profile_id=PROFILE, set_id=set_id,
                            basis=Basis.certificate(cert), flags=killed)
    assert _case_states(db) == []
    done = proposals.apply_set(conn, profile_id=PROFILE, set_id=set_id,
                               basis=Basis.certificate(cert), flags=_on())
    assert done["basis"] == {"kind": "certificate", "ref": cert}
    assert _case_states(db) == ["applied"]


@pytest.mark.parametrize("make,basis_kind,why", [
    ({"voided": True}, "certificate", "voided"),
    ({"profile": "other"}, "certificate", "another profile"),
    ({"question": "DUPLICATE", "verdict": "duplicate"}, "certificate", "does not cover a retire"),
    ({"verdict": "same"}, "certificate", "does not cover a retire"),
    ({"tier": "rules"}, "certificate", "does not match"),
    ({}, "rule_certificate", "does not match"),
])
def test_a_certificate_must_be_open_in_this_profile_and_cover_the_change(
        store, conn, make, basis_kind, why) -> None:
    db, ids = store
    if make.get("profile") == "other":
        db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('other', 'other')")
    cert = _certificate(db, **make)
    set_id = proposals.create_set(conn, profile_id=PROFILE, origin="dream",
                                  proposals=[_retire(ids)])
    with pytest.raises(BasisRefused):
        proposals.apply_set(conn, profile_id=PROFILE, set_id=set_id,
                            basis=Basis(basis_kind, cert), flags=_on())
    assert _case_states(db) == []


def test_a_rule_certificate_covers_a_retire_through_the_rules_tier(store, conn) -> None:
    db, ids = store
    cert = _certificate(db, tier="rules", cert_id="rule-1")
    set_id = proposals.create_set(conn, profile_id=PROFILE, origin="dream",
                                  proposals=[_retire(ids)])
    done = proposals.apply_set(conn, profile_id=PROFILE, set_id=set_id,
                               basis=Basis.rule_certificate(cert), flags=_on())
    assert done["basis"]["kind"] == "rule_certificate"


def test_a_rollback_needs_a_basis_too(store, conn) -> None:
    db, ids = store
    set_id = proposals.create_set(conn, profile_id=PROFILE, origin="dream",
                                  proposals=[_retire(ids)])
    proposals.apply_set(conn, profile_id=PROFILE, set_id=set_id, basis=PERSON)
    with pytest.raises(BasisRefused):
        proposals.rollback_set(conn, profile_id=PROFILE, set_id=set_id, basis=None)
    assert _case_states(db) == ["applied"]


# -- states, rejection, the ledger -----------------------------------------------------------------

def test_a_set_is_decided_once(store, conn) -> None:
    _db, ids = store
    set_id = proposals.create_set(conn, profile_id=PROFILE, origin="dream",
                                  proposals=[_retire(ids)])
    proposals.apply_set(conn, profile_id=PROFILE, set_id=set_id, basis=PERSON)
    with pytest.raises(StateRefused):
        proposals.apply_set(conn, profile_id=PROFILE, set_id=set_id, basis=PERSON)
    with pytest.raises(StateRefused):
        proposals.reject_set(conn, profile_id=PROFILE, set_id=set_id, actor="alice",
                             channel="cli")
    proposals.rollback_set(conn, profile_id=PROFILE, set_id=set_id, basis=PERSON)
    with pytest.raises(StateRefused):
        proposals.rollback_set(conn, profile_id=PROFILE, set_id=set_id, basis=PERSON)


def test_rejecting_changes_no_memory(store, conn) -> None:
    db, ids = store
    before = _snapshot(db)
    set_id = proposals.create_set(conn, profile_id=PROFILE, origin="dream",
                                  proposals=[_retire(ids)])
    done = proposals.reject_set(conn, profile_id=PROFILE, set_id=set_id, actor="alice",
                                channel="dashboard")
    assert done["state"] == "rejected" and done["decided_by"] == "alice"
    assert _snapshot(db) == before and _case_states(db) == []
    with pytest.raises(StateRefused):
        proposals.apply_set(conn, profile_id=PROFILE, set_id=set_id, basis=PERSON)


def _case(db, ids, status="proposed") -> str:
    db.execute(
        "INSERT INTO correction_cases (case_id, profile_id, scope, predecessor_fact_id, "
        "successor_fact_id, reason_code, status, version, idempotency_key, proposed_by_actor_id, "
        "proposed_by_actor_kind, proposed_by_trust_tier, created_at, updated_at) VALUES "
        "('case-1', 'default', 'personal', ?, ?, 'direct_content_correction', ?, 0, 'k1', 'w', "
        "'host_authenticated', 'trusted', '2026-10-01T00:00:00+00:00', '2026-10-01T00:00:00+00:00')",
        (ids[0], ids[1], status))
    return "case-1"


def test_a_ledger_case_waiting_for_review_is_a_retire_proposal_with_its_basis(store, conn) -> None:
    db, ids = store
    case_id = _case(db, ids)
    shown = proposals.case_as_proposal(conn, PROFILE, case_id)
    assert shown["kind"] == "retire" and shown["state"] == "proposed" and shown["basis"] is None
    assert shown["payload"] == {"predecessor": ids[0], "successor": ids[1],
                                "reason": "direct_content_correction"}
    before = _snapshot(db)
    set_id = proposals.adopt_case(conn, profile_id=PROFILE, case_id=case_id)
    proposals.apply_set(conn, profile_id=PROFILE, set_id=set_id, basis=PERSON)
    assert _case_states(db) == ["applied"]
    after = proposals.case_as_proposal(conn, PROFILE, case_id)
    assert after["state"] == "applied" and after["basis"] == {"kind": "person", "ref": "alice"}
    proposals.rollback_set(conn, profile_id=PROFILE, set_id=set_id, basis=PERSON)
    assert _snapshot(db) == before and _case_states(db) == ["rolled_back"]


def test_rejecting_an_adopted_case_declines_it_in_the_ledger(store, conn) -> None:
    db, ids = store
    set_id = proposals.adopt_case(conn, profile_id=PROFILE, case_id=_case(db, ids))
    proposals.reject_set(conn, profile_id=PROFILE, set_id=set_id, actor="alice", channel="cli")
    assert _case_states(db) == ["rejected"]


def test_only_a_case_still_waiting_can_be_adopted(store, conn) -> None:
    db, ids = store
    case_id = _case(db, ids, status="applied")
    with pytest.raises(StateRefused):
        proposals.adopt_case(conn, profile_id=PROFILE, case_id=case_id)
    with pytest.raises(LookupError):
        proposals.adopt_case(conn, profile_id=PROFILE, case_id="no-such-case")


# -- what can be proposed ----------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    {"kind": "retire", "payload": {"predecessor": "nope", "successor": "also-nope"}},
    {"kind": "retire", "payload": {}},
    {"kind": "merge", "payload": {"keep": "x", "retire": []}},
    {"kind": "edit_derived", "payload": {"artifact": "persona", "key": {"profile_id": PROFILE},
                                         "set": {"summary": "x"}}},
    {"kind": "edit_derived", "payload": {"artifact": "persona", "key": {}, "set": {"profile_id": "x"}}},
    {"kind": "create_page", "payload": {}},
    {"kind": "teleport", "payload": {}},
])
def test_a_proposal_that_cannot_be_applied_is_refused_when_proposed(store, conn, bad) -> None:
    with pytest.raises(proposals.ProposalError):
        proposals.create_set(conn, profile_id=PROFILE, origin="dream", proposals=[bad])
    assert proposals.list_sets(conn, PROFILE) == []


def test_only_declared_text_columns_can_be_edited(store, conn) -> None:
    db, ids = store
    edit = _profile_edit(db, ids)
    edit["payload"]["set"] = {"fact_ids_json": "[]"}
    with pytest.raises(proposals.ProposalError, match="can be edited"):
        proposals.create_set(conn, profile_id=PROFILE, origin="p", proposals=[edit])


def test_a_retire_of_a_memory_with_an_open_correction_is_refused(store, conn) -> None:
    db, ids = store
    _case(db, ids)
    with pytest.raises(proposals.ProposalError, match="open correction"):
        proposals.create_set(conn, profile_id=PROFILE, origin="dream", proposals=[_retire(ids)])


def test_a_kind_registered_by_another_module_is_applied_and_undone(store, conn, monkeypatch) -> None:
    """A module adds a kind with ``register_handler`` (pages and wiki edits do). The registry is
    copied first, so the test's stub never replaces a real handler for the tests after it."""
    from superlocalmemory.compiler import proposal_kinds as kinds

    seen = {}

    def apply(c, ctx, payload):
        seen["applied"] = payload
        return {"page": "p1"}

    monkeypatch.setattr(kinds, "HANDLERS", dict(kinds.HANDLERS), raising=True)
    kinds.register_handler("stub_kind", kinds.Handler(
        lambda c, p, payload: [], lambda c, p, payload: "Add a stub",
        apply, lambda c, ctx, payload, inverse: seen.update(rolled=inverse)))
    set_id = proposals.create_set(conn, profile_id=PROFILE, origin="dream", proposals=[
        {"kind": "stub_kind", "payload": {"title": "t"}}])
    proposals.apply_set(conn, profile_id=PROFILE, set_id=set_id, basis=PERSON)
    proposals.rollback_set(conn, profile_id=PROFILE, set_id=set_id, basis=PERSON)
    assert seen == {"applied": {"title": "t"}, "rolled": {"page": "p1"}}


def test_the_pages_and_wiki_kinds_are_registered_by_their_modules() -> None:
    from superlocalmemory.compiler import proposal_kinds as kinds

    kinds.load_optional_handlers()
    assert {"create_page", "wiki_edit"} <= set(kinds.HANDLERS)


def test_listing_shows_pending_sets_in_plain_words(store, conn) -> None:
    db, ids = store
    proposals.create_set(conn, profile_id=PROFILE, origin="dream", title="Tidy",
                         proposals=[_retire(ids)])
    listed = proposals.list_sets(conn, PROFILE, states=["proposed"])
    assert len(listed) == 1 and proposals.count_pending(conn, PROFILE) == 1
    summary = listed[0]["proposals"][0]["summary"]
    assert summary.startswith("Retire “") and "replaces it" in summary
    assert proposals.list_sets(conn, PROFILE, states=["applied"]) == []
    assert proposals.list_sets(conn, "somebody-else") == []


# -- I-11: erasing a fact -------------------------------------------------------------------------

def _plant(db, conn, ids):
    """A pending set, a decided set, and a version, each naming ids[0]."""
    pending = proposals.create_set(conn, profile_id=PROFILE, origin="dream",
                                   proposals=[_retire(ids, 0, 1)])
    decided = proposals.create_set(conn, profile_id=PROFILE, origin="dream", proposals=[
        {"kind": "merge", "payload": {"keep": ids[2], "retire": [ids[1]]}},
        _profile_edit(db, [ids[0], ids[2]])])
    proposals.apply_set(conn, profile_id=PROFILE, set_id=decided, basis=PERSON)
    db.store_core_block("b1", PROFILE, "user_profile", "text naming it", json.dumps([ids[0]]), 14)
    db.store_core_block("b2", PROFILE, "user_profile", "later text", json.dumps([ids[2]]), 10)
    return pending, decided


def test_erasing_a_fact_removes_its_proposals_and_versions_at_tombstone_time(store, conn) -> None:
    db, ids = store
    pending, decided = _plant(db, conn, ids)
    assert len(versions_of(db, "core_block")) == 1
    counts = erasure.erase_for_fact(conn, PROFILE, ids[0])
    assert counts["versions"] >= 1 and counts["proposals"] == 2
    assert versions_of(db, "core_block") == []
    with pytest.raises(LookupError):
        proposals.get_set(conn, PROFILE, pending)  # a waiting set goes whole
    kept = proposals.get_set(conn, PROFILE, decided)  # a decided set loses only that proposal
    assert [p["kind"] for p in kept["proposals"]] == ["merge"]


def test_deleting_the_fact_row_removes_the_same_with_no_help(store, conn) -> None:
    db, ids = store
    pending, decided = _plant(db, conn, ids)
    db.execute("DELETE FROM correction_cases WHERE predecessor_fact_id = ?", (ids[0],))
    db.execute("DELETE FROM atomic_facts WHERE fact_id = ?", (ids[0],))
    assert db.execute("SELECT COUNT(*) FROM proposal_sets WHERE set_id = ?", (pending,))[0][0] == 0
    assert db.execute("SELECT COUNT(*) FROM proposal_facts WHERE fact_id = ?", (ids[0],))[0][0] == 0
    assert db.execute("SELECT COUNT(*) FROM artifact_version_facts WHERE fact_id = ?",
                      (ids[0],))[0][0] == 0
    assert versions_of(db, "core_block") == []
    assert [p["kind"] for p in proposals.get_set(conn, PROFILE, decided)["proposals"]] == ["merge"]


def test_the_last_proposal_of_a_decided_set_takes_the_set_with_it(store, conn) -> None:
    db, ids = store
    set_id = proposals.create_set(conn, profile_id=PROFILE, origin="p", proposals=[
        _profile_edit(db, [ids[0]])])
    proposals.apply_set(conn, profile_id=PROFILE, set_id=set_id, basis=PERSON)
    db.execute("DELETE FROM atomic_facts WHERE fact_id = ?", (ids[0],))
    assert db.execute("SELECT COUNT(*) FROM proposal_sets")[0][0] == 0
    assert db.execute("SELECT COUNT(*) FROM proposals")[0][0] == 0


def test_the_versions_api_does_not_resurrect_an_erased_fact(store) -> None:
    db, ids = store
    db.store_core_block("b1", PROFILE, "user_profile", "quotes the fact", json.dumps([ids[0]]), 15)
    db.store_core_block("b2", PROFILE, "user_profile", "newer text", json.dumps(ids[1:]), 10)
    assert len(versions.list_versions(db, PROFILE, "core_block")) == 1
    db.execute("DELETE FROM atomic_facts WHERE fact_id = ?", (ids[0],))
    assert versions.list_versions(db, PROFILE, "core_block") == []


def test_the_tombstone_step_of_an_erasure_clears_them_while_the_fact_still_exists(store, conn) -> None:
    from superlocalmemory.core.transactions.tombstones import TOMBSTONE_WRITTEN, write_tombstones_status

    db, ids = store
    pending, decided = _plant(db, conn, ids)
    assert write_tombstones_status(db, PROFILE, (ids[0],), "erasure-1", 1.0) == TOMBSTONE_WRITTEN
    assert db.execute("SELECT COUNT(*) FROM atomic_facts WHERE fact_id = ?", (ids[0],))[0][0] == 1
    assert db.execute("SELECT COUNT(*) FROM proposal_facts WHERE fact_id = ?", (ids[0],))[0][0] == 0
    assert db.execute("SELECT COUNT(*) FROM artifact_version_facts WHERE fact_id = ?",
                      (ids[0],))[0][0] == 0
    assert db.execute("SELECT COUNT(*) FROM proposal_sets WHERE set_id = ?", (pending,))[0][0] == 0


def test_a_failure_clearing_versions_never_blocks_the_erasure(store, monkeypatch) -> None:
    from superlocalmemory.core.transactions.tombstones import TOMBSTONE_WRITTEN, write_tombstones_status

    db, ids = store

    def broken(conn, profile_id, fact_id):
        raise RuntimeError("versions table is locked")

    monkeypatch.setattr(erasure, "erase_for_fact", broken)
    assert write_tombstones_status(db, PROFILE, (ids[0],), "erasure-2", 1.0) == TOMBSTONE_WRITTEN
    assert db.execute("SELECT COUNT(*) FROM projection_tombstones WHERE fact_id = ?",
                      (ids[0],))[0][0] == 1
