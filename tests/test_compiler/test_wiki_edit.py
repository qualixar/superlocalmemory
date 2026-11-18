# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""An edit in a Markdown file is a proposal (D-17): admitted for review, applied by compare-and-swap."""

from __future__ import annotations

import pytest

from superlocalmemory.compiler import page_sources, proposals, wiki
from superlocalmemory.compiler.proposal_kinds import ProposalError
from superlocalmemory.compiler.proposals import Basis, BasisRefused
from superlocalmemory.compiler.wiki_edit import edit_text
from superlocalmemory.truth import disputes
from tests.test_compiler.conftest import PROFILE, raw, table_rows
from tests.test_compiler.helpers import add_fact, admit_like_the_route, bind_slot

PERSON = Basis.person("alice", "cli")


@pytest.fixture(autouse=True)
def _taint_observed(monkeypatch, tmp_path):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SLM_FEATURE_TRUST_TAINT", "shadow")


@pytest.fixture
def conn(store):
    db, _ids = store
    c = raw(db)
    yield c
    c.close()


def _publish(db, conn, root) -> None:
    drafts = page_sources.build_drafts(conn, PROFILE)
    set_id = proposals.create_set(conn, profile_id=PROFILE, origin="dream",
                                  proposals=[{"kind": "create_page", "payload": {"page": d}}
                                             for d in drafts])
    proposals.apply_set(conn, profile_id=PROFILE, set_id=set_id, basis=PERSON)
    wiki.build(conn, conn, PROFILE, root)


@pytest.fixture
def slotted(store, conn, tmp_path):
    """A slot with one value, projected; the person edits its value line in the file."""
    db, _ids = store
    head = add_fact(db, "SLM ships every week", "2026-03-01T00:00:00+00:00", entity=None)
    bind_slot(db, head, "slm", "cadence", "weekly", "2026-03-01T00:00:00+00:00")
    root = tmp_path / "wiki"
    _publish(db, conn, root)
    path = next((root / PROFILE / "current_state").glob("*.md"))
    path.write_text(path.read_text().replace("slm cadence: SLM ships every week",
                                             "slm cadence: SLM ships every day"))
    return db, head, root


def _queue(db, conn, root):
    plan = wiki.sync_plan(conn, PROFILE, root)
    done = wiki.queue_edits(db, PROFILE, plan["edits"], lambda e: admit_like_the_route(db, e))
    return plan, done["queued"]


def _slot(db):
    return dict(db.execute("SELECT * FROM memory_slots WHERE subject_key = 'name:slm'")[0])


def test_the_edit_text_rules() -> None:
    span = {"type": "span", "text": "Two.", "facts": ["f"]}
    assert edit_text(span, "Deux.", "One. Two. Three.") == "One. Deux. Three."
    assert edit_text(span, "Two.", "One. Two.") is None  # unchanged
    assert edit_text(span, "", "One. Two.") is None
    assert edit_text(span, "x", "no such sentence") is None
    value = {"type": "template", "template": "slot_value", "text": "slm cadence: weekly ok"}
    assert edit_text(value, "slm cadence: daily", "weekly ok") == "daily"
    assert edit_text(value, "daily", "weekly ok") is None  # the frame was changed
    story = {"type": "template", "template": "story_step", "text": "2026-03-01: We chose X."}
    assert edit_text(story, "2026-03-01: We chose Y.", "We chose X. More.") == "We chose Y. More."
    assert edit_text(story, "2026-04-01: We chose Y.", "We chose X.") is None
    since = {"type": "template", "template": "slot_since", "text": "x has had this value since 2026."}
    assert edit_text(since, "y", "z") is None


def test_an_edit_is_a_proposal_and_changes_no_live_memory(slotted, conn) -> None:
    db, head, root = slotted
    live_before = {t: table_rows(db, t) for t in ("memory_slots", "slot_revisions", "slot_disputes")}
    valid_before = [r for r in table_rows(db, "fact_temporal_validity") if r["fact_id"] == head]
    plan, queued = _queue(db, conn, root)
    assert len(queued) == 1 and plan["edits"][0]["base"]["slot_id"]
    shown = proposals.get_set(conn, PROFILE, queued[0]["set_id"])
    assert shown["state"] == "proposed" and shown["origin"] == "wiki"
    assert shown["proposals"][0]["kind"] == "wiki_edit"
    assert "SLM ships every day" in shown["proposals"][0]["summary"]
    assert {t: table_rows(db, t) for t in live_before} == live_before
    assert [r for r in table_rows(db, "fact_temporal_validity") if r["fact_id"] == head] == valid_before
    case = db.execute("SELECT status, predecessor_fact_id, successor_fact_id FROM correction_cases")[0]
    assert case["status"] == "proposed" and case["predecessor_fact_id"] == head
    hidden = db.get_nonapplied_correction_successor_ids([case["successor_fact_id"]], PROFILE)
    assert hidden == {case["successor_fact_id"]}  # never searchable until a person applies it


def test_a_pending_edits_hidden_words_never_reach_a_page(slotted, conn) -> None:
    db, head, root = slotted
    _queue(db, conn, root)
    succ = db.execute("SELECT successor_fact_id FROM correction_cases")[0][0]
    cited = {f for d in page_sources.build_drafts(conn, PROFILE) for s in d["sentences"]
             for f in s["facts"]}
    assert succ not in cited and head in cited


def test_the_same_edit_is_proposed_once(slotted, conn) -> None:
    db, _head, root = slotted
    _queue(db, conn, root)
    again = wiki.sync_plan(conn, PROFILE, root)
    assert again["edits"] == [] and again["already_proposed"] == 1


def test_edited_text_enters_admission_with_origin_external_edit(slotted, conn) -> None:
    from superlocalmemory.trust import origin_class as oc

    db, _head, root = slotted
    _queue(db, conn, root)
    succ = db.execute("SELECT successor_fact_id FROM correction_cases")[0][0]
    origin = db.execute("SELECT channel, transform FROM fact_origin WHERE fact_id = ?", (succ,))[0]
    assert (origin["channel"], origin["transform"]) == ("external_edit", "verbatim")
    trust = db.execute("SELECT origin_class, basis FROM fact_trust WHERE fact_id = ?", (succ,))[0]
    assert trust["origin_class"] == oc.AGENT_INFERRED and "edit.external" in trust["basis"]


def test_a_person_applies_it_and_the_new_words_are_confirmed(slotted, conn) -> None:
    db, head, root = slotted
    _plan, queued = _queue(db, conn, root)
    set_id = queued[0]["set_id"]
    done = proposals.apply_set(conn, profile_id=PROFILE, set_id=set_id, basis=PERSON)
    assert done["state"] == "applied"
    case = db.execute("SELECT status, successor_fact_id FROM correction_cases")[0]
    assert case["status"] == "applied"
    expired = db.execute("SELECT system_expired_at FROM fact_temporal_validity WHERE fact_id = ?",
                         (head,))[0][0]
    assert expired is not None
    trust = db.execute("SELECT origin_class, basis FROM fact_trust WHERE fact_id = ?",
                       (case["successor_fact_id"],))[0]
    assert trust["origin_class"] == "user_confirmed" and "alice" in trust["basis"]
    proposals.rollback_set(conn, profile_id=PROFILE, set_id=set_id, basis=PERSON)
    assert db.execute("SELECT status FROM correction_cases")[0][0] == "rolled_back"
    assert db.execute("SELECT system_expired_at FROM fact_temporal_validity WHERE fact_id = ?",
                      (head,))[0][0] is None
    again = db.execute("SELECT origin_class FROM fact_trust WHERE fact_id = ?",
                       (case["successor_fact_id"],))[0][0]
    assert again == "agent_inferred"  # the class it had before the approval


def test_no_certificate_covers_an_edit_of_a_persons_words(slotted, conn) -> None:
    db, _head, root = slotted
    _plan, queued = _queue(db, conn, root)
    with pytest.raises(BasisRefused):
        proposals.apply_set(conn, profile_id=PROFILE, set_id=queued[0]["set_id"], basis=None)
    with pytest.raises(BasisRefused):
        proposals.apply_set(conn, profile_id=PROFILE, set_id=queued[0]["set_id"],
                            basis=Basis.certificate("cert-1"))
    assert db.execute("SELECT status FROM correction_cases")[0][0] == "proposed"


def test_declining_the_set_declines_the_pending_correction_too(slotted, conn) -> None:
    db, head, root = slotted
    _plan, queued = _queue(db, conn, root)
    proposals.reject_set(conn, profile_id=PROFILE, set_id=queued[0]["set_id"], actor="alice",
                         channel="cli")
    assert db.execute("SELECT status FROM correction_cases")[0][0] == "rejected"
    assert db.execute("SELECT system_expired_at FROM fact_temporal_validity WHERE fact_id = ?",
                      (head,))[0][0] is None


def _move_the_slot(db) -> str:
    other = add_fact(db, "SLM ships once a month", "2026-06-01T00:00:00+00:00", entity=None)
    bind_slot(db, other, "slm", "cadence", "monthly", "2026-06-01T00:00:00+00:00")
    return other


def test_a_slot_that_moved_turns_the_edit_into_a_dispute_not_a_swap(slotted, conn) -> None:
    db, head, root = slotted
    _plan, queued = _queue(db, conn, root)
    version = _slot(db)["version"]
    _move_the_slot(db)
    assert _slot(db)["version"] != version  # the compare-and-swap will fail
    open_before = db.execute("SELECT COUNT(*) FROM slot_disputes WHERE state = 'open'")[0][0]
    done = proposals.apply_set(conn, profile_id=PROFILE, set_id=queued[0]["set_id"], basis=PERSON)
    assert done["state"] == "applied"
    # the replaced memory is untouched; the edit stands beside it, and a dispute asks which is true
    assert db.execute("SELECT system_expired_at FROM fact_temporal_validity WHERE fact_id = ?",
                      (head,))[0][0] is None
    rows = db.execute("SELECT dispute_id, reason_code FROM slot_disputes WHERE state = 'open' "
                      "AND reason_code = 'external_edit_conflict'")
    assert len(rows) == 1
    assert db.execute("SELECT COUNT(*) FROM slot_disputes WHERE state = 'open'")[0][0] == open_before + 1
    succ = db.execute("SELECT successor_fact_id FROM correction_cases")[0][0]
    assert db.execute("SELECT class FROM (SELECT origin_class AS class FROM fact_trust WHERE fact_id = ?)",
                      (succ,))[0][0] == "agent_inferred"  # not confirmed: nobody chose it yet


def test_a_conflict_dispute_can_be_settled_for_the_edit_and_the_edit_is_current(slotted, conn) -> None:
    db, head, root = slotted
    _plan, queued = _queue(db, conn, root)
    _move_the_slot(db)
    proposals.apply_set(conn, profile_id=PROFILE, set_id=queued[0]["set_id"], basis=PERSON)
    succ = db.execute("SELECT successor_fact_id FROM correction_cases")[0][0]
    dispute = db.execute("SELECT dispute_id, revision_a FROM slot_disputes WHERE state = 'open' "
                         "AND reason_code = 'external_edit_conflict'")[0]
    head_fact = db.execute("SELECT fact_id FROM slot_revisions WHERE revision_id = ?",
                           (dispute["revision_a"],))[0][0]
    result = disputes.settle(conn, profile_id=PROFILE, dispute_id=dispute["dispute_id"],
                             keep=succ, channel="cli", actor="alice")
    assert result["status"] == "settled" and result["retired"] == [head_fact]
    hidden = db.get_nonapplied_correction_successor_ids([succ], PROFILE)
    assert hidden == set()  # the edit is a normal, searchable memory now


def test_rolling_back_a_conflict_dismisses_its_dispute(slotted, conn) -> None:
    db, head, root = slotted
    _plan, queued = _queue(db, conn, root)
    _move_the_slot(db)
    set_id = queued[0]["set_id"]
    proposals.apply_set(conn, profile_id=PROFILE, set_id=set_id, basis=PERSON)
    proposals.rollback_set(conn, profile_id=PROFILE, set_id=set_id, basis=PERSON)
    assert db.execute("SELECT COUNT(*) FROM slot_disputes WHERE reason_code = "
                      "'external_edit_conflict' AND state = 'open'")[0][0] == 0
    assert db.execute("SELECT status FROM correction_cases")[0][0] == "rolled_back"
    assert db.execute("SELECT state FROM slot_revisions WHERE reason_code = "
                      "'external_edit_conflict'")[0][0] == "unbound"


def test_a_conflict_already_settled_is_not_rolled_back_blindly(slotted, conn) -> None:
    db, _head, root = slotted
    _plan, queued = _queue(db, conn, root)
    _move_the_slot(db)
    set_id = queued[0]["set_id"]
    proposals.apply_set(conn, profile_id=PROFILE, set_id=set_id, basis=PERSON)
    dispute = db.execute("SELECT dispute_id FROM slot_disputes WHERE reason_code = "
                         "'external_edit_conflict'")[0][0]
    disputes.settle(conn, profile_id=PROFILE, dispute_id=dispute, keep="both", channel="cli")
    with pytest.raises(ProposalError, match="settled since"):
        proposals.rollback_set(conn, profile_id=PROFILE, set_id=set_id, basis=PERSON)


def test_a_slot_that_is_gone_declines_the_edit(slotted, conn) -> None:
    db, head, root = slotted
    _plan, queued = _queue(db, conn, root)
    row = db.execute("SELECT payload_json, proposal_id FROM proposals WHERE kind = 'wiki_edit'")[0]
    import json

    payload = json.loads(row["payload_json"])
    payload["base"]["slot_id"] = "slot-that-never-was"
    db.execute("UPDATE proposals SET payload_json = ? WHERE proposal_id = ?",
               (json.dumps(payload), row["proposal_id"]))
    proposals.apply_set(conn, profile_id=PROFILE, set_id=queued[0]["set_id"], basis=PERSON)
    assert db.execute("SELECT status FROM correction_cases")[0][0] == "rejected"
    assert db.execute("SELECT COUNT(*) FROM slot_disputes WHERE reason_code = "
                      "'external_edit_conflict'")[0][0] == 0


def test_a_memory_outside_any_slot_is_compared_by_whether_it_is_still_current(store, conn, tmp_path) -> None:
    db, ids = store
    root = tmp_path / "wiki"
    _publish(db, conn, root)
    path = next((root / PROFILE / "topic").glob("*.md"))
    path.write_text(path.read_text().replace("release 0 with a measured recall gain",
                                             "release 0 with a better recall gain"))
    _plan, queued = _queue(db, conn, root)
    db.execute("UPDATE fact_temporal_validity SET system_expired_at = '2026-10-01' WHERE fact_id = ?",
               (ids[0],))
    proposals.apply_set(conn, profile_id=PROFILE, set_id=queued[0]["set_id"], basis=PERSON)
    assert db.execute("SELECT status FROM correction_cases")[0][0] == "rejected"


def test_erasing_the_edited_fact_takes_the_pending_edit_with_it(slotted, conn) -> None:
    db, head, root = slotted
    _plan, queued = _queue(db, conn, root)
    from superlocalmemory.compiler import erasure

    erasure.erase_for_fact(conn, PROFILE, head)  # at tombstone time, before the row goes
    assert [s["origin"] for s in proposals.list_sets(conn, PROFILE, states=["proposed"])] == []
    assert table_rows(db, "wiki_edits") != []  # the record is dropped when the row itself goes


def test_validation_refuses_a_foreign_or_decided_case(slotted, conn) -> None:
    db, head, root = slotted
    plan = wiki.sync_plan(conn, PROFILE, root)
    got = admit_like_the_route(db, plan["edits"][0])
    base = {"fact_id": head, "case_id": got["case_id"], "successor_fact_id": got["successor_fact_id"],
            "page_id": "p", "block_hash": "h", "base": {}, "path": "x"}
    for bad in ({**base, "case_id": "nope"}, {**base, "fact_id": got["successor_fact_id"]}):
        with pytest.raises(ProposalError):
            proposals.create_set(conn, profile_id=PROFILE, origin="wiki",
                                 proposals=[{"kind": "wiki_edit", "payload": bad}])
    proposals.create_set(conn, profile_id=PROFILE, origin="wiki",
                         proposals=[{"kind": "wiki_edit", "payload": base}])
    conn.execute("UPDATE correction_cases SET status = 'rejected'")
    with pytest.raises(ProposalError, match="still waiting"):
        proposals.create_set(conn, profile_id=PROFILE, origin="wiki",
                             proposals=[{"kind": "wiki_edit", "payload": base}])
