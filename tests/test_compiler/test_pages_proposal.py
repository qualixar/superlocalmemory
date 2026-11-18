# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``create_page`` (D-18, I-10): only through a ProposalSet, verified at apply and at read."""

from __future__ import annotations

import json

import pytest

from superlocalmemory.compiler import page_sources, pages, proposals
from superlocalmemory.compiler.proposal_kinds import ProposalError
from superlocalmemory.compiler.proposals import Basis, BasisRefused
from tests.test_compiler.conftest import ENTITY, PROFILE, raw, table_rows, versions_of
from tests.test_compiler.helpers import add_fact

PERSON = Basis.person("varun", "cli")
PAGE_TABLES = ("compiled_pages", "page_citations", "artifact_versions", "atomic_facts")


def _topic(db) -> dict:
    conn = raw(db)
    try:
        return next(d for d in page_sources.build_drafts(conn, PROFILE, ("topic",))
                    if d["subject_key"] == f"entity:{ENTITY}")
    finally:
        conn.close()


def _propose(conn, draft) -> str:
    return proposals.create_set(conn, profile_id=PROFILE, origin="dream", title="Pages",
                                proposals=[{"kind": "create_page", "payload": {"page": draft}}])


def _snap(db) -> dict:
    return {t: table_rows(db, t) for t in PAGE_TABLES}


@pytest.fixture
def conn(store):
    db, _ids = store
    c = raw(db)
    yield c
    c.close()


def test_a_page_is_only_a_proposal_until_a_person_applies_it(store, conn) -> None:
    db, _ids = store
    before = _snap(db)
    set_id = _propose(conn, _topic(db))
    assert _snap(db) == before and table_rows(db, "compiled_pages") == []
    shown = proposals.get_set(conn, PROFILE, set_id)
    assert shown["proposals"][0]["summary"].startswith("Add the topic page")
    proposals.apply_set(conn, profile_id=PROFILE, set_id=set_id, basis=PERSON)
    row = table_rows(db, "compiled_pages")[0]
    assert row["origin"] == "derived" and row["version"] == 1 and row["set_id"] == set_id
    assert table_rows(db, "atomic_facts") == before["atomic_facts"]  # never admitted as memory
    assert len(table_rows(db, "page_citations")) == len(json.loads(row["body_json"]))


def test_applying_needs_a_basis_and_no_certificate_covers_a_page(store, conn) -> None:
    db, _ids = store
    set_id = _propose(conn, _topic(db))
    with pytest.raises(BasisRefused):
        proposals.apply_set(conn, profile_id=PROFILE, set_id=set_id, basis=None)
    with pytest.raises(BasisRefused):  # a certificate needs the flag, and none exists anyway
        proposals.apply_set(conn, profile_id=PROFILE, set_id=set_id,
                            basis=Basis.certificate("cert-none"))
    assert table_rows(db, "compiled_pages") == []


def test_rolling_back_a_new_page_leaves_every_table_as_it_was(store, conn) -> None:
    db, _ids = store
    before = _snap(db)
    set_id = _propose(conn, _topic(db))
    proposals.apply_set(conn, profile_id=PROFILE, set_id=set_id, basis=PERSON)
    assert _snap(db) != before
    proposals.rollback_set(conn, profile_id=PROFILE, set_id=set_id, basis=PERSON)
    assert _snap(db) == before


def test_a_rebuild_keeps_the_old_page_as_a_version_and_rolls_back_exactly(store, conn) -> None:
    db, _ids = store
    first = proposals.apply_set(conn, profile_id=PROFILE, set_id=_propose(conn, _topic(db)),
                                basis=PERSON)
    assert first["state"] == "applied"
    page_before = {t: table_rows(db, t) for t in ("compiled_pages", "page_citations")}
    add_fact(db, "A brand new fact about the project.", "2026-09-01T00:00:00+00:00")
    second = _propose(conn, _topic(db))
    proposals.apply_set(conn, profile_id=PROFILE, set_id=second, basis=PERSON)
    row = table_rows(db, "compiled_pages")[0]
    assert row["version"] == 2
    assert [v["reason"] for v in versions_of(db, "compiled_page")] == ["edit"]
    proposals.rollback_set(conn, profile_id=PROFILE, set_id=second, basis=PERSON)
    assert {t: table_rows(db, t) for t in ("compiled_pages", "page_citations")} == page_before


def test_an_unchanged_rebuild_changes_nothing(store, conn) -> None:
    db, _ids = store
    proposals.apply_set(conn, profile_id=PROFILE, set_id=_propose(conn, _topic(db)), basis=PERSON)
    snap = _snap(db)
    again = _propose(conn, _topic(db))
    proposals.apply_set(conn, profile_id=PROFILE, set_id=again, basis=PERSON)
    assert _snap(db) == snap
    proposals.rollback_set(conn, profile_id=PROFILE, set_id=again, basis=PERSON)
    assert _snap(db) == snap


def test_apply_rechecks_sentences_and_counts_the_ones_that_failed(store, conn) -> None:
    db, ids = store
    draft = _topic(db)
    set_id = _propose(conn, draft)
    db.execute("UPDATE atomic_facts SET content = 'Rewritten after the draft.' WHERE fact_id = ?",
               (ids[0],))
    proposals.apply_set(conn, profile_id=PROFILE, set_id=set_id, basis=PERSON)
    row = table_rows(db, "compiled_pages")[0]
    assert row["shown_count"] == len(draft["sentences"]) - 1
    assert json.loads(row["withheld_json"]) == {"unverifiable": 1}


def test_a_page_with_nothing_verifiable_is_refused(store, conn) -> None:
    db, ids = store
    set_id = _propose(conn, _topic(db))
    for fid in ids:
        db.execute("UPDATE atomic_facts SET content = 'Gone.' WHERE fact_id = ?", (fid,))
    with pytest.raises(ProposalError, match="None of the page's sentences"):
        proposals.apply_set(conn, profile_id=PROFILE, set_id=set_id, basis=PERSON)
    assert proposals.get_set(conn, PROFILE, set_id)["state"] == "proposed"
    assert table_rows(db, "compiled_pages") == []


def test_every_shown_sentence_is_verified_when_the_page_is_read(store, conn) -> None:
    db, ids = store
    proposals.apply_set(conn, profile_id=PROFILE, set_id=_propose(conn, _topic(db)), basis=PERSON)
    pid = table_rows(db, "compiled_pages")[0]["page_id"]
    shown = pages.show_page(conn, PROFILE, pid)
    assert shown["origin"] == "derived" and not shown["stale"] and shown["withheld"]["count"] == 0
    assert all(c["memory_id"] for s in shown["sentences"] for c in s["citations"])
    db.execute("UPDATE atomic_facts SET content = 'Changed under the page.' WHERE fact_id = ?",
               (ids[1],))
    again = pages.show_page(conn, PROFILE, pid)
    assert again["stale"] and len(again["sentences"]) == len(shown["sentences"]) - 1
    assert again["withheld"]["count"] == 1
    assert all(ids[1] not in [c["fact_id"] for c in s["citations"]] for s in again["sentences"])
    assert "unverifiable" in again["withheld"]["reasons"]


def test_a_retired_fact_leaves_the_page_when_read(store, conn) -> None:
    db, ids = store
    proposals.apply_set(conn, profile_id=PROFILE, set_id=_propose(conn, _topic(db)), basis=PERSON)
    pid = table_rows(db, "compiled_pages")[0]["page_id"]
    db.execute("UPDATE fact_temporal_validity SET system_expired_at = '2026-10-01' WHERE fact_id = ?",
               (ids[2],))
    assert pages.show_page(conn, PROFILE, pid)["stale"]


def test_erasing_a_cited_fact_removes_the_page_that_quotes_it(store, conn) -> None:
    db, ids = store
    proposals.apply_set(conn, profile_id=PROFILE, set_id=_propose(conn, _topic(db)), basis=PERSON)
    db.execute("DELETE FROM atomic_facts WHERE fact_id = ?", (ids[0],))
    assert table_rows(db, "compiled_pages") == [] and table_rows(db, "page_citations") == []
    assert versions_of(db, "compiled_page") == []


def test_a_pending_page_proposal_goes_with_an_erased_fact(store, conn) -> None:
    db, ids = store
    set_id = _propose(conn, _topic(db))
    db.execute("DELETE FROM atomic_facts WHERE fact_id = ?", (ids[0],))
    assert proposals.list_sets(conn, PROFILE, states=["proposed"]) == []
    with pytest.raises(LookupError):
        proposals.get_set(conn, PROFILE, set_id)


def test_validation_refuses_malformed_pages(store, conn) -> None:
    db, _ids = store
    good = _topic(db)
    for bad in (
        {**good, "kind": "essay"},
        {**good, "page_id": "0" * 24},
        {**good, "sentences": []},
        {**good, "title": " "},
        {**good, "sentences": [{**good["sentences"][0], "facts": ["no-such-fact"]}]},
        {**good, "sentences": [{"text": "x"}]},
    ):
        with pytest.raises(ProposalError):
            _propose(conn, bad)
    with pytest.raises(ProposalError):
        proposals.create_set(conn, profile_id=PROFILE, origin="dream",
                             proposals=[{"kind": "create_page", "payload": {}}])


def test_a_page_rebuilt_after_the_proposal_is_not_rolled_back_blindly(store, conn) -> None:
    db, _ids = store
    first = _propose(conn, _topic(db))
    proposals.apply_set(conn, profile_id=PROFILE, set_id=first, basis=PERSON)
    add_fact(db, "Another new fact for the topic.", "2026-09-02T00:00:00+00:00")
    proposals.apply_set(conn, profile_id=PROFILE, set_id=_propose(conn, _topic(db)), basis=PERSON)
    with pytest.raises(ProposalError, match="rebuilt after"):
        proposals.rollback_set(conn, profile_id=PROFILE, set_id=first, basis=PERSON)


def test_list_and_status_read_the_pages(store, conn) -> None:
    db, _ids = store
    assert pages.list_pages(conn, PROFILE) == [] and pages.status(conn, PROFILE)["pages"] == 0
    set_id = _propose(conn, _topic(db))
    assert pages.status(conn, PROFILE)["pending_sets"] == 1
    proposals.apply_set(conn, profile_id=PROFILE, set_id=set_id, basis=PERSON)
    listed = pages.list_pages(conn, PROFILE)
    assert [p["kind"] for p in listed] == ["topic"] and "body_json" not in listed[0]
    assert pages.list_pages(conn, PROFILE, kind="story") == []
    assert pages.status(conn, PROFILE) == {"pages": 1, "by_kind": {"topic": 1}, "pending_sets": 0}
    assert pages.show_page(conn, PROFILE, "nope") is None
