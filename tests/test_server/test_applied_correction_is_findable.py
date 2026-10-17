# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""An edited memory, once its correction is applied, is findable by meaning.

4.2.9, found on the author's live store: eleven memories edited through
``update_memory`` (PATCH /api/memories/{id}) and applied with
``review_correction`` had their embedding saved in ``atomic_facts`` but no row
in the vector index, so a question never found them; only word matches did,
until a daemon restart rebuilt the index. The edit had made recall worse than
leaving the memory alone.

A proposal stays out of the indexes until it is applied (see
``test_authorized_update_creates_review_required_successor``); recall also
hides a proposed, rejected or rolled-back successor by its case status.
"""

from __future__ import annotations

from tests.test_server.test_canonical_remember_route import _client

OLD = "The platform release train leaves on Tuesday at 14:00 UTC."
NEW = "The platform release train now leaves on Thursday at 09:00 UTC."


def _in_vector_index(engine, fact_id: str) -> bool:
    return bool(engine._db.execute(
        "SELECT 1 FROM embedding_metadata WHERE fact_id = ?", (fact_id,)))


def _edit_and_apply(client, fact_id: str) -> str:
    edited = client.patch(f"/api/memories/{fact_id}", json={"content": NEW},
                          headers={"X-Idempotency-Key": "edit-train"})
    assert edited.status_code == 202, edited.text
    body = edited.json()
    case = body["correction_case"]
    applied = client.post(f"/api/corrections/{case['case_id']}/apply",
                          json={"expected_version": case["version"]})
    assert applied.status_code == 200, applied.text
    assert applied.json()["correction_case"]["status"] == "applied"
    return body["successor_fact_id"]


def test_an_applied_edit_is_in_the_vector_index(engine_with_mock_deps) -> None:
    engine = engine_with_mock_deps
    with _client(engine) as client:
        [old_id] = client.post("/remember", json={
            "content": OLD, "idempotency_key": "train-old"}).json()["fact_ids"]
        successor = _edit_and_apply(client, old_id)

    assert _in_vector_index(engine, successor), "applied edit is not findable by meaning"


def test_recall_returns_an_applied_edit(engine_with_mock_deps) -> None:
    """4.2.10 on the live store: indexed, valid, found by search, yet never
    returned by recall."""
    from tests.test_server.test_per_request_profile import _daemon

    with _daemon(engine_with_mock_deps, profiles=("work",)) as (client, _app):
        saved = client.post("/remember", json={
            "content": OLD, "profile_id": "work", "idempotency_key": "train-old-3"})
        [old_id] = saved.json()["fact_ids"]
        edited = client.patch(f"/api/memories/{old_id}",
                              json={"content": NEW, "profile_id": "work"},
                              headers={"X-Idempotency-Key": "edit-train-3"})
        assert edited.status_code == 202, edited.text
        case = edited.json()["correction_case"]
        applied = client.post(f"/api/corrections/{case['case_id']}/apply",
                              json={"expected_version": case["version"], "profile_id": "work"})
        assert applied.status_code == 200, applied.text
        successor = edited.json()["successor_fact_id"]
        found = client.get("/recall", params={
            "q": "When does the platform release train leave?",
            "profile_id": "work", "answer_check": "skip"}).json()

    ids = [r["fact_id"] for r in found["results"]]
    assert successor in ids, found
    assert old_id not in ids


def test_a_proposed_edit_stays_out_of_the_vector_index(engine_with_mock_deps) -> None:
    engine = engine_with_mock_deps
    with _client(engine) as client:
        [old_id] = client.post("/remember", json={
            "content": OLD, "idempotency_key": "train-old-2"}).json()["fact_ids"]
        edited = client.patch(f"/api/memories/{old_id}", json={"content": NEW},
                              headers={"X-Idempotency-Key": "edit-train-2"})
        assert edited.status_code == 202, edited.text

    assert not _in_vector_index(engine, edited.json()["successor_fact_id"])
