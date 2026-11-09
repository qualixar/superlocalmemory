# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""A memory replaced while its enrichment is still running: the facts that
enrichment derives from it afterwards are retired as they are written, never
reach recall or session context, and come back when the replacement is undone.

This is the ordinary flow, not an edge case: an agent saves a checkpoint and
seconds later saves the update with ``replaces=<the id it was given>``, while
the checkpoint's background enrichment has not run yet."""

from __future__ import annotations

from superlocalmemory.core.engine_ingestion import build_engine_ingestion_command
from superlocalmemory.core.standing_rules import standing_facts
from tests.test_server.test_canonical_remember_route import _client

OLD = ("Checkpoint: Alice is migrating the billing service to Postgres. "
       "The cutover is planned for Friday. Bob reviews the schema.")
NEW = ("Checkpoint: the billing migration moved to MySQL. "
       "The cutover is planned for Monday. Carol reviews the schema.")
QUERY = "Who reviews the billing schema and when is the cutover?"
ACTOR = "daemon-capability:" + "a" * 64


def _save(client, content, key, **extra) -> dict:
    response = client.post("/remember", json={"content": content, "idempotency_key": key,
                                              "kind": "decision", **extra})
    assert response.status_code == 200, response.text
    return response.json()


def _memory_of(engine, fact_id: str) -> str:
    return dict(engine._db.execute("SELECT memory_id FROM atomic_facts WHERE fact_id=?",
                                   (fact_id,))[0])["memory_id"]


def _facts_of(engine, memory_id: str) -> set[str]:
    return {dict(r)["fact_id"] for r in engine._db.execute(
        "SELECT fact_id FROM atomic_facts WHERE memory_id=?", (memory_id,))}


def _current(engine, fact_ids) -> set[str]:
    retired = {dict(r)["fact_id"] for r in engine._db.execute(
        "SELECT fact_id FROM fact_temporal_validity WHERE system_expired_at IS NOT NULL")}
    return set(fact_ids) - retired


def _recalled(client) -> list[str]:
    body = client.get("/recall", params={"q": QUERY, "answer_check": "skip",
                                         "limit": 50}).json()
    return [r["fact_id"] for r in body["results"]]


def _session(engine) -> set[str]:
    return {f.fact_id for f in standing_facts(engine._db, "default")}


def _undo(client, case: dict) -> None:
    client.app.state.canonical_remember_runtime.transition_correction(
        "default", case["case_id"], action="rollback", expected_version=case["version"],
        actor_id=ACTOR, idempotency_key="undo-" + case["case_id"][:12])


def test_facts_derived_after_the_replacement_never_surface(engine_with_mock_deps) -> None:
    engine = engine_with_mock_deps
    with _client(engine) as client:
        old = _save(client, OLD, "enrich-old-1")
        [root] = old["fact_ids"]
        memory_id = _memory_of(engine, root)
        assert _facts_of(engine, memory_id) == {root}          # enrichment has not run
        new = _save(client, NEW, "enrich-new-1", replaces=root)
        assert new["replaced"]["ok"] is True, new["replaced"]

        build_engine_ingestion_command(engine).materialize(old["operation_id"])
        derived = _facts_of(engine, memory_id) - {root}
        assert derived, "the materializer derived no facts; the test proves nothing"

        assert _current(engine, derived | {root}) == set()
        assert not set(_recalled(client)) & (derived | {root})
        assert not _session(engine) & (derived | {root})
        assert new["fact_ids"][0] in _session(engine)

        [case] = new["replaced"]["cases"]
        _undo(client, case)
        assert _current(engine, derived | {root}) == derived | {root}
        assert set(_recalled(client)) & (derived | {root})
        # 4.2.14: a session shows one entry per memory (its whole-text fact),
        # not one per sentence, so the memory is back by any of its facts.
        assert _session(engine) & (derived | {root})


def test_undone_before_enrichment_the_derived_facts_stay_current(engine_with_mock_deps) -> None:
    engine = engine_with_mock_deps
    with _client(engine) as client:
        old = _save(client, OLD, "enrich-old-2")
        [root] = old["fact_ids"]
        new = _save(client, NEW, "enrich-new-2", replaces=root)
        _undo(client, new["replaced"]["cases"][0])
        build_engine_ingestion_command(engine).materialize(old["operation_id"])
        memory_id = _memory_of(engine, root)
        assert _current(engine, _facts_of(engine, memory_id)) == _facts_of(engine, memory_id)


def test_replacing_one_extracted_fact_leaves_later_siblings_alone(engine_with_mock_deps) -> None:
    from superlocalmemory.storage.models import AtomicFact, FactType

    engine = engine_with_mock_deps
    with _client(engine) as client:
        old = _save(client, OLD, "enrich-old-3")
        [root] = old["fact_ids"]
        memory_id = _memory_of(engine, root)
        one = engine._db.store_fact(AtomicFact(profile_id="default", memory_id=memory_id,
                                               content="Bob reviews the billing schema.",
                                               fact_type=FactType.SEMANTIC))
        new = _save(client, NEW, "enrich-new-3", replaces=one)
        assert new["replaced"]["fact_ids"] == [one]
        build_engine_ingestion_command(engine).materialize(old["operation_id"])
        later = _facts_of(engine, memory_id) - {root, one}
        assert later and _current(engine, later | {root}) == later | {root}
