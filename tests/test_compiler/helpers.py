"""Builders for the page and wiki tests: facts, decisions and slots on the migrated store."""

from __future__ import annotations

import json

from superlocalmemory.truth import slot_store
from superlocalmemory.truth.slot_keys import EXPLICIT, SlotKey
from tests.test_compiler.conftest import ENTITY, PROFILE, raw
from tests.test_truth.test_slot_pass import _fact


def add_fact(db, content: str, created: str = "2026-05-01T10:00:00+00:00", *, kind: str | None = None,
             entity: str | None = ENTITY) -> str:
    fact_id = _fact(db, content, created)
    if entity:
        db.execute("UPDATE atomic_facts SET canonical_entities_json = ? WHERE fact_id = ?",
                   (json.dumps([entity]), fact_id))
    if kind:
        db.execute("UPDATE atomic_facts SET memory_kind = ? WHERE fact_id = ?", (kind, fact_id))
    return fact_id


def bind_slot(db, fact_id: str, subject: str, relation: str, value: str, when: str, *,
              qualifier: str = "") -> object:
    key = SlotKey(subject_key=f"name:{subject}", relation_key=relation, qualifier=qualifier,
                  cardinality="single", key_source=EXPLICIT, confidence=1.0,
                  value_hash=f"h-{value}")
    conn = raw(db)
    try:
        conn.execute("BEGIN")
        got = slot_store.bind(conn, profile_id=PROFILE, scope="personal", fact_id=fact_id, key=key,
                              valid_from=when)
        conn.execute("COMMIT")
        return got
    finally:
        conn.close()


def admit_like_the_route(db, edit: dict, *, actor: str = "varun") -> dict:
    """What ``update_fact_authorized`` does, at the storage layer (no engine, no embedder):
    the edited words become the successor of a correction case waiting for review, with
    origin channel ``external_edit``."""
    import uuid

    from superlocalmemory.core.remember_runtime import _propose_correction_successor

    payload = {"fact_id": edit["fact_id"], "successor_fact_id": uuid.uuid4().hex[:16],
               "content": edit["new_content"], "trusted_actor_id": actor,
               "idempotency_key": edit["edit_key"], "origin_channel": "external_edit"}
    with db.raw_connection() as conn:
        got = _propose_correction_successor(db, PROFILE, payload, connection=conn)
    return {"case_id": got["case_id"], "successor_fact_id": got["successor_fact_id"]}
