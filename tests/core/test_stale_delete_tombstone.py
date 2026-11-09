# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A memory left with a delete tombstone and no delete finishing it is healed.

A delete writes the memory's erasure tombstone and removes its search entries
before the writer deletes it. If the writer times out (or the process dies) the
memory stays stored with the tombstone, unfindable. When correction history
protects it, every later delete is refused up front, so nothing would ever
finish that erasure: before 4.1.22 the memory stayed unfindable for good. Now
the refused delete rebuilds its search entries. An unprotected memory needs no
heal: the user's next delete completes the erasure (the user's intent).
"""

from __future__ import annotations

import uuid

import pytest
from tests.test_compliance.test_gdpr_erasure_wins import _live_index_has, _person_edits

_WORDS = "Quorvantel stores the synthetic copper compass in the attic chest."


def _actor() -> str:
    from superlocalmemory.core.engine_ingestion import local_trusted_actor_id

    return local_trusted_actor_id("python-api")


def _store(engine, text: str) -> str:
    from superlocalmemory.core.engine_ingestion import canonical_store

    receipt = canonical_store(engine, text, source_type="python-api", trusted_actor_id=_actor(),
                              require_complete=True, return_receipt=True)
    return list(receipt.final_fact_ids)[0]


def _n(engine, sql: str, args: tuple = ()) -> int:
    return int(dict(engine._db.execute(sql, args)[0])["n"])


def _half_deleted(engine, fact_id: str) -> None:
    """What a delete leaves when its writer times out after removing entries."""
    from superlocalmemory.core.transactions import OperationContext
    from superlocalmemory.core.transactions.concrete_owners import build_erasure_service

    ctx = OperationContext(operation_id=uuid.uuid4().hex, profile_id=engine._profile_id,
                           subject_id=fact_id, fact_ids=(fact_id,))
    assert build_erasure_service(engine).remove(engine._db, ctx).tombstoned


def _age(engine, fact_id: str, seconds: float) -> None:
    with engine._db.raw_connection() as conn:
        conn.execute("UPDATE projection_tombstones SET created_at = created_at - ? "
                     "WHERE fact_id = ?", (seconds, fact_id))


def _state(engine, fact_id: str) -> dict:
    return {
        "fact": _n(engine, "SELECT COUNT(*) AS n FROM atomic_facts WHERE fact_id = ?", (fact_id,)),
        "bm25": _n(engine, "SELECT COUNT(*) AS n FROM bm25_tokens WHERE fact_id = ?", (fact_id,)),
        "temporal": _n(engine, "SELECT COUNT(*) AS n FROM fact_temporal_validity "
                       "WHERE fact_id = ?", (fact_id,)),
        "in_live_index": _live_index_has(engine, fact_id),
        "tombstone": _n(engine, "SELECT COUNT(*) AS n FROM projection_tombstones "
                        "WHERE fact_id = ?", (fact_id,)),
        "open_erasures": _n(engine, "SELECT COUNT(*) AS n FROM projection_obligations "
                            "WHERE kind = 'erase' AND subject_id = ?", (fact_id,)),
    }


def test_a_refused_delete_heals_a_protected_memory_left_unfindable(engine_with_mock_deps):
    from superlocalmemory.core.mutations import delete_fact_authorized
    from superlocalmemory.core.remember_runtime import CanonicalMutationConflict

    engine = engine_with_mock_deps
    kept = _store(engine, _WORDS)
    assert _person_edits(engine, kept, "succ" + kept[:12], "edit-stale")["ok"]
    healthy = _state(engine, kept)
    _half_deleted(engine, kept)
    broken = _state(engine, kept)
    assert broken["tombstone"] == 1 and broken["bm25"] == 0 and not broken["in_live_index"]

    from superlocalmemory.core.delete_refusal import STALE_AFTER_S

    # A young tombstone may be an erasure still running elsewhere: left alone.
    with pytest.raises(CanonicalMutationConflict, match="protected by correction history"):
        delete_fact_authorized(engine, kept, trusted_actor_id=_actor(), source_agent_id="test")
    assert _state(engine, kept) == broken
    _age(engine, kept, STALE_AFTER_S + 1)

    with pytest.raises(CanonicalMutationConflict, match="protected by correction history"):
        delete_fact_authorized(engine, kept, trusted_actor_id=_actor(), source_agent_id="test")

    assert _state(engine, kept) == {**healthy, "open_erasures": 0}
    assert healthy["in_live_index"] and healthy["bm25"] == 1


def test_the_next_delete_finishes_an_unprotected_half_deleted_memory(engine_with_mock_deps):
    from superlocalmemory.core.mutations import delete_fact_authorized

    engine = engine_with_mock_deps
    doomed = _store(engine, _WORDS)
    _half_deleted(engine, doomed)
    result = delete_fact_authorized(engine, doomed, trusted_actor_id=_actor(),
                                    source_agent_id="test")
    assert result.get("ok") is True and result.get("erasure_verified") is True, result
    state = _state(engine, doomed)
    assert state["fact"] == 0 and state["bm25"] == 0 and not state["in_live_index"]
