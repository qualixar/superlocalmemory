# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Failed saves that a retry could never clear (GitHub issue #325, reported on 4.2.19).

Two shapes, both measured on the reporter's store:

* "incomplete derivation stages: embeddings". The save recorded its facts' embeddings as
  missing because they landed a moment later. A retry resumes the checkpoint, which repaired
  only provenance and the post-store hooks, so ``embeddings`` stayed False forever even
  though every fact has its embedding now (one save was retried 37 times).
* "external projection facts are missing". A fact of the save was archived afterwards, which
  is legitimate. The projector counted only live facts and called the archived one missing,
  so every retry failed forever.

A fact that is truly gone, or that sits in another profile, is still a real fault and must
keep failing.
"""

from __future__ import annotations

import json

from superlocalmemory.core.engine_ingestion import build_engine_ingestion_command
from superlocalmemory.core.ingestion_command import IngestionState
from superlocalmemory.core.ops_remediation import _action_retry
from superlocalmemory.storage.migrations import M031_dead_letter_operations
from tests.test_server.test_canonical_remember_route import _client

TEXT = "The billing cutover is on Friday. Bob reviews the schema. Carol signs the plan."


def _saved_and_complete(engine, client, key: str):
    body = client.post("/remember", json={"content": TEXT, "idempotency_key": key}).json()
    op = build_engine_ingestion_command(engine).materialize(body["operation_id"])
    assert op.state is IngestionState.COMPLETE, op.last_error
    assert len(op.final_fact_ids) >= 2, "the save needs at least two facts to prove anything"
    return op


def _stuck(engine, op, *, state: dict, error: str) -> None:
    """Put the save back the way 4.2.19 left it: FAILED, exhausted, and dead-lettered."""
    with engine._db.raw_connection() as conn:
        M031_dead_letter_operations.apply(conn)
    engine._db.execute(
        "UPDATE ingestion_operations SET state='failed', attempt_count=5, last_error=?, "
        "derivation_state_json=?, lease_owner='', lease_expires_at=0 WHERE operation_id=?",
        (error, json.dumps(state), op.operation_id))
    engine._db.execute(
        "INSERT INTO dead_letter_operations (original_op_id, error, attempt_count, profile_id) "
        "VALUES (?, ?, 5, ?)", (op.operation_id, error, op.profile_id))


def _dead_letters(engine, op) -> int:
    return int(dict(engine._db.execute(
        "SELECT COUNT(*) AS n FROM dead_letter_operations WHERE original_op_id=?",
        (op.operation_id,))[0])["n"])


def _state(engine, op) -> str:
    return dict(engine._db.execute(
        "SELECT state FROM ingestion_operations WHERE operation_id=?",
        (op.operation_id,))[0])["state"]


def _error(engine, op) -> str:
    return dict(engine._db.execute(
        "SELECT last_error FROM ingestion_operations WHERE operation_id=?",
        (op.operation_id,))[0])["last_error"]


def _embeddings_late(op) -> dict:
    return {**op.derivation_state, "embeddings": False}


EMB_ERROR = "incomplete derivation stages: embeddings"
PROJ_ERROR = "external projection facts are missing"


# -- bug 1: embeddings that arrived after the save ------------------------------------------

def test_retry_clears_a_save_whose_embeddings_arrived_late(engine_with_mock_deps) -> None:
    engine = engine_with_mock_deps
    with _client(engine) as client:
        op = _saved_and_complete(engine, client, "late-emb-1")
        _stuck(engine, op, state=_embeddings_late(op), error=EMB_ERROR)

        result = _action_retry(engine._db.db_path, engine, op.operation_id)

        assert result["success"] is True, result
        assert _state(engine, op) == "complete"
        assert _dead_letters(engine, op) == 0


def test_retry_still_fails_while_an_embedding_is_really_missing(engine_with_mock_deps) -> None:
    engine = engine_with_mock_deps
    with _client(engine) as client:
        op = _saved_and_complete(engine, client, "late-emb-2")
        engine._db.execute("UPDATE atomic_facts SET embedding=NULL WHERE fact_id=?",
                           (op.final_fact_ids[0],))
        _stuck(engine, op, state=_embeddings_late(op), error=EMB_ERROR)

        result = _action_retry(engine._db.db_path, engine, op.operation_id)

        assert result["success"] is False
        assert _state(engine, op) == "failed"
        assert _dead_letters(engine, op) == 1
        assert _error(engine, op) == EMB_ERROR, "the resume itself must still see it missing"


def test_a_store_without_an_embedder_clears(engine_with_mock_deps) -> None:
    """No embedder configured (Mode A): embeddings are not owed, as at save time."""
    engine = engine_with_mock_deps
    with _client(engine) as client:
        op = _saved_and_complete(engine, client, "late-emb-4")
        engine._db.execute("UPDATE atomic_facts SET embedding=NULL WHERE fact_id=?",
                           (op.final_fact_ids[0],))
        _stuck(engine, op, state=_embeddings_late(op), error=EMB_ERROR)
        engine._embedder = None

        result = _action_retry(engine._db.db_path, engine, op.operation_id)

        assert result["success"] is True, result
        assert _state(engine, op) == "complete"


def test_an_archived_fact_does_not_count_against_late_embeddings(engine_with_mock_deps) -> None:
    engine = engine_with_mock_deps
    with _client(engine) as client:
        op = _saved_and_complete(engine, client, "late-emb-3")
        engine._db.execute(
            "UPDATE atomic_facts SET embedding=NULL, archive_status='archived' WHERE fact_id=?",
            (op.final_fact_ids[0],))
        _stuck(engine, op, state=_embeddings_late(op), error=EMB_ERROR)

        result = _action_retry(engine._db.db_path, engine, op.operation_id)

        assert result["success"] is True, result
        assert _state(engine, op) == "complete"


# -- bug 2: a fact archived after the save ---------------------------------------------------

def test_retry_clears_a_save_with_a_fact_archived_after_it(engine_with_mock_deps) -> None:
    engine = engine_with_mock_deps
    with _client(engine) as client:
        op = _saved_and_complete(engine, client, "archived-1")
        engine._db.execute("UPDATE atomic_facts SET archive_status='archived' WHERE fact_id=?",
                           (op.final_fact_ids[0],))
        _stuck(engine, op, state=dict(op.derivation_state), error=PROJ_ERROR)

        result = _action_retry(engine._db.db_path, engine, op.operation_id)

        assert result["success"] is True, result
        assert _state(engine, op) == "complete"
        assert _dead_letters(engine, op) == 0


def test_retry_still_fails_when_a_fact_is_truly_gone(engine_with_mock_deps) -> None:
    engine = engine_with_mock_deps
    with _client(engine) as client:
        op = _saved_and_complete(engine, client, "archived-2")
        engine._db.execute("DELETE FROM atomic_facts WHERE fact_id=?", (op.final_fact_ids[0],))
        _stuck(engine, op, state=dict(op.derivation_state), error=PROJ_ERROR)

        result = _action_retry(engine._db.db_path, engine, op.operation_id)

        assert result["success"] is False
        assert _state(engine, op) == "failed"
        assert _dead_letters(engine, op) == 1


def test_retry_still_fails_when_a_fact_moved_to_another_profile(engine_with_mock_deps) -> None:
    engine = engine_with_mock_deps
    with _client(engine) as client:
        op = _saved_and_complete(engine, client, "archived-3")
        with engine._db.raw_connection() as conn:  # a cross-profile row, as on a damaged store
            conn.execute("PRAGMA foreign_keys=OFF")
            conn.execute("UPDATE atomic_facts SET profile_id='someone-else', "
                         "archive_status='archived' WHERE fact_id=?", (op.final_fact_ids[0],))
            conn.commit()
            conn.execute("PRAGMA foreign_keys=ON")
        engine._db.execute(  # even when this profile's erasure records name it
            "INSERT INTO projection_tombstones (profile_id, fact_id, erasure_id, created_at) "
            "VALUES (?, ?, 'erasure-test', 0)", (op.profile_id, op.final_fact_ids[0]))
        _stuck(engine, op, state=dict(op.derivation_state), error=PROJ_ERROR)

        result = _action_retry(engine._db.db_path, engine, op.operation_id)

        assert result["success"] is False
        assert _state(engine, op) == "failed"


def test_both_shapes_at_once_clear(engine_with_mock_deps) -> None:
    """Late embeddings on a save one of whose facts was archived since (the resume path)."""
    engine = engine_with_mock_deps
    with _client(engine) as client:
        op = _saved_and_complete(engine, client, "both-1")
        engine._db.execute("UPDATE atomic_facts SET archive_status='archived' WHERE fact_id=?",
                           (op.final_fact_ids[-1],))
        _stuck(engine, op, state=_embeddings_late(op), error=EMB_ERROR)

        result = _action_retry(engine._db.db_path, engine, op.operation_id)

        assert result["success"] is True, result
        assert _state(engine, op) == "complete"


def test_a_fact_erased_since_the_save_does_not_block_it(engine_with_mock_deps) -> None:
    """An erasure removes the row and leaves a tombstone; that is retired, not missing."""
    engine = engine_with_mock_deps
    with _client(engine) as client:
        op = _saved_and_complete(engine, client, "erased-1")
        gone = op.final_fact_ids[0]
        engine._db.execute(
            "INSERT INTO projection_tombstones (profile_id, fact_id, erasure_id, created_at) "
            "VALUES (?, ?, 'erasure-test', 0)", (op.profile_id, gone))
        with engine._db.raw_connection() as conn:
            conn.execute("PRAGMA foreign_keys=OFF")
            conn.execute("DELETE FROM atomic_facts WHERE fact_id=?", (gone,))
            conn.commit()
            conn.execute("PRAGMA foreign_keys=ON")
        _stuck(engine, op, state=dict(op.derivation_state), error=PROJ_ERROR)

        result = _action_retry(engine._db.db_path, engine, op.operation_id)

        assert result["success"] is True, result
        assert _state(engine, op) == "complete"
