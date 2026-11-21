# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""A JSON text column that holds bytes which are not UTF-8 must not fail saves.

SQLite keeps whatever a writer binds: a ``bytes`` value lands in a TEXT column
as a BLOB, and Python reads it back as ``bytes``. ``json.loads(bytes)`` decodes
strictly, so one such row in ``atomic_facts.entities_json`` (or any of the
columns below) made every later save that read the row fail its enrichment with
``'utf-8' codec can't decode byte 0x81 in position 2: invalid start byte``; ten
attempts later each of those saves was dead-lettered. This is the only path
found that gives that exact text on macOS and Linux.

The fix reads such a value as absent (the column's empty value) and reports it
once; reading never changes the row, and a BLOB that does hold UTF-8 JSON is
still read. A later save that writes one of these values does not build it from
the empty read: it rebuilds it from its source (an entity profile from the facts
of the entity) and keeps the original bytes recoverable, in a version row
(entity profiles, scenes) or the change ledger (fact columns). Rows no save
writes keep their bytes exactly as they were.
"""

from __future__ import annotations

import json
import logging

import pytest

from superlocalmemory.core.engine_ingestion import (
    build_engine_ingestion_command,
    canonical_store,
    local_trusted_actor_id,
)
from superlocalmemory.core.ingestion_command import IngestionState

CODEC_ERROR = "'utf-8' codec can't decode byte 0x81 in position 2: invalid start byte"
NOT_TEXT = bytes.fromhex("61628163")  # b"ab\x81c"
EARLIER = [
    "Alice met Bob in Tokyo to discuss the Postgres billing migration on Friday.",
    "Alice and Bob reviewed the Postgres billing schema in Tokyo again.",
    "Bob told Alice the Postgres billing cutover in Tokyo moves to Monday.",
]
LATER = "Alice and Bob met in Tokyo about the Postgres billing migration, 最終版."

# Every column whose damage failed a later save (found by corrupting each JSON
# column an enrichment reads, one at a time).
COLUMNS = [
    ("atomic_facts", "entities_json"),
    ("atomic_facts", "canonical_entities_json"),
    ("atomic_facts", "source_turn_ids_json"),
    ("atomic_facts", "langevin_position"),
    ("atomic_facts", "shared_with"),
    ("memory_scenes", "entity_ids_json"),
    ("memory_scenes", "fact_ids_json"),
    ("entity_profiles", "fact_ids_json"),
    ("memories", "metadata_json"),
    ("memories", "shared_with"),
]


def _save(engine, text: str):
    receipt = canonical_store(
        engine, text, source_type="python-api",
        trusted_actor_id=local_trusted_actor_id("python-api"),
        metadata={"project": "demo"}, require_complete=False, return_receipt=True,
    )
    return build_engine_ingestion_command(engine).materialize(receipt.operation_id)


@pytest.mark.parametrize(("table", "column"), COLUMNS)
def test_a_later_save_completes_past_a_column_holding_non_utf8_bytes(
    engine_with_mock_deps, table, column,
) -> None:
    engine = engine_with_mock_deps
    for text in EARLIER:
        assert _save(engine, text).state is IngestionState.COMPLETE
    profiles_before = {
        dict(r)["entity_id"]: json.loads(dict(r)["fact_ids_json"])
        for r in engine._db.execute("SELECT entity_id, fact_ids_json FROM entity_profiles")}
    damaged = engine._db.execute(
        f"UPDATE {table} SET {column} = ? RETURNING rowid", (NOT_TEXT,))
    assert damaged, f"no {table} rows to damage; the test proves nothing"

    result = _save(engine, LATER)

    assert result.state is IngestionState.COMPLETE, result.last_error
    assert CODEC_ERROR not in result.last_error
    assert result.raw_content == LATER
    # Nothing is invented: every value is either the damaged bytes, left as they
    # were, or real JSON (a row the save legitimately rebuilt, such as an entity
    # profile that now lists the new fact).
    values = [dict(r)["v"] for r in engine._db.execute(f"SELECT {column} AS v FROM {table}")]
    still_damaged = sum(1 for v in values if v == NOT_TEXT)
    for value in values:
        if value is None or value == NOT_TEXT:
            continue
        json.loads(value)
    kept = [
        hex_value
        for row in engine._db.execute(
            "SELECT row_json FROM artifact_versions WHERE artifact IN ('entity_profile','scene')")
        for value in json.loads(dict(row)["row_json"]).values()
        if isinstance(value, dict) and "__blob_hex__" in value
        for hex_value in [value["__blob_hex__"]]]

    if table != "entity_profiles":
        # No save rewrites these here: each damaged row keeps its bytes exactly.
        assert still_damaged == len(damaged)
        return
    # An entity profile the save rewrote: rebuilt from the facts of its entity,
    # so it still lists every earlier fact, and its old bytes are in the version row.
    new_ids = set(result.fact_ids)
    after = {dict(r)["entity_id"]: json.loads(dict(r)["fact_ids_json"])
             for r in engine._db.execute(
                 "SELECT entity_id, fact_ids_json FROM entity_profiles WHERE fact_ids_json != ?",
                 (NOT_TEXT,))}
    rewritten = {e: ids for e, ids in after.items() if new_ids & set(ids)}
    assert rewritten, "the new fact is in no entity profile"
    for entity_id, ids in rewritten.items():
        assert set(profiles_before[entity_id]) <= set(ids), entity_id
    assert still_damaged + len(rewritten) == len(damaged)
    assert kept == [NOT_TEXT.hex()] * len(rewritten)


def test_a_blob_holding_utf8_json_is_still_read(caplog) -> None:
    from superlocalmemory.storage.json_columns import load_json_column

    assert load_json_column('["東京"]'.encode(), [], column="t.c") == ["東京"]
    with caplog.at_level(logging.WARNING):
        assert load_json_column(NOT_TEXT, [], column="t.c", row="r1") == []
    assert any("t.c holds bytes that are not UTF-8" in r.getMessage() for r in caplog.records)
    assert load_json_column(None, {}, column="t.c") == {}
    assert load_json_column("", None, column="t.c") is None
    assert load_json_column('{"a": 1}', {}, column="t.c") == {"a": 1}


def _strict(raw, default, **_kwargs):
    """The reader before the fix: ``json.loads`` on whatever SQLite returned."""
    return default if raw is None or raw in ("", b"") else json.loads(raw)


def _strict_checked(raw, default, **kwargs):
    return _strict(raw, default, **kwargs), False


_READERS = (
    "superlocalmemory.storage.json_columns.load_json_column",
    "superlocalmemory.encoding.scene_builder.load_json_column",
    "superlocalmemory.encoding.observation_builder.load_json_column",
    "superlocalmemory.storage.fact_context.load_json_column",
    "superlocalmemory.storage.source_identity.load_json_column",
)
_CHECKED_READERS = (
    "superlocalmemory.storage.json_columns.load_json_column_checked",
    "superlocalmemory.encoding.scene_builder.load_json_column_checked",
    "superlocalmemory.encoding.observation_builder.load_json_column_checked",
)


def test_a_save_dead_lettered_on_a_damaged_row_retries_and_clears(engine_with_mock_deps) -> None:
    import sqlite3
    from contextlib import ExitStack
    from unittest.mock import patch

    from superlocalmemory.core.ingestion_command import _MAX_AUTOMATIC_MATERIALIZATION_ATTEMPTS
    from superlocalmemory.core.ops_remediation import resolve_operation

    engine = engine_with_mock_deps
    for text in EARLIER:
        assert _save(engine, text).state is IngestionState.COMPLETE
    engine._db.execute("UPDATE atomic_facts SET entities_json = ?", (NOT_TEXT,))
    engine._db.execute("UPDATE memory_scenes SET entity_ids_json = ?", (NOT_TEXT,))
    receipt = canonical_store(
        engine, LATER, source_type="python-api",
        trusted_actor_id=local_trusted_actor_id("python-api"),
        metadata={"project": "demo"}, require_complete=False, return_receipt=True,
    )
    operation_id = receipt.operation_id
    command = build_engine_ingestion_command(engine)

    # The reader before the fix: every automatic attempt fails on the damaged rows.
    with ExitStack() as stack:
        for target in _READERS:
            stack.enter_context(patch(target, _strict, create=True))
        for target in _CHECKED_READERS:
            stack.enter_context(patch(target, _strict_checked, create=True))
        for _ in range(_MAX_AUTOMATIC_MATERIALIZATION_ATTEMPTS):
            failed = command.materialize(operation_id)
    assert failed.state is IngestionState.FAILED
    assert failed.last_error == CODEC_ERROR
    db_path = engine._config.db_path
    with sqlite3.connect(str(db_path)) as conn:
        errors = [r[0] for r in conn.execute(
            "SELECT error FROM dead_letter_operations WHERE original_op_id=?", (operation_id,))]
    assert errors == [CODEC_ERROR]

    # With the fix, the operator's retry completes it and clears the row.
    outcome = resolve_operation(db_path, engine, operation_id, "retry")

    assert outcome["success"] is True, outcome
    done = command.repository.get(operation_id)
    assert done.state is IngestionState.COMPLETE, done.last_error
    assert done.raw_content == LATER
    with sqlite3.connect(str(db_path)) as conn:
        left = conn.execute(
            "SELECT COUNT(*) FROM dead_letter_operations WHERE original_op_id=?",
            (operation_id,)).fetchone()[0]
        still_damaged = conn.execute(
            "SELECT COUNT(*) FROM atomic_facts WHERE entities_json = ?", (NOT_TEXT,)).fetchone()[0]
    assert left == 0
    # Exactly the earlier facts: no save rewrites an older fact, so each keeps its bytes.
    assert still_damaged == len(EARLIER)
