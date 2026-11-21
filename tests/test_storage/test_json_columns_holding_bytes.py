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
once; the damaged row itself is left as it is (nothing is rewritten or
invented), and a BLOB that does hold UTF-8 JSON is still read.
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
    for value in values:
        if value is None or value == NOT_TEXT:
            continue
        json.loads(value)


def test_a_blob_holding_utf8_json_is_still_read(caplog) -> None:
    from superlocalmemory.storage.json_columns import load_json_column

    assert load_json_column('["東京"]'.encode(), [], column="t.c") == ["東京"]
    with caplog.at_level(logging.WARNING):
        assert load_json_column(NOT_TEXT, [], column="t.c", row="r1") == []
    assert any("t.c holds bytes that are not UTF-8" in r.getMessage() for r in caplog.records)
    assert load_json_column(None, {}, column="t.c") == {}
    assert load_json_column("", None, column="t.c") is None
    assert load_json_column('{"a": 1}', {}, column="t.c") == {"a": 1}
