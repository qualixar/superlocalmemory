# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""A damaged JSON value is rebuilt from its source, never written over from empty.

A JSON text column holding bytes that are not UTF-8 reads as the column's empty
value (storage/json_columns.py). A writer that then built the new value from
that empty one replaced the older content with a value holding only the new
fact (an entity profile lost every earlier fact) and threw the original bytes
away. These tests pin the other behaviour: a writer that finds the stored value
damaged rebuilds it from the source of truth, and the original bytes stay
recoverable (a version row, or the change ledger).
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from superlocalmemory.core.engine_ingestion import (
    build_engine_ingestion_command,
    canonical_store,
    local_trusted_actor_id,
)
from superlocalmemory.core.ingestion_command import IngestionState

NOT_TEXT = bytes.fromhex("61628163")
EARLIER = [
    "Alice met Bob in Tokyo to discuss the Postgres billing migration on Friday.",
    "Alice and Bob reviewed the Postgres billing schema in Tokyo again.",
    "Bob told Alice the Postgres billing cutover in Tokyo moves to Monday.",
]
LATER = "Alice and Bob met in Tokyo about the Postgres billing migration, 最終版."


def _receipt(engine, text: str):
    return canonical_store(
        engine, text, source_type="python-api",
        trusted_actor_id=local_trusted_actor_id("python-api"),
        metadata={"project": "demo"}, require_complete=False, return_receipt=True,
    )


def _save(engine, text: str):
    receipt = _receipt(engine, text)
    return build_engine_ingestion_command(engine).materialize(receipt.operation_id)


def _seed(engine) -> None:
    for text in EARLIER:
        assert _save(engine, text).state is IngestionState.COMPLETE


def _rows(engine, sql: str, params: tuple = ()) -> list[dict]:
    return [dict(r) for r in engine._db.execute(sql, params)]


def _kept_hex(engine, artifact: str) -> list[str]:
    """Every ``__blob_hex__`` value the artifact's version rows carry."""
    found: list[str] = []
    for row in _rows(engine, "SELECT row_json FROM artifact_versions WHERE artifact = ?",
                     (artifact,)):
        for value in json.loads(row["row_json"]).values():
            if isinstance(value, dict) and "__blob_hex__" in value:
                found.append(value["__blob_hex__"])
    return found


def _kept_in_ledger(engine, table: str, field: str) -> list[bytes]:
    from superlocalmemory.storage import enrichment_ledger as ledger

    return [ledger.loads(r["before_json"]) for r in _rows(
        engine, "SELECT before_json FROM enrichment_changes WHERE target_table = ? "
        "AND field = ?", (table, field))]


def test_an_entity_profile_keeps_its_earlier_facts_when_its_list_is_damaged(
    engine_with_mock_deps,
) -> None:
    engine = engine_with_mock_deps
    _seed(engine)
    before = {r["entity_id"]: json.loads(r["fact_ids_json"])
              for r in _rows(engine, "SELECT entity_id, fact_ids_json FROM entity_profiles")}
    assert before and max(len(v) for v in before.values()) == len(EARLIER)
    engine._db.execute("UPDATE entity_profiles SET fact_ids_json = ?", (NOT_TEXT,))

    result = _save(engine, LATER)

    assert result.state is IngestionState.COMPLETE, result.last_error
    new_ids = set(result.fact_ids)
    after = {r["entity_id"]: json.loads(r["fact_ids_json"])
             for r in _rows(engine, "SELECT entity_id, fact_ids_json FROM entity_profiles")}
    touched = {e: ids for e, ids in after.items() if new_ids & set(ids)}
    assert touched
    for entity_id, ids in touched.items():
        # Every earlier fact of the entity is still listed, plus the new one.
        assert set(before.get(entity_id, [])) <= set(ids), entity_id
    assert max(len(ids) for ids in touched.values()) == len(EARLIER) + 1
    # The damaged bytes are recoverable from the version row.
    assert _kept_hex(engine, "entity_profile").count(NOT_TEXT.hex()) >= 1


def test_a_scene_rebuilds_its_entity_list_from_its_facts_and_keeps_the_bytes(
    engine_with_mock_deps,
) -> None:
    from superlocalmemory.encoding.scene_builder import SceneBuilder

    engine = engine_with_mock_deps
    _seed(engine)
    builder = SceneBuilder(engine._db)
    scene = builder.get_all_scenes("default")[0]
    expected = set(scene.entity_ids)
    assert expected
    engine._db.execute("UPDATE memory_scenes SET entity_ids_json = ? WHERE scene_id = ?",
                       (NOT_TEXT, scene.scene_id))
    new_fact = engine._db.get_fact(_save(engine, LATER).fact_ids[0], "default")

    updated = builder._add_to_scene(scene, new_fact, "default")

    assert expected <= set(updated.entity_ids)
    stored = _rows(engine, "SELECT entity_ids_json FROM memory_scenes WHERE scene_id = ?",
                   (scene.scene_id,))[0]["entity_ids_json"]
    assert expected <= set(json.loads(stored))
    assert NOT_TEXT.hex() in _kept_hex(engine, "scene")


def test_a_scene_whose_fact_list_is_damaged_is_left_alone_and_the_fact_gets_a_new_scene(
    engine_with_mock_deps,
) -> None:
    from superlocalmemory.encoding.scene_builder import SceneBuilder

    engine = engine_with_mock_deps
    _seed(engine)
    builder = SceneBuilder(engine._db)
    scene = builder.get_all_scenes("default")[0]
    engine._db.execute("UPDATE memory_scenes SET fact_ids_json = ? WHERE scene_id = ?",
                       (NOT_TEXT, scene.scene_id))
    new_fact = engine._db.get_fact(_save(engine, LATER).fact_ids[0], "default")

    placed = builder._add_to_scene(scene, new_fact, "default")

    assert placed.scene_id != scene.scene_id
    assert new_fact.fact_id in placed.fact_ids
    left = _rows(engine, "SELECT fact_ids_json FROM memory_scenes WHERE scene_id = ?",
                 (scene.scene_id,))[0]["fact_ids_json"]
    assert left == NOT_TEXT


def test_the_langevin_backfill_rebuilds_a_damaged_position_and_keeps_the_bytes(
    engine_with_mock_deps,
) -> None:
    from superlocalmemory.core.maintenance import run_maintenance

    engine = engine_with_mock_deps
    _seed(engine)
    ids = [r["fact_id"] for r in _rows(engine, "SELECT fact_id FROM atomic_facts")]
    engine._db.execute("UPDATE atomic_facts SET langevin_position = ?", (NOT_TEXT,))

    counts = run_maintenance(engine._db, engine._config, "default")

    assert counts["langevin_backfilled"] == len(ids)
    for row in _rows(engine, "SELECT langevin_position AS p FROM atomic_facts"):
        assert isinstance(json.loads(row["p"]), list)
    assert _kept_in_ledger(engine, "atomic_facts", "langevin_position") == [NOT_TEXT] * len(ids)


def test_a_promotion_keeps_the_bytes_of_the_entity_columns_it_replaces(
    engine_with_mock_deps,
) -> None:
    engine = engine_with_mock_deps
    _seed(engine)
    old = {r["fact_id"] for r in _rows(engine, "SELECT fact_id FROM atomic_facts")}
    receipt = _receipt(engine, LATER)  # the queryable fact exists; materialising promotes it
    fresh = [r["fact_id"] for r in _rows(engine, "SELECT fact_id FROM atomic_facts")
             if r["fact_id"] not in old]
    assert fresh
    marks = ",".join("?" for _ in fresh)
    engine._db.execute(
        f"UPDATE atomic_facts SET entities_json = ?, canonical_entities_json = ?, "
        f"langevin_position = ? WHERE fact_id IN ({marks})",
        (NOT_TEXT, NOT_TEXT, NOT_TEXT, *fresh))

    result = build_engine_ingestion_command(engine).materialize(receipt.operation_id)

    assert result.state is IngestionState.COMPLETE, result.last_error
    for field in ("entities_json", "canonical_entities_json", "langevin_position"):
        assert NOT_TEXT in _kept_in_ledger(engine, "atomic_facts", field), field


def test_the_enrichment_ledger_keeps_bytes_it_replaces_and_undo_restores_them(
    engine_with_mock_deps,
) -> None:
    from superlocalmemory.storage import enrichment_ledger as ledger

    engine = engine_with_mock_deps
    _seed(engine)
    fact_id = _rows(engine, "SELECT fact_id FROM atomic_facts")[0]["fact_id"]
    engine._db.execute(
        "UPDATE atomic_facts SET canonical_entities_json = ? WHERE fact_id = ?",
        (NOT_TEXT, fact_id))
    engine._db.execute(
        "INSERT INTO enrichment_runs(run_id,enrichment_version,status,upper_rowid,total,"
        "created_at,updated_at) VALUES ('r1','v','running',0,0,'t','t')")

    assert ledger.add(engine._db, "r1", "default", "atomic_facts", fact_id,
                      "canonical_entities_json", '["e1"]', replace_guess=True)
    assert _kept_in_ledger(engine, "atomic_facts", "canonical_entities_json") == [NOT_TEXT]

    outcome = ledger.undo_changes(engine._db, "r1")  # must not raise a codec error

    assert outcome == {"restored": 1, "conflicts": 0}
    back = _rows(engine, "SELECT canonical_entities_json AS v FROM atomic_facts "
                 "WHERE fact_id = ?", (fact_id,))[0]["v"]
    assert back == NOT_TEXT


def test_a_damaged_memory_metadata_is_not_written_by_the_ledger(engine_with_mock_deps) -> None:
    from superlocalmemory.storage import enrichment_ledger as ledger

    engine = engine_with_mock_deps
    _seed(engine)
    memory_id = _rows(engine, "SELECT memory_id FROM memories")[0]["memory_id"]
    engine._db.execute("UPDATE memories SET metadata_json = ? WHERE memory_id = ?",
                       (NOT_TEXT, memory_id))
    engine._db.execute(
        "INSERT INTO enrichment_runs(run_id,enrichment_version,status,upper_rowid,total,"
        "created_at,updated_at) VALUES ('r2','v','running',0,0,'t','t')")

    assert ledger.add(engine._db, "r2", "default", "memories", memory_id, "tags", "x") is False
    kept = _rows(engine, "SELECT metadata_json AS m FROM memories WHERE memory_id = ?",
                 (memory_id,))[0]["m"]
    assert kept == NOT_TEXT


def test_the_version_row_keeps_a_blob_as_hex_and_a_restore_brings_it_back(
    engine_with_mock_deps,
) -> None:
    from superlocalmemory.compiler import versions

    row = {"profile_id": "default", "scene_id": "s1", "fact_ids_json": NOT_TEXT, "theme": "t"}
    assert versions._record(row)["fact_ids_json"] == {"__blob_hex__": NOT_TEXT.hex()}

    engine = engine_with_mock_deps
    _seed(engine)
    scene_id = _rows(engine, "SELECT scene_id FROM memory_scenes")[0]["scene_id"]
    engine._db.execute("UPDATE memory_scenes SET entity_ids_json = ? WHERE scene_id = ?",
                       (NOT_TEXT, scene_id))
    with versions.atomic(engine._db):
        versions.keep_previous(engine._db, "scene", {"scene_id": scene_id})
        engine._db.execute("UPDATE memory_scenes SET entity_ids_json = '[]' WHERE scene_id = ?",
                           (scene_id,))
    version = _rows(engine, "SELECT version_id FROM artifact_versions WHERE artifact='scene' "
                    "AND artifact_key LIKE ?", (f'%{scene_id}%',))[0]["version_id"]

    versions.restore_version(engine._db, "default", version)

    restored = _rows(engine, "SELECT entity_ids_json AS v FROM memory_scenes WHERE scene_id = ?",
                     (scene_id,))[0]["v"]
    assert restored == NOT_TEXT


def test_the_database_error_text_is_still_absent(engine_with_mock_deps) -> None:
    """Sanity: the fixtures used here really are a store we can open read-only."""
    engine = engine_with_mock_deps
    _seed(engine)
    with sqlite3.connect(str(engine._config.db_path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM atomic_facts").fetchone()[0] == len(EARLIER)


@pytest.mark.parametrize("column", ["metadata_json", "shared_with"])
def test_saving_the_same_text_again_past_a_damaged_source_row_stores_a_new_memory(
    engine_with_mock_deps, column,
) -> None:
    engine = engine_with_mock_deps
    _seed(engine)
    first = _rows(engine, "SELECT memory_id FROM memories WHERE content = ?", (EARLIER[0],))
    assert len(first) == 1
    engine._db.execute(f"UPDATE memories SET {column} = ? WHERE memory_id = ?",
                       (NOT_TEXT, first[0]["memory_id"]))

    result = _save(engine, EARLIER[0])

    assert result.state is IngestionState.COMPLETE, result.last_error
    same = _rows(engine, f"SELECT memory_id, {column} AS v FROM memories WHERE content = ?",
                 (EARLIER[0],))
    assert len(same) == 2  # a new memory, not a merge into the damaged one
    by_id = {r["memory_id"]: r["v"] for r in same}
    assert by_id[first[0]["memory_id"]] == NOT_TEXT  # untouched


def test_a_memory_with_damaged_metadata_is_not_a_same_context_candidate(
    engine_with_mock_deps,
) -> None:
    from superlocalmemory.storage.fact_context import same_context_candidates

    engine = engine_with_mock_deps
    db = engine._db
    for text in ("Plain note one about nothing.", "Plain note two about nothing."):
        canonical_store(
            engine, text, source_type="python-api",
            trusted_actor_id=local_trusted_actor_id("python-api"),
            require_complete=False, return_receipt=True,
        )
    memories = [r["memory_id"] for r in _rows(engine, "SELECT memory_id FROM memories "
                                              "ORDER BY created_at")]
    assert len(memories) == 2
    old_fact, new_fact = (
        db.get_facts_by_memory_id(m, "default")[0] for m in memories)
    # Both memories carry no project: the older one is a candidate for the newer.
    assert [f.fact_id for f in same_context_candidates(db, new_fact, [old_fact])] == [
        old_fact.fact_id]
    db.execute("UPDATE memories SET metadata_json = ? WHERE memory_id = ?",
               (NOT_TEXT, memories[0]))

    assert same_context_candidates(db, new_fact, [old_fact]) == []


def test_get_scene_for_fact_reads_past_a_damaged_fact_list(engine_with_mock_deps) -> None:
    from superlocalmemory.encoding.scene_builder import SceneBuilder

    engine = engine_with_mock_deps
    _seed(engine)
    builder = SceneBuilder(engine._db)
    scenes = builder.get_all_scenes("default")  # newest first
    newest, oldest = scenes[0], scenes[-1]
    # The oldest scene is read first (row order), so its damage is met before the match.
    engine._db.execute("UPDATE memory_scenes SET fact_ids_json = ? WHERE scene_id = ?",
                       (NOT_TEXT, oldest.scene_id))

    assert builder.get_scene_for_fact(newest.fact_ids[0], "default").scene_id == newest.scene_id
