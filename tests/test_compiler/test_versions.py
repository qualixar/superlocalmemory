# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Versions first (WS-6): a derived artifact keeps its previous row before every overwrite.

One test per artifact, each driving the module that really writes it, so the
test fails if the reroute is removed.
"""

from __future__ import annotations

import json

import pytest

from superlocalmemory.compiler import versions
from superlocalmemory.storage.models import EntityProfile, MemoryScene
from tests.test_compiler.conftest import ENTITY, PROFILE, raw, versions_of


def _row_of(version) -> dict:
    return json.loads(version["row_json"])


# -- entity profiles -------------------------------------------------------------------------------

def test_store_entity_profile_keeps_the_profile_it_replaces(store) -> None:
    db, ids = store
    first = EntityProfile(entity_id=ENTITY, profile_id=PROFILE, knowledge_summary="first text",
                          fact_ids=ids[:2])
    db.store_entity_profile(first)
    assert versions_of(db, "entity_profile") == []  # nothing was overwritten yet
    db.store_entity_profile(EntityProfile(
        profile_entry_id=first.profile_entry_id, entity_id=ENTITY, profile_id=PROFILE,
        knowledge_summary="second text", fact_ids=ids))
    kept = versions_of(db, "entity_profile")
    assert len(kept) == 1 and _row_of(kept[0])["knowledge_summary"] == "first text"
    live = db.execute("SELECT knowledge_summary FROM entity_profiles")[0]["knowledge_summary"]
    assert live == "second text"
    links = {r["fact_id"] for r in db.execute("SELECT fact_id FROM artifact_version_facts")}
    assert links == set(ids[:2])


def test_the_observation_builder_keeps_the_profile_it_replaces(store) -> None:
    from superlocalmemory.encoding.observation_builder import ObservationBuilder

    db, ids = store
    builder = ObservationBuilder(db)
    base = EntityProfile(entity_id=ENTITY, profile_id=PROFILE, knowledge_summary="v1",
                         fact_ids=ids)
    builder._save_profile(base)
    builder._save_profile(EntityProfile(
        profile_entry_id=base.profile_entry_id, entity_id=ENTITY, profile_id=PROFILE,
        knowledge_summary="v2", fact_ids=ids))
    assert [_row_of(v)["knowledge_summary"] for v in versions_of(db, "entity_profile")] == ["v1"]


def test_the_entity_compiler_keeps_the_compiled_view_it_rewrites(store) -> None:
    from superlocalmemory.learning.entity_compiler import EntityCompiler

    db, _ids = store
    compiler = EntityCompiler(db.db_path)
    compiler._compile_single(PROFILE, "", ENTITY, "SuperLocalMemory")
    assert versions_of(db, "entity_profile") == []
    db.execute("UPDATE atomic_facts SET confidence = 0.2 WHERE fact_id IN "
               "(SELECT fact_id FROM atomic_facts LIMIT 1)")
    db.execute("INSERT INTO fact_importance (fact_id, profile_id, pagerank_score) "
               "SELECT fact_id, profile_id, 0.5 FROM atomic_facts LIMIT 1")
    compiler._compile_single(PROFILE, "", ENTITY, "SuperLocalMemory")
    assert len(versions_of(db, "entity_profile")) == 1
    assert "SuperLocalMemory" in _row_of(versions_of(db, "entity_profile")[0])["compiled_truth"]


def test_maintenance_keeps_the_full_text_before_it_cuts_a_summary(store) -> None:
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.core.maintenance import run_maintenance

    db, ids = store
    long_text = "x" * 3000
    db.store_entity_profile(EntityProfile(entity_id=ENTITY, profile_id=PROFILE,
                                          knowledge_summary=long_text, fact_ids=ids))
    run_maintenance(db, SLMConfig.default(), PROFILE)
    kept = versions_of(db, "entity_profile")
    assert kept and len(_row_of(kept[0])["knowledge_summary"]) == 3000
    assert len(db.execute("SELECT knowledge_summary FROM entity_profiles")[0][0]) == 2048


# -- summaries and persona -------------------------------------------------------------------------

def _communities(db, ids):
    db.execute("INSERT OR REPLACE INTO entity_communities (profile_id, entity_id, community_id) "
               "VALUES (?, ?, 1)", (PROFILE, ENTITY))
    db.execute("INSERT OR REPLACE INTO entity_communities (profile_id, entity_id, community_id) "
               "VALUES (?, 'ent-other', 1)", (PROFILE,))


def test_community_summaries_are_kept_before_they_are_rebuilt(store) -> None:
    from superlocalmemory.core.community_summary import CommunitySummaryBuilder

    db, ids = store
    _communities(db, ids)
    builder = CommunitySummaryBuilder(db)
    assert builder.compute_and_store(PROFILE)["summaries_written"] == 1
    assert versions_of(db, "community_summary") == []
    db.execute("UPDATE community_summaries SET summary = 'older words' WHERE profile_id = ?",
               (PROFILE,))
    builder.compute_and_store(PROFILE)
    kept = versions_of(db, "community_summary")
    assert len(kept) == 1 and _row_of(kept[0])["summary"] == "older words"
    assert {r["fact_id"] for r in db.execute(
        "SELECT fact_id FROM artifact_version_facts")} == set(ids)


def test_the_persona_is_kept_before_it_is_rebuilt(store) -> None:
    from superlocalmemory.core.community_summary import CommunitySummaryBuilder
    from superlocalmemory.core.progressive_abstraction import ProgressiveAbstraction

    db, ids = store
    _communities(db, ids)
    CommunitySummaryBuilder(db).compute_and_store(PROFILE)
    persona = ProgressiveAbstraction(db)
    persona.compute_and_store(PROFILE)
    assert versions_of(db, "persona") == []
    db.execute("UPDATE persona_summary SET summary = 'old persona' WHERE profile_id = ?", (PROFILE,))
    persona.compute_and_store(PROFILE)
    kept = versions_of(db, "persona")
    assert len(kept) == 1 and _row_of(kept[0])["summary"] == "old persona"
    # the persona version names the facts of the communities it stitched
    assert {r["fact_id"] for r in db.execute(
        "SELECT f.fact_id FROM artifact_version_facts f JOIN artifact_versions v "
        "ON v.version_id = f.version_id WHERE v.artifact = 'persona'")} == set(ids)


def test_a_consolidated_summary_is_kept_when_a_second_pass_rewrites_its_coverage(store) -> None:
    from superlocalmemory.core.fact_consolidator import _consolidate_cluster

    db, ids = store
    text = "SuperLocalMemory shipped three releases, each with a measured recall gain."
    conn = raw(db)
    try:
        _consolidate_cluster(conn, PROFILE, ENTITY, "SuperLocalMemory", ids, False,
                             _presummary=text)
        assert versions_of(db, "consolidated_summary") == []
        # a fact joins the cluster later: the coverage window of the same text moves
        conn.execute("UPDATE atomic_facts SET created_at = '2026-12-01T00:00:00+00:00' "
                     "WHERE fact_id = ?", (ids[-1],))
        _consolidate_cluster(conn, PROFILE, ENTITY, "SuperLocalMemory", ids, False,
                             _presummary=text)
    finally:
        conn.close()
    kept = versions_of(db, "consolidated_summary")
    assert len(kept) == 1
    assert json.loads(_row_of(kept[0])["source_fact_ids"]) == ids
    assert _row_of(kept[0])["source_latest"] != "2026-12-01T00:00:00+00:00"


# -- scenes and core blocks ------------------------------------------------------------------------

def test_a_scene_is_kept_before_its_theme_or_members_change(store) -> None:
    db, ids = store
    scene = MemoryScene(profile_id=PROFILE, theme="launch", fact_ids=ids[:1], entity_ids=[ENTITY])
    db.store_scene(scene)
    assert versions_of(db, "scene") == []
    db.store_scene(MemoryScene(scene_id=scene.scene_id, profile_id=PROFILE, theme="launch week",
                               fact_ids=ids, entity_ids=[ENTITY], created_at=scene.created_at))
    kept = versions_of(db, "scene")
    assert len(kept) == 1 and _row_of(kept[0])["theme"] == "launch"


def test_the_scene_builder_keeps_the_scene_it_replaces(store) -> None:
    from superlocalmemory.encoding.scene_builder import SceneBuilder

    db, ids = store
    scene = MemoryScene(profile_id=PROFILE, theme="a", fact_ids=ids[:1], entity_ids=[])
    builder = SceneBuilder.__new__(SceneBuilder)
    builder._db = db
    builder._save_scene(scene)
    builder._save_scene(MemoryScene(scene_id=scene.scene_id, profile_id=PROFILE, theme="b",
                                    fact_ids=ids, entity_ids=[], created_at=scene.created_at))
    assert [_row_of(v)["theme"] for v in versions_of(db, "scene")] == ["a"]


def test_a_core_block_is_kept_when_replaced_and_when_deleted(store) -> None:
    db, ids = store
    db.store_core_block("b1", PROFILE, "user_profile", "first block", json.dumps(ids), 11)
    db.store_core_block("b2", PROFILE, "user_profile", "second block", json.dumps(ids), 12)
    assert [_row_of(v)["content"] for v in versions_of(db, "core_block")] == ["first block"]
    db.delete_core_blocks(PROFILE)
    assert [_row_of(v)["content"] for v in versions_of(db, "core_block")] == [
        "first block", "second block"]


def test_a_person_editing_a_core_block_over_http_keeps_the_old_text(store, monkeypatch) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from superlocalmemory.server.routes import helpers, v3_api

    db, ids = store
    db.store_core_block("b1", PROFILE, "user_profile", "original block", json.dumps(ids), 14)
    monkeypatch.setattr(helpers, "DB_PATH", db.db_path)
    monkeypatch.setattr(helpers, "get_active_profile", lambda: PROFILE)
    monkeypatch.setattr(v3_api, "authorize_route_mutation", lambda *a, **k: _Done())
    app = FastAPI()
    app.include_router(v3_api.router)
    reply = TestClient(app).put("/api/v3/core-memory/b1", json={"content": "a person's text"})
    assert reply.status_code == 200, reply.text
    assert [_row_of(v)["content"] for v in versions_of(db, "core_block")] == ["original block"]


class _Done:
    def complete(self) -> None:
        return None


# -- the module itself -----------------------------------------------------------------------------

def test_an_unchanged_row_is_not_copied_again(store) -> None:
    db, ids = store
    db.store_core_block("b1", PROFILE, "user_profile", "same text", json.dumps(ids), 9)
    for n in range(4):
        db.store_core_block(f"b{n + 2}", PROFILE, "user_profile", "same text", json.dumps(ids), 9)
    assert len(versions_of(db, "core_block")) == 1


def test_only_the_last_few_versions_of_a_key_are_kept(store) -> None:
    db, ids = store
    for n in range(versions.KEEP_LAST + 3):
        db.store_core_block(f"b{n}", PROFILE, "user_profile", f"text {n}", json.dumps(ids), 6)
    kept = versions_of(db, "core_block")
    assert len(kept) == versions.KEEP_LAST
    assert [v["seq"] for v in kept] == list(range(3, 3 + versions.KEEP_LAST))
    # pruning removed the links of the dropped versions too
    assert db.execute("SELECT COUNT(*) AS n FROM artifact_version_facts")[0]["n"] == (
        versions.KEEP_LAST * len(ids))


def test_a_version_restores_exactly_and_keeps_the_row_it_replaces(store) -> None:
    db, ids = store
    db.store_core_block("b1", PROFILE, "user_profile", "old text", json.dumps(ids), 8)
    db.store_core_block("b2", PROFILE, "user_profile", "new text", json.dumps(ids), 8)
    version = versions_of(db, "core_block")[0]["version_id"]
    conn = raw(db)
    try:
        versions.restore_version(conn, PROFILE, version)
    finally:
        conn.close()
    assert db.execute("SELECT content FROM core_memory_blocks")[0]["content"] == "old text"
    assert [_row_of(v)["content"] for v in versions_of(db, "core_block")] == [
        "old text", "new text"]


def test_a_store_without_the_tables_still_writes(store) -> None:
    db, ids = store
    db.execute("DROP TABLE artifact_version_facts")
    db.execute("DROP TABLE artifact_versions")
    db.store_core_block("b1", PROFILE, "user_profile", "text", json.dumps(ids), 4)
    db.store_core_block("b2", PROFILE, "user_profile", "other", json.dumps(ids), 5)
    assert db.execute("SELECT content FROM core_memory_blocks")[0]["content"] == "other"


def test_it_never_copies_a_whole_table_and_rejects_unknown_artifacts(store) -> None:
    db, _ids = store
    conn = raw(db)
    try:
        with pytest.raises(ValueError, match="never copies a whole table"):
            versions.keep_previous(conn, "persona")
        with pytest.raises(ValueError, match="unknown derived artifact"):
            versions.keep_previous(conn, "mystery", {"profile_id": PROFILE})
        with pytest.raises(ValueError, match="bad column"):
            versions.keep_previous(conn, "persona", {"profile_id; DROP TABLE x": 1})
    finally:
        conn.close()


def test_a_version_quoting_a_fact_being_erased_is_not_shown_or_restorable(store) -> None:
    db, ids = store
    db.store_core_block("b1", PROFILE, "user_profile", "quotes the fact", json.dumps([ids[0]]), 15)
    db.store_core_block("b2", PROFILE, "user_profile", "newer", json.dumps(ids[1:]), 5)
    version = versions_of(db, "core_block")[0]["version_id"]
    assert len(versions.list_versions(db, PROFILE, "core_block")) == 1
    db.execute("INSERT INTO projection_tombstones (profile_id, fact_id, erasure_id, memory_id, "
               "created_at) VALUES (?, ?, 'e1', NULL, 1.0)", (PROFILE, ids[0]))
    assert versions.list_versions(db, PROFILE, "core_block") == []
    assert versions.get_version(db, PROFILE, version) is None
    conn = raw(db)
    try:
        with pytest.raises(LookupError):
            versions.restore_version(conn, PROFILE, version)
    finally:
        conn.close()
