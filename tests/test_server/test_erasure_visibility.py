"""Dashboard reads must not display a fact after erasure has started."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from superlocalmemory.core.delete_refusal import STALE_AFTER_S, _drop_stale_erasure


@pytest.fixture(autouse=True)
def _origin_firewall_off(monkeypatch):
    """These tests are about what session start selects, not the origin firewall: they save through agent
    doors (``agent_inferred``), which the firewall (on by default since 19 Nov) holds back until a promotion.
    tests/test_trust_firewall covers that rule."""
    monkeypatch.setenv("SLM_FEATURE_TRUST_TAINT", "off")


@pytest.fixture()
def dashboard(engine_with_mock_deps):
    from superlocalmemory.server.profile_runtime import bind_profile_runtime
    from superlocalmemory.server.unified_daemon import create_app

    engine = engine_with_mock_deps
    db = engine._db
    for name in ("erasuremarker", "controlmarker"):
        db.execute("INSERT INTO memories(memory_id,profile_id,content) VALUES (?, 'default',?)",
                   (name, name + " source text"))
        db.execute("INSERT INTO atomic_facts(fact_id,memory_id,profile_id,content,fact_type,"
                   "created_at) VALUES (?, ?, 'default', ?, 'opinion', datetime('now'))",
                   (name, name, name))
    app = create_app()
    app.state.engine = engine
    app.state.config = engine._config
    bind_profile_runtime(app.state, engine, engine._config)
    descriptor = app.state.daemon_descriptor
    headers = {"X-SLM-Daemon-Capability": descriptor.capability,
               "X-SLM-Target-Instance": descriptor.instance_id}
    yield TestClient(app), headers, db


@pytest.mark.parametrize("path", [
    "/api/memories?limit=50", "/api/graph", "/api/v3/timeline/?range=30d",
    "/api/facts/erasuremarker", "/api/memories/erasuremarker/detail",
    "/api/memories/erasuremarker/facts",
    "/api/v3/insights/changed_this_week", "/api/v3/insights/opinions",
])
def test_dashboard_reads_hide_tombstone_until_heal(dashboard, path):
    client, headers, db = dashboard
    before = client.get(path, headers=headers)
    assert before.status_code == 200, before.text
    assert "erasuremarker" in before.text, before.text
    db.execute("INSERT INTO projection_tombstones(profile_id,fact_id,erasure_id,created_at) "
               "VALUES ('default', 'erasuremarker', 'synthetic-erasure', ?)",
               (time.time() - STALE_AFTER_S - 1,))
    during = client.get(path, headers=headers)
    assert during.status_code in (200, 404), during.text
    assert "erasuremarker" not in during.text, during.text
    if path in {"/api/memories?limit=50", "/api/graph", "/api/v3/timeline/?range=30d"}:
        assert "controlmarker" in during.text, during.text
    assert _drop_stale_erasure(db, "default", "erasuremarker")
    after = client.get(path, headers=headers)
    assert after.status_code == 200 and "erasuremarker" in after.text, after.text


def test_visible_sibling_detail_does_not_reveal_erasing_source(dashboard):
    client, headers, db = dashboard
    db.execute("UPDATE atomic_facts SET memory_id='erasuremarker' "
               "WHERE fact_id='controlmarker'")
    before = client.get('/api/facts/controlmarker', headers=headers)
    assert before.status_code == 200 and 'erasuremarker source text' in before.text
    db.execute("INSERT INTO projection_tombstones(profile_id,fact_id,erasure_id,created_at) "
               "VALUES ('default','erasuremarker','synthetic-erasure',?)", (time.time(),))
    during = client.get('/api/facts/controlmarker', headers=headers)
    assert during.status_code == 200 and 'controlmarker' in during.text
    assert 'erasuremarker source text' not in during.text


@pytest.mark.parametrize('path', ['/api/v3/insights/contradictions','/api/v3/associations'])
def test_retained_edge_cannot_return_erased_endpoint_id(dashboard, path):
    client, headers, db = dashboard
    if path.endswith('contradictions'):
        db.execute("INSERT INTO graph_edges(edge_id,profile_id,source_id,target_id,edge_type) "
                   "VALUES ('edge','default','erasuremarker','controlmarker','contradiction')")
    else:
        db.execute("INSERT INTO association_edges(edge_id,profile_id,source_fact_id,target_fact_id,"
                   "association_type) VALUES ('edge','default','erasuremarker','controlmarker',"
                   "'user_defined')")
    before = client.get(path, headers=headers)
    assert before.status_code == 200 and 'erasuremarker' in before.text, before.text
    db.execute("INSERT INTO projection_tombstones(profile_id,fact_id,erasure_id,created_at) "
               "VALUES ('default','erasuremarker','synthetic-erasure',?)", (time.time(),))
    db.execute("DELETE FROM atomic_facts WHERE fact_id='erasuremarker'")
    during = client.get(path, headers=headers)
    assert during.status_code == 200 and 'erasuremarker' not in during.text, during.text


@pytest.mark.parametrize('path', ['/api/v3/insights/contradictions', '/api/v3/associations'])
def test_cross_profile_erased_endpoint_is_hidden_after_canonical_deletion(dashboard, path):
    client, headers, db = dashboard
    db.execute("INSERT INTO profiles(profile_id,name) VALUES ('other','other')")
    db.execute("INSERT INTO memories(memory_id,profile_id,content) "
               "VALUES ('foreign','other','synthetic foreign source')")
    db.execute("INSERT INTO atomic_facts(fact_id,memory_id,profile_id,content,scope) "
               "VALUES ('foreign','foreign','other','synthetic foreign fact','global')")
    if path.endswith('contradictions'):
        db.execute("INSERT INTO graph_edges(edge_id,profile_id,source_id,target_id,edge_type) "
                   "VALUES ('edge','default','controlmarker','foreign','contradiction')")
    else:
        db.execute("INSERT INTO association_edges(edge_id,profile_id,source_fact_id,target_fact_id,"
                   "association_type) VALUES ('edge','default','controlmarker','foreign',"
                   "'user_defined')")
    before = client.get(path, headers=headers)
    assert before.status_code == 200 and 'foreign' in before.text, before.text
    db.execute("INSERT INTO projection_tombstones(profile_id,fact_id,erasure_id,created_at) "
               "VALUES ('other','foreign','synthetic-erasure',?)", (time.time(),))
    db.execute("DELETE FROM atomic_facts WHERE fact_id='foreign'")
    during = client.get(path, headers=headers)
    assert during.status_code == 200 and 'foreign' not in during.text, during.text


_SUMMARY_TEXT = "The copper compass location is in the attic chest beside the old lantern and maps."


def _add_summary(db, source_ids):
    import json

    db.execute(
        "INSERT INTO consolidated_summaries(summary_id,profile_id,entity_id,entity_name,content,"
        "source_fact_ids,source_count,char_count,generated_by) VALUES "
        "('sum1','default','e1','Erasure Entity',?,?,?,?,'extractive')",
        (_SUMMARY_TEXT, json.dumps(source_ids), len(source_ids), len(_SUMMARY_TEXT)))


@pytest.mark.parametrize("flag", ["", "&include_unusable=true"])
def test_consolidated_summary_is_withheld_while_a_source_is_erasing(dashboard, flag):
    client, headers, db = dashboard
    path = "/api/v3/abstraction/consolidated?limit=10" + flag
    _add_summary(db, ["erasuremarker", "controlmarker"])
    assert "copper compass" in client.get(path, headers=headers).text
    db.execute("INSERT INTO projection_tombstones(profile_id,fact_id,erasure_id,created_at) "
               "VALUES ('default','erasuremarker','synthetic-erasure',?)",
               (time.time() - STALE_AFTER_S - 1,))
    during = client.get(path, headers=headers)
    assert during.status_code == 200 and "copper compass" not in during.text, during.text
    assert _drop_stale_erasure(db, "default", "erasuremarker")
    assert "copper compass" in client.get(path, headers=headers).text


def test_consolidated_summary_with_untouched_sources_stays_visible(dashboard):
    client, headers, db = dashboard
    _add_summary(db, ["controlmarker"])
    db.execute("INSERT INTO projection_tombstones(profile_id,fact_id,erasure_id,created_at) "
               "VALUES ('default','erasuremarker','synthetic-erasure',?)", (time.time(),))
    assert "copper compass" in client.get(
        "/api/v3/abstraction/consolidated?limit=10", headers=headers).text


def test_tombstone_write_invalidates_summaries_that_include_the_fact(dashboard):
    from superlocalmemory.core.transactions.tombstones import write_tombstones

    _, _, db = dashboard
    _add_summary(db, ["erasuremarker", "controlmarker"])
    assert write_tombstones(db, "default", ("erasuremarker",), "e-1", time.time(), "erasuremarker")
    left = db.execute("SELECT COUNT(*) AS n FROM consolidated_summaries")
    assert dict(list(left)[0])["n"] == 0


# -- other derived stores that quote a fact's words ---------------------------

_WORDS = "The copper compass is hidden in the attic chest."


def _erase(db, fact_id="erasuremarker"):
    db.execute("INSERT INTO projection_tombstones(profile_id,fact_id,erasure_id,created_at) "
               "VALUES ('default',?, 'synthetic-erasure', ?)", (fact_id, time.time()))


def _add_block(db):
    import json

    db.execute("INSERT INTO core_memory_blocks(block_id,profile_id,block_type,content,"
               "source_fact_ids,char_count) VALUES ('b1','default','user_profile',?,?,?)",
               (_WORDS, json.dumps(["erasuremarker", "controlmarker"]), len(_WORDS)))


def test_core_memory_blocks_are_withheld_while_a_source_is_erasing(dashboard):
    client, headers, db = dashboard
    _add_block(db)
    assert "copper compass" in client.get("/api/v3/core-memory", headers=headers).text
    _erase(db)
    during = client.get("/api/v3/core-memory", headers=headers)
    assert during.status_code == 200 and "copper compass" not in during.text, during.text


def test_soft_prompt_extractor_ignores_blocks_with_an_erasing_source(dashboard):
    from superlocalmemory.parameterization.pattern_extractor import PatternExtractor

    _, _, db = dashboard
    _add_block(db)
    extractor = PatternExtractor.__new__(PatternExtractor)
    extractor._db = db
    extractor._config = __import__("types").SimpleNamespace(min_confidence=0.1)
    assert extractor._extract_from_core_memory("default")
    _erase(db)
    assert extractor._extract_from_core_memory("default") == []


def _add_entity_profile(db):
    import json

    db.execute("INSERT INTO canonical_entities(entity_id,canonical_name,profile_id,entity_type) "
               "VALUES ('ent1','Compass Entity','default','concept')")
    db.execute("INSERT INTO entity_profiles(profile_entry_id,entity_id,profile_id,"
               "knowledge_summary,fact_ids_json,compiled_truth,project_name) "
               "VALUES ('ep1','ent1','default',?,?,?, '')",
               (_WORDS, json.dumps(["erasuremarker"]), _WORDS))


def test_entity_profile_text_is_withheld_while_a_source_is_erasing(dashboard):
    client, headers, db = dashboard
    _add_entity_profile(db)
    for path in ("/api/entity/Compass%20Entity", "/api/entity/list"):
        assert "copper compass" in client.get(path, headers=headers).text, path
    _erase(db)
    for path in ("/api/entity/Compass%20Entity", "/api/entity/list",
                 "/api/entity/list?search=copper"):
        during = client.get(path, headers=headers)
        assert during.status_code in (200, 404), during.text
        assert "copper compass" not in during.text, (path, during.text)


def test_database_scene_reads_withhold_erasing_sources(dashboard):
    import json

    _, _, db = dashboard
    _add_entity_profile(db)
    db.execute("INSERT INTO memory_scenes(scene_id,profile_id,theme,fact_ids_json) "
               "VALUES ('s1','default',?,?)", (_WORDS, json.dumps(["erasuremarker"])))
    assert db.get_scenes_for_fact("erasuremarker", "default")
    assert db.get_scenes_for_facts_batch(["erasuremarker"], "default")
    _erase(db)
    assert db.get_scenes_for_facts_batch(["erasuremarker"], "default") == {}
    assert db.get_scenes_for_fact("erasuremarker", "default", include_erasing=True)
    assert db.get_scenes_for_fact("erasuremarker", "default") == []


def test_persona_is_withheld_while_a_community_source_is_erasing(dashboard):
    import json

    client, headers, db = dashboard
    db.execute("INSERT INTO community_summaries(profile_id,community_id,summary,fact_ids_json) "
               "VALUES ('default',7,?,?)", (_WORDS, json.dumps(["erasuremarker"])))
    db.execute("INSERT INTO persona_summary(profile_id,summary,community_ids_json) "
               "VALUES ('default',?,?)", (_WORDS, json.dumps([7])))
    assert "copper compass" in client.get("/api/v3/abstraction/persona", headers=headers).text
    _erase(db)
    during = client.get("/api/v3/abstraction/persona", headers=headers)
    assert during.status_code == 200 and "copper compass" not in during.text, during.text


def test_tombstone_write_invalidates_soft_prompts_built_from_a_quoting_block(dashboard):
    import json

    from superlocalmemory.core.transactions.tombstones import write_tombstones

    _, _, db = dashboard
    _add_block(db)
    for pid, sources in (("sp-from-block", ["b1"]), ("sp-unrelated", ["other-pattern"])):
        db.execute("INSERT INTO soft_prompt_templates(prompt_id,profile_id,category,content,"
                   "source_pattern_ids) VALUES (?, 'default', ?, ?, ?)",
                   (pid, "identity" if pid == "sp-from-block" else "custom", _WORDS,
                    json.dumps(sources)))
    assert write_tombstones(db, "default", ("erasuremarker",), "e-2", time.time(), "erasuremarker")
    left = {dict(r)["prompt_id"] for r in db.execute("SELECT prompt_id FROM soft_prompt_templates")}
    assert left == {"sp-unrelated"}


def test_cli_context_injection_skips_blocks_with_an_erasing_source(dashboard):
    from superlocalmemory.cli.commands import _core_block_rows

    _, _, db = dashboard
    _add_block(db)
    assert len(_core_block_rows(db.db_path, "default")) == 1
    _erase(db)
    assert _core_block_rows(db.db_path, "default") == []
