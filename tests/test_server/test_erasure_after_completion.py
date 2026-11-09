"""Words of an erased fact must not survive in the persona or a scene theme.

The read-path guards in ``test_erasure_visibility`` hold while a tombstone names
a *live* member of the derived row. Once an erasure completes the community row
and the fact's id are gone from the derived row but the persona text and the
scene theme (the first 200 characters of the anchor fact) are not, so these
tests cover the moment after completion too.
"""

# ruff: noqa: F811 (pytest fixtures are imported and redeclared as parameters)

from __future__ import annotations

import json
import time

import pytest

from tests.test_server.test_erasure_visibility import _WORDS, _erase, dashboard  # noqa: F401

_SECRET = "copper compass"
_CONTROL_TEXT = "controlmarker"


def _add_persona(db, community_facts=("erasuremarker",), community_id=7):
    db.execute("INSERT INTO community_summaries(profile_id,community_id,summary,fact_ids_json) "
               "VALUES ('default',?,?,?)", (community_id, _WORDS, json.dumps(list(community_facts))))
    db.execute("INSERT OR REPLACE INTO persona_summary(profile_id,summary,community_ids_json) "
               "VALUES ('default',?,?)", (_WORDS, json.dumps([community_id])))


def _persona_rows(db) -> int:
    return dict(list(db.execute("SELECT COUNT(*) AS n FROM persona_summary"))[0])["n"]


def _persona_text(client, headers) -> str:
    return client.get("/api/v3/abstraction/persona", headers=headers).text


def test_persona_never_returns_erased_text_before_or_after_scrub(dashboard):
    from superlocalmemory.core import erasure_scrub

    client, headers, db = dashboard
    _add_persona(db)
    assert _SECRET in _persona_text(client, headers)
    _erase(db)
    assert _SECRET not in _persona_text(client, headers)
    # completion: the scrub drops the community row, the fact goes, the tombstone stays
    erasure_scrub.scrub(db, "default", "erasuremarker", None)
    db.execute("DELETE FROM atomic_facts WHERE fact_id='erasuremarker'")
    assert _SECRET not in _persona_text(client, headers)
    assert _persona_rows(db) == 0


def test_scrub_removes_a_persona_built_on_the_erased_communities(dashboard):
    from superlocalmemory.core import erasure_scrub

    _, _, db = dashboard
    _add_persona(db, ("erasuremarker", "controlmarker"))
    counts = erasure_scrub.scrub(db, "default", "erasuremarker", None)
    assert _persona_rows(db) == 0
    assert counts["derived_summaries_removed"] >= 2


def test_tombstone_write_marks_the_persona_stale_for_rebuild(dashboard):
    from superlocalmemory.core.transactions.tombstones import write_tombstones

    _, _, db = dashboard
    _add_persona(db, ("erasuremarker", "controlmarker"))
    assert write_tombstones(db, "default", ("erasuremarker",), "e-1", time.time(), "erasuremarker")
    assert _persona_rows(db) == 0


def test_tombstone_leaves_a_persona_of_unrelated_communities(dashboard):
    from superlocalmemory.core.transactions.tombstones import write_tombstones

    client, headers, db = dashboard
    _add_persona(db, ("controlmarker",))
    assert write_tombstones(db, "default", ("erasuremarker",), "e-2", time.time(), "erasuremarker")
    assert _persona_rows(db) == 1
    assert _SECRET in _persona_text(client, headers)


def test_other_profiles_persona_is_untouched(dashboard):
    from superlocalmemory.core import erasure_scrub

    _, _, db = dashboard
    db.execute("INSERT INTO profiles(profile_id,name) VALUES ('other','other')")
    db.execute("INSERT INTO community_summaries(profile_id,community_id,summary,fact_ids_json) "
               "VALUES ('other',7,?,?)", (_WORDS, json.dumps(["erasuremarker"])))
    db.execute("INSERT INTO persona_summary(profile_id,summary,community_ids_json) "
               "VALUES ('other',?,?)", (_WORDS, json.dumps([7])))
    erasure_scrub.scrub(db, "default", "erasuremarker", None)
    assert _persona_rows(db) == 1


# -- scene themes -------------------------------------------------------------

_SCENE_PATHS = ("/api/clusters", "/api/graph", "/api/clusters/s1")


def _add_scene(db, members=("erasuremarker", "controlmarker")):
    db.execute("INSERT INTO memory_scenes(scene_id,profile_id,theme,fact_ids_json) "
               "VALUES ('s1','default',?,?)", (_WORDS, json.dumps(list(members))))


def _theme(db) -> str:
    return dict(list(db.execute("SELECT theme FROM memory_scenes WHERE scene_id='s1'"))[0])["theme"]


@pytest.mark.parametrize("path", _SCENE_PATHS)
def test_scene_theme_never_returned_before_or_after_the_fact_is_removed(dashboard, path):
    client, headers, db = dashboard
    _add_scene(db)
    _erase(db)
    during = client.get(path, headers=headers)
    assert during.status_code in (200, 404) and _SECRET not in during.text, during.text
    db.remove_fact_from_scenes("erasuremarker", "default")
    db.execute("DELETE FROM atomic_facts WHERE fact_id='erasuremarker'")
    after = client.get(path, headers=headers)
    assert after.status_code in (200, 404), after.text
    assert _SECRET not in after.text, after.text


def test_removing_the_anchor_rethemes_from_the_first_remaining_member(dashboard):
    _, _, db = dashboard
    _add_scene(db)
    db.remove_fact_from_scenes("erasuremarker", "default")
    assert _theme(db) == _CONTROL_TEXT


def test_removing_the_anchor_after_its_row_is_gone_still_rethemes(dashboard):
    _, _, db = dashboard
    _add_scene(db)
    db.execute("DELETE FROM atomic_facts WHERE fact_id='erasuremarker'")
    db.remove_fact_from_scenes("erasuremarker", "default")
    assert _theme(db) == _CONTROL_TEXT


def test_removing_a_non_anchor_keeps_the_theme(dashboard):
    _, _, db = dashboard
    _add_scene(db, ("controlmarker", "erasuremarker"))
    db.execute("UPDATE memory_scenes SET theme='control theme' WHERE scene_id='s1'")
    db.remove_fact_from_scenes("erasuremarker", "default")
    assert _theme(db) == "control theme"


def test_theme_is_blanked_when_no_member_has_text_left(dashboard):
    _, _, db = dashboard
    _add_scene(db, ("erasuremarker", "ghost"))
    db.remove_fact_from_scenes("erasuremarker", "default")
    assert _theme(db) == ""


def test_tombstone_write_rethemes_a_scene_whose_theme_quotes_the_fact(dashboard):
    from superlocalmemory.core.transactions.tombstones import write_tombstones

    client, headers, db = dashboard
    _add_scene(db)
    db.execute("UPDATE atomic_facts SET content=? WHERE fact_id='erasuremarker'", (_WORDS,))
    assert write_tombstones(db, "default", ("erasuremarker",), "e-3", time.time(), "erasuremarker")
    assert _SECRET not in _theme(db)
    assert _theme(db) == _CONTROL_TEXT


def test_tombstone_theme_skips_members_that_are_themselves_erasing(dashboard):
    from superlocalmemory.core.transactions.tombstones import write_tombstones

    _, _, db = dashboard
    _add_scene(db, ("erasuremarker", "controlmarker", "thirdmarker"))
    db.execute("INSERT INTO memories(memory_id,profile_id,content) VALUES ('thirdmarker','default','t')")
    db.execute("INSERT INTO atomic_facts(fact_id,memory_id,profile_id,content,fact_type,created_at) "
               "VALUES ('thirdmarker','thirdmarker','default','thirdmarker','opinion',datetime('now'))")
    _erase(db, "controlmarker")
    assert write_tombstones(db, "default", ("erasuremarker",), "e-4", time.time(), "erasuremarker")
    assert _theme(db) == "thirdmarker"


def test_scene_theme_is_cut_to_the_builders_length(dashboard):
    _, _, db = dashboard
    _add_scene(db)
    db.execute("UPDATE atomic_facts SET content=? WHERE fact_id='controlmarker'", ("x" * 500,))
    db.remove_fact_from_scenes("erasuremarker", "default")
    assert _theme(db) == "x" * 200
