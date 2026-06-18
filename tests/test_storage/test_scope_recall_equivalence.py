"""Equivalence proof for the multi-scope feature (PRs #42/#43): for EXISTING
data (every row scope='personal'), the new scope-filtered DB queries must return
exactly the same facts as the old profile_id-only queries. This is the core
"must not impact our functionality" guarantee — scope is purely additive.
"""
import sqlite3

import pytest

from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager, _scope_where
from superlocalmemory.storage.migrations import M016_add_scope_support as M016


@pytest.fixture
def db(tmp_path):
    d = DatabaseManager(tmp_path / "memory.db")
    with d.raw_connection() as conn:
        schema.create_all_tables(conn)
        M016.apply(conn)
        # Seed the profiles referenced by the test facts (FK parents).
        for pid in ("default", "other"):
            conn.execute(
                "INSERT OR IGNORE INTO profiles (profile_id, name, description) "
                "VALUES (?, ?, '')", (pid, pid),
            )
    return d


def _insert_fact(db: DatabaseManager, fid: str, profile: str, scope: str = "personal",
                 shared_with: str | None = None, entity: str | None = None):
    cej = f'["{entity}"]' if entity else "[]"
    mid = f"m_{fid}"
    with db.raw_connection() as conn:
        # parent memory row (FK target for atomic_facts.memory_id)
        conn.execute(
            "INSERT OR IGNORE INTO memories (memory_id, profile_id, scope, content) "
            "VALUES (?, ?, ?, ?)", (mid, profile, scope, f"mem {fid}"),
        )
        conn.execute(
            "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, scope, shared_with, "
            "content, fact_type, confidence, importance, evidence_count, access_count, "
            "canonical_entities_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 'semantic', 0.9, 0.5, 1, 0, ?, datetime('now'))",
            (fid, mid, profile, scope, shared_with, f"content {fid}", cej),
        )


def test_scope_where_collapses_to_profile_for_personal_data():
    # The clause is a strict superset of profile_id=?; the global/shared branches
    # match no rows when all data is 'personal'.
    where, params = _scope_where("p1")
    assert "profile_id = ?" in where
    assert params[0] == "p1"


def test_get_all_facts_equivalent_for_personal_only(db):
    for i in range(8):
        _insert_fact(db, f"f{i}", "default")
    # NEW path (scope defaults)
    new = {f.fact_id for f in db.get_all_facts("default")}
    # OLD behaviour: raw profile_id filter
    old = {r[0] for r in db.execute(
        "SELECT fact_id FROM atomic_facts WHERE profile_id = ?", ("default",))}
    assert new == old == {f"f{i}" for i in range(8)}


def test_profile_isolation_preserved(db):
    _insert_fact(db, "a", "default")
    _insert_fact(db, "b", "other")          # different profile, personal
    got = {f.fact_id for f in db.get_all_facts("default")}
    assert got == {"a"}                       # 'other' NOT leaked into default


def test_global_scope_is_the_only_new_visibility(db):
    _insert_fact(db, "mine", "default")
    _insert_fact(db, "glob", "other", scope="global")   # global from another profile
    got = {f.fact_id for f in db.get_all_facts("default")}
    # additive: global becomes visible; personal isolation still holds
    assert got == {"mine", "glob"}
    # and with include_global=False we get back the strict old behaviour
    strict = {f.fact_id for f in db.get_all_facts("default", include_global=False, include_shared=False)}
    assert strict == {"mine"}


def test_shared_scope_visibility(db):
    _insert_fact(db, "s1", "other", scope="shared", shared_with='["default"]')
    _insert_fact(db, "s2", "other", scope="shared", shared_with='["someone_else"]')
    got = {f.fact_id for f in db.get_all_facts("default")}
    assert "s1" in got and "s2" not in got   # only facts shared WITH default


def test_get_facts_by_entity_equivalent_for_personal(db):
    _insert_fact(db, "e1", "default", entity="ent_x")
    _insert_fact(db, "e2", "default", entity="ent_x")
    _insert_fact(db, "e3", "other", entity="ent_x")   # different profile
    got = {f.fact_id for f in db.get_facts_by_entity("ent_x", "default")}
    assert got == {"e1", "e2"}
