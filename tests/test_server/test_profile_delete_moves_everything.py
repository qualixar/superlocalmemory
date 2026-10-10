# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Deleting a dashboard profile moves ALL of its memories to 'default'.

DELETE /api/profiles/{name} said "Moves its memories to 'default'" but moved
two tables (atomic_facts, memories) and then deleted the profile row with
foreign keys on. Every table tied to ``profiles`` by ON DELETE CASCADE lost
the moved memories' rows on the spot -- their keyword tokens, vector index
rows, entities, graph edges, temporal rows, scenes -- and every table without
that key (the correction ledger, entity links, the vectors themselves) was left
filed under a profile that no longer existed. The memories survived as text
that the keyword, entity, graph and correction surfaces could no longer reach
under 'default'.

These tests drive the real route function on a real engine store.
"""

from __future__ import annotations

import asyncio
import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.helpers.env_capabilities import (
    NO_VECTOR_SEARCH_REASON,
    vector_search_available,
)

# The fixtures build stores with sqlite-vec loaded. An interpreter whose sqlite3
# cannot load extensions (the python.org builds on macOS) has the package but
# cannot use it, which an import check does not see.
pytestmark = pytest.mark.skipif(
    not vector_search_available(), reason=NO_VECTOR_SEARCH_REASON,
)

_X = "kestrelwork"
_TEXT = "Brindlemoor Quayle keeps the synthetic pewter kettle in the Ostrava archive."
_EDIT = "Brindlemoor Quayle keeps the synthetic pewter kettle in the Brno archive."
_DEFAULT_TEXT = "Brindlemoor Quayle also chairs the synthetic harbour committee."


def _vec_conn(db_path):
    import sqlite_vec

    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.row_factory = sqlite3.Row
    return conn


#: Rows that keep naming a deleted profile ON PURPOSE (the test's own oracle,
#: independent of the module's table): immutable write receipts, erasure
#: receipts, and the ids-only change feed that tells cached indexes to drop it.
_KEEP = {"write_commits", "erasure_receipts", "fact_search_changes"}


def _naming(db_path, profile: str) -> dict[str, int]:
    """Every table (and vector partition) still holding a row for ``profile``."""
    conn = _vec_conn(db_path)
    try:
        out = {}
        for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
            if table == "profiles" or table in _KEEP or table.startswith("sqlite_"):
                continue
            try:
                cols = {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')}
            except sqlite3.OperationalError:
                continue
            if "profile_id" not in cols:
                continue
            n = conn.execute(f'SELECT COUNT(*) FROM "{table}" WHERE profile_id = ?',
                             (profile,)).fetchone()[0]
            if n:
                out[table] = n
        return out
    finally:
        conn.close()


def _count(db_path, sql: str, args: tuple = ()) -> int:
    conn = _vec_conn(db_path)
    try:
        return int(conn.execute(sql, args).fetchone()[0])
    finally:
        conn.close()


def _store(engine, text: str, profile: str) -> str:
    from superlocalmemory.core.engine_ingestion import canonical_store, local_trusted_actor_id

    receipt = canonical_store(engine, text, source_type="python-api",
                              trusted_actor_id=local_trusted_actor_id("python-api"),
                              require_complete=True, return_receipt=True, profile_id=profile)
    return list(receipt.final_fact_ids)[0]


def _correct(engine, fact_id: str, profile: str) -> None:
    from superlocalmemory.core.remember_runtime import _execute_mutation
    from superlocalmemory.storage.write_coordinator import CommandKind

    with engine._db.raw_connection() as conn:
        _execute_mutation(engine._db, CommandKind.PROPOSE_CORRECTION, profile, {
            "fact_id": fact_id, "successor_fact_id": "c" * 16, "content": _EDIT,
            "trusted_actor_id": "person-test", "idempotency_key": "fold-1"},
            connection=conn)


def _delete_via_route(engine, tmp_path, monkeypatch, name: str) -> dict:
    from superlocalmemory.server.routes import helpers, profiles

    monkeypatch.setattr(helpers, "DB_PATH", engine._config.db_path)
    monkeypatch.setattr(profiles, "DB_PATH", engine._config.db_path)
    monkeypatch.setattr(helpers, "MEMORY_DIR", tmp_path / "dash-memory-dir")
    authorization = MagicMock()
    request = MagicMock()
    request.app.state = SimpleNamespace()
    runtime = SimpleNamespace(snapshot=SimpleNamespace(profile_id="default"))
    with patch.object(profiles, "sync_profiles",
                      return_value=[{"profile_id": "default"}, {"profile_id": name}]), \
            patch.object(profiles, "get_profile_runtime", return_value=runtime), \
            patch.object(profiles, "authorize_route_mutation", return_value=authorization), \
            patch("superlocalmemory.server.rbac_enforce.require_manage", lambda *a, **k: None):
        result = asyncio.run(profiles.delete_profile(name, request))
    authorization.complete.assert_called_once()
    return result


@pytest.fixture
def folded(engine_with_mock_deps, tmp_path, monkeypatch):
    """Profile X with an entity shared by name with 'default', edges and a
    correction case; then X is deleted through the dashboard route."""
    engine = engine_with_mock_deps
    engine._db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)", (_X, _X))
    default_fact = _store(engine, _DEFAULT_TEXT, "default")
    fact_id = _store(engine, _TEXT, _X)
    _correct(engine, fact_id, _X)
    db = engine._config.db_path
    before = _naming(db, _X)
    for table in ("atomic_facts", "memories", "bm25_tokens", "canonical_entities",
                  "fact_entity_associations", "correction_cases", "correction_events",
                  "embedding_metadata"):
        assert before.get(table), f"fixture did not produce {table} rows for {_X}: {before}"
    result = _delete_via_route(engine, tmp_path, monkeypatch, _X)
    return SimpleNamespace(engine=engine, db=db, fact_id=fact_id, default_fact=default_fact,
                           before=before, result=result)


def test_no_row_anywhere_is_left_filed_under_the_deleted_profile(folded):
    assert folded.result["success"] is True
    assert _naming(folded.db, _X) == {}
    assert _count(folded.db, "SELECT COUNT(*) FROM profiles WHERE profile_id = ?", (_X,)) == 0


def test_the_moved_memory_keeps_its_search_indexes_under_default(folded):
    db, fid = folded.db, folded.fact_id
    assert _count(db, "SELECT COUNT(*) FROM atomic_facts WHERE fact_id=? AND profile_id='default'",
                  (fid,)) == 1
    assert _count(db, "SELECT COUNT(*) FROM bm25_tokens WHERE fact_id=? AND profile_id='default'",
                  (fid,)) == 1, "keyword tokens of the moved memory were lost"
    assert _count(db, "SELECT COUNT(*) FROM embedding_metadata WHERE fact_id=? "
                      "AND profile_id='default'", (fid,)) == 1, "vector index row was lost"
    assert _count(db, "SELECT COUNT(*) FROM fact_embeddings fe JOIN embedding_metadata em "
                      "ON em.vec_rowid = fe.rowid WHERE em.fact_id=? AND fe.profile_id='default'",
                  (fid,)) == 1, "the vector is not in default's partition"


def test_the_moved_memory_is_linked_to_default_s_entity_of_the_same_name(folded):
    db, fid = folded.db, folded.fact_id
    linked = _count(db, "SELECT COUNT(DISTINCT fea.entity_id) FROM fact_entity_associations fea "
                        "JOIN canonical_entities ce ON ce.entity_id = fea.entity_id "
                        "WHERE fea.fact_id=? AND ce.profile_id='default' "
                        "AND fea.profile_id='default'", (fid,))
    assert linked >= 1, "the moved memory lost its entity links"
    dupes = _count(db, "SELECT COUNT(*) FROM (SELECT LOWER(canonical_name) n FROM "
                       "canonical_entities WHERE profile_id='default' GROUP BY n HAVING COUNT(*)>1)")
    assert dupes == 0, "the same entity now exists twice under default"


def test_the_moved_memory_is_recalled_from_default(folded):
    response = folded.engine.recall("pewter kettle Ostrava archive", profile_id="default")
    assert folded.fact_id in [r.fact.fact_id for r in response.results]


def test_its_correction_case_is_listed_under_default(folded):
    from superlocalmemory.storage.correction_cases import CorrectionCaseStore

    store = CorrectionCaseStore(folded.db, is_profile_active=lambda p: p == "default",
                                is_actor_trusted=lambda _a: False)
    cases = store.list_cases("default", limit=50)
    assert any(getattr(c, "predecessor_fact_id", None) == folded.fact_id
               or (isinstance(c, dict) and c.get("predecessor_fact_id") == folded.fact_id)
               for c in cases), cases


@pytest.fixture
def unfolded(engine_with_mock_deps, tmp_path, monkeypatch):
    engine = engine_with_mock_deps
    engine._db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)", (_X, _X))
    _store(engine, _DEFAULT_TEXT, "default")
    fact_id = _store(engine, _TEXT, _X)
    learning = engine._config.db_path.parent / "learning.db"
    from superlocalmemory.storage import migration_runner

    migration_runner.apply_all(learning, engine._config.db_path)
    with sqlite3.connect(str(learning)) as conn:
        conn.execute("INSERT INTO bandit_arms (profile_id, stratum, arm_id, alpha, beta, plays) "
                     "VALUES (?, 's', 'a', 1, 1, 0)", (_X,))
    return SimpleNamespace(engine=engine, db=engine._config.db_path, fact_id=fact_id,
                           learning=learning, before=_naming(engine._config.db_path, _X),
                           tmp_path=tmp_path, monkeypatch=monkeypatch)


def _bandit_rows(learning, profile):
    with sqlite3.connect(str(learning)) as conn:
        return conn.execute("SELECT COUNT(*) FROM bandit_arms WHERE profile_id=?",
                            (profile,)).fetchone()[0]


def test_a_refused_delete_changes_nothing_anywhere(unfolded):
    from fastapi import HTTPException

    conn = _vec_conn(unfolded.db)
    with conn:
        from superlocalmemory.storage import embedding_spaces as sp

        sp.ensure_control_tables(conn)
        conn.execute("INSERT INTO embedding_reindex_jobs (kind, state, from_signature, "
                     "to_signature, from_config, to_config, created_at, updated_at) "
                     "VALUES ('switch', 'running', 'a', 'b', '{}', '{}', 0, 0)")
    conn.close()
    with pytest.raises(HTTPException) as raised:
        _delete_via_route(unfolded.engine, unfolded.tmp_path, unfolded.monkeypatch, _X)
    assert raised.value.status_code == 409 and "re-index" in str(raised.value.detail)
    assert _naming(unfolded.db, _X) == unfolded.before
    assert _bandit_rows(unfolded.learning, _X) == 1, "learned state purged by a refused delete"


def test_a_failure_part_way_rolls_the_whole_store_back(unfolded):
    from fastapi import HTTPException

    from superlocalmemory.storage import profile_fold

    def _boom(*_a, **_k):
        raise sqlite3.OperationalError("disk I/O error (injected)")

    unfolded.monkeypatch.setattr(profile_fold, "_move_vectors", _boom)
    with pytest.raises(HTTPException) as raised:
        _delete_via_route(unfolded.engine, unfolded.tmp_path, unfolded.monkeypatch, _X)
    assert raised.value.status_code == 500
    assert _naming(unfolded.db, _X) == unfolded.before, "a half-moved profile was committed"
    assert _count(unfolded.db, "SELECT COUNT(*) FROM profiles WHERE profile_id = ?", (_X,)) == 1
