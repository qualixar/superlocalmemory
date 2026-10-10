# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The profile fold's decision table is complete, and its edge cases hold.

A profile-scoped table without a decision is a table a profile delete would
silently leave behind (or, before 4.1.22, silently CASCADE-delete). The fold
refuses such a table, so these tests keep the table complete against the
live schema, including the side tables an embedding-model switch creates.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

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

_X = "foldsrc"


def _vec(conn: sqlite3.Connection) -> sqlite3.Connection:
    import sqlite_vec

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.row_factory = sqlite3.Row
    return conn


def _open(db: Path) -> sqlite3.Connection:
    return _vec(sqlite3.connect(str(db)))


@pytest.fixture
def store(engine_with_mock_deps):
    """A fully migrated memory.db + learning.db, with the model-switch side tables."""
    from superlocalmemory.storage import embedding_spaces as sp
    from superlocalmemory.storage import migration_runner

    db = Path(engine_with_mock_deps._config.db_path)
    migration_runner.apply_all(db.parent / "learning.db", db)
    conn = _open(db)
    with conn:
        sp.ensure_control_tables(conn)
        sp.ensure_side_tables(conn)
        for table in (sp.NEXT_VEC, sp.PREV_VEC, sp.TRASH_VEC):
            sp.create_vec(conn, table, 8)
        conn.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)", (_X, _X))
    conn.close()
    return db


def test_every_profile_scoped_memory_table_has_a_decision(store):
    from superlocalmemory.storage.profile_fold import DECISIONS, profile_scoped_tables

    conn = _open(store)
    try:
        tables = profile_scoped_tables(conn)
    finally:
        conn.close()
    assert "fact_embeddings" in tables and "reembed_prev_vec" in tables, tables
    assert [t for t in tables if t not in DECISIONS] == []
    for table, (action, why) in DECISIONS.items():
        assert action in {"move", "merge", "rekey", "vec", "delete", "keep"} and why, table


def test_every_profile_scoped_learning_table_has_a_decision(store):
    from superlocalmemory.storage.profile_fold_sidecars import LEARNING_DECISIONS, _scoped

    conn = sqlite3.connect(str(store.parent / "learning.db"))
    try:
        tables = _scoped(conn)
    finally:
        conn.close()
    assert len(tables) >= 15, tables
    assert [t for t in tables if t not in LEARNING_DECISIONS] == []


def test_an_unclassified_table_stops_the_fold_before_anything_changes(store):
    from superlocalmemory.storage.profile_fold import ProfileFoldError, fold_profile

    conn = _open(store)
    try:
        conn.execute("CREATE TABLE brand_new_feature (id TEXT, profile_id TEXT)")
        conn.execute("INSERT INTO memories (memory_id, profile_id, content) VALUES ('m1', ?, 'x')",
                     (_X,))
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(ProfileFoldError, match="brand_new_feature"):
            fold_profile(conn, _X)
        conn.rollback()
        assert conn.execute("SELECT profile_id FROM memories WHERE memory_id='m1'").fetchone()[0] == _X
    finally:
        conn.close()


def _seed_collisions(conn: sqlite3.Connection) -> None:
    for profile, fid in ((_X, "fx"), ("default", "fd")):
        conn.execute("INSERT INTO memories (memory_id, profile_id, content) VALUES (?, ?, 'c')",
                     ("m" + fid, profile))
        conn.execute("INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content) "
                     "VALUES (?, ?, ?, 'c')", (fid, "m" + fid, profile))
        conn.execute("INSERT INTO ingestion_log (profile_id, source_type, dedup_key, fact_ids, "
                     "ingested_at) VALUES (?, 'api', 'same-key', ?, 'now')", (profile, fid))
        conn.execute("INSERT INTO trust_scores (trust_id, profile_id, target_type, target_id, "
                     "trust_score) VALUES (?, ?, 'agent', 'agent-1', ?)",
                     ("t" + fid, profile, 0.9 if profile == "default" else 0.1))
    conn.execute("INSERT INTO fact_embeddings (rowid, profile_id, embedding) VALUES (77, ?, ?)",
                 (_X, b"\x00\x00\x80?" + b"\x00" * (768 * 4 - 4)))
    conn.execute("INSERT INTO reembed_prev_vec (rowid, profile_id, embedding) VALUES (5, ?, ?)",
                 (_X, b"\x00\x00\x80?" + b"\x00" * 28))
    conn.execute("INSERT INTO reembed_prev_map (fact_id, profile_id, vec_rowid, content_hash) "
                 "VALUES ('fx', ?, 5, 'h')", (_X,))


def test_collisions_keep_default_s_row_and_lose_nothing_else(store):
    from superlocalmemory.storage.profile_fold import fold_profile

    conn = _open(store)
    try:
        _seed_collisions(conn)
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")
        counts = fold_profile(conn, _X)
        conn.commit()
        rows = conn.execute("SELECT profile_id, fact_ids FROM ingestion_log "
                            "WHERE dedup_key='same-key'").fetchall()
        assert [tuple(r) for r in rows] == [("default", "fd")]
        trust = conn.execute("SELECT profile_id, trust_score FROM trust_scores "
                             "WHERE target_id='agent-1'").fetchall()
        assert [tuple(r) for r in trust] == [("default", 0.9)]
        assert conn.execute("SELECT profile_id FROM atomic_facts WHERE fact_id='fx'"
                            ).fetchone()[0] == "default"
        assert conn.execute("SELECT profile_id FROM fact_embeddings WHERE rowid=77"
                            ).fetchone()[0] == "default"
        assert conn.execute("SELECT profile_id FROM reembed_prev_vec WHERE rowid=5"
                            ).fetchone()[0] == "default"
        assert counts["fact_embeddings"] == 1 and counts["reembed_prev_vec"] == 1
        queued = conn.execute("SELECT profile_id, op FROM projection_outbox WHERE fact_id='fx'"
                              ).fetchone()
        assert tuple(queued) == ("default", "upsert"), "the projection was not told to re-file it"
    finally:
        conn.close()


def test_a_shared_idempotency_key_is_kept_apart_not_dropped(store):
    from superlocalmemory.storage.profile_fold import fold_profile

    conn = _open(store)
    try:
        for profile, case in ((_X, "case-x"), ("default", "case-d")):
            fid = "f" + case
            conn.execute("INSERT INTO memories (memory_id, profile_id, content) VALUES (?, ?, 'c')",
                         ("m" + fid, profile))
            for f in (fid, fid + "s"):
                conn.execute("INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content) "
                             "VALUES (?, ?, ?, 'c')", (f, "m" + fid, profile))
            conn.execute(
                "INSERT INTO correction_cases (case_id, profile_id, scope, predecessor_fact_id, "
                "successor_fact_id, reason_code, status, version, idempotency_key, "
                "proposed_by_actor_id, proposed_by_actor_kind, proposed_by_trust_tier, "
                "created_at, updated_at) VALUES (?, ?, 'personal', ?, ?, 'user_edit', "
                "'proposed', 0, 'same-key', 'a', 'person', 'trusted', 'now', 'now')",
                (case, profile, fid, fid + "s"))
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")
        fold_profile(conn, _X)
        conn.commit()
        rows = dict(conn.execute("SELECT case_id, idempotency_key FROM correction_cases "
                                 "WHERE profile_id='default'").fetchall())
        assert rows == {"case-d": "same-key", "case-x": "same-key:folded:case-x"}
    finally:
        conn.close()


def test_a_running_reindex_refuses_the_fold(store):
    from superlocalmemory.storage.profile_fold import ProfileFoldError, fold_profile

    conn = _open(store)
    try:
        conn.execute("INSERT INTO embedding_reindex_jobs (kind, state, from_signature, "
                     "to_signature, from_config, to_config, created_at, updated_at) "
                     "VALUES ('switch', 'running', 'a', 'b', '{}', '{}', 0, 0)")
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(ProfileFoldError, match="re-index is running"):
            fold_profile(conn, _X)
        conn.rollback()
    finally:
        conn.close()


def test_sidecars_move_pending_memories_and_drop_learned_state(store):
    from superlocalmemory.storage import profile_fold_sidecars as sidecars

    learning = store.parent / "learning.db"
    with sqlite3.connect(str(learning)) as conn:
        conn.execute("INSERT INTO bandit_arms (profile_id, stratum, arm_id, alpha, beta, plays) "
                     "VALUES (?, 's', 'a', 1, 1, 0)", (_X,))
        conn.execute("INSERT INTO bandit_arms (profile_id, stratum, arm_id, alpha, beta, plays) "
                     "VALUES ('default', 's', 'a', 1, 1, 0)")
    pending = store.parent / "pending.db"
    with sqlite3.connect(str(pending)) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS pending_memories (id INTEGER PRIMARY KEY, "
                     "profile_id TEXT, content TEXT, created_at TEXT NOT NULL)")
        conn.execute("DELETE FROM pending_memories")
        conn.execute("INSERT INTO pending_memories (profile_id, content, created_at) "
                     "VALUES (?, 'later', 'now')", (_X,))
    assert sidecars.purge_learned_state(learning, _X) == 1
    assert sidecars.move_pending(pending, _X) == 1
    with sqlite3.connect(str(learning)) as conn:
        assert conn.execute("SELECT profile_id FROM bandit_arms").fetchall() == [("default",)]
    with sqlite3.connect(str(pending)) as conn:
        assert conn.execute("SELECT profile_id FROM pending_memories").fetchall() == [("default",)]


def test_the_docstring_decision_table_matches_the_code():
    import re

    from superlocalmemory.storage import profile_fold

    doc = profile_fold.__doc__.split("DECISION TABLE", 1)[1].split("Canonical entities", 1)[0]
    listed: dict[str, str] = {}
    action = None
    for line in doc.splitlines():
        head = re.match(r"^(MOVE|MERGE|REKEY|VEC|DELETE|KEEP)\s+(.*)$", line)
        if head:
            action, line = head.group(1).lower(), head.group(2)
        elif action is None or not line.startswith(" "):
            continue
        for name in re.findall(r"[a-z_0-9]+", line):
            listed[name] = action
    assert listed == {t: a for t, (a, _why) in profile_fold.DECISIONS.items()}


def test_an_edge_default_already_has_is_not_doubled(store):
    """After a same-name entity merge both profiles can hold the same edge; a
    doubled edge would count the relation twice in every graph walk."""
    from superlocalmemory.storage.profile_fold import fold_profile

    conn = _open(store)
    try:
        for profile, edge in ((_X, "e-x"), ("default", "e-d"), (_X, "e-x-own")):
            target = "fact-own" if edge == "e-x-own" else "fact-shared"
            conn.execute("INSERT INTO graph_edges (edge_id, profile_id, source_id, target_id, "
                         "edge_type, weight) VALUES (?, ?, 'ent-1', ?, 'entity', 1.0)",
                         (edge, profile, target))
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")
        fold_profile(conn, _X)
        conn.commit()
        rows = sorted(tuple(r) for r in conn.execute(
            "SELECT edge_id, profile_id FROM graph_edges"))
        assert rows == [("e-d", "default"), ("e-x-own", "default")]
    finally:
        conn.close()
