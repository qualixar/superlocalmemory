# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""4.1.22 erasure and delete integrity, on a real engine and store.

The daemon-level proof is tests/test_integration/test_erasure_integrity_e2e.py;
these drive the same functions directly so each behaviour can be reverted and
watched fail in seconds.
"""

from __future__ import annotations

import json
import sqlite3
import uuid

import pytest

from tests.helpers.env_capabilities import (
    NO_SECURE_DELETE_REASON,
    keyword_index_forgets_at_once,
    purge_keyword_index_on_old_sqlite,
)


def _actor() -> str:
    from superlocalmemory.core.engine_ingestion import local_trusted_actor_id

    return local_trusted_actor_id("python-api")


def _store(engine, text: str):
    from superlocalmemory.core.engine_ingestion import canonical_store

    return canonical_store(engine, text, source_type="python-api", trusted_actor_id=_actor(),
                           require_complete=True, return_receipt=True)


def _count(engine, sql: str, args: tuple = ()) -> int:
    return int(dict(engine._db.execute(sql, args)[0])["n"])


def _text_copies(db_path, needle: str) -> dict[str, int]:
    hits: dict[str, int] = {}
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
            try:
                cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]
            except sqlite3.Error:
                continue
            for col in cols:
                try:
                    n = conn.execute(f'SELECT COUNT(*) FROM "{table}" WHERE instr(lower(CAST('
                                     f'"{col}" AS TEXT)), ?) > 0', (needle,)).fetchone()[0]
                except sqlite3.Error:
                    continue
                if n:
                    hits[f"{table}.{col}"] = n
    finally:
        conn.close()
    return hits


def _delete(engine, fact_id: str) -> dict:
    from superlocalmemory.core.mutations import delete_fact_authorized

    return delete_fact_authorized(engine, fact_id, trusted_actor_id=_actor(),
                                  source_agent_id="test")


def _protect(engine, predecessor: str, successor: str, status: str = "rejected") -> None:
    with engine._db.raw_connection() as conn:
        conn.execute(
            "INSERT INTO correction_cases (case_id, profile_id, scope, predecessor_fact_id, "
            "successor_fact_id, reason_code, status, version, idempotency_key, "
            "proposed_by_actor_id, proposed_by_actor_kind, proposed_by_trust_tier, "
            "created_at, updated_at) VALUES (?, ?, 'personal', ?, ?, "
            "'consolidation_update', ?, 1, ?, 'a', 'system', 'low', 't', 't')",
            ("case-g07", engine._profile_id, predecessor, successor, status,
             uuid.uuid4().hex))
        conn.commit()


# -- refusal before anything is touched ---------------------------------------

def test_a_protected_fact_is_refused_before_any_projection_changes(engine_with_mock_deps):
    from superlocalmemory.core.remember_runtime import CanonicalMutationConflict

    engine = engine_with_mock_deps
    keep = list(_store(engine, "Vellmora keeps the synthetic brass key in drawer four.")
                .final_fact_ids)[0]
    other = list(_store(engine, "Tarrowin waters the synthetic fern every Tuesday.")
                 .final_fact_ids)[0]
    _protect(engine, keep, other)

    def footprint() -> tuple:
        return (
            _count(engine, "SELECT COUNT(*) AS n FROM atomic_facts WHERE fact_id = ?", (keep,)),
            _count(engine, "SELECT COUNT(*) AS n FROM bm25_tokens WHERE fact_id = ?", (keep,)),
            _count(engine, "SELECT COUNT(*) AS n FROM projection_tombstones"),
            _count(engine, "SELECT COUNT(*) AS n FROM erasure_receipts"),
            _count(engine, "SELECT COUNT(*) AS n FROM temporal_events WHERE fact_id = ?", (keep,)),
        )

    before = footprint()
    assert before[0] == 1 and before[1] == 1, before
    with pytest.raises(CanonicalMutationConflict) as refused:
        _delete(engine, keep)
    assert "protected by correction history" in str(refused.value)
    assert "case-g07 (rejected)" in str(refused.value)
    assert "Nothing was changed" in str(refused.value)
    assert footprint() == before


# -- a fact delete leaves no keyed residue ------------------------------------

def test_deleting_a_fact_removes_the_rows_no_foreign_key_removes(engine_with_mock_deps):
    engine = engine_with_mock_deps
    fact_id = list(_store(engine, "Odrennic paints synthetic lighthouses in ochre.")
                   .final_fact_ids)[0]
    pid = engine._profile_id
    with engine._db.raw_connection() as conn:
        conn.execute("INSERT OR REPLACE INTO vector_row_map (fact_id, profile_id, vec_rowid) "
                     "VALUES (?, ?, 987654)", (fact_id, pid))
        conn.execute("INSERT INTO fact_outcome_score (fact_id, profile_id, score, play_count, "
                     "updated_at) VALUES (?, ?, 0.5, 1, 't')", (fact_id, pid))
        conn.execute("INSERT INTO activation_cache (cache_id, profile_id, query_hash, node_id, "
                     "activation_value, iteration, created_at, expires_at) "
                     "VALUES ('c', ?, 'q', ?, 0.1, 1, 't', 't')", (pid, fact_id))
        conn.commit()
    assert _count(engine, "SELECT COUNT(*) AS n FROM bm25_tokens WHERE fact_id = ?", (fact_id,))

    engine._db.delete_fact(fact_id, profile_id=pid)

    for table, column in (("bm25_tokens", "fact_id"), ("vector_row_map", "fact_id"),
                          ("fact_outcome_score", "fact_id"), ("activation_cache", "node_id"),
                          ("embedding_metadata", "fact_id")):
        assert _count(engine, f"SELECT COUNT(*) AS n FROM {table} WHERE {column} = ?",
                      (fact_id,)) == 0, table


# -- erasure reaches the copies that are not projections ----------------------

def test_full_erasure_removes_the_words_from_every_table(engine_with_mock_deps):
    engine = engine_with_mock_deps
    marker = f"quorvane{uuid.uuid4().hex[:6]}"
    receipt = _store(engine, f"{marker} Haverlin audits the Brisk depot on 3 March 2026.")
    facts = list(receipt.final_fact_ids)
    # The HTTP event path files every event under "default", whatever the
    # memory's profile: plant one the way it does, under another profile.
    with engine._db.raw_connection() as conn:
        conn.execute(  # the event bus creates it on first use (infra/event_bus.py)
            "CREATE TABLE IF NOT EXISTS memory_events (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "profile_id TEXT NOT NULL, event_type TEXT NOT NULL, memory_id INTEGER, "
            "source_agent TEXT, source_protocol TEXT, payload TEXT, importance INTEGER, "
            "tier TEXT, created_at TIMESTAMP)")
        conn.execute(
            "INSERT INTO memory_events (profile_id, event_type, source_agent, source_protocol, "
            "payload, importance, tier, created_at) VALUES ('elsewhere', 'memory.stored', "
            "'materializer', 'http', ?, 5, 'hot', 't')",
            (json.dumps({"operation_id": receipt.operation_id, "fact_ids": [],
                         "content_preview": f"{marker} Haverlin audits"}),))
        conn.commit()
        conn.execute(  # a soft-forget copy (memories route /forget) of the same fact
            "INSERT INTO memory_archive (archive_id, fact_id, profile_id, payload_json, "
            "archived_at, reason) VALUES ('a1', ?, ?, ?, 't', 'forget')",
            (facts[0], engine._profile_id, json.dumps({"content": f"{marker} Haverlin"})))
        conn.commit()
    assert "ingestion_operations.raw_content" in _text_copies(engine._db.db_path, marker)
    assert "memory_archive.payload_json" in _text_copies(engine._db.db_path, marker)

    for fact_id in facts:
        result = _delete(engine, fact_id)
        assert result.get("deleted") == fact_id, result

    purge_keyword_index_on_old_sqlite(engine._db.db_path)
    assert _text_copies(engine._db.db_path, marker) == {}
    journal = dict(engine._db.execute(
        "SELECT raw_content, attempt_count, last_error FROM ingestion_operations "
        "WHERE operation_id = ?", (receipt.operation_id,))[0])
    assert journal["raw_content"] == "" and journal["attempt_count"] >= 10, journal
    assert journal["last_error"] == "source text removed by erasure"


def test_partial_erasure_keeps_the_source_of_a_surviving_fact(engine_with_mock_deps):
    """The journal text is the source of the facts that remain: never scrubbed."""
    engine = engine_with_mock_deps
    receipt = _store(engine, "Marrowick sells synthetic pears. Marrowick also repairs clocks "
                             "on Fridays in the old synthetic arcade.")
    facts = list(receipt.final_fact_ids)
    if len(facts) < 2:
        pytest.skip("extraction produced one fact; nothing survives a partial erasure")
    _delete(engine, facts[0])
    raw = dict(engine._db.execute("SELECT raw_content FROM ingestion_operations WHERE "
                                  "operation_id = ?", (receipt.operation_id,))[0])["raw_content"]
    assert "Marrowick" in raw


# -- the keyword index forgets deleted words ------------------------------------

def _fts_store(tmp_path, *, secure: bool):
    from superlocalmemory.storage.fts_residue import ensure_secure_delete

    conn = sqlite3.connect(tmp_path / "fts.db")
    conn.execute("CREATE TABLE facts (id INTEGER PRIMARY KEY, content TEXT)")
    conn.execute("CREATE VIRTUAL TABLE atomic_facts_fts USING fts5(content, content='facts', "
                 "content_rowid='id')")
    conn.execute("CREATE TRIGGER fd AFTER DELETE ON facts BEGIN INSERT INTO atomic_facts_fts("
                 "atomic_facts_fts, rowid, content) VALUES('delete', old.id, old.content); END")
    conn.execute("CREATE TRIGGER fi AFTER INSERT ON facts BEGIN INSERT INTO atomic_facts_fts("
                 "rowid, content) VALUES(new.id, new.content); END")
    if secure:
        assert ensure_secure_delete(conn)["atomic_facts_fts"] == "enabled"
    for i in range(50):
        conn.execute("INSERT INTO facts (content) VALUES (?)", (f"ordinary words {i}",))
    conn.execute("INSERT INTO facts (id, content) VALUES (999, 'xyloquent audits depot')")
    conn.commit()
    conn.execute("DELETE FROM facts WHERE id = 999")
    conn.commit()
    return conn


def _index_bytes_hold(conn, word: bytes) -> bool:
    return any(word in bytes(b) for (b,) in conn.execute("SELECT block FROM atomic_facts_fts_data"))


@pytest.mark.skipif(not keyword_index_forgets_at_once(), reason=NO_SECURE_DELETE_REASON)
def test_with_secure_delete_a_deleted_word_leaves_the_index(tmp_path):
    from superlocalmemory.storage.fts_residue import ensure_secure_delete

    conn = _fts_store(tmp_path, secure=True)
    assert not _index_bytes_hold(conn, b"xyloquent")
    assert ensure_secure_delete(conn)["atomic_facts_fts"] == "on"  # idempotent


def test_purge_removes_words_deleted_before_secure_delete(tmp_path):
    from superlocalmemory.storage.fts_residue import purge_deleted_terms

    conn = _fts_store(tmp_path, secure=False)
    assert _index_bytes_hold(conn, b"xyloquent")  # the residue an old store has
    purge_deleted_terms(conn, "atomic_facts_fts")
    conn.commit()
    assert not _index_bytes_hold(conn, b"xyloquent")
    assert conn.execute("SELECT COUNT(*) FROM atomic_facts_fts WHERE atomic_facts_fts "
                        "MATCH 'ordinary'").fetchone()[0] == 50


@pytest.mark.skipif(not keyword_index_forgets_at_once(), reason=NO_SECURE_DELETE_REASON)
def test_a_new_store_has_secure_delete_from_the_start(engine_with_mock_deps):
    from superlocalmemory.storage.fts_residue import secure_delete_on

    assert secure_delete_on(engine_with_mock_deps._db, "atomic_facts_fts")
