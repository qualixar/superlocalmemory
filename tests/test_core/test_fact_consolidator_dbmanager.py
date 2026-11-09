# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for the DatabaseManager path of consolidate_facts (v3.8.4).

ALL THREE existing tests in test_fact_consolidator_dedup.py pass a str path,
which takes the LEGACY backward-compat branch that holds ONE write connection
for the entire pass — meaning an Ollama call inside a held write lock in Modes
B/C.

The DatabaseManager branch — the v3.8.4 concurrency-safe path — does:
  1. Discover clusters in a short memory_read() (no write lock).
  2. For each cluster: load facts in another short memory_read().
  3. Generate the summary OUTSIDE any lock.
  4. Write the result in a short memory_write() (lock held only for SQL).

This file exercises THAT branch, ensuring production behaviour is tested.
"""

from __future__ import annotations

import pytest

from superlocalmemory.core.fact_consolidator import consolidate_facts
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import MemoryRecord

_NOW = "2026-01-01T00:00:00+00:00"

_CLUSTER = [
    "Omega is a distributed systems architect at a major research institute.",
    "Omega has published ten papers on consensus protocols and fault tolerance.",
    "Omega leads a team of twenty engineers focused on cloud infrastructure.",
    "Omega pioneered a new approach to linearizable distributed transactions.",
]


@pytest.fixture()
def dbmanager_db(tmp_path):
    """Return a DatabaseManager with the full migration chain applied.

    Mirrors the fixture in test_fact_consolidator_dedup.py but returns the
    DatabaseManager directly so tests can pass it to consolidate_facts().
    """
    db_file = str(tmp_path / "consol_dbm.db")
    mgr = DatabaseManager(db_file)
    mgr.initialize(real_schema)

    # Apply the migration chain the engine applies — creates pinned_facts,
    # fact_consolidations, lifecycle tables, and association_edges.
    from superlocalmemory.storage.schema_v343 import apply_v343_schema, apply_v346_schema
    from superlocalmemory.storage.schema_v347 import apply_v347_schema
    from superlocalmemory.storage.schema_v3410 import apply_v3410_schema
    from superlocalmemory.storage.schema_v3411 import apply_v3411_schema
    for _apply in (apply_v343_schema, apply_v346_schema, apply_v347_schema,
                   apply_v3410_schema, apply_v3411_schema):
        _apply(db_file)

    # Seed: parent memory + canonical entity + warm facts.
    mgr.store_memory(MemoryRecord(
        memory_id="mem_dbm", profile_id="default", content="dbm cluster source"
    ))
    mgr.execute(
        "INSERT INTO canonical_entities "
        "(entity_id, profile_id, canonical_name, entity_type, first_seen, last_seen, fact_count) "
        "VALUES ('omega','default','Omega','person',?,?,4)",
        (_NOW, _NOW),
    )
    return mgr


def _insert_warm_fact(mgr: DatabaseManager, fid: str, content: str) -> None:
    mgr.execute(
        "INSERT INTO atomic_facts "
        "(fact_id, memory_id, profile_id, content, fact_type, "
        " canonical_entities_json, entities_json, confidence, importance, "
        " evidence_count, access_count, created_at, lifecycle) "
        "VALUES (?, 'mem_dbm', 'default', ?, 'semantic', '[\"omega\"]', '[\"omega\"]', "
        " 0.8, 0.5, 1, 0, ?, 'warm')",
        (fid, content, _NOW),
    )


def _count_active(mgr: DatabaseManager) -> int:
    rows = mgr.execute(
        "SELECT COUNT(*) AS c FROM atomic_facts "
        "WHERE lifecycle='active' AND profile_id='default'"
    )
    return dict(rows[0])["c"]


def _count_archived(mgr: DatabaseManager) -> int:
    rows = mgr.execute(
        "SELECT COUNT(*) AS c FROM atomic_facts "
        "WHERE lifecycle='archived' AND profile_id='default'"
    )
    return dict(rows[0])["c"]


def _count_consolidation_records(mgr: DatabaseManager) -> int:
    rows = mgr.execute(
        "SELECT COUNT(*) AS c FROM fact_consolidations WHERE profile_id='default'"
    )
    return dict(rows[0])["c"]


def test_databasemanager_path_consolidates_cluster(dbmanager_db) -> None:
    """The DatabaseManager branch of consolidate_facts produces a consolidated fact.

    This exercises the v3.8.4 per-cluster write path: discovery in memory_read,
    summary generation outside any lock, write inside short memory_write.
    """
    mgr = dbmanager_db
    for i, c in enumerate(_CLUSTER):
        _insert_warm_fact(mgr, f"dm{i}", c)

    # Pass the DatabaseManager — takes the v3.8.4 concurrency-safe branch.
    stats = consolidate_facts(mgr, profile_id="default", config=None)

    assert stats["clusters_found"] >= 1, "DatabaseManager path found no clusters"
    assert stats["consolidated"] >= 1, (
        "DatabaseManager path found clusters but consolidated 0 — "
        "the per-cluster write loop is broken"
    )
    assert stats["facts_summarized"] >= len(_CLUSTER), (
        "the cluster's facts were not counted as summarized"
    )
    assert stats["facts_archived"] == 0, (
        "this path archives nothing as of 4.0.10; a non-zero count means the "
        "corpus write came back"
    )
    assert stats["errors"] == 0, f"consolidation errors: {stats['error_detail']}"


def test_databasemanager_path_leaves_originals_untouched(dbmanager_db) -> None:
    """Original facts must survive, and now also stay reachable.

    "Never deleted" was the old bar and it is still met. 4.0.10 raises it:
    the sources are not archived either, because the display summary that
    replaced the corpus row does not stand in for them in recall.
    """
    mgr = dbmanager_db
    for i, c in enumerate(_CLUSTER):
        _insert_warm_fact(mgr, f"da{i}", c)

    consolidate_facts(mgr, profile_id="default", config=None)

    assert _count_archived(mgr) == 0, "a source fact was archived"
    ids = [f"da{i}" for i in range(len(_CLUSTER))]
    live = mgr.execute(
        "SELECT COUNT(*) AS c FROM atomic_facts "
        f"WHERE fact_id IN ({','.join('?' * len(ids))}) AND lifecycle='warm'",
        tuple(ids),
    )
    assert dict(live[0])["c"] == len(_CLUSTER), (
        "source facts are neither warm nor accounted for — they must remain "
        "exactly as they were"
    )


def test_databasemanager_path_writes_provenance(dbmanager_db) -> None:
    """Every consolidation must record its provenance in fact_consolidations."""
    mgr = dbmanager_db
    for i, c in enumerate(_CLUSTER):
        _insert_warm_fact(mgr, f"dp{i}", c)

    consolidate_facts(mgr, profile_id="default", config=None)

    records = _count_consolidation_records(mgr)
    assert records >= 1, (
        "fact_consolidations has 0 rows after DatabaseManager consolidation — "
        "provenance tracking is broken on the v3.8.4 path"
    )


def test_databasemanager_path_dry_run_does_not_write(dbmanager_db) -> None:
    """dry_run=True must not touch the database."""
    mgr = dbmanager_db
    for i, c in enumerate(_CLUSTER):
        _insert_warm_fact(mgr, f"dr{i}", c)

    before_active = _count_active(mgr)
    stats = consolidate_facts(mgr, profile_id="default", config=None, dry_run=True)

    assert stats["clusters_found"] >= 1
    assert stats["consolidated"] >= 1
    after_active = _count_active(mgr)
    # In dry_run mode no facts are archived so active count is unchanged.
    assert after_active == before_active, (
        "dry_run=True modified the database — the DatabaseManager dry_run path "
        "must not perform any writes"
    )
    assert _count_consolidation_records(mgr) == 0, (
        "dry_run=True wrote provenance records — must not write anything"
    )


def test_databasemanager_path_returns_mode_in_stats(dbmanager_db) -> None:
    """Stats dict must include a 'mode' key (extractive = 'a')."""
    mgr = dbmanager_db
    for i, c in enumerate(_CLUSTER):
        _insert_warm_fact(mgr, f"dm2_{i}", c)

    stats = consolidate_facts(mgr, profile_id="default", config=None)
    assert "mode" in stats, "stats dict missing 'mode' key on DatabaseManager path"
