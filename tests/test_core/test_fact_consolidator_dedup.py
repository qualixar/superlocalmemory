# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""P0-3 (dedup-complete-01): the fact consolidator must obey the same
content-idempotency invariant as storage.database.store_fact.

Before v3.6.4 the consolidator wrote consolidated summary facts via a raw
`INSERT INTO atomic_facts`, bypassing dedup entirely — re-consolidating an
equivalent cluster spawned duplicate summary facts and never reinforced
evidence. These tests pin the corrected behaviour using deterministic
extractive summarisation (mode 'a', config=None — no LLM).
"""

from __future__ import annotations

import pytest

from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import MemoryRecord
from superlocalmemory.core.fact_consolidator import consolidate_facts

_NOW = "2026-01-01T00:00:00+00:00"

_CLUSTER = [
    "Zeta is a senior reliability engineer based in Berlin Germany.",
    "Zeta leads the distributed systems team at the research company.",
    "Zeta has fifteen years of experience building fault tolerant services.",
]


@pytest.fixture()
def consolidator_db(tmp_path):
    path = str(tmp_path / "consol.db")
    mgr = DatabaseManager(path)
    mgr.initialize(real_schema)
    # Apply the migration chain the engine applies (creates pinned_facts +
    # fact_consolidations + lifecycle tables the consolidator depends on).
    from superlocalmemory.storage.schema_v343 import apply_v343_schema, apply_v346_schema
    from superlocalmemory.storage.schema_v347 import apply_v347_schema
    from superlocalmemory.storage.schema_v3410 import apply_v3410_schema
    from superlocalmemory.storage.schema_v3411 import apply_v3411_schema
    for _apply in (apply_v343_schema, apply_v346_schema, apply_v347_schema,
                   apply_v3410_schema, apply_v3411_schema):
        _apply(path)
    # Parent memory row (atomic_facts.memory_id FK → memories; DatabaseManager
    # enforces FKs).
    mgr.store_memory(MemoryRecord(memory_id="mem0", profile_id="default",
                                  content="cluster source"))
    mgr.execute(
        "INSERT INTO canonical_entities "
        "(entity_id, profile_id, canonical_name, entity_type, first_seen, last_seen, fact_count) "
        "VALUES ('zeta','default','Zeta','person',?,?,3)",
        (_NOW, _NOW),
    )
    return path, mgr


def _insert_warm_fact(mgr: DatabaseManager, fid: str, content: str) -> None:
    mgr.execute(
        "INSERT INTO atomic_facts "
        "(fact_id, memory_id, profile_id, content, fact_type, "
        " canonical_entities_json, entities_json, confidence, importance, "
        " evidence_count, access_count, created_at, lifecycle) "
        "VALUES (?, 'mem0', 'default', ?, 'semantic', '[\"zeta\"]', '[\"zeta\"]', "
        " 0.8, 0.5, 1, 0, ?, 'warm')",
        (fid, content, _NOW),
    )


def _live_count(mgr: DatabaseManager, content: str) -> int:
    rows = mgr.execute(
        "SELECT COUNT(*) AS c FROM atomic_facts "
        "WHERE content = ? AND lifecycle IN ('active','warm','cold')",
        (content,),
    )
    return dict(rows[0])["c"]


def test_consolidator_dedups_identical_summary(consolidator_db) -> None:
    path, mgr = consolidator_db

    # Cluster A → consolidate → exactly one active summary fact, originals archived.
    for i, c in enumerate(_CLUSTER):
        _insert_warm_fact(mgr, f"a{i}", c)
    consolidate_facts(path, profile_id="default", config=None)

    active = mgr.execute(
        "SELECT content, evidence_count FROM atomic_facts "
        "WHERE lifecycle='active' AND profile_id='default'"
    )
    assert len(active) == 1, "run 1 must create exactly one consolidated summary"
    summary = dict(active[0])["content"]
    assert summary, "summary should be non-empty"

    # Cluster B: identical fresh warm facts → identical extractive summary.
    for i, c in enumerate(_CLUSTER):
        _insert_warm_fact(mgr, f"b{i}", c)
    consolidate_facts(path, profile_id="default", config=None)

    # The identical summary must DEDUP, not duplicate.
    assert _live_count(mgr, summary) == 1, \
        "consolidator created a duplicate summary fact (dedup bypassed)"


def test_consolidator_reinforces_evidence_on_dedup(consolidator_db) -> None:
    path, mgr = consolidator_db
    for i, c in enumerate(_CLUSTER):
        _insert_warm_fact(mgr, f"a{i}", c)
    consolidate_facts(path, profile_id="default", config=None)
    summary = dict(mgr.execute(
        "SELECT content FROM atomic_facts WHERE lifecycle='active' AND profile_id='default'"
    )[0])["content"]
    ev_before = dict(mgr.execute(
        "SELECT evidence_count AS e FROM atomic_facts WHERE content=?", (summary,)
    )[0])["e"]

    for i, c in enumerate(_CLUSTER):
        _insert_warm_fact(mgr, f"b{i}", c)
    consolidate_facts(path, profile_id="default", config=None)

    ev_after = dict(mgr.execute(
        "SELECT evidence_count AS e FROM atomic_facts WHERE content=? "
        "AND lifecycle IN ('active','warm','cold')", (summary,)
    )[0])["e"]
    assert ev_after > ev_before, "dedup on a consolidated summary must reinforce evidence"


def test_archival_removes_edges_and_sets_retention_zone(consolidator_db) -> None:
    # P1-4 (graph-integrity-01): when consolidation archives the source facts,
    # their association_edges must be removed (so spreading_activation stops
    # ranking on them) and fact_retention.lifecycle_zone set to 'archive'.
    path, mgr = consolidator_db
    for i, c in enumerate(_CLUSTER):
        _insert_warm_fact(mgr, f"a{i}", c)
    # An association edge between two cluster facts + a retention row.
    mgr.execute(
        "INSERT INTO association_edges "
        "(edge_id, profile_id, source_fact_id, target_fact_id, association_type, weight) "
        "VALUES ('e1','default','a0','a1','hebbian',0.7)"
    )
    mgr.execute(
        "INSERT INTO fact_retention (fact_id, profile_id, lifecycle_zone) "
        "VALUES ('a0','default','warm')"
    )

    consolidate_facts(path, profile_id="default", config=None)

    edges = mgr.execute(
        "SELECT COUNT(*) AS c FROM association_edges "
        "WHERE source_fact_id='a0' OR target_fact_id='a0'"
    )
    assert dict(edges[0])["c"] == 0, "archived fact's association edges not removed"
    zone = mgr.execute("SELECT lifecycle_zone AS z FROM fact_retention WHERE fact_id='a0'")
    assert dict(zone[0])["z"] == "archive", "retention zone not set to archive"
    archived = mgr.execute(
        "SELECT COUNT(*) AS c FROM atomic_facts af "
        "JOIN fact_retention fr ON fr.fact_id = af.fact_id "
        "WHERE af.fact_id IN ('a0','a1','a2') "
        "AND af.lifecycle = 'archived' AND fr.lifecycle_zone = 'archive'"
    )
    assert dict(archived[0])["c"] == 3, "every archived source needs a retention row"
