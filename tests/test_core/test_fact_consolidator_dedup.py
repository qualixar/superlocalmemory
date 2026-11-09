# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Re-consolidating an unchanged cluster must converge, not accumulate.

HISTORY, because it explains the shape of this file. Before v3.6.4 the
consolidator wrote its summary into ``atomic_facts`` with a raw INSERT that
bypassed dedup, so an equivalent cluster spawned a duplicate summary fact on
every pass. v3.6.4 fixed the duplication by adding a reinforce-or-insert dance
against ``atomic_facts`` — and left the summary in the retrieval corpus, which
was the larger mistake and cost 1,195 model-written rows before anyone noticed.

4.0.10 removes the corpus write entirely. The idempotency requirement these
tests were written to protect is unchanged and still worth protecting; only its
target moved, to ``consolidated_summaries`` and its
``UNIQUE (profile_id, entity_id, content)`` constraint.

Deterministic extractive summarisation throughout (mode 'a', config=None — no
LLM), so a second pass over an unchanged cluster produces byte-identical text
and the constraint is actually exercised.
"""

from __future__ import annotations

import pytest

from superlocalmemory.core.fact_consolidator import consolidate_facts
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import MemoryRecord

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




def test_an_unchanged_cluster_yields_one_summary_however_often_it_runs(
    consolidator_db,
) -> None:
    """Maintenance runs on a schedule. Convergence is the whole requirement."""
    path, mgr = consolidator_db

    for i, c in enumerate(_CLUSTER):
        _insert_warm_fact(mgr, f"a{i}", c)
    consolidate_facts(path, profile_id="default", config=None)

    summaries = mgr.execute(
        "SELECT summary_id, content FROM consolidated_summaries "
        "WHERE profile_id='default'"
    )
    assert len(summaries) == 1, "run 1 must write exactly one display summary"
    summary = dict(summaries[0])["content"]
    assert summary, "summary should be non-empty"

    # A second identical cluster produces byte-identical extractive text, which
    # is what the UNIQUE constraint has to absorb.
    for i, c in enumerate(_CLUSTER):
        _insert_warm_fact(mgr, f"b{i}", c)
    consolidate_facts(path, profile_id="default", config=None)

    again = mgr.execute(
        "SELECT COUNT(*) AS c FROM consolidated_summaries "
        "WHERE profile_id='default' AND content=?", (summary,),
    )
    assert dict(again[0])["c"] == 1, \
        "an identical summary was stored twice; the display table must converge"


def test_a_repeat_pass_refreshes_the_coverage_window(consolidator_db) -> None:
    """Convergence must not mean the row goes stale.

    The second pass covers more facts than the first, and a reader needs to see
    that rather than a snapshot frozen at whatever the first pass happened to
    find.
    """
    path, mgr = consolidator_db
    for i, c in enumerate(_CLUSTER):
        _insert_warm_fact(mgr, f"a{i}", c)
    consolidate_facts(path, profile_id="default", config=None)
    first = dict(mgr.execute(
        "SELECT source_count, source_fact_ids FROM consolidated_summaries "
        "WHERE profile_id='default'"
    )[0])

    for i, c in enumerate(_CLUSTER):
        _insert_warm_fact(mgr, f"b{i}", c)
    consolidate_facts(path, profile_id="default", config=None)
    second = dict(mgr.execute(
        "SELECT source_count, source_fact_ids FROM consolidated_summaries "
        "WHERE profile_id='default'"
    )[0])

    assert second["source_count"] > first["source_count"], (
        "the refreshed summary still reports the first pass's source count, "
        "so the ON CONFLICT update is not firing"
    )
    assert second["source_fact_ids"] != first["source_fact_ids"]


def test_consolidation_leaves_edges_and_retention_zones_alone(
    consolidator_db,
) -> None:
    """The inverse of the rule this test used to assert.

    Removing an archived fact's association edges was right while a retrievable
    summary stood in for it — spreading activation would otherwise have kept
    ranking on a fact the user could no longer reach. A display-only summary
    stands in for nothing, so the sources stay live, and taking their edges or
    pushing them into retention zone 'archive' would delete a retrieval signal
    and hide a real memory for no gain.
    """
    path, mgr = consolidator_db
    for i, c in enumerate(_CLUSTER):
        _insert_warm_fact(mgr, f"a{i}", c)
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
    assert dict(edges[0])["c"] == 1, "a live fact's association edge was deleted"
    zone = mgr.execute(
        "SELECT lifecycle_zone AS z FROM fact_retention WHERE fact_id='a0'"
    )
    assert dict(zone[0])["z"] == "warm", \
        "a live fact was pushed into retention zone 'archive'"
    still_live = mgr.execute(
        "SELECT COUNT(*) AS c FROM atomic_facts "
        "WHERE fact_id IN ('a0','a1','a2') AND lifecycle='warm'"
    )
    assert dict(still_live[0])["c"] == 3, "source facts were archived"
