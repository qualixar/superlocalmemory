# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""A summary is a view of memory. It must not become a memory.

v3.6 introduced consolidated summaries to show a non-technical reader what
their store contains, deliberately kept out of the retrieval corpus. v3.6.4
then ended each cluster with a raw ``INSERT INTO atomic_facts`` and the
boundary was gone -- silently, because nothing asserted it.

What that cost, measured on the author's 5,089-fact store before this change:

    consolidator-authored rows in atomic_facts       1,195
      ...retrieval-eligible                            307
      ...that are summaries of summaries               353
      ...with a temporal_events row                      0
    genuine memories archived to make room for them     528
      ...archived by anything other than consolidation    0

and, asked "what am I working on", ranks 1, 2 and 3 all read "Unfortunately,
there is no information available about 'Gateway', 'State', 'Bounded', or
'Claude' in the provided text."

These tests pin the boundary in four independent ways, so restoring any one
piece of the old behaviour fails at least one of them.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

import pytest

from superlocalmemory.core.fact_consolidator import (
    _MIN_CLUSTER_SIZE,
    _consolidate_cluster,
    consolidate_facts,
)

# ---------------------------------------------------------------------------
# A store with one entity and enough warm facts to form a cluster.
# ---------------------------------------------------------------------------

_PROFILE = "default"
_ENTITY_ID = "ent-slm"
_ENTITY_NAME = "SuperLocalMemory"


def _seed(db_path: Path, *, n_facts: int = 4) -> list[str]:
    # The full chain the engine applies. The consolidator reads pinned_facts
    # and writes fact_retention, neither of which create_all_tables owns, so a
    # partial bootstrap makes every assertion below vacuous.
    from superlocalmemory.storage import schema as real_schema
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage.schema_v343 import (
        apply_v343_schema,
        apply_v346_schema,
    )
    from superlocalmemory.storage.schema_v347 import apply_v347_schema
    from superlocalmemory.storage.schema_v3410 import apply_v3410_schema
    from superlocalmemory.storage.schema_v3411 import apply_v3411_schema

    mgr = DatabaseManager(str(db_path))
    mgr.initialize(real_schema)
    for _apply in (apply_v343_schema, apply_v346_schema, apply_v347_schema,
                   apply_v3410_schema, apply_v3411_schema):
        _apply(str(db_path))

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(
        "INSERT OR IGNORE INTO canonical_entities "
        "(entity_id, profile_id, canonical_name) VALUES (?,?,?)",
        (_ENTITY_ID, _PROFILE, _ENTITY_NAME),
    )
    memory_id = uuid.uuid4().hex[:16]
    conn.execute(
        "INSERT INTO memories (memory_id, profile_id, content) VALUES (?,?,?)",
        (memory_id, _PROFILE, "seed conversation"),
    )
    fact_ids: list[str] = []
    for i in range(n_facts):
        fid = f"fact-{i:03d}"
        fact_ids.append(fid)
        conn.execute(
            "INSERT INTO atomic_facts "
            "(fact_id, memory_id, profile_id, content, fact_type, "
            " canonical_entities_json, confidence, importance, lifecycle, "
            " created_at) "
            "VALUES (?,?,?,?,'semantic',?,?,?, 'warm', ?)",
            (
                fid, memory_id, _PROFILE,
                f"SuperLocalMemory shipped release {i} with a measured "
                f"improvement to recall repeatability.",
                json.dumps([_ENTITY_ID]), 0.9, 0.3,
                f"2026-0{i + 1}-15T10:00:00+00:00",
            ),
        )
    conn.commit()
    conn.close()
    return fact_ids


@pytest.fixture()
def store(tmp_path: Path) -> Path:
    db = tmp_path / "memory.db"
    _seed(db)
    return db


def _rows(db: Path, sql: str, *params: object) -> list[dict]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, params)]
    finally:
        conn.close()


class TestNothingIsWrittenToAtomicFacts:
    """The row count is the assertion. It cannot be satisfied by intent."""

    def test_consolidation_adds_no_fact(self, store: Path) -> None:
        before = _rows(store, "SELECT fact_id FROM atomic_facts")
        stats = consolidate_facts(str(store), profile_id=_PROFILE)
        after = _rows(store, "SELECT fact_id FROM atomic_facts")

        assert stats["consolidated"] >= 1, (
            f"the fixture did not form a cluster, so this test proves "
            f"nothing: {stats}"
        )
        assert {r["fact_id"] for r in after} == {r["fact_id"] for r in before}, (
            "consolidation created or removed a row in atomic_facts; the "
            "summary must live in consolidated_summaries only"
        )

    def test_no_fact_carries_an_empty_memory_id(self, store: Path) -> None:
        """The signature of the old back door, asserted directly.

        ``memory_id=''`` is what a writer that skipped the ingestion pipeline
        left behind -- and it also violates the declared foreign key to
        ``memories``, which only went unnoticed because the connection the old
        code wrote through never enabled foreign_keys.
        """
        consolidate_facts(str(store), profile_id=_PROFILE)
        orphans = _rows(
            store,
            "SELECT fact_id, content FROM atomic_facts WHERE memory_id = '' "
            "   OR memory_id NOT IN (SELECT memory_id FROM memories)",
        )
        assert orphans == [], (
            f"{len(orphans)} fact(s) reference no memory row: "
            f"{[o['fact_id'] for o in orphans]}"
        )

    def test_source_facts_are_not_archived(self, store: Path) -> None:
        """Archiving made sense only while a retrievable summary replaced them.

        A display-only summary replaces nothing, so archiving the sources takes
        real memories out of recall and puts nothing in their place.
        """
        consolidate_facts(str(store), profile_id=_PROFILE)
        lifecycles = {r["lifecycle"] for r in _rows(
            store, "SELECT lifecycle FROM atomic_facts",
        )}
        assert lifecycles == {"warm"}, (
            f"source facts changed lifecycle to {lifecycles - {'warm'}}; "
            "consolidation must be purely additive to the store"
        )
        zones = _rows(store, "SELECT lifecycle_zone FROM fact_retention")
        assert not any(z["lifecycle_zone"] == "archive" for z in zones), (
            "a source fact was pushed into retention zone 'archive', which "
            "the forgetting filter excludes from every normal recall"
        )

    def test_association_edges_survive(self, store: Path) -> None:
        """The old path deleted edges touching the facts it archived.

        Spreading activation reads those edges. Deleting them for facts that
        are still live removes a retrieval signal for no reason.
        """
        conn = sqlite3.connect(str(store))
        conn.execute(
            "INSERT INTO association_edges "
            "(edge_id, profile_id, source_fact_id, target_fact_id, "
            " association_type, weight) "
            "VALUES ('e1', ?, 'fact-000', 'fact-001', 'hebbian', 0.5)",
            (_PROFILE,),
        )
        conn.commit()
        conn.close()

        consolidate_facts(str(store), profile_id=_PROFILE)
        edges = _rows(store, "SELECT edge_id FROM association_edges")
        assert [e["edge_id"] for e in edges] == ["e1"], (
            "consolidation deleted an association edge between two live facts"
        )


class TestTheSummaryLandsInTheDisplayTable:
    def test_a_row_is_written_with_its_provenance(self, store: Path) -> None:
        fact_ids = sorted(r["fact_id"] for r in _rows(
            store, "SELECT fact_id FROM atomic_facts",
        ))
        consolidate_facts(str(store), profile_id=_PROFILE)
        summaries = _rows(store, "SELECT * FROM consolidated_summaries")
        assert len(summaries) == 1, f"expected one summary, got {len(summaries)}"
        row = summaries[0]
        assert row["entity_id"] == _ENTITY_ID
        assert row["entity_name"] == _ENTITY_NAME
        assert row["content"].strip()
        assert sorted(json.loads(row["source_fact_ids"])) == fact_ids
        assert row["source_count"] == len(fact_ids)
        assert row["generated_by"] in {"extractive", "ollama", "cloud"}

    def test_the_coverage_window_comes_from_the_sources(self, store: Path) -> None:
        """A summary has no observation date -- it was never observed.

        The honest dates are the span of what it covers, which is also what
        lets the dashboard say which stretch of work a summary is about.
        """
        consolidate_facts(str(store), profile_id=_PROFILE)
        row = _rows(store, "SELECT * FROM consolidated_summaries")[0]
        created = sorted(
            r["created_at"] for r in _rows(
                store, "SELECT created_at FROM atomic_facts",
            )
        )
        assert row["source_earliest"] == created[0]
        assert row["source_latest"] == created[-1]

    def test_a_second_pass_refreshes_rather_than_duplicates(
        self, store: Path,
    ) -> None:
        """Maintenance runs repeatedly. Without this it accumulates near-copies.

        The old code needed a reinforce-or-insert dance against atomic_facts
        for exactly this reason; the UNIQUE constraint replaces it.
        """
        consolidate_facts(str(store), profile_id=_PROFILE)
        consolidate_facts(str(store), profile_id=_PROFILE)
        consolidate_facts(str(store), profile_id=_PROFILE)
        summaries = _rows(store, "SELECT summary_id FROM consolidated_summaries")
        assert len(summaries) == 1, (
            f"three passes produced {len(summaries)} rows; the display table "
            "must converge, not grow"
        )

    def test_the_summary_does_not_inherit_the_clusters_entity_pool(
        self, store: Path,
    ) -> None:
        """Pooling the cluster's entities is what made these rows out-rank facts.

        A summary carrying ten facts' entity lists has more entity links than
        any single real memory, so the entity channel ranks it first. The
        display table has one entity column for the entity that seeded the
        cluster and no pooled list to inherit.
        """
        columns = {
            r["name"] for r in _rows(
                store, "SELECT name FROM pragma_table_info('consolidated_summaries')",
            )
        }
        assert "canonical_entities_json" not in columns
        assert "entities_json" not in columns
        assert "entity_id" in columns


class TestANonAnswerIsRefusedBeforeItIsWritten:
    def test_a_refusal_supplied_as_a_presummary_is_rejected(
        self, store: Path,
    ) -> None:
        """The write path must not trust that its caller vetted the text.

        ``_consolidate_cluster`` is reachable with a caller-supplied summary
        (the short-write-lock path does exactly that), so "the caller checked"
        is the assumption to distrust -- it is the one that let 1,195
        non-answers through.
        """
        fact_ids = sorted(r["fact_id"] for r in _rows(
            store, "SELECT fact_id FROM atomic_facts",
        ))
        assert len(fact_ids) >= _MIN_CLUSTER_SIZE

        conn = sqlite3.connect(str(store))
        conn.row_factory = sqlite3.Row
        try:
            result = _consolidate_cluster(
                conn, _PROFILE, _ENTITY_ID, _ENTITY_NAME, fact_ids,
                dry_run=False, config=None,
                _presummary=(
                    "Unfortunately, there is no information available about "
                    "'Gateway', 'State', 'Bounded', or 'Claude' in the "
                    "provided text."
                ),
            )
            conn.commit()
        finally:
            conn.close()

        assert result is None, "a refusal was accepted as a summary"
        assert _rows(store, "SELECT summary_id FROM consolidated_summaries") == []

    def test_a_real_summary_supplied_the_same_way_is_accepted(
        self, store: Path,
    ) -> None:
        """The negative control for the test above.

        Without this, the rejection test would also pass if
        ``_consolidate_cluster`` had simply stopped working.
        """
        fact_ids = sorted(r["fact_id"] for r in _rows(
            store, "SELECT fact_id FROM atomic_facts",
        ))
        conn = sqlite3.connect(str(store))
        conn.row_factory = sqlite3.Row
        try:
            result = _consolidate_cluster(
                conn, _PROFILE, _ENTITY_ID, _ENTITY_NAME, fact_ids,
                dry_run=False, config=None,
                _presummary=(
                    "SuperLocalMemory shipped four releases in 2026, each "
                    "measured against recall repeatability on a real store."
                ),
            )
            conn.commit()
        finally:
            conn.close()

        assert result is not None
        assert len(_rows(store, "SELECT summary_id FROM consolidated_summaries")) == 1

    def test_scaffolding_around_a_real_summary_is_stripped_not_discarded(
        self, store: Path,
    ) -> None:
        """"Here is a concise summary: <content>" is salvageable.

        20 of the author's stored summaries open with that phrase because this
        module never called the stripper that has existed since 3.6. Rejecting
        them outright would throw away the content along with the scaffolding,
        so the order is strip-then-judge.
        """
        fact_ids = sorted(r["fact_id"] for r in _rows(
            store, "SELECT fact_id FROM atomic_facts",
        ))
        body = (
            "SuperLocalMemory shipped four releases in 2026, each measured "
            "against recall repeatability on a real store."
        )
        conn = sqlite3.connect(str(store))
        conn.row_factory = sqlite3.Row
        try:
            _consolidate_cluster(
                conn, _PROFILE, _ENTITY_ID, _ENTITY_NAME, fact_ids,
                dry_run=False, config=None,
                _presummary=f"Sure! Here is a concise summary:\n\n{body}",
            )
            conn.commit()
        finally:
            conn.close()

        rows = _rows(store, "SELECT content FROM consolidated_summaries")
        assert len(rows) == 1, "a salvageable summary was discarded"
        assert "concise summary" not in rows[0]["content"].lower(), (
            f"scaffolding survived into the stored summary: "
            f"{rows[0]['content'][:120]!r}"
        )
        assert body in rows[0]["content"]

    def test_refused_clusters_are_counted_apart_from_errors(
        self, store: Path,
    ) -> None:
        """Refusing junk is the guard working; a failure is not.

        Collapsing the two would make a run where every cluster was refused
        look identical to a healthy run with nothing to do.
        """
        stats = consolidate_facts(str(store), profile_id=_PROFILE)
        assert "rejected" in stats
        assert stats["errors"] == 0
        assert stats["facts_summarized"] >= _MIN_CLUSTER_SIZE
