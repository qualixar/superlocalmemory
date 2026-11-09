# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""A memory that reached SQLite must reach the graph and vector stores too.

The graph and the embeddings live in CozoDB and LanceDB, which are separate
storage engines from SQLite. No transaction can span them, so "write the fact,
then write the projection" has a window: a crash between the two leaves a
memory that exists and cannot be recalled, with nothing recording that it
happened.

The outbox closes that window. A row naming the fact is written *inside the
same SQLite transaction as the fact*, so the two commit or roll back together.
A worker drains the row into the projections and deletes it only once they have
accepted it. Crash anywhere and the row survives, so the work is retried.

These tests pin the invariant the whole design exists for: **every fact in
SQLite is in the projection or in the outbox, and never in neither.**
"""

from __future__ import annotations

import pytest

from superlocalmemory.storage import projection_outbox
from superlocalmemory.storage.models import (
    AtomicFact,
    EdgeType,
    GraphEdge,
    MemoryRecord,
)
from tests.conftest import force_sync_enrichment

_PARENT_MEMORY = "m-outbox"


def _outbox(db) -> dict[str, dict]:
    """Every outbox row, keyed by fact_id."""
    return {
        dict(r)["fact_id"]: dict(r)
        for r in db.execute("SELECT * FROM projection_outbox")
    }


def _fact(content: str, profile_id: str = "default") -> AtomicFact:
    """A fact with a real id — the model generates one when none is given."""
    return AtomicFact(
        memory_id=_PARENT_MEMORY, profile_id=profile_id, content=content,
    )


@pytest.fixture
def db(engine_with_mock_deps):
    """A live store with the parent memory row every fact below belongs to.

    ``atomic_facts.memory_id`` is a foreign key and these connections enforce
    them, so a fact built by hand needs its parent to exist first.
    """
    manager = engine_with_mock_deps._db
    manager.store_memory(MemoryRecord(
        memory_id=_PARENT_MEMORY, profile_id="default",
        content="parent record for the outbox tests",
    ))
    manager.execute("DELETE FROM projection_outbox")
    return manager


# ---------------------------------------------------------------------------
# The invariant, through the door a user actually comes in
# ---------------------------------------------------------------------------

class TestTheRealRememberPath:
    """The path `remember` takes, not a unit-test shortcut around it."""

    def test_the_main_remember_path_enqueues_what_it_stores(
        self, engine_with_mock_deps,
    ) -> None:
        """A memory stored through the real ingestion path is queued to project.

        This is the defect this work exists for. ``run_store`` — the path every
        ``remember`` call takes — never touched the projection at all: the only
        call to ``sync_new_fact`` in the codebase sat in
        ``run_store_fact_direct``, a fire-and-forget entry point that ordinary
        ingestion does not use. Switching the backends on without this would
        have produced a graph missing every memory stored since promotion.
        """
        engine = force_sync_enrichment(engine_with_mock_deps)
        fact_ids = engine.store("CozoDB holds the graph and LanceDB holds the vectors")
        assert fact_ids, "ingestion stored nothing, so this proves nothing"

        queued = set(_outbox(engine._db))
        assert set(fact_ids) <= queued, (
            f"stored {sorted(fact_ids)} but only {sorted(queued)} are queued to "
            "project — the missing ones would be unrecallable from the graph"
        )

    def test_every_queued_row_names_a_fact_that_exists(
        self, engine_with_mock_deps,
    ) -> None:
        """The outbox never points at a fact SQLite does not have."""
        engine = force_sync_enrichment(engine_with_mock_deps)
        engine.store("The outbox is drained by a worker, not by the caller")

        db = engine._db
        for fact_id, row in _outbox(db).items():
            if row["op"] == "delete":
                continue
            assert db.get_fact(fact_id) is not None, (
                f"outbox names {fact_id[:12]} which is not in atomic_facts"
            )


# ---------------------------------------------------------------------------
# Atomicity — the reason the row lives in SQLite and not in a queue
# ---------------------------------------------------------------------------

class TestTheRowCommitsWithTheFact:
    """One transaction, or the guarantee is worthless."""

    def test_a_rolled_back_fact_leaves_no_queued_row(self, db) -> None:
        """If the fact did not commit, neither did its outbox row."""
        fact = _fact("this store is going to be rolled back")

        with pytest.raises(RuntimeError, match="deliberate"):
            with db.transaction():
                db.store_fact(fact)
                raise RuntimeError("deliberate failure inside the transaction")

        assert db.get_fact(fact.fact_id) is None
        assert fact.fact_id not in _outbox(db), (
            "an outbox row outlived the fact it names — the drain would then "
            "project a memory that does not exist"
        )

    def test_a_committed_fact_always_has_its_row(self, db) -> None:
        """The other half: commit the fact, and the row is there."""
        fact = _fact("this store commits")
        with db.transaction():
            db.store_fact(fact)

        assert db.get_fact(fact.fact_id) is not None
        assert fact.fact_id in _outbox(db)


# ---------------------------------------------------------------------------
# What else changes a projection
# ---------------------------------------------------------------------------

class TestEverythingTheProjectionIsDerivedFrom:
    """A projection is stale unless every input re-queues it."""

    def test_an_edge_write_requeues_both_of_its_endpoints(self, db) -> None:
        """Edges arrive after the fact, so the fact must be projected again.

        Ingestion is queryable-first: the fact commits immediately and its
        graph edges are written later by the materializer. A row queued only at
        fact-insert time would therefore be drained before a single edge
        existed, and the projected graph would have the node and none of its
        connections.
        """
        left, right = _fact("left endpoint"), _fact("right endpoint")
        db.store_fact(left)
        db.store_fact(right)
        db.execute("DELETE FROM projection_outbox")

        db.store_edge(GraphEdge(
            profile_id="default", source_id=left.fact_id, target_id=right.fact_id,
            edge_type=EdgeType.SEMANTIC, weight=0.8,
        ))

        queued = set(_outbox(db))
        assert {left.fact_id, right.fact_id} <= queued, (
            "an edge was written without re-queueing its endpoints, so the "
            "projected graph keeps the old adjacency"
        )

    def test_an_embedding_update_requeues_the_fact(self, db) -> None:
        """The vector store holds the embedding, so a new one must reach it."""
        fact = _fact("its embedding is about to change")
        db.store_fact(fact)
        db.execute("DELETE FROM projection_outbox")

        db.update_fact(fact.fact_id, {"embedding": [0.5] * 8})

        assert fact.fact_id in _outbox(db)

    def test_an_access_count_bump_does_not_requeue(self, db) -> None:
        """Recall touches counters on every hit; those are not projected.

        Queueing on any update at all would put one outbox row per recalled
        fact per recall on the drain worker's back, for a column neither Cozo
        nor Lance stores.
        """
        fact = _fact("this one only gets read")
        db.store_fact(fact)
        db.execute("DELETE FROM projection_outbox")

        db.update_fact(fact.fact_id, {"access_count": 7})

        assert fact.fact_id not in _outbox(db), (
            "a counter bump queued a projection write; at recall volume this "
            "is a self-inflicted denial of service on the drain worker"
        )

    def test_a_deleted_fact_is_queued_for_removal(self, db) -> None:
        """Forgetting must reach the projections, or the memory survives there."""
        fact = _fact("this one gets deleted")
        db.store_fact(fact)
        db.execute("DELETE FROM projection_outbox")

        db.delete_fact(fact.fact_id)

        row = _outbox(db).get(fact.fact_id)
        assert row is not None, "a deleted fact was never queued for removal"
        assert row["op"] == "delete"


# ---------------------------------------------------------------------------
# Queue mechanics
# ---------------------------------------------------------------------------

class TestTheQueueItself:
    """Coalescing, ordering, and the races a drain worker creates."""

    def test_repeated_queueing_coalesces_to_one_row(self, db) -> None:
        """One fact touched ten times is one unit of work, not ten.

        The drain re-derives the fact's whole projected state from SQLite, so
        only the latest intent matters. Without coalescing, a bulk edit would
        queue a row per write and the depth metric would read as a backlog when
        it is one fact.
        """
        fact = _fact("touched repeatedly")
        db.store_fact(fact)
        for i in range(10):
            db.update_fact(fact.fact_id, {"embedding": [float(i)] * 8})

        rows = db.execute(
            "SELECT COUNT(*) AS n FROM projection_outbox WHERE fact_id = ?",
            (fact.fact_id,),
        )
        assert dict(rows[0])["n"] == 1

    def test_a_delete_supersedes_a_pending_upsert(self, db) -> None:
        """Latest intent wins, so a drain can never resurrect a deleted fact."""
        fact = _fact("stored then immediately deleted")
        db.store_fact(fact)
        db.delete_fact(fact.fact_id)

        rows = [dict(r) for r in db.execute(
            "SELECT * FROM projection_outbox WHERE fact_id = ?", (fact.fact_id,),
        )]
        assert len(rows) == 1, f"expected one intent, found {len(rows)}"
        assert rows[0]["op"] == "delete"

    def test_requeueing_during_a_drain_is_not_swallowed(self, db) -> None:
        """A write that lands mid-drain must survive the drain's own cleanup.

        The drain claims a row, projects it, then deletes it. An unconditional
        delete would discard an intent queued in between — the fact would be
        projected at its older state and nothing would say so. The delete is
        therefore conditional on the row not having moved.
        """
        fact = _fact("written twice, once mid-drain")
        db.store_fact(fact)

        claimed = projection_outbox.claim_batch(db, limit=10)
        assert any(c["fact_id"] == fact.fact_id for c in claimed)

        # A concurrent writer touches the same fact before the drain finishes.
        db.update_fact(fact.fact_id, {"embedding": [1.0] * 8})

        for c in claimed:
            projection_outbox.resolve(db, c["fact_id"], c["revision"])

        assert fact.fact_id in _outbox(db), (
            "the drain deleted an intent it never projected"
        )

    def test_resolving_an_unchanged_row_clears_it(self, db) -> None:
        """The ordinary case still drains, or the queue only grows."""
        fact = _fact("drains cleanly")
        db.store_fact(fact)

        for c in projection_outbox.claim_batch(db, limit=10):
            projection_outbox.resolve(db, c["fact_id"], c["revision"])

        assert _outbox(db) == {}

    def test_a_failed_projection_keeps_the_row_and_counts_the_attempt(self, db) -> None:
        """A projection that raises must leave evidence, not silence."""
        fact = _fact("its projection is going to fail")
        db.store_fact(fact)

        projection_outbox.record_failure(db, fact.fact_id, "cozo unavailable")

        row = _outbox(db)[fact.fact_id]
        assert row["attempts"] == 1
        assert "cozo unavailable" in (row["last_error"] or "")

    def test_depth_is_the_number_of_pending_rows(self, db) -> None:
        """The health metric is the queue length, and nothing else."""
        assert projection_outbox.depth(db) == 0
        for i in range(3):
            db.store_fact(_fact(f"pending {i}"))
        assert projection_outbox.depth(db) == 3


# ---------------------------------------------------------------------------
# A store that has no projection
# ---------------------------------------------------------------------------

class TestWhenThereIsNothingToProject:
    """The queue must not become a leak on a store with no backends."""

    def test_a_store_without_the_table_still_accepts_writes(self, db) -> None:
        """An un-migrated store keeps working; it simply queues nothing.

        The table's presence is what enables queueing, so a store that has not
        reached the migration must not fail every write with "no such table".
        """
        db.execute("DROP TABLE projection_outbox")
        projection_outbox.forget_availability(db)

        fact = _fact("stored on a store with no outbox table")
        db.store_fact(fact)

        assert db.get_fact(fact.fact_id) is not None
        assert projection_outbox.depth(db) == 0
