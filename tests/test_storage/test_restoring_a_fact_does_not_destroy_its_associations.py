# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Storing a fact twice under one id must not delete the rows hanging off it.

``store_fact`` used ``INSERT OR REPLACE``, which SQLite implements as a DELETE
followed by an INSERT. Eight tables carry
``FOREIGN KEY (fact_id) REFERENCES atomic_facts (fact_id) ON DELETE CASCADE``,
so re-storing a fact under an id that already existed dropped its retention
row, its access history, its context and its importance — returned the fact_id
as though it had succeeded, and raised nothing.

This is the same defect already fixed in ``store_memory``. It was left in place
here because no caller had been *shown* to reach it: ``store_fact`` dedups on
content and returns early, and ``fact_id`` normally comes from a uuid factory.

A caller does reach it. ``MemoryEngine.store_fact_direct`` exists to persist a
caller-chosen id — ``canonical_store_fact`` is documented as ingesting "a
caller-built fact without changing its public ID" and raises if the id is not
preserved. Within one profile an idempotency key of ``prebuilt:<fact_id>``
catches a second store of different content. But that key is scoped
``(profile_id, source_type, idempotency_key)`` while ``atomic_facts.fact_id``
is a bare ``TEXT PRIMARY KEY`` — global. So the same id stored from a second
profile misses the idempotency guard, then misses the content dedup (which
filters on ``profile_id`` too), and lands on the replace.

``test_a_second_profile_cannot_destroy_the_first_profiles_fact`` pins that
path; it fails on the unfixed code with the first profile's fact gone.
"""

from __future__ import annotations

import pytest

from superlocalmemory.storage.database import ProfileOwnershipConflict
from superlocalmemory.storage.models import AtomicFact, MemoryRecord

# Four of the eight tables that cascade off atomic_facts.fact_id. Enough to
# prove the delete happened; each is written by a different subsystem
# (lifecycle, auto-invoke, access accounting, graph importance).
CASCADE_TABLES = ("fact_retention", "fact_context", "fact_access_log", "fact_importance")


@pytest.fixture
def db(engine_with_mock_deps):
    return engine_with_mock_deps._db


def _associate(db, fact_id: str, profile_id: str = "default") -> None:
    """Write one row into each cascade table for *fact_id*."""
    db.execute(
        "INSERT OR REPLACE INTO fact_retention (fact_id, profile_id) VALUES (?,?)",
        (fact_id, profile_id),
    )
    db.execute(
        "INSERT OR REPLACE INTO fact_context "
        "(fact_id, profile_id, contextual_description) VALUES (?,?,?)",
        (fact_id, profile_id, "what this fact is about"),
    )
    db.execute(
        "INSERT OR REPLACE INTO fact_access_log (log_id, fact_id, profile_id) VALUES (?,?,?)",
        (f"log-{fact_id}", fact_id, profile_id),
    )
    db.execute(
        "INSERT OR REPLACE INTO fact_importance (fact_id, profile_id) VALUES (?,?)",
        (fact_id, profile_id),
    )


def _associations(db, fact_id: str) -> dict[str, int]:
    return {
        table: len(db.execute(
            f"SELECT 1 FROM {table} WHERE fact_id = ?", (fact_id,),
        ))
        for table in CASCADE_TABLES
    }


def _fact_row(db, fact_id: str) -> dict:
    rows = db.execute("SELECT * FROM atomic_facts WHERE fact_id = ?", (fact_id,))
    return dict(rows[0]) if rows else {}


def test_restoring_a_fact_under_the_same_id_keeps_its_associations(db) -> None:
    """The defect, reproduced. Content differs, so the dedup does not catch it."""
    db.store_memory(MemoryRecord(
        memory_id="m1", profile_id="default", content="the source turn",
    ))
    db.store_fact(AtomicFact(
        fact_id="f1", memory_id="m1", profile_id="default",
        content="Alex works at Acme",
    ))
    _associate(db, "f1")
    before = _associations(db, "f1")
    assert all(v == 1 for v in before.values()), before

    db.store_fact(AtomicFact(
        fact_id="f1", memory_id="m1", profile_id="default",
        content="Varun works at Qualixar",
    ))

    assert _associations(db, "f1") == before, (
        "re-storing a fact under an existing id deleted the rows that "
        "reference it through ON DELETE CASCADE"
    )


def test_the_first_observation_time_is_not_rewritten(db) -> None:
    """When the fact was first seen is history; a re-store is not a new sighting."""
    db.store_memory(MemoryRecord(
        memory_id="m2", profile_id="default", content="turn",
    ))
    db.store_fact(AtomicFact(
        fact_id="f2", memory_id="m2", profile_id="default", content="first content",
        created_at="2020-01-01T00:00:00+00:00",
    ))
    db.store_fact(AtomicFact(
        fact_id="f2", memory_id="m2", profile_id="default", content="second content",
        created_at="2026-08-22T00:00:00+00:00",
    ))

    assert _fact_row(db, "f2")["created_at"] == "2020-01-01T00:00:00+00:00"


def test_restoring_still_updates_the_row(db) -> None:
    """Last write wins is preserved — the fix must not turn it into a no-op."""
    db.store_memory(MemoryRecord(
        memory_id="m3", profile_id="default", content="turn",
    ))
    db.store_fact(AtomicFact(
        fact_id="f3", memory_id="m3", profile_id="default",
        content="first content", importance=0.1, confidence=0.2,
    ))
    db.store_fact(AtomicFact(
        fact_id="f3", memory_id="m3", profile_id="default",
        content="second content", importance=0.9, confidence=0.8,
    ))

    row = _fact_row(db, "f3")
    assert row["content"] == "second content"
    assert row["importance"] == pytest.approx(0.9)
    assert row["confidence"] == pytest.approx(0.8)
    assert len(db.execute(
        "SELECT 1 FROM atomic_facts WHERE fact_id = ?", ("f3",),
    )) == 1, "the upsert must not leave a duplicate row"


def test_identical_content_still_reinforces_instead_of_inserting(db) -> None:
    """The dedup fast path is untouched: same content reinforces, never replaces."""
    db.store_memory(MemoryRecord(
        memory_id="m4", profile_id="default", content="turn",
    ))
    first = db.store_fact(AtomicFact(
        memory_id="m4", profile_id="default", content="a repeated fact",
    ))
    _associate(db, first)
    before = _associations(db, first)

    second = db.store_fact(AtomicFact(
        memory_id="m4", profile_id="default", content="a repeated fact",
    ))

    assert second == first, "identical content must collapse onto the canonical id"
    assert _associations(db, first) == before
    row = _fact_row(db, first)
    assert row["evidence_count"] == 2, "the reinforcement bump must still happen"


def test_a_new_fact_id_still_inserts(db) -> None:
    """The ordinary path — the overwhelming majority of calls."""
    db.store_memory(MemoryRecord(
        memory_id="m5", profile_id="default", content="turn",
    ))
    fact_id = db.store_fact(AtomicFact(
        fact_id="f5", memory_id="m5", profile_id="default", content="brand new fact",
    ))

    assert fact_id == "f5"
    assert _fact_row(db, "f5")["content"] == "brand new fact"


def test_a_second_profile_cannot_destroy_the_first_profiles_fact(
    engine_with_mock_deps,
) -> None:
    """The reachable caller path, end to end through the public engine API.

    ``profile_id`` is a public settable property and ``store_fact_direct``
    contracts to preserve the caller's id, so two profiles sharing one store
    can be handed the same caller-chosen fact_id — an importer keyed on an
    external record id syncing one source into two workspaces does exactly
    this. The per-profile idempotency key does not span profiles and the
    content dedup filters on profile_id, so nothing upstream stops it.

    On the unfixed code the second store won: the first profile's fact was
    replaced outright — new owner, new content — and its associations were
    cascaded away, silently.

    The upsert alone did not close this. It stopped the cascade, but it still
    listed ``profile_id = excluded.profile_id``, so the second profile still
    took the row. What happened next depended on something unrelated: where a
    scene referenced the fact, ``scene_fact_members`` — a composite
    ``(profile_id, fact_id) -> atomic_facts (profile_id, fact_id)`` foreign key
    — was orphaned by the ownership change and SQLite refused the write with a
    bare ``FOREIGN KEY constraint failed``; where no scene referenced it, the
    identical write succeeded and the fact changed hands in silence. Tenant
    isolation was resting on whether a projection happened to exist.

    So the refusal is now the store's own, not the schema's by accident: the
    owner is read before the row is touched and a mismatch raises
    ``ProfileOwnershipConflict``, naming both profiles. Same outcome with or
    without a scene, and legible when it fires.
    """
    engine = engine_with_mock_deps
    db = engine._db

    engine.store_fact_direct(AtomicFact(
        fact_id="external-record-42", profile_id="default",
        content="Alex works at Acme as an architect",
    ))
    _associate(db, "external-record-42")
    before_associations = _associations(db, "external-record-42")
    before_row = _fact_row(db, "external-record-42")

    db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?,?)",
        ("work", "work"),
    )
    engine.profile_id = "work"
    with pytest.raises(ProfileOwnershipConflict) as refusal:
        engine.store_fact_direct(AtomicFact(
            fact_id="external-record-42", profile_id="work",
            content="An unrelated fact in a different workspace",
        ))
    # Both owners named: a refusal nobody can read gets worked around.
    assert "default" in str(refusal.value)
    assert "work" in str(refusal.value)

    assert _associations(db, "external-record-42") == before_associations, (
        "a store from a second profile deleted the first profile's "
        "fact associations through the cascade"
    )
    after_row = _fact_row(db, "external-record-42")
    assert after_row["profile_id"] == before_row["profile_id"], (
        "a second profile took ownership of the first profile's fact"
    )
    assert after_row["content"] == before_row["content"]
