"""Erasing a chosen set of facts goes through the same erasure service as erasing an entity."""

from __future__ import annotations

import pytest

from superlocalmemory.compliance.gdpr import GDPRCompliance
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, MemoryRecord


@pytest.fixture()
def db(tmp_path):
    d = DatabaseManager(tmp_path / "g.db")
    d.initialize(real_schema)
    d.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('u', 'U')")
    d.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('v', 'V')")
    for pid, mid, fid in (("u", "m1", "f1"), ("u", "m2", "f2"), ("v", "m3", "f3")):
        d.store_memory(MemoryRecord(memory_id=mid, profile_id=pid, content="text " + fid))
        d.store_fact(AtomicFact(fact_id=fid, memory_id=mid, profile_id=pid, content="fact " + fid))
    return d


def _left(db):
    return sorted(dict(r)["fact_id"] for r in db.execute("SELECT fact_id FROM atomic_facts"))


def test_only_the_named_facts_of_that_profile_are_erased(db):
    out = GDPRCompliance(db).forget_facts(["f1", "f3"], "u", subject_id="doc-1")
    assert out["facts"] == 1 and out["erasure_complete"] == 1
    assert _left(db) == ["f2", "f3"]


def test_nothing_named_changes_nothing(db):
    out = GDPRCompliance(db).forget_facts([], "u", subject_id="doc-1")
    assert out["facts"] == 0 and _left(db) == ["f1", "f2", "f3"]
