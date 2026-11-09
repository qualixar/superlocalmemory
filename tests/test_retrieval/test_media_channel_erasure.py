"""The picture channel never returns a fact whose erasure has started."""

from __future__ import annotations

import time

import pytest
from benchmark.recall_channels import migrate

from superlocalmemory.retrieval.media_channel import MediaChannel
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, MemoryRecord

from ._media_support import FakeClient, make_store, mid, unit


@pytest.fixture
def world(tmp_path):
    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(schema)
    migrate(db)
    store = make_store(tmp_path, [(mid(1), "image", "pic-a", unit(0)),
                                  (mid(2), "image", "pic-b", unit(0))])
    for name in ("pic-a", "pic-b"):
        db.store_memory(MemoryRecord(
            memory_id=name, content="[Image without text]",
            metadata={"_slm_source": {"type": "media", "media_id": mid(1)}}))
        db.store_fact(AtomicFact(fact_id=name + "-fact", memory_id=name,
                                 content="[Image without text]"))
    channel = MediaChannel(lambda: store, lambda: FakeClient(unit(0)), db)
    try:
        yield db, channel
    finally:
        store.close()
        db.close()


def _tombstone(db, fact_id):
    db.execute("INSERT INTO projection_tombstones(profile_id,fact_id,erasure_id,created_at) "
               "VALUES ('default',?, 'synthetic-erasure', ?)", (fact_id, time.time()))


def test_local_media_search_hides_a_tombstoned_fact_until_heal(world):
    db, channel = world
    before = {f for f, _ in channel.search(unit(0), "default", 10)}
    assert before == {"pic-a-fact", "pic-b-fact"}
    _tombstone(db, "pic-a-fact")
    during = {f for f, _ in channel.search(unit(0), "default", 10)}
    assert during == {"pic-b-fact"}
    db.execute("DELETE FROM projection_tombstones WHERE fact_id='pic-a-fact'")
    assert {f for f, _ in channel.search(unit(0), "default", 10)} == before


def test_shared_media_search_hides_a_tombstoned_fact(world):
    db, channel = world
    _tombstone(db, "pic-a-fact")
    found = {f for f, _ in channel.search(unit(0), "default", 10, include_global=True)}
    assert "pic-a-fact" not in found and "pic-b-fact" in found
