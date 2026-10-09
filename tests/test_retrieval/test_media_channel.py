"""The picture channel: when it runs, how it scores, and which memories it finds."""

from __future__ import annotations

import time

import pytest

from superlocalmemory.media import open_media_store
from superlocalmemory.retrieval import channel_status as chstat
from superlocalmemory.retrieval.media_channel import MediaChannel

from ._media_support import FakeClient, MemoryDb, blend, make_store, mid, unit


@pytest.fixture()
def world(tmp_path):
    db = MemoryDb()
    db.add("m1", "f1")
    db.add("m2", "f2")
    db.add("m3", "f3a")
    db.conn.execute("INSERT INTO atomic_facts VALUES ('f3b', 'm3', 'default')")
    store = make_store(tmp_path, [
        (mid(1), "image", "m1", unit(0)),
        (mid(2), "image", "m2", blend(0, 1, 0.5)),
        (mid(3), "image", "m3", unit(2)),
    ])
    yield db, store
    store.close()


def channel(store, client, db, **kw):
    return MediaChannel(lambda: store, lambda: client, db, **kw)


def test_score_is_one_minus_distance_clamped(world):
    db, store = world
    found = dict(channel(store, FakeClient(unit(0)), db).search(unit(0), "default", 10))
    assert found["f1"] == pytest.approx(1.0, abs=1e-4)
    assert 0.0 <= found["f2"] < 1.0
    assert found["f3a"] == 0.0  # orthogonal: 1 - 1 = 0, never negative


def test_two_images_map_to_their_own_memories_and_facts(world):
    db, store = world
    found = dict(channel(store, FakeClient(unit(0)), db).search(unit(0), "default", 10))
    assert {"f1", "f2"} <= set(found)


def test_one_memory_with_two_facts_gives_both_the_score_and_one_lookup(world):
    db, store = world
    found = dict(channel(store, FakeClient(unit(2)), db).search(unit(2), "default", 10))
    assert found["f3a"] == found["f3b"] > 0.9
    assert len(db.queries) == 1


def test_a_fact_found_by_two_images_keeps_the_better_score(tmp_path):
    db = MemoryDb()
    db.add("m1", "f1")
    store = make_store(tmp_path, [(mid(1), "image", "m1", unit(0)),
                                  (mid(2), "image", "m1", blend(0, 1, 0.5))])
    try:
        out = channel(store, FakeClient(unit(0)), db).search(unit(0), "default", 10)
    finally:
        store.close()
    assert [f for f, _ in out] == ["f1"]
    assert out[0][1] == pytest.approx(1.0, abs=1e-4)


def test_page_rows_map_to_the_page_memories(tmp_path):
    db = MemoryDb()
    db.add("pm1", "pf1")
    db.add("pm2", "pf2")
    store = make_store(tmp_path, [(mid(7), "page", None, unit(1))])
    try:
        with store._write() as conn:  # a page row as the document step writes it
            conn.execute("UPDATE media_items SET document_id='d1', page_no=1 WHERE media_id=?", (mid(7),))
            conn.execute("INSERT INTO doc_pages(document_id, page_no, media_id, memory_ids_json, text_origin)"
                         " VALUES ('d1', 1, ?, '[\"pm1\", \"pm2\"]', 'ocr')", (mid(7),))
        out = dict(channel(store, FakeClient(unit(1)), db).search(unit(1), "default", 10))
    finally:
        store.close()
    assert set(out) == {"pf1", "pf2"}


def test_a_tombstoned_image_is_not_found(world):
    db, store = world
    store.set_state(mid(1), "tombstoned")
    found = dict(channel(store, FakeClient(unit(0)), db).search(unit(0), "default", 10))
    assert "f1" not in found


def test_inactive_without_a_store_a_client_or_vectors(tmp_path):
    db = MemoryDb()
    assert not MediaChannel(lambda: None, lambda: FakeClient(unit(0)), db).is_active("default")
    empty = open_media_store(create=True, data_root=tmp_path)
    try:
        assert not MediaChannel(lambda: empty, lambda: FakeClient(unit(0)), db).is_active("default")
        assert not MediaChannel(lambda: empty, lambda: None, db).is_active("default")
    finally:
        empty.close()


def test_active_state_is_cached_for_thirty_seconds(world):
    db, store = world
    now = [100.0]
    calls = []
    ch = MediaChannel(lambda: calls.append(1) or store, lambda: FakeClient(unit(0)), db,
                      clock=lambda: now[0])
    assert ch.is_active("default") and ch.is_active("default")
    assert len(calls) == 1
    now[0] += 31
    assert ch.is_active("default")
    assert len(calls) == 2


def test_prepare_warming_only_when_otherwise_active(world, tmp_path):
    db, store = world
    cold = FakeClient(None)
    assert channel(store, cold, db).prepare("a cat", "default") == (None, chstat.WARMING)
    assert cold.calls and cold.calls[0][1] <= 0.3  # the embed is bounded
    assert MediaChannel(lambda: None, lambda: cold, db).prepare("a cat", "default") == (None, None)


def test_prepare_returns_the_vector_and_remembers_it(world):
    db, store = world
    client = FakeClient(unit(0))
    ch = channel(store, client, db)
    assert ch.prepare("a cat", "default") == (unit(0), None)
    ch.prepare("a cat", "default")
    assert len(client.calls) == 1


def test_a_cold_worker_costs_a_recall_almost_nothing(world):
    db, store = world
    client = FakeClient(None)
    started = time.monotonic()
    channel(store, client, db).prepare("a cat", "default")
    assert time.monotonic() - started < 0.35
    assert client.warmups == 1
