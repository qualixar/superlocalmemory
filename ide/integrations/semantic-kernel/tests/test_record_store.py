"""Functional + security tests for the framework-free record store core.

Run without ``semantic-kernel`` installed, against the real SLM engine on a
temp database.
"""

import pytest


def test_collection_lifecycle(record_store):
    assert record_store.collection_exists("docs") is False
    record_store.create_collection("docs")
    assert record_store.collection_exists("docs") is True
    record_store.delete_collection("docs")
    assert record_store.collection_exists("docs") is False


def test_upsert_and_get(record_store):
    record_store.upsert("docs", "d1", {"id": "d1", "text": "hello", "tags": ["a"]})
    rec = record_store.get("docs", "d1")
    assert rec == {"id": "d1", "text": "hello", "tags": ["a"]}
    assert record_store.get("docs", "missing") is None


def test_upsert_replaces(record_store):
    record_store.upsert("docs", "d1", {"id": "d1", "text": "v1"})
    record_store.upsert("docs", "d1", {"id": "d1", "text": "v2"})
    assert record_store.get("docs", "d1")["text"] == "v2"
    assert record_store.list_keys("docs") == ["d1"]


def test_get_many_and_list(record_store):
    record_store.upsert("docs", "d1", {"id": "d1"})
    record_store.upsert("docs", "d2", {"id": "d2"})
    assert len(record_store.get_many("docs", ["d1", "d2", "nope"])) == 2
    assert sorted(record_store.list_keys("docs")) == ["d1", "d2"]
    assert len(record_store.list_records("docs")) == 2


def test_collection_isolation(record_store):
    record_store.upsert("a", "k", {"id": "k"})
    record_store.upsert("b", "k", {"id": "k"})
    assert record_store.list_keys("a") == ["k"]
    assert record_store.list_keys("b") == ["k"]
    record_store.delete_collection("a")
    assert record_store.list_keys("a") == []
    assert record_store.list_keys("b") == ["k"]


def test_delete_record(record_store):
    record_store.upsert("docs", "d1", {"id": "d1"})
    record_store.upsert("docs", "d2", {"id": "d2"})
    record_store.delete("docs", "d1")
    assert record_store.get("docs", "d1") is None
    assert record_store.get("docs", "d2") is not None


def test_sql_injection_in_collection_and_key(record_store):
    coll = "'; DROP TABLE memories; --"
    record_store.upsert(coll, "'; DELETE FROM memories; --", {"ok": True})
    assert record_store.get(coll, "'; DELETE FROM memories; --")["ok"] is True
    record_store.upsert("safe", "k", {"v": 1})
    assert record_store.get("safe", "k")["v"] == 1


def test_like_wildcards_are_literal(record_store):
    record_store.upsert("a%b", "k", {"n": 1})
    record_store.upsert("axb", "k", {"n": 2})
    assert record_store.list_keys("a%b") == ["k"]
    assert len(record_store.list_records("a%b")) == 1


def test_oversized_record_raises(record_store):
    with pytest.raises(ValueError, match="exceeds maximum size"):
        record_store.upsert("c", "k", {"blob": "x" * 1_000_001})


def test_unicode_roundtrip(record_store):
    rec = {"id": "u", "text": "Ada — 日本語 — 🎯"}
    record_store.upsert("c", "u", rec)
    assert record_store.get("c", "u") == rec
