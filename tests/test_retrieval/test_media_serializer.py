"""Results that are pictures or pages say so, found with one lookup (or none)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from superlocalmemory.retrieval.media_channel import memory_sources
from superlocalmemory.server.recall_serializer import serialize_recall_response

from ._media_support import MemoryDb


def _result(fid, mem):
    fact = SimpleNamespace(fact_id=fid, memory_id=mem, content="c", created_at="", fact_type="semantic",
                           lifecycle="active", access_count=0, memory_kind="", kind_confidence=0.0)
    return SimpleNamespace(fact=fact, score=0.5, channel_scores={}, confidence=0.5)


def _resp(*pairs):
    return SimpleNamespace(results=[_result(f, m) for f, m in pairs], no_confident_match=False)


SOURCES = {
    "m1": {"type": "media", "media_id": "a" * 32, "origin": "tool"},
    "m2": {"type": "document", "media_id": "b" * 32, "document_id": "d1", "page": 3,
           "citation": "Report, p. 3"},
}


def test_media_and_document_results_carry_a_media_object():
    out, _ = serialize_recall_response(_resp(("f1", "m1"), ("f2", "m2"), ("f3", "m3")), source_map=SOURCES)
    assert out[0]["media"] == {"media_id": "a" * 32, "kind": "image",
                               "thumbnail_uri": f"slm://media/{'a' * 32}/thumb",
                               "page": None, "document_id": None, "citation": ""}
    assert out[1]["media"]["kind"] == "page"
    assert (out[1]["media"]["page"], out[1]["media"]["document_id"]) == (3, "d1")
    assert out[1]["media"]["citation"] == "Report, p. 3"
    assert "media" not in out[2]


def test_without_a_source_map_the_output_is_what_it_was():
    plain, _ = serialize_recall_response(_resp(("f1", "m1")))
    assert "media" not in plain[0]
    mapped, _ = serialize_recall_response(_resp(("f1", "m1")), source_map={"m1": {"type": "note"}})
    assert mapped == plain


@pytest.fixture()
def db_on(tmp_path, monkeypatch):
    db = MemoryDb()
    db.db_path = tmp_path / "memory.db"
    for n, src in enumerate([SOURCES["m1"], SOURCES["m2"], None]):
        db.add(f"m{n + 1}", f"f{n + 1}", src)
    return db


def _turn_on(tmp_path, monkeypatch, enabled=True):
    (tmp_path / "media.db").write_bytes(b"")
    monkeypatch.setattr("superlocalmemory.runtimes.features.media_enabled", lambda root=None: enabled)


def test_one_batched_lookup_when_media_is_on(db_on, tmp_path, monkeypatch):
    _turn_on(tmp_path, monkeypatch)
    found = memory_sources(db_on, ["m1", "m2", "m3", "m1"])
    assert set(found) == {"m1", "m2"} and found["m2"]["document_id"] == "d1"
    assert len(db_on.queries) == 1


def test_no_lookup_at_all_when_media_is_off(db_on, tmp_path, monkeypatch):
    assert memory_sources(db_on, ["m1", "m2"]) == {}  # no media.db
    _turn_on(tmp_path, monkeypatch, enabled=False)
    assert memory_sources(db_on, ["m1", "m2"]) == {}  # file there, feature off
    assert db_on.queries == []


def test_a_database_without_a_path_costs_nothing():
    db = MemoryDb()
    assert memory_sources(db, ["m1"]) == {} and db.queries == []
