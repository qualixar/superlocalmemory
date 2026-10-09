"""What a caller may not see is decided in one module, and costs nothing when nothing is hidden."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from superlocalmemory.retrieval import visibility
from superlocalmemory.retrieval.fusion import FusionResult

from ._media_support import MemoryDb


def _fr(fid, score=0.5):
    return FusionResult(fact_id=fid, fused_score=score, channel_ranks={}, channel_scores={})


def _fact(fid, mem="m0"):
    return SimpleNamespace(fact_id=fid, memory_id=mem)


class _CountingDb:
    def __init__(self):
        self.calls = 0

    def execute(self, *a, **k):
        self.calls += 1
        return []


def test_the_default_context_hides_nothing_and_is_empty():
    ctx = visibility.current()
    assert ctx.hidden_fact_ids == frozenset() and ctx.hide_media is False
    assert visibility.is_empty() and not visibility.hides_media()


def test_use_sets_the_context_and_restores_it():
    with visibility.use(visibility.VisibilityContext(hidden_fact_ids=frozenset({"a"}))):
        assert not visibility.is_empty()
        assert visibility.current().hidden_fact_ids == {"a"}
    assert visibility.is_empty()


def test_use_restores_the_context_after_an_error():
    with pytest.raises(RuntimeError):
        with visibility.use(visibility.VisibilityContext(hide_media=True)):
            raise RuntimeError("boom")
    assert visibility.is_empty()


def test_another_thread_does_not_see_the_context():
    seen = []
    with visibility.use(visibility.VisibilityContext(hidden_fact_ids=frozenset({"a"}))):
        t = threading.Thread(target=lambda: seen.append(visibility.is_empty()))
        t.start()
        t.join()
    assert seen == [True]


def test_the_default_context_returns_the_same_objects_and_asks_nothing():
    db = _CountingDb()
    fused, facts = [_fr("a"), _fr("b")], {"a": _fact("a")}
    assert visibility.drop_hidden_results(fused, db, "default") is fused
    assert visibility.drop_hidden_facts(facts, db) is facts
    top = [_fr("a")]
    assert visibility.keep_loaded(top, facts) is top
    assert db.calls == 0


def test_hidden_ids_are_dropped_from_results_facts_and_the_final_cut():
    ctx = visibility.VisibilityContext(hidden_fact_ids=frozenset({"b"}))
    db = _CountingDb()
    with visibility.use(ctx):
        out = visibility.drop_hidden_results([_fr("a", 0.9), _fr("b", 0.8), _fr("c", 0.7)], db, "default")
        assert [(f.fact_id, f.fused_score) for f in out] == [("a", 0.9), ("c", 0.7)]
        facts = visibility.drop_hidden_facts({"a": _fact("a"), "b": _fact("b")}, db)
        assert list(facts) == ["a"]
        top = visibility.keep_loaded([_fr("a"), _fr("b"), _fr("c")], {"a": _fact("a")})
        assert [f.fact_id for f in top] == ["a"]  # b and c are not loaded
    assert db.calls == 0  # ids alone need no lookup


@pytest.fixture()
def media_db(tmp_path, monkeypatch):
    db = MemoryDb()
    db.db_path = tmp_path / "memory.db"
    db.add("m_img", "f_img", {"type": "media", "media_id": "a" * 32})
    db.add("m_page", "f_page", {"type": "document", "document_id": "d1", "page": 1})
    db.add("m_note", "f_note", {"type": "note"})
    db.add("m_plain", "f_plain")
    (tmp_path / "media.db").write_bytes(b"")
    monkeypatch.setattr("superlocalmemory.runtimes.features.media_enabled", lambda root=None: True)
    return db


def test_hide_media_drops_picture_and_page_facts_with_one_source_query(media_db):
    ctx = visibility.VisibilityContext(hide_media=True)
    facts = {f: _fact(f, m) for f, m in
             [("f_img", "m_img"), ("f_page", "m_page"), ("f_note", "m_note"), ("f_plain", "m_plain")]}
    with visibility.use(ctx):
        kept = visibility.drop_hidden_facts(facts, media_db)
    assert list(kept) == ["f_note", "f_plain"]
    assert len(media_db.queries) == 1


def test_hide_media_on_results_looks_up_each_fact_once(media_db):
    with visibility.use(visibility.VisibilityContext(hide_media=True)):
        out = visibility.drop_hidden_results(
            [_fr("f_img"), _fr("f_note"), _fr("f_page"), _fr("f_plain")], media_db, "default")
    assert [f.fact_id for f in out] == ["f_note", "f_plain"]
    assert len(media_db.queries) == 2  # fact -> memory, then memory -> source


def test_hiding_does_not_depend_on_the_feature_switch(media_db, tmp_path, monkeypatch):
    (tmp_path / "media.db").unlink()  # no media.db
    monkeypatch.setattr("superlocalmemory.runtimes.features.media_enabled", lambda root=None: False)
    with visibility.use(visibility.VisibilityContext(hide_media=True)):
        out = visibility.drop_hidden_results(
            [_fr("f_img"), _fr("f_page"), _fr("f_note")], media_db, "default")
        facts = visibility.drop_hidden_facts(
            {"f_img": _fact("f_img", "m_img"), "f_note": _fact("f_note", "m_note")}, media_db)
    assert [f.fact_id for f in out] == ["f_note"]
    assert list(facts) == ["f_note"]


def test_unparseable_metadata_counts_as_hidden_but_plain_json_is_visible(media_db):
    media_db.conn.execute("UPDATE memories SET metadata_json = '{not json' WHERE memory_id = 'm_note'")
    with visibility.use(visibility.VisibilityContext(hide_media=True)):
        out = visibility.drop_hidden_results([_fr("f_note"), _fr("f_plain")], media_db, "default")
    assert [f.fact_id for f in out] == ["f_plain"]


class _BrokenDb:
    def execute(self, *a, **k):
        raise RuntimeError("disk")


def test_a_failed_lookup_hides_everything(caplog):
    with visibility.use(visibility.VisibilityContext(hide_media=True)):
        assert visibility.drop_hidden_results([_fr("a")], _BrokenDb(), "default") == []
        assert visibility.drop_hidden_facts({"a": _fact("a")}, _BrokenDb()) == {}
    assert "RuntimeError" in caplog.text
