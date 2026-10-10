"""``hide_sources`` hides folder memories, failing closed."""

from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import pytest

from superlocalmemory.retrieval import visibility as vis
from superlocalmemory.retrieval.visibility import VisibilityContext


class Db:
    def __init__(self, memories, facts, broken=False):
        self.c = sqlite3.connect(":memory:")
        self.c.row_factory = sqlite3.Row
        self.c.execute("CREATE TABLE memories(memory_id TEXT, metadata_json TEXT)")
        self.c.execute("CREATE TABLE atomic_facts(fact_id TEXT, memory_id TEXT, profile_id TEXT)")
        for mid, meta in memories.items():
            self.c.execute("INSERT INTO memories VALUES (?,?)", (mid, meta))
        for fid, mid in facts.items():
            self.c.execute("INSERT INTO atomic_facts VALUES (?,?,?)", (fid, mid, "p"))
        self.broken = broken

    def execute(self, sql, args=()):
        if self.broken:
            raise sqlite3.OperationalError("boom")
        return self.c.execute(sql, args).fetchall()


def meta(**source):
    return json.dumps({"_slm_source": source})


MEMORIES = {
    "m_folder": meta(type="folder", source_id="s", relpath="a.md", version="v"),
    "m_doc": meta(type="document", document_id="d", origin="folder"),
    "m_img": meta(type="media", media_id="i", origin="folder"),
    "m_plain": json.dumps({}),
    "m_pic": meta(type="media", media_id="x", origin="tool"),
}
FACTS = {"f_folder": "m_folder", "f_doc": "m_doc", "f_img": "m_img", "f_plain": "m_plain", "f_pic": "m_pic"}


def fused(*ids):
    return [SimpleNamespace(fact_id=i) for i in ids]


def test_default_context_is_empty():
    assert VisibilityContext().hide_sources is False
    assert vis.is_empty()


def test_hide_sources_drops_folder_memories_only():
    db = Db(MEMORIES, FACTS)
    with vis.use(VisibilityContext(hide_sources=True)):
        assert not vis.is_empty()
        kept = vis.drop_hidden_results(fused(*FACTS), db, "p")
    assert [f.fact_id for f in kept] == ["f_plain", "f_pic"]


def test_hide_media_does_not_hide_folder_text():
    db = Db(MEMORIES, FACTS)
    with vis.use(VisibilityContext(hide_media=True)):
        kept = vis.drop_hidden_results(fused(*FACTS), db, "p")
    assert "f_folder" in [f.fact_id for f in kept]


def test_both_flags_combine():
    db = Db(MEMORIES, FACTS)
    with vis.use(VisibilityContext(hide_media=True, hide_sources=True)):
        kept = vis.drop_hidden_results(fused(*FACTS), db, "p")
    assert [f.fact_id for f in kept] == ["f_plain"]


def test_loaded_facts_are_filtered():
    db = Db(MEMORIES, FACTS)
    facts = {fid: SimpleNamespace(memory_id=mid) for fid, mid in FACTS.items()}
    with vis.use(VisibilityContext(hide_sources=True)):
        kept = vis.drop_hidden_facts(facts, db)
    assert sorted(kept) == ["f_pic", "f_plain"]


def test_lookup_failure_hides_everything():
    db = Db(MEMORIES, FACTS, broken=True)
    with vis.use(VisibilityContext(hide_sources=True)):
        assert vis.drop_hidden_results(fused(*FACTS), db, "p") == []
        assert vis.drop_hidden_facts({"f_plain": SimpleNamespace(memory_id="m_plain")}, db) == {}


def test_unreadable_metadata_counts_as_hidden():
    db = Db({"m_bad": "{not json"}, {"f_bad": "m_bad"})
    with vis.use(VisibilityContext(hide_sources=True)):
        assert vis.drop_hidden_results(fused("f_bad"), db, "p") == []
