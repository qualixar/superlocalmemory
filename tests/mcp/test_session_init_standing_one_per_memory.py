# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A standing rule is one entry, the right rules win the slots, recall keeps its rows.

Found on the author's store (9 Nov 2026): since 4.2.5 every sentence of a memory
is also a fact of its own, and each inherits the memory's kind. One saved rule
of seven sentences became seven "rules", fragments like "cost him credits."
among them; they took 7 of the 10 rule slots, and the session's ``memories``
list was all rules, with recall's relevant memories pushed out.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from superlocalmemory.core import standing_rules
from superlocalmemory.mcp.tools_active import register_active_tools
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord
from tests.mcp.test_session_init_prospective_surface import (
    _FakeRecallResponse,
    _patches,
    _ToolServer,
)

RULE = ("RULE: never put a real name in a video prompt. Prompts naming a person were "
        "rejected. They cost him credits. Test one prompt before a batch.")
SENTENCES = ["RULE: never put a real name in a video prompt.",
             "Prompts naming a person were rejected.", "They cost him credits.",
             "Test one prompt before a batch."]


@pytest.fixture()
def db(tmp_path) -> DatabaseManager:
    mgr = DatabaseManager(tmp_path / "memory.db")
    mgr.initialize(schema)
    return mgr


def _memory(db, content: str, facts: list[str], *, kind="rule",
            created="2026-10-01T10:00:00+00:00", project: str | None = None) -> str:
    meta = {"project": project} if project else {}
    memory_id = db.store_memory(MemoryRecord(profile_id="default", content=content,
                                             metadata=meta))
    for text in facts:
        fact = AtomicFact(profile_id="default", memory_id=memory_id, content=text,
                          fact_type=FactType.SEMANTIC, created_at=created)
        fact.memory_kind, fact.memory_kind_source = kind, "caller"
        db.store_fact(fact)
    return memory_id


def _session(db, results=()) -> dict:
    engine = SimpleNamespace(profile_id="default", mode="B", db=db, _db=None, config=None)
    server = _ToolServer()
    register_active_tools(server, lambda: engine)
    p = _patches(engine)
    response = _FakeRecallResponse()
    response.results = list(results)
    with p[0], patch("superlocalmemory.mcp._pool_adapter.pool_recall",
                     return_value=response), p[2]:
        out = asyncio.run(server.tools["session_init"](project_path="/test/proj",
                                                       max_results=5))
    assert out["success"] is True, out
    return out


def test_a_rule_saved_with_its_sentences_is_one_entry(db) -> None:
    _memory(db, RULE, [RULE, *SENTENCES])
    facts = standing_rules.standing_facts(db, "default")
    assert [f.content for f in facts] == [RULE]


def test_without_a_whole_text_fact_the_longest_sentence_stands_for_it(db) -> None:
    _memory(db, RULE, SENTENCES)
    assert [f.content for f in standing_rules.standing_facts(db, "default")] == [SENTENCES[0]]


def test_the_cap_counts_rules_not_sentences(db) -> None:
    for i in range(12):
        text = f"Rule {i}: always do thing {i}. It matters. Check it twice."
        _memory(db, text, [text, f"Rule {i}: always do thing {i}.", "It matters."],
                created=f"2026-10-{i + 1:02d}T10:00:00+00:00")
    facts = standing_rules.standing_facts(db, "default")
    assert len(facts) == standing_rules.MAX_RULES
    assert len({f.memory_id for f in facts}) == standing_rules.MAX_RULES


def test_over_the_cap_the_project_then_relevance_wins_not_age(db) -> None:
    relevant = _memory(db, "Never publish SLM without the live-store check.",
                       ["Never publish SLM without the live-store check."],
                       created="2026-09-01T10:00:00+00:00")
    mine = _memory(db, "In this repo, run gitnexus impact first.",
                   ["In this repo, run gitnexus impact first."],
                   created="2026-09-02T10:00:00+00:00", project="/test/proj")
    for i in range(12):  # newer, unrelated
        _memory(db, f"Video rule {i}.", [f"Video rule {i}."],
                created=f"2026-10-{i + 1:02d}T10:00:00+00:00")
    facts = standing_rules.standing_facts(db, "default", project_path="/test/proj",
                                          relevance={relevant: 0.8})
    assert facts[0].memory_id == mine
    assert facts[1].memory_id == relevant
    assert len(facts) == standing_rules.MAX_RULES


def test_recall_rows_keep_their_place_in_the_session(db) -> None:
    for i in range(12):
        _memory(db, f"Rule {i}.", [f"Rule {i}."])
    found = AtomicFact(profile_id="default", memory_id="m-recall", fact_id="f-recall",
                       content="SLM 4.2.13 shipped on 29 Oct.", fact_type=FactType.EPISODIC,
                       created_at="2026-11-01T10:00:00+00:00")
    hit = SimpleNamespace(fact=found, score=0.71, tier="supported")
    out = _session(db, results=[hit])
    sources = [m["source_type"] for m in out["memories"]]
    assert "recall" in sources
    assert sum(s.startswith("standing-") for s in sources) == standing_rules.MAX_RULES


def test_newer_rules_of_other_projects_never_crowd_out_this_projects(db, monkeypatch) -> None:
    # Codex review, 4.2.14: a read window of the newest N confirmed facts was
    # filled by another project's rules before this project's were reached.
    mine = _memory(db, "In this repo, run gitnexus impact first.",
                   ["In this repo, run gitnexus impact first."],
                   created="2026-01-01T10:00:00+00:00", project="/test/proj")
    for i in range(40):
        _memory(db, f"Other project rule {i}.", [f"Other project rule {i}."],
                created=f"2026-10-{(i % 28) + 1:02d}T10:00:00+00:00", project="/other/elsewhere")
    facts = standing_rules.standing_facts(db, "default", project_path="/test/proj")
    assert [f.memory_id for f in facts] == [mine]
