# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Jev orders a session start's memories, and only orders them.

A person who chose Jev to judge relevance got it on every recall but not on
the memories an agent starts with: a session start skips the answer check,
and Jev's ordering rides in that request. These tests use a stand-in judge;
nothing here reaches the network.
"""

from __future__ import annotations

import json
import time
from argparse import Namespace
from types import SimpleNamespace

import pytest

from superlocalmemory.core import judge_selection, session_jev
from superlocalmemory.retrieval.jev_rerank import RerankVerdict

ITEMS = ["alpha note", "beta note", "gamma note", "delta note"]


class _Judge:
    def __init__(self, order=(2, 0, 1), verdict="no-answer", delay=0.0, boom=False):
        self.order, self.verdict, self.delay, self.boom = order, verdict, delay, boom
        self.calls, self.closed, self.deadline = [], False, None

    def rerank_and_judge(self, query, docs, *, deadline=None):
        self.calls.append((query, [d.content for d in docs]))
        self.deadline = deadline
        if self.boom:
            raise RuntimeError("provider down")
        time.sleep(self.delay)
        return RerankVerdict(order=self.order, verdict=self.verdict, status="ok")

    def shutdown(self):
        self.closed = True


def _order(judge, items=ITEMS, query="project context api"):
    return session_jev.jev_order(query, items, str, SimpleNamespace(),
                                 judge_factory=lambda cfg, t: judge)


def test_jev_order_is_applied_and_the_rest_keep_their_place():
    judge = _Judge(order=(2, 0, 1))
    assert _order(judge) == ["gamma note", "alpha note", "beta note", "delta note"]
    assert judge.calls == [("project context api", ITEMS)]
    assert judge.closed


def test_the_verdict_is_ignored_nothing_is_hidden():
    assert len(_order(_Judge(order=(1, 0), verdict="nothing answers"))) == len(ITEMS)


@pytest.mark.parametrize("judge", [None, _Judge(boom=True), _Judge(order=None),
                                   _Judge(order=(0, 0, 1)), _Judge(order=(0, 9))])
def test_anything_but_a_clean_order_keeps_the_original(judge):
    assert _order(judge) == ITEMS


def test_the_request_carries_the_session_budget_as_its_deadline():
    judge = _Judge()
    before = time.monotonic()
    _order(judge)
    assert before + session_jev.BUDGET_S - 0.1 <= judge.deadline <= time.monotonic() + session_jev.BUDGET_S


def test_no_query_or_one_memory_sends_nothing():
    judge = _Judge()
    _order(judge, items=["only"])
    _order(judge, query="  ")
    assert judge.calls == []


@pytest.mark.parametrize("cfg", [
    {"sufficiency_judge": "laya", "sufficiency_jev_rerank": True, "sufficiency_jev_rerank_consent": True},
    {"sufficiency_judge": "jev", "sufficiency_jev_rerank": True, "sufficiency_jev_rerank_consent": False},
    {"sufficiency_judge": "jev", "sufficiency_jev_rerank": False, "sufficiency_jev_rerank_consent": True},
    {"sufficiency_judge": "off"},
])
def test_without_jev_reordering_and_its_consent_no_judge_is_built(cfg, monkeypatch):
    built = []
    monkeypatch.setattr(judge_selection, "_construct_jev", lambda *a: built.append(a))
    assert judge_selection.session_rank_judge(SimpleNamespace(**cfg), 1.5) is None
    assert built == []


def test_the_hook_path_orders_its_memories_with_the_project_query(tmp_path, monkeypatch, capsys):
    from superlocalmemory.cli.commands import cmd_session_context
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path / "billing-api"))
    (tmp_path / "config.json").write_text(json.dumps({"mode": "a"}), encoding="utf-8")
    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(schema)
    for i, text in enumerate(("Decision one about caching.", "Decision two about billing.")):
        mid = db.store_memory(MemoryRecord(profile_id="default", content=text))
        db.store_fact(AtomicFact(profile_id="default", memory_id=mid, content=text,
                                 fact_type=FactType.SEMANTIC, importance=0.5 - i / 10))
    judge = _Judge(order=(1, 0))
    monkeypatch.setattr(session_jev, "_default_judge", lambda cfg, t: judge)
    cmd_session_context(Namespace(json=False, full=False, query="", max_age_days=0))
    out = capsys.readouterr().out
    assert judge.calls and judge.calls[0][0] == "project context billing-api"
    assert out.index("about billing") < out.index("about caching")

    # What the hooks actually run: `slm session-context <folder name>`.
    judge.calls.clear()
    cmd_session_context(Namespace(json=False, full=False, query="billing-api", max_age_days=0))
    assert judge.calls[0][0] == "project context billing-api"
    judge.calls.clear()
    cmd_session_context(Namespace(json=False, full=False, query="what did we decide on billing",
                                  max_age_days=0))
    assert judge.calls[0][0] == "what did we decide on billing"
