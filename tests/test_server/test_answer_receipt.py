# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W3-7b (WS-4): the ``why`` block on every recall result."""

from __future__ import annotations

from superlocalmemory.server import answer_receipt as ar


def _item(**kw) -> dict:
    base = {"fact_id": "f1", "content": "x", "channel_scores": {}, "age_label": "3 days ago"}
    return {**base, **kw}


def test_found_by_is_the_strongest_positive_channels_capped_at_three() -> None:
    got = ar.why(_item(channel_scores={"bm25": 0.2, "semantic": 0.9, "graph": 0.0,
                                       "temporal": 0.5, "entity": 0.3, "bad": "n/a"}))
    assert got["found_by"] == ["semantic", "temporal", "entity"]


def test_ties_are_broken_by_name_so_the_order_is_stable() -> None:
    assert ar.why(_item(channel_scores={"b": 0.5, "a": 0.5}))["found_by"] == ["a", "b"]


def test_answer_check_reads_the_tier_and_says_when_it_did_not_run() -> None:
    assert ar.why(_item(tier="supported"))["answer_check"] == "supported"
    assert ar.why(_item(tier="related"))["answer_check"] == "related"
    assert ar.why(_item(tier=""))["answer_check"] == "not_checked"
    assert ar.why(_item())["answer_check"] == "not_checked"


def test_a_weak_supported_memory_is_reported_weak() -> None:
    assert ar.why(_item(tier="supported", weak=True))["answer_check"] == "weak"
    assert ar.why(_item(tier="related", weak=True))["answer_check"] == "related"


def test_standing_puts_a_dispute_before_the_source_file() -> None:
    assert ar.why(_item())["standing"] == "current"
    assert ar.why(_item(source={"state": "fresh"}))["standing"] == "current"
    assert ar.why(_item(source={"state": "changed"}))["standing"] == "source changed"
    assert ar.why(_item(source={"state": "missing"}))["standing"] == "source missing"
    assert ar.why(_item(disputed=True, source={"state": "changed"}))["standing"] == "disputed"
    assert ar.why(_item(source="not-a-dict"))["standing"] == "current"


def test_said_is_the_age_label_or_empty() -> None:
    assert ar.why(_item())["said"] == "3 days ago"
    assert ar.why(_item(age_label=None))["said"] == ""


def test_with_why_adds_the_block_without_touching_the_rest() -> None:
    src = [_item(tier="supported", channel_scores={"semantic": 0.7})]
    out = ar.with_why(src)
    assert "why" not in src[0]                       # input not mutated
    assert out[0]["why"] == {"found_by": ["semantic"], "answer_check": "supported",
                             "standing": "current", "said": "3 days ago"}
    assert {k: v for k, v in out[0].items() if k != "why"} == src[0]


def test_with_why_never_fails_a_recall() -> None:
    broken = _item(channel_scores=["not", "a", "dict"])
    out = ar.with_why([broken, "not-a-dict", _item()])
    assert out[0] is broken and out[1] == "not-a-dict" and "why" in out[2]


def test_a_channel_that_scored_zero_did_not_find_it() -> None:
    assert ar.why(_item(channel_scores={"semantic": 0.4, "graph": 0.0}))["found_by"] == ["semantic"]


# ---- every recall path carries the block (run_recall is the one exit) ----

import asyncio  # noqa: E402

import pytest  # noqa: E402

from superlocalmemory.server import recall_cache, recall_core  # noqa: E402
from superlocalmemory.server.recall_core import RecallCall  # noqa: E402


def _run(envelope, monkeypatch):
    async def once(_call, _profile, _fn):
        return envelope

    monkeypatch.setattr(recall_cache, "recall_once", once)
    monkeypatch.setattr(recall_core, "_receipt", lambda *a, **k: None)
    engine = type("E", (), {"profile_id": "default"})()
    call = RecallCall(query="q", limit=5, session_id="", agent_id="", fast=True)
    return asyncio.run(recall_core.run_recall(engine, call, app_state=None))


def test_a_full_recall_answers_why_from_its_channels(monkeypatch) -> None:
    env = {"ok": True, "results": [_item(tier="supported", channel_scores={"semantic": 0.8})]}
    got = _run(env, monkeypatch)
    assert got["results"][0]["why"]["found_by"] == ["semantic"]
    assert "why" not in env["results"][0]          # a cached envelope is never mutated


def test_the_keyword_fallback_says_keyword_found_it(monkeypatch) -> None:
    env = {"ok": True, "retrieval_mode": "degraded_lexical",
           "results": [{"fact_id": "f", "content": "x", "score": None}]}
    why = _run(env, monkeypatch)["results"][0]["why"]
    assert why == {"found_by": ["keyword"], "answer_check": "not_checked",
                   "standing": "current", "said": ""}


def test_a_partial_answer_without_scores_names_no_channel(monkeypatch) -> None:
    env = {"ok": True, "retrieval_mode": "partial_context",
           "results": [{"fact_id": "f", "content": "x", "score": 0.4}]}
    assert _run(env, monkeypatch)["results"][0]["why"]["found_by"] == []


@pytest.mark.parametrize("env", [{"ok": False, "code": "RECALL_BUSY"}, {"ok": True}])
def test_a_reply_without_results_is_returned_as_is(env, monkeypatch) -> None:
    assert _run(env, monkeypatch) == env


from tests.test_modes.test_mode_a_never_fails import mode_a_engine  # noqa: E402,F401


def test_the_hook_adapter_recall_carries_why(mode_a_engine) -> None:  # noqa: F811
    from superlocalmemory.server.unified_daemon import EngineRecallAdapter

    mode_a_engine.store("The invoice run happens on the last working day of the month.")
    out = EngineRecallAdapter(mode_a_engine).recall("invoice run", 5, "conv-why")
    assert out["results"] and all(set(r["why"]) == {"found_by", "answer_check", "standing",
                                                     "said"} for r in out["results"])
