# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Ladder questions on the wire: Laya ``ask`` requests, Jev bodies, strict replies."""

from __future__ import annotations

import json

import pytest

from superlocalmemory.core.judge_ladder import Item
from superlocalmemory.core.judge_questions import SLOT_RELATION
from superlocalmemory.retrieval import judge_ask

_PAIR = Item("slot_dispute", "d1", "p", {"older": "a" * 5000, "newer": "b"})


def test_laya_requests_are_chunked_cut_and_typed() -> None:
    reqs = judge_ask.laya_requests(SLOT_RELATION, [_PAIR] * 3)
    assert [len(r["items"]) for r in reqs] == [2, 1]
    item = reqs[0]["items"][0]
    assert set(item["state"]) == {"older", "newer"} and len(item["state"]["older"]) == 1800
    assert item["questions"]["q"]["type"] == "choice"
    assert set(item["questions"]["q"]["criteria"]) == set(SLOT_RELATION.criteria)


def test_read_choice_is_strict() -> None:
    ok = {"type": "choice", "choice": "same", "probabilities": {"same": 0.8, "replaces": 0.1},
          "confidence": 0.8}
    a = judge_ask.read_choice(ok, SLOT_RELATION, "m", "c")
    assert a is not None and (a.verdict, a.confidence) == ("same", 0.8)
    for bad in ({**ok, "choice": "nope"}, {**ok, "probabilities": {"nope": 0.1}},
                {**ok, "confidence": 2.0}, {**ok, "probabilities": {"same": float("nan")}},
                {**ok, "type": "noul"}, "text", None):
        assert judge_ask.read_choice(bad, SLOT_RELATION, "m", "c") is None


def test_jev_body_names_each_item_and_redacts() -> None:
    key = "sk-" + "A" * 40
    body = judge_ask.jev_body("jev-1", SLOT_RELATION,
                              [Item("x", "1", "p", {"older": f"token {key} here", "newer": "n"})])
    assert body is not None and key not in json.dumps(body)
    assert "Judge ONLY item i0" in body["questions"]["i0"]["instructions"]


def test_jev_body_refuses_a_text_that_is_only_a_credential() -> None:
    key = "sk-" + "A" * 40
    assert judge_ask.jev_body("jev-1", SLOT_RELATION,
                              [Item("x", "1", "p", {"older": key, "newer": "n"})]) is None


def test_read_jev_needs_every_answer_and_the_right_model() -> None:
    body = judge_ask.jev_body("jev-1", SLOT_RELATION, [_PAIR, _PAIR])
    answer = {"type": "choice", "choice": "replaces", "probabilities": {"replaces": 0.9},
              "confidence": 0.9}
    good = json.dumps({"model": "jev-1", "answers": {"i0": answer, "i1": answer}}).encode()
    out = judge_ask.read_jev(good, body, SLOT_RELATION, "jev-1", "c", lambda m: m == "jev-1")
    assert out is not None and [a.verdict for a in out] == ["replaces", "replaces"]
    short = json.dumps({"model": "jev-1", "answers": {"i0": answer}}).encode()
    assert judge_ask.read_jev(short, body, SLOT_RELATION, "jev-1", "c", lambda m: True) is None
    assert judge_ask.read_jev(good, body, SLOT_RELATION, "jev-1", "c", lambda m: False) is None
    assert judge_ask.read_jev(b"not json", body, SLOT_RELATION, "jev-1", "c",
                              lambda m: True) is None


def test_laya_is_never_asked_outside_background_work() -> None:
    class Judge:
        ready = True

    assert judge_ask.ask_laya(Judge(), SLOT_RELATION, [_PAIR], model="m",
                              calibration_id="c") is None


def test_a_laya_worker_that_gives_no_reply_is_a_failed_ask(monkeypatch) -> None:
    """R-35: asked and failed raises (receipted by the ladder); not ready declines."""
    from superlocalmemory.core import recall_gate
    from superlocalmemory.core.judge_ladder import Item, TierFailure
    from superlocalmemory.retrieval import judge_ask, laya_background

    class Laya:
        ready = True

    item = Item("x", "1", "p", {"older": "a", "newer": "b"})
    with recall_gate.background_work():
        monkeypatch.setattr(laya_background, "exchange", lambda *a, **k: None)
        with pytest.raises(TierFailure) as failed:
            judge_ask.ask_laya_statement(Laya(), "s", ("older", "newer"), [item])
        assert failed.value.kind == "worker"
        monkeypatch.setattr(laya_background, "exchange", lambda *a, **k: {"ok": True, "answers": []})
        with pytest.raises(TierFailure) as failed:
            judge_ask.ask_laya_statement(Laya(), "s", ("older", "newer"), [item])
        assert failed.value.kind == "malformed"
        Laya.ready = False
        assert judge_ask.ask_laya_statement(Laya(), "s", ("older", "newer"), [item]) is None
