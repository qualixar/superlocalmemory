# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``JevSufficiencyJudge.ask_ladder``: the judge ladder's only hosted egress.

Over ``httpx.MockTransport`` only. Pins: one request at most, never retried;
nothing sent without literal consent or outside background work; a request
that went out and failed raises ``TierFailure`` with its kind (so the ladder
receipts it, I-09 / R-35), while a request that was never sent returns None.
"""

from __future__ import annotations

import json
import threading

import httpx
import pytest

from superlocalmemory.core import recall_gate
from superlocalmemory.core.judge_keys import JudgeKeyStore
from superlocalmemory.core.judge_ladder import Item, TierFailure
from superlocalmemory.core.judge_questions import SLOT_RELATION
from superlocalmemory.retrieval.jev_judge import JEV_ENDPOINTS, JevSufficiencyJudge

FAKE_KEY = "sk-test-" + "a1B2c3D4" * 4
MODEL = JEV_ENDPOINTS["typesafe"][1]
ITEMS = [Item("slot_dispute", "d1", "p", {"older": "port is 8080", "newer": "port is 9090"}),
         Item("slot_dispute", "d2", "p", {"older": "owner is Alex", "newer": "owner is Sam"})]


@pytest.fixture
def key_store(tmp_path) -> JudgeKeyStore:
    store = JudgeKeyStore(slm_home=tmp_path)
    store.set_key("typesafe", FAKE_KEY)
    return store


class _Recorder:
    def __init__(self, handler):
        self.requests: list[httpx.Request] = []
        self._handler = handler
        self.transport = httpx.MockTransport(self._wrapped)

    def _wrapped(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._handler(request)


def _choices(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    probs = {lab: (0.9 if lab == "replaces" else 0.025) for lab in SLOT_RELATION.criteria}
    answers = {k: {"type": "choice", "choice": "replaces", "confidence": 0.9,
                   "probabilities": probs} for k in body["questions"]}
    return httpx.Response(200, json={"model": MODEL, "answers": answers,
                                     "usage": {"input_tokens": 1, "output_tokens": 1}})


def _bg(fn):
    box: dict = {}

    def run():
        with recall_gate.background_work():
            try:
                box["value"] = fn()
            except BaseException as exc:  # noqa: BLE001 - handed back to the test thread
                box["error"] = exc

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(10)
    if "error" in box:
        raise box["error"]
    return box["value"]


def _judge(key_store, rec) -> JevSufficiencyJudge:
    return JevSufficiencyJudge(provider="typesafe", key_store=key_store, timeout_s=2.0,
                               transport=rec.transport)


def test_one_request_one_answer_per_item(key_store) -> None:
    rec = _Recorder(_choices)
    out = _bg(lambda: _judge(key_store, rec).ask_ladder(SLOT_RELATION, ITEMS, consent=True))
    assert [a.verdict for a in out] == ["replaces", "replaces"] and len(rec.requests) == 1


@pytest.mark.parametrize("handler,kind", [
    (lambda r: httpx.Response(503, json={}), "http"),
    (lambda r: (_ for _ in ()).throw(httpx.ReadTimeout("slow", request=r)), "timeout"),
    (lambda r: (_ for _ in ()).throw(httpx.ConnectError("down", request=r)), "transport"),
    (lambda r: httpx.Response(200, json={"model": MODEL, "answers": {}}), "malformed"),
    (lambda r: httpx.Response(200, content=b"not json"), "malformed"),
])
def test_a_request_that_went_out_and_failed_raises_its_kind(key_store, handler, kind) -> None:
    rec = _Recorder(handler)
    with pytest.raises(TierFailure) as failed:
        _bg(lambda: _judge(key_store, rec).ask_ladder(SLOT_RELATION, ITEMS, consent=True))
    assert failed.value.kind == kind
    assert len(rec.requests) == 1  # never retried


@pytest.mark.parametrize("consent", [False, None, 1, "yes"])
def test_nothing_is_sent_without_literal_consent(key_store, consent) -> None:
    rec = _Recorder(_choices)
    assert _bg(lambda: _judge(key_store, rec).ask_ladder(
        SLOT_RELATION, ITEMS, consent=consent)) is None
    assert rec.requests == []


def test_nothing_is_sent_from_a_foreground_thread(key_store) -> None:
    rec = _Recorder(_choices)
    assert _judge(key_store, rec).ask_ladder(SLOT_RELATION, ITEMS, consent=True) is None
    assert rec.requests == []
