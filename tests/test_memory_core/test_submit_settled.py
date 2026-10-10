"""submit_memory_settled: an accepted (queued) save is re-sent under the same key until it commits."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from superlocalmemory.memory_core import ContentOrigin
from superlocalmemory.memory_core.submit import SavePending, SaveRequest, submit_memory_settled

ACCEPTED = {"status": "accepted", "operation_id": None, "fact_ids": []}
DONE = {"status": "queryable", "operation_id": "op1", "fact_ids": ["f1"], "memory_id": "m1"}


class Runtime:
    """Answers with the payloads in order, repeating the last one."""

    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.keys = []

    def remember(self, admission, actor, *, deadline_ms, accept_after_ms):
        self.keys.append(admission.idempotency_key)
        payload = self.payloads.pop(0) if len(self.payloads) > 1 else self.payloads[0]
        return SimpleNamespace(payload=payload)


class Clock:
    def __init__(self):
        self.now = 0.0
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.slept.append(s)
        self.now += s


def req(key="doc:d:1:1"):
    return SaveRequest(segments=(("words", ContentOrigin.USER_TEXT),), profile_id="p1",
                       source_type="document", trusted_actor_id="a", idempotency_key=key)


CFG = SimpleNamespace(pii_redaction=False)


def test_a_committed_save_returns_at_once():
    rt, clock = Runtime(DONE), Clock()
    out = submit_memory_settled(rt, req(), config=CFG, clock=clock, sleep=clock.sleep)
    assert out.fact_ids == ("f1",) and out.memory_id == "m1" and rt.keys == ["doc:d:1:1"] and clock.slept == []


def test_an_accepted_save_is_resent_with_the_same_key_until_it_commits():
    rt, clock = Runtime(ACCEPTED, ACCEPTED, DONE), Clock()
    out = submit_memory_settled(rt, req(), config=CFG, clock=clock, sleep=clock.sleep)
    assert out.fact_ids == ("f1",)
    assert rt.keys == ["doc:d:1:1"] * 3


def test_a_save_still_queued_after_the_budget_raises_pending():
    rt, clock = Runtime(ACCEPTED), Clock()
    with pytest.raises(SavePending):
        submit_memory_settled(rt, req(), config=CFG, wait_s=5.0, clock=clock, sleep=clock.sleep)
    assert clock.now <= 5.0 + 1e-9 and len(set(rt.keys)) == 1


def test_a_request_without_a_key_gets_one_fixed_key_so_a_resend_is_never_a_second_save():
    rt, clock = Runtime(ACCEPTED, DONE), Clock()
    submit_memory_settled(rt, req(key=""), config=CFG, clock=clock, sleep=clock.sleep)
    assert len(rt.keys) == 2 and rt.keys[0] == rt.keys[1] and rt.keys[0]
