# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The online answer check never sends another profile's memories.

Consent to the online check is given once for the install, but a recall that
includes shared or global memories can return memories another profile owns —
in a team, memories another person wrote. Those must never leave the machine
on this profile's say-so. When the memories the online check would read
include one from another profile, the check is not asked for that recall: the
response says ``skipped``, and the results are exactly what retrieval returned.
The on-device check sends nothing anywhere, so it judges as before.
"""

# ruff: noqa: F811 (pytest fixtures are imported and redeclared as parameters)

from __future__ import annotations

from superlocalmemory.core.answer_check_stage import run_answer_check
from superlocalmemory.retrieval import answer_check_status as acs
from superlocalmemory.retrieval.sufficiency import SufficiencyVerdict
from superlocalmemory.storage.models import AtomicFact, RecallResponse, RetrievalResult

_VERDICT = SufficiencyVerdict((0.9,), 0.5, "test:judge")


class _Judge:
    top_k = 3

    def __init__(self, backend: str, *, rerank_k: int = 0) -> None:
        self.backend = backend
        self.rerank_k = rerank_k
        self.rerank_enabled = rerank_k > 0
        self.asked: list[int] = []

    def assess(self, query, documents, *, deadline=None):
        self.asked.append(len(documents))
        return acs.JudgeOutcome(_VERDICT, acs.STATUS_JUDGED)

    def judge(self, query, documents):
        return self.assess(query, documents).verdict

    def rerank_and_judge(self, query, documents, *, deadline=None):
        self.asked.append(len(documents))
        return None


def _engine(judge):
    return type("Engine", (), {"_sufficiency_judge": judge})()


def _response(owners: list[str]) -> RecallResponse:
    return RecallResponse(results=[
        RetrievalResult(fact=AtomicFact(fact_id=f"f{i}", content=f"memory {i}",
                                        profile_id=owner),
                        score=0.9 - i * 0.01, confidence=1.0)
        for i, owner in enumerate(owners)
    ])


def test_another_profiles_memory_in_the_top_three_is_never_sent_online():
    judge = _Judge("jev")
    response = _response(["me", "team-hr", "me", "me"])
    before = [r.fact.fact_id for r in response.results]

    outcome = run_answer_check(_engine(judge), "q?", response, profile_id="me")

    assert judge.asked == [], "another profile's memory was sent to the provider"
    assert outcome.status == acs.STATUS_SKIPPED
    assert outcome.verdict is None
    assert [r.fact.fact_id for r in response.results] == before


def test_only_this_profiles_memories_are_judged_online_as_before():
    judge = _Judge("jev")
    outcome = run_answer_check(_engine(judge), "q?", _response(["me"] * 4),
                               profile_id="me")
    assert judge.asked == [3]
    assert outcome.status == acs.STATUS_JUDGED


def test_a_reorder_that_would_read_another_profiles_memory_is_not_sent():
    judge = _Judge("jev", rerank_k=20)
    owners = ["me"] * 12 + ["team-hr"] + ["me"] * 7   # foreign at rank 13 of 20
    outcome = run_answer_check(_engine(judge), "q?", _response(owners),
                               profile_id="me")
    assert judge.asked == []
    assert outcome.status == acs.STATUS_SKIPPED


def test_a_foreign_memory_below_what_is_sent_does_not_block_the_check():
    judge = _Judge("jev")
    outcome = run_answer_check(_engine(judge), "q?",
                               _response(["me", "me", "me", "team-hr"]), profile_id="me")
    assert judge.asked == [3]
    assert outcome.status == acs.STATUS_JUDGED


def test_the_on_device_check_sends_nothing_and_judges_as_before():
    judge = _Judge("laya")
    outcome = run_answer_check(_engine(judge), "q?",
                               _response(["me", "team-hr", "me"]), profile_id="me")
    assert judge.asked == [3]
    assert outcome.status == acs.STATUS_JUDGED


# -- end to end, through the real recall path and a real hosted judge ----------

from types import SimpleNamespace  # noqa: E402

from superlocalmemory.core import recall_pipeline  # noqa: E402
from tests.test_core.test_the_answer_check_on_the_recall_path import (  # noqa: E402,F401
    _jev,
    _no_live_judge,
    _Provider,
    key_store,
)


def _recall(judge, config, monkeypatch, owners):
    from tests._recall_sources import recall_sources

    response = _response(owners)
    engine = SimpleNamespace(_sufficiency_judge=judge, recall=lambda *a, **k: response)
    monkeypatch.setattr(recall_pipeline, "apply_ranking", lambda resp, *a, **k: resp)
    with recall_sources(config.base_dir, response.results) as db:
        return recall_pipeline.run_recall(
            "which one?", "me", fast=True, config=config, retrieval_engine=engine,
            trust_scorer=None, embedder=None, db=db, llm=None,
            hooks=None, include_shared=True)


def test_a_shared_recall_sends_nothing_of_another_profile_online(
        key_store, mode_a_config, monkeypatch):
    provider = _Provider()
    out = _recall(_jev(key_store, provider), mode_a_config, monkeypatch,
                  ["me", "team-hr", "me"])
    assert provider.bodies == [], "another profile's memory reached the provider"
    assert out.answer_check_status == acs.STATUS_SKIPPED
    assert [r.fact.fact_id for r in out.results] == ["f0", "f1", "f2"]


def test_a_recall_of_only_this_profile_is_still_checked_online(
        key_store, mode_a_config, monkeypatch):
    provider = _Provider()
    out = _recall(_jev(key_store, provider), mode_a_config, monkeypatch, ["me"] * 3)
    assert len(provider.bodies) == 1
    assert out.answer_check_status == acs.STATUS_JUDGED
