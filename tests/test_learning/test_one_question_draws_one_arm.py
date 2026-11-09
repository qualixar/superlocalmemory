# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The same question over the same memories comes back in the same order (R1).

The bandit's arm sets the channel weights, and those decide the order. A fresh
system-random draw per recall meant one question asked 40 times came back in 5
orders, the top answer changing 14 times — and the answer-check memo, keyed by
the order it read, never hit. The draw is now keyed by the question and the
learned posterior: repeatable, still exploring across questions, still moving
when a reward moves the posterior.
"""

from __future__ import annotations

import collections
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from superlocalmemory.core import answer_check_memo as memo
from superlocalmemory.core import judge_selection, recall_pipeline
from superlocalmemory.core import security_primitives as sp
from superlocalmemory.learning import pcos
from superlocalmemory.learning.arm_catalog import ARM_CATALOG
from superlocalmemory.learning.bandit import ContextualBandit
from superlocalmemory.learning.bandit_cache import _BanditCache
from superlocalmemory.retrieval import answer_check_status as acs
from superlocalmemory.retrieval.sufficiency import SufficiencyVerdict
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.migrations import M005_bandit_tables as m5
from superlocalmemory.storage.models import AtomicFact, RecallResponse, RetrievalResult
from tests._recall_sources import recall_sources

CTX = {"query_type": "single_hop", "entity_count": 0, "time_bucket": "morning"}
QUESTION = "which database does the billing service use?"


@pytest.fixture(autouse=True)
def _install_token(tmp_path, monkeypatch):
    token = tmp_path / ".install_token"
    token.write_text("t" * 64, encoding="utf-8")
    monkeypatch.setattr(sp, "_install_token_path", lambda: token)


@pytest.fixture()
def learning_db(tmp_path: Path) -> Path:
    path = tmp_path / "learning.db"
    conn = sqlite3.connect(path)
    conn.executescript(m5.DDL)
    conn.commit()
    conn.close()
    return path


def _bandit(db: Path) -> ContextualBandit:
    return ContextualBandit(db, "default", cache=_BanditCache(max_entries=8))


def _set_posterior(db: Path, arm: str, alpha: float, beta: float) -> None:
    stratum = "single_hop|0|morning"
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT OR REPLACE INTO bandit_arms (profile_id, stratum, arm_id, alpha, beta,"
        " plays, last_played_at) VALUES ('default', ?, ?, ?, ?, 1, '2026-10-04')",
        (stratum, arm, alpha, beta))
    conn.commit()
    conn.close()


class TestTheDrawIsRepeatable:
    def test_twenty_identical_asks_draw_one_arm(self, learning_db) -> None:
        arms = {_bandit(learning_db).choose_readonly(CTX, draw_key=QUESTION).arm_id
                for _ in range(20)}
        assert len(arms) == 1

    def test_recorded_plays_draw_one_arm_too(self, learning_db) -> None:
        b = _bandit(learning_db)
        arms = {b.choose(CTX, f"qid-{i}", draw_key=QUESTION).arm_id for i in range(20)}
        assert len(arms) == 1

    def test_retyped_question_is_the_same_question(self, learning_db) -> None:
        b = _bandit(learning_db)
        a = b.choose_readonly(CTX, draw_key=QUESTION).arm_id
        assert b.choose_readonly(CTX, draw_key="  Which DATABASE does the billing\tservice use? ").arm_id == a


class TestLearningStillWorks:
    def test_different_questions_still_explore(self, learning_db) -> None:
        b = _bandit(learning_db)
        arms = {b.choose_readonly(CTX, draw_key=f"question {i}").arm_id for i in range(200)}
        # At the uniform prior every arm is equally likely; 200 independent
        # draws over 40 arms reach far more than 10 of them.
        assert len(arms) >= 10

    def test_a_reward_reaches_the_repeat_question(self, learning_db) -> None:
        b = _bandit(learning_db)
        before = b.choose_readonly(CTX, draw_key=QUESTION).arm_id
        winner = next(a for a in ARM_CATALOG if a != before)
        _set_posterior(learning_db, winner, 1000.0, 1.0)
        assert b.choose_readonly(CTX, draw_key=QUESTION).arm_id == winner

    def test_any_posterior_change_redraws(self, learning_db) -> None:
        from superlocalmemory.learning.bandit_draw import draw_rng
        r1 = draw_rng("default", "s", QUESTION, {"a": (1.0, 1.0)}).random()
        r2 = draw_rng("default", "s", QUESTION, {"a": (2.0, 1.0)}).random()
        r3 = draw_rng("default", "s", QUESTION, {"a": (1.0, 1.0)}).random()
        assert r1 == r3 and r1 != r2


# -- the ensemble path the auditor measured ----------------------------------------

def _results():
    spec = [("A", {"semantic": 0.030, "bm25": 0.010}),
            ("B", {"semantic": 0.010, "bm25": 0.029}),
            ("C", {"semantic": 0.015, "bm25": 0.015, "entity_graph": 0.005}),
            ("f_other", {"semantic": 0.001})]
    return [RetrievalResult(fact=AtomicFact(fact_id=f, content=f"memory {f}"),
                            score=0.05 - i * 0.001, channel_scores=dict(cs))
            for i, (f, cs) in enumerate(spec)]


@pytest.fixture()
def memory_db(tmp_path: Path) -> Path:
    path = tmp_path / "memory.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE fact_outcome_score (fact_id TEXT, profile_id TEXT,"
                 " score REAL, play_count INTEGER, PRIMARY KEY(fact_id, profile_id))")
    conn.execute("INSERT INTO fact_outcome_score VALUES ('f_other','default',0.55,1)")
    conn.commit()
    conn.close()
    return path


def test_ensemble_order_is_repeatable(learning_db, memory_db) -> None:
    orders = collections.Counter()
    for i in range(20):
        pcos.RECENT_TOPS.forget("default")
        resp = RecallResponse(query=QUESTION, results=_results(), query_type="factual")
        out = recall_pipeline.apply_v2_bandit_ensemble(
            resp, QUESTION, "default", f"qid{i}", learning_db_path=learning_db,
            record_plays=True, memory_db_path=memory_db)
        orders[tuple(r.fact.fact_id for r in out.results)] += 1
    assert len(orders) == 1, orders


# -- end to end: run_recall, and the answer-check memo then hits --------------------

class _Laya:
    backend = "laya"
    top_k = 3
    threshold = 0.5
    calibration_id = "laya:test"

    def __init__(self) -> None:
        self.asks = 0

    def assess(self, query, documents, *, deadline=None):
        self.asks += 1
        return acs.JudgeOutcome(SufficiencyVerdict((0.7,), 0.5, "laya:test"),
                                acs.STATUS_JUDGED)

    def assess_if_idle(self, query, documents):
        return self.assess(query, documents)

    def shutdown(self) -> None: ...


def test_twenty_identical_recalls_one_order_and_the_memo_hits(
        learning_db, memory_db, mode_a_config, monkeypatch) -> None:
    from superlocalmemory.infra import data_root

    monkeypatch.setenv("SLM_RANKING", "v2-ensemble")
    monkeypatch.setattr(judge_selection, "_live", None)
    real_state_path = data_root.state_path
    monkeypatch.setattr(data_root, "state_path",
                        lambda *p, **k: learning_db if p == ("learning.db",)
                        else real_state_path(*p, **k))
    judge = _Laya()
    engine = SimpleNamespace(
        _sufficiency_judge=judge,
        recall=lambda *a, **k: RecallResponse(query=QUESTION, results=_results(),
                                              query_type="factual"))
    orders = collections.Counter()
    details = []
    with DatabaseManager(memory_db) as db:
        for _ in range(20):
            pcos.RECENT_TOPS.forget("default")
            response = engine.recall(QUESTION)
            backed_engine = SimpleNamespace(_sufficiency_judge=judge,
                                            recall=lambda *a, **k: response)
            with recall_sources(mode_a_config.base_dir, response.results, db=db):
                out = recall_pipeline.run_recall(
                    QUESTION, "default", fast=True, config=mode_a_config,
                    retrieval_engine=backed_engine, trust_scorer=None, embedder=None,
                    db=db, llm=None, hooks=None)
            orders[tuple(r.fact.fact_id for r in out.results)] += 1
            details.append(out.answer_check_detail)
    memo.clear(judge)
    assert len(orders) == 1, orders
    # One genuine check; the other nineteen reuse it.
    assert judge.asks == 1
    assert details[1:] == [acs.DETAIL_REUSED] * 19
