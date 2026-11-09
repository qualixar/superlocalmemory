# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Where the sufficiency judge meets the recall path, and when it stays away.

The recall path has 13 dependents — the daemon, session open, the MCP
pre-stage, the CLI, benchmarks and the IDE integrations — so the guard that
matters most is the boring one: with no judge, nothing changes.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from superlocalmemory.core import engine_wiring, judge_selection

# Imported at collection time: tests/conftest.py replaces this attribute for the
# whole session so no engine a test builds can start a real Laya worker.
from superlocalmemory.core.engine_wiring import (
    init_sufficiency_judge as _real_init_sufficiency_judge,
)
from superlocalmemory.core.recall_pipeline import _judge_sufficiency
from superlocalmemory.core.score_contract import finalize_score_contract
from superlocalmemory.retrieval import sufficiency
from superlocalmemory.retrieval.judge_recipe import JudgeDocument
from superlocalmemory.retrieval.sufficiency import SufficiencyVerdict
from superlocalmemory.storage.models import (
    AtomicFact,
    FactType,
    RecallResponse,
    RetrievalResult,
)


@pytest.fixture(autouse=True)
def _no_live_judge(monkeypatch):
    """The process-wide live judge is module state; no test may inherit one."""
    monkeypatch.setattr(judge_selection, "_live", None)


def _response(*contents: str) -> RecallResponse:
    return RecallResponse(results=[
        RetrievalResult(fact=AtomicFact(content=c, confidence=0.8), score=0.7, confidence=1.0)
        for c in contents
    ])


class _Judge:
    top_k = 3

    def __init__(self, verdict=None, raises=False):
        self.verdict = verdict
        self.raises = raises
        self.calls = []

    def judge(self, query, documents):
        self.calls.append((query, documents))
        if self.raises:
            raise RuntimeError("worker exploded")
        return self.verdict


def _engine(judge):
    return SimpleNamespace(_sufficiency_judge=judge)


_VERDICT = SufficiencyVerdict((0.2, 0.1), 0.6, "laya-mlx:t:sufficiency:top3")


class TestNoJudgeChangesNothing:
    def test_without_a_judge_the_contract_output_is_unchanged(self) -> None:
        judged, plain = _response("a", "b"), _response("a", "b")
        finalize_score_contract(judged, verdict=_judge_sufficiency(_engine(None), "q", judged))
        finalize_score_contract(plain)
        for field in ("calibration_status", "calibration_id", "answer_confidence",
                      "abstained", "abstention_reason", "score_contract_version"):
            assert getattr(judged, field) == getattr(plain, field), field
        assert [r.rank_position for r in judged.results] == [r.rank_position for r in plain.results]

    def test_an_engine_double_cannot_inject_a_verdict(self) -> None:
        """getattr on a MagicMock returns another MagicMock, not None."""
        assert _judge_sufficiency(MagicMock(), "q", _response("a")) is None


class TestTheJudgeIsAskedOnlyWhenItShouldBe:
    def test_a_real_recall_is_judged_on_its_top_k(self) -> None:
        judge = _Judge(verdict=_VERDICT)
        response = _response("one", "two", "three", "four", "five")
        assert _judge_sufficiency(_engine(judge), "why?", response) is _VERDICT
        assert [d.content for d in judge.calls[0][1]] == ["one", "two", "three"]
        assert judge.calls[0][0] == "why?" and len(judge.calls) == 1

    def test_each_memory_reaches_the_judge_as_the_text_the_model_reads(self) -> None:
        """F12: the recipe sends text only, so the document carries text only."""
        judge = _Judge(verdict=_VERDICT)
        fact = AtomicFact(content="Alice moved to Paris", fact_type=FactType.EPISODIC,
                          observation_date="2026-03-01", entities=["Alice", "Paris"],
                          canonical_entities=["9f3a1c2b7d4e5f60", "0c1d2e3f4a5b6c7d"])
        response = RecallResponse(results=[RetrievalResult(fact=fact, score=0.7, confidence=1.0)])
        _judge_sufficiency(_engine(judge), "where?", response)
        assert judge.calls[0][1] == [JudgeDocument("Alice moved to Paris")]

    def test_the_judge_is_read_off_the_engine_exactly_once(self) -> None:
        """A switch can replace it between two reads; one read means a recall
        works with one judge from start to finish."""
        judge = _Judge(verdict=_VERDICT)

        class _Counting:
            reads = 0

            @property
            def _sufficiency_judge(self):
                type(self).reads += 1
                return judge

        engine = _Counting()
        assert _judge_sufficiency(engine, "q", _response("a")) is _VERDICT
        assert _Counting.reads == 1

    def test_background_work_never_spends_inference_on_it(self) -> None:
        from superlocalmemory.core.recall_gate import background_work

        judge = _Judge(verdict=_VERDICT)
        with background_work():
            assert _judge_sufficiency(_engine(judge), "q", _response("a")) is None
        assert judge.calls == []

    def test_an_empty_recall_is_not_judged(self) -> None:
        judge = _Judge(verdict=_VERDICT)
        assert _judge_sufficiency(_engine(judge), "q", RecallResponse(results=[])) is None
        assert judge.calls == []


class TestAJudgeNeverBreaksARecall:
    def test_a_judge_that_raises_leaves_the_recall_unjudged(self) -> None:
        assert _judge_sufficiency(_engine(_Judge(raises=True)), "q", _response("a")) is None

    @pytest.mark.parametrize("bogus", [MagicMock(), {"answer_confidence": 0.9}, 0.9, "yes"])
    def test_anything_but_a_real_verdict_is_ignored(self, bogus) -> None:
        assert _judge_sufficiency(_engine(_Judge(verdict=bogus)), "q", _response("a")) is None


class TestWiring:
    def _cfg(self, **kw):
        base = dict(sufficiency_judge="auto", sufficiency_python="",
                    sufficiency_model="aac6fef/laya-mlx", sufficiency_hf_home="",
                    sufficiency_timeout_s=1.5)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_off_means_off(self, monkeypatch) -> None:
        monkeypatch.setattr(sufficiency, "laya_supported", lambda: True)
        assert _real_init_sufficiency_judge(self._cfg(sufficiency_judge="off")) is None

    def test_not_apple_silicon_means_off(self, monkeypatch) -> None:
        """Everything else present, so only the platform check can refuse."""
        monkeypatch.setattr(sufficiency, "laya_supported", lambda: False)
        monkeypatch.setattr(sufficiency, "LayaSufficiencyJudge", lambda **kw: object())
        monkeypatch.setattr(judge_selection, "interpreter_has_module", lambda py, m: True)
        assert _real_init_sufficiency_judge(self._cfg()) is None

    def test_an_interpreter_without_laya_means_off(self, monkeypatch) -> None:
        monkeypatch.setattr(sufficiency, "laya_supported", lambda: True)
        monkeypatch.setattr(judge_selection, "interpreter_has_module", lambda py, m: False)
        assert _real_init_sufficiency_judge(self._cfg()) is None

    def test_everything_present_builds_a_configured_judge(self, monkeypatch, tmp_path) -> None:
        built = {}

        class _Stub:
            def __init__(self, **kw):
                built.update(kw)

        from superlocalmemory.core import laya_runtime

        monkeypatch.setattr(sufficiency, "laya_supported", lambda: True)
        monkeypatch.setattr(sufficiency, "LayaSufficiencyJudge", _Stub)
        monkeypatch.setattr(laya_runtime, "check_interpreter",
                            lambda python, strict_location=True: "")  # a fictitious path
        monkeypatch.setattr(judge_selection, "interpreter_has_module", lambda py, m: True)
        # "laya" by name: "auto" needs a checked install (M-2). The repo id
        # loads the measured revision's snapshot only.
        from superlocalmemory.core.laya_runtime import LAYA_MODEL_REVISION

        snapshot = (tmp_path / "hub" / "models--aac6fef--laya-mlx" / "snapshots"
                    / LAYA_MODEL_REVISION)
        snapshot.mkdir(parents=True)
        judge = _real_init_sufficiency_judge(self._cfg(
            sufficiency_judge="laya",
            sufficiency_python="/opt/laya/bin/python", sufficiency_hf_home=str(tmp_path),
            sufficiency_timeout_s=0.8))
        assert isinstance(judge, _Stub)
        assert built == {"python": "/opt/laya/bin/python", "model": str(snapshot),
                         "hf_home": str(tmp_path), "timeout_s": 0.8, "start": False}

    def test_the_module_check_never_imports_it(self, tmp_path) -> None:
        """find_spec only: a daemon start must not pay for loading MLX here."""
        import sys

        check = judge_selection.interpreter_has_module
        assert check(sys.executable, "json") is True
        assert check(sys.executable, "no_such_module_x") is False
        assert check(str(tmp_path / "missing-python"), "json") is False

    def test_no_test_engine_starts_a_real_judge(self, monkeypatch) -> None:
        """The session guard: building an engine in a test never spawns Laya.

        Forced to the case where everything is present, so it holds on a
        machine whose test venv has no laya-mlx too — otherwise it would pass
        there with the guard removed, and prove nothing."""
        monkeypatch.setattr(sufficiency, "laya_supported", lambda: True)
        monkeypatch.setattr(sufficiency, "LayaSufficiencyJudge", lambda **kw: object())
        monkeypatch.setattr(judge_selection, "interpreter_has_module", lambda py, m: True)
        assert engine_wiring.init_sufficiency_judge(self._cfg()) is None
