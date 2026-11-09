# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The sufficiency judge must fail open, and must never strand itself.

The reranker taught both lessons the hard way: a worker that died stayed dead
for seven days because the code that re-warms it sat behind a readiness check,
and a recycle sent work to a worker still loading its model. Every test here
pins one way the judge could end up silent, stuck, or wrong — against a fake
worker that speaks the real protocol, so no model is ever loaded.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from superlocalmemory.core.score_contract import finalize_score_contract
from superlocalmemory.retrieval import sufficiency as mod
from superlocalmemory.retrieval.judge_recipe import RECIPE_V1, JudgeDocument, JudgeRecipe
from superlocalmemory.retrieval.sufficiency import LayaSufficiencyJudge, SufficiencyVerdict
from superlocalmemory.storage.models import AtomicFact, RecallResponse, RetrievalResult
from tests.helpers.owned_python import owned_python

_FAKE_WORKER = r'''
import json, os, sys, time

mode = os.environ.get("FAKE_LAYA_MODE", "normal")
log = os.environ.get("FAKE_LAYA_LOG")
for raw in sys.stdin:
    req = json.loads(raw)
    if log:
        with open(log, "a") as fh:
            fh.write(json.dumps(req) + "\n")
    cmd = req.get("cmd")
    if cmd == "quit":
        break
    if cmd == "load":
        if mode == "fail_load":
            print(json.dumps({"ok": False, "error": "weights not found"}), flush=True)
        else:
            print(json.dumps({"ok": True, "model": req.get("model")}), flush=True)
        continue
    if cmd == "judge":
        docs = req["documents"]
        if mode == "slow":
            time.sleep(5)
        if mode == "die":
            os._exit(1)
        if mode == "malformed":
            print(json.dumps({"ok": True, "probabilities": [1.5] * len(docs)}), flush=True)
            continue
        if mode == "short":
            print(json.dumps({"ok": True, "probabilities": [0.9]}), flush=True)
            continue
        print(json.dumps({"ok": True, "probabilities":
                          [0.9 if "answer" in d else 0.1 for d in docs]}), flush=True)
        continue
    print(json.dumps({"ok": False, "error": "unknown"}), flush=True)
'''


@pytest.fixture()
def worker(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "fake_laya_worker.py"
    path.write_text(_FAKE_WORKER, encoding="utf-8")
    monkeypatch.setattr(mod, "_WARMUP_BACKOFF_S", 0.01)
    return path


def _judge(worker: Path, **kw) -> LayaSufficiencyJudge:
    kw.setdefault("timeout_s", 2.0)
    return LayaSufficiencyJudge(python=str(owned_python(worker.parent)), worker_path=worker,
                                **kw)


def _wait(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class TestItNeverWaitsForAColdWorker:
    def test_a_cold_judge_answers_nothing_and_starts_warming(self, worker) -> None:
        judge = _judge(worker, start=False)
        try:
            assert judge.judge("q", ["the answer"]) is None
            assert _wait(lambda: judge.ready), "the cold call did not start the warm-up"
        finally:
            judge.shutdown()

    def test_a_warm_judge_judges_only_the_top_k(self, worker) -> None:
        judge = _judge(worker, top_k=3)
        try:
            assert _wait(lambda: judge.ready)
            verdict = judge.judge("q", ["the answer", "noise", "noise", "the answer", "x"])
            assert isinstance(verdict, SufficiencyVerdict)
            assert verdict.probabilities == (0.9, 0.1, 0.1)
            assert verdict.answer_confidence == 0.9
            assert verdict.insufficient is False
        finally:
            judge.shutdown()

    def test_a_set_with_nothing_relevant_is_insufficient(self, worker, tmp_path) -> None:
        # Only the measured weights may abstain (F16), so name that snapshot.
        judge = _judge(worker, model=str(
            tmp_path / "hub" / "models--aac6fef--laya-mlx" / "snapshots" / _SNAPSHOT_SHA))
        try:
            assert _wait(lambda: judge.ready)
            verdict = judge.judge("q", ["noise", "more noise"])
            assert verdict is not None and verdict.insufficient is True
        finally:
            judge.shutdown()


class TestItFailsOpenAndHealsItself:
    def test_a_slow_worker_is_cut_off_and_re_warmed(self, worker, monkeypatch) -> None:
        monkeypatch.setenv("FAKE_LAYA_MODE", "slow")
        judge = _judge(worker, timeout_s=0.3)
        try:
            assert _wait(lambda: judge.ready)
            started = time.monotonic()
            assert judge.judge("q", ["the answer"]) is None
            assert time.monotonic() - started < 2.0, "a recall waited on a stuck judge"
            assert _wait(lambda: judge.ready or judge.loading), "it did not try to recover"
        finally:
            judge.shutdown()

    def test_a_dead_worker_is_noticed_and_re_warmed(self, worker, monkeypatch) -> None:
        monkeypatch.setenv("FAKE_LAYA_MODE", "die")
        judge = _judge(worker)
        try:
            assert _wait(lambda: judge.ready)
            assert judge.judge("q", ["the answer"]) is None
            assert _wait(lambda: judge.ready or judge.loading), "a dead worker stranded the judge"
        finally:
            judge.shutdown()

    @pytest.mark.parametrize("mode", ["malformed", "short"])
    def test_a_malformed_answer_is_never_trusted(self, worker, monkeypatch, mode) -> None:
        monkeypatch.setenv("FAKE_LAYA_MODE", mode)
        judge = _judge(worker)
        try:
            assert _wait(lambda: judge.ready)
            assert judge.judge("q", ["the answer", "noise"]) is None
        finally:
            judge.shutdown()

    def test_a_load_that_never_succeeds_never_claims_readiness(self, worker, monkeypatch) -> None:
        monkeypatch.setenv("FAKE_LAYA_MODE", "fail_load")
        judge = _judge(worker)
        try:
            assert _wait(lambda: not judge.loading), "the warm-up never finished"
            assert judge.ready is False
            assert judge.judge("q", ["the answer"]) is None
        finally:
            judge.shutdown()

    def test_nothing_to_judge_is_not_a_verdict(self, worker) -> None:
        judge = _judge(worker, start=False)
        assert judge.judge("", ["the answer"]) is None
        assert judge.judge("q", []) is None
        assert judge.loading is False, "an empty request must not start a worker"


def test_shutdown_returns_promptly(worker) -> None:
    judge = _judge(worker)
    assert _wait(lambda: judge.ready)
    started = time.monotonic()
    judge.shutdown()
    assert time.monotonic() - started < 3.0
    assert judge.judge("q", ["the answer"]) is None


def test_laya_is_only_offered_on_apple_silicon(monkeypatch) -> None:
    monkeypatch.setattr(mod.sys, "platform", "linux")
    assert mod.laya_supported() is False
    monkeypatch.setattr(mod.sys, "platform", "darwin")
    monkeypatch.setattr(mod.platform, "machine", lambda: "x86_64")
    assert mod.laya_supported() is False
    monkeypatch.setattr(mod.platform, "machine", lambda: "arm64")
    assert mod.laya_supported() is True


# ---------------------------------------------------------------------------
# 4.1.18: the recipe reaches the worker, and the verdict says what it is
# ---------------------------------------------------------------------------

_SNAPSHOT_SHA = "20aed815fc6acde75733882e7ec0e3f28aeb9717"


def _requests(log: Path, cmd: str) -> list[dict]:
    if not log.exists():
        return []
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r.get("cmd") == cmd]


class TestTheRecipeReachesTheWorker:
    def test_the_recipe_question_and_rendering_are_what_the_worker_receives(
        self, worker, tmp_path, monkeypatch,
    ) -> None:
        log = tmp_path / "requests.jsonl"
        monkeypatch.setenv("FAKE_LAYA_LOG", str(log))
        recipe = JudgeRecipe("sufficiency-test", question="Custom wording?",
                             max_document_chars=6)
        judge = _judge(worker, recipe=recipe)
        try:
            assert _wait(lambda: judge.ready)
            judge.judge("q", [JudgeDocument("the answer is long")])
        finally:
            judge.shutdown()
        sent = _requests(log, "judge")
        assert len(sent) == 1
        assert sent[0]["question"] == "Custom wording?"
        assert sent[0]["documents"] == ["the an"], "the recipe renders each document"

    def test_the_default_recipe_is_the_measured_one(self, worker, tmp_path, monkeypatch) -> None:
        log = tmp_path / "requests.jsonl"
        monkeypatch.setenv("FAKE_LAYA_LOG", str(log))
        judge = _judge(worker)
        try:
            assert _wait(lambda: judge.ready)
            assert judge.judge("q", ["the answer"]) is not None
        finally:
            judge.shutdown()
        assert _requests(log, "judge")[0]["question"] == RECIPE_V1.question


class TestTheVerdictNamesItsSource:
    def test_a_snapshot_path_becomes_repo_and_revision_never_the_path(
        self, worker, tmp_path,
    ) -> None:
        snapshot = (tmp_path / "hf" / "hub" / "models--aac6fef--laya-mlx"
                    / "snapshots" / _SNAPSHOT_SHA)
        judge = _judge(worker, model=str(snapshot))
        try:
            assert _wait(lambda: judge.ready)
            verdict = judge.judge("q", ["the answer"])
        finally:
            judge.shutdown()
        assert verdict.calibration_id == "laya:aac6fef/laya-mlx@20aed815fc6a:sufficiency-v1:top3"
        assert str(tmp_path) not in verdict.calibration_id
        assert verdict.calibration_id.count("/") == 1, "only the repo id's own slash"
        assert verdict.backend == "laya"
        assert verdict.calibration_status == "measured_small_sample_not_calibrated"

    def test_a_plain_folder_is_local_and_a_repo_id_is_unpinned(self, tmp_path) -> None:
        folder = LayaSufficiencyJudge(model=str(tmp_path / "my-laya"), start=False)
        assert folder.calibration_id == "laya:local@unpinned:sufficiency-v1:top3"
        repo = LayaSufficiencyJudge(model="aac6fef/laya-mlx", start=False)
        assert repo.calibration_id == "laya:aac6fef/laya-mlx@unpinned:sufficiency-v1:top3"
        home = LayaSufficiencyJudge(model="~/models/laya", start=False)
        assert home.calibration_id == "laya:local@unpinned:sufficiency-v1:top3"

    def test_an_unmeasured_recipe_reports_confidence_but_cannot_abstain(self, worker) -> None:
        recipe = JudgeRecipe("sufficiency-never-measured", question="Does it answer?")
        judge = _judge(worker, recipe=recipe)
        try:
            assert _wait(lambda: judge.ready)
            verdict = judge.judge("q", ["noise", "more noise"])
        finally:
            judge.shutdown()
        assert verdict is not None and verdict.answer_confidence == 0.1
        assert verdict.insufficient is False, "an unmeasured pair abstained on its own say-so"
        assert verdict.calibration_status == "not_measured_cannot_abstain"
        response = RecallResponse(results=[RetrievalResult(
            fact=AtomicFact(content="noise"), score=0.5, confidence=1.0)])
        finalize_score_contract(response, verdict=verdict)
        assert response.abstained is False
        assert response.answer_confidence == 0.1

    def test_the_judge_names_its_backend(self) -> None:
        assert LayaSufficiencyJudge(start=False).backend == "laya"


class TestShutdownLeavesNothingRunning:
    def test_a_spawn_that_shutdown_did_not_see_is_stopped(self, worker) -> None:
        """shutdown() lands after the warm-up's first check but before the worker
        exists: it finds no process to stop. The warm-up must notice and stop
        the one it just started, or a second model runs beside the next judge."""
        judge = _judge(worker, start=False)
        spawned: list = []
        real_spawn = judge._spawn

        def spawn_after_shutdown() -> None:
            judge.shutdown()
            real_spawn()
            spawned.append(judge._proc)

        judge._spawn = spawn_after_shutdown
        resp = judge._request({"cmd": "load", "model": "m", "hf_home": "",
                               "memory_limit_mb": 0}, deadline=time.monotonic() + 5.0)
        assert resp is None
        assert spawned and spawned[0] is not None
        assert _wait(lambda: spawned[0].poll() is not None, timeout=5.0), \
            "a worker started after shutdown is still running"
        assert judge._proc is None and judge.ready is False

    def test_the_worker_vanishing_mid_request_is_none_not_a_crash(self) -> None:
        """shutdown() swaps the process out from under a recall that already
        checked it. The recall must get None, never an exception."""
        judge = LayaSufficiencyJudge(start=False, timeout_s=0.1)
        read_end, write_end = os.pipe()
        stdin = os.fdopen(os.dup(write_end), "w")
        stdout = os.fdopen(read_end, "r")

        class _Vanishing:
            def __init__(self) -> None:
                self.stdin, self.stdout = stdin, stdout

            def poll(self):
                judge._proc = None  # shutdown landed right after the check
                return None

            def kill(self) -> None: ...

            def wait(self, timeout=None) -> int:
                return 0

        judge._proc = _Vanishing()
        judge._ready = True
        try:
            assert judge.judge("q", ["the answer"]) is None
        finally:
            judge.shutdown()
            for fh in (stdin, stdout):
                try:
                    fh.close()
                except OSError:
                    pass
            os.close(write_end)

    def test_recalls_racing_a_shutdown_never_raise(self, worker) -> None:
        judge = _judge(worker)
        assert _wait(lambda: judge.ready)
        errors: list[BaseException] = []
        after: list = []
        stop = threading.Event()

        def recall_loop() -> None:
            while not stop.is_set():
                try:
                    verdict = judge.judge("q", ["the answer"])
                    if judge.closed:
                        after.append(verdict)
                except BaseException as exc:  # noqa: BLE001 — that is the assertion
                    errors.append(exc)

        thread = threading.Thread(target=recall_loop)
        thread.start()
        time.sleep(0.2)
        judge.shutdown()
        time.sleep(0.2)
        stop.set()
        thread.join(5)
        assert errors == []
        assert after and all(v is None for v in after[1:])


class TestOneLayaPerDataFolder:
    """Every SLM process can build an engine; only one may load the model."""

    @pytest.mark.skipif(sys.platform == "win32",
                        reason="the lock holder uses fcntl; Laya runs only on Apple silicon")
    def test_a_judge_does_not_load_while_another_process_runs_laya(self, worker) -> None:
        import subprocess as sp

        from superlocalmemory.infra.data_root import state_path

        path = state_path(".laya-judge.lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        holder = sp.Popen(
            [sys.executable, "-c",
             "import fcntl, sys, time; f = open(sys.argv[1], 'a+'); "
             "fcntl.flock(f.fileno(), fcntl.LOCK_EX); print('held', flush=True); time.sleep(30)",
             str(path)],
            stdout=sp.PIPE, text=True)
        try:
            assert holder.stdout.readline().strip() == "held"
            judge = _judge(worker)
            try:
                time.sleep(0.3)
                assert judge.loading is False and judge.ready is False
                assert judge._proc is None, "a second copy of the model was started"
                assert judge.judge("q", ["the answer"]) is None
            finally:
                judge.shutdown()
        finally:
            holder.kill()
            holder.wait(5)

    def test_the_slot_passes_to_the_next_judge_after_shutdown(self, worker, monkeypatch) -> None:
        monkeypatch.setattr(mod, "_SLOT_RETRY_S", 0.01)
        first = _judge(worker)
        second = None
        try:
            assert _wait(lambda: first.ready)
            second = _judge(worker)
            time.sleep(0.1)
            assert second.ready is False and second._proc is None
            first.shutdown()
            time.sleep(0.05)
            second.start_warmup()
            assert _wait(lambda: second.ready), "the freed slot was never taken"
        finally:
            first.shutdown()
            if second is not None:
                second.shutdown()

