# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Exactly one answer check runs, and only the one someone chose.

The product rule: switch on Jev and Laya does not run; switch on Laya and Jev
does not run. "auto" means Laya when it is installed, and never Jev — memory
text leaves the machine only when someone chose that. Every test below fakes
the things the selector consults (platform, runtime, interpreter, key store,
both judges), so each one can force exactly the case it is about.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from types import SimpleNamespace

import pytest

from superlocalmemory.core import engine_wiring, judge_keys, judge_selection, laya_runtime
from superlocalmemory.core.config import RetrievalConfig

# Imported at collection time: tests/conftest.py replaces this attribute for the
# whole session so no engine a test builds can start a real Laya worker.
from superlocalmemory.core.engine_wiring import (
    init_sufficiency_judge as _real_init_sufficiency_judge,
)
from superlocalmemory.core.recall_pipeline import _judge_sufficiency
from superlocalmemory.retrieval import jev_judge, sufficiency
from superlocalmemory.retrieval.sufficiency import SufficiencyVerdict
from superlocalmemory.storage.models import AtomicFact, RecallResponse, RetrievalResult

_SHA = "20aed815fc6acde75733882e7ec0e3f28aeb9717"
_MANAGED_PY = "/managed/laya/bin/python"
_SNAPSHOT = f"/managed/hf/hub/models--aac6fef--laya-mlx/snapshots/{_SHA}"
_VERDICT = SufficiencyVerdict((0.9,), 0.6, "laya:t@x:sufficiency-v1:top3")


class _World:
    """What the selector can see, and what it did."""

    def __init__(self) -> None:
        self.supported = True
        self.detect: object = "ready"      # a state, or an exception class to raise
        self.has_laya: set[str] = {_MANAGED_PY}
        self.keys: set[str] = {"typesafe"}
        self.key_store_error: type[BaseException] | None = None
        self.jev_error: type[BaseException] | None = None
        self.events: list[str] = []
        self.stores: list[object] = []
        self.live = 0
        self.max_live = 0
        self.build_delay_s = 0.0   # widens race windows in the concurrency test
        self.lock = threading.Lock()

    def started(self, name: str) -> None:
        with self.lock:
            self.events.append(f"{name}.init")
            self.live += 1
            self.max_live = max(self.max_live, self.live)

    def stopped(self, name: str) -> None:
        with self.lock:
            self.events.append(f"{name}.shutdown")
            self.live -= 1


def _fake_judge(world: _World, name: str):
    class _Fake:
        backend = name
        top_k = 3

        def __init__(self, **kwargs) -> None:
            if name == "jev" and world.jev_error is not None:
                raise world.jev_error("not in this build")
            self.kwargs = kwargs
            self.closed = False
            time.sleep(world.build_delay_s)
            world.started(name)

        @property
        def ready(self) -> bool:
            return not self.closed

        def judge(self, query, documents):
            return None if self.closed else _VERDICT

        def shutdown(self) -> None:
            if not self.closed:
                self.closed = True
                world.stopped(name)

    return _Fake


@pytest.fixture()
def world(monkeypatch) -> _World:
    w = _World()

    class _Store:
        def __init__(self, slm_home=None) -> None:
            if w.key_store_error is not None:
                raise w.key_store_error("not in this build")
            w.stores.append(self)

        def has_key(self, provider: str) -> bool:
            return provider in w.keys

    def detect(retrieval_config=None):
        if isinstance(w.detect, type) and issubclass(w.detect, BaseException):
            raise w.detect("runtime check unavailable")
        return laya_runtime.LayaRuntimeStatus(
            state=str(w.detect), managed=True, python=_MANAGED_PY,
            hf_home="/managed/hf", model_path=_SNAPSHOT, model_revision=_SHA)

    monkeypatch.setattr(sufficiency, "laya_supported", lambda: w.supported)
    monkeypatch.setattr(sufficiency, "LayaSufficiencyJudge", _fake_judge(w, "laya"))
    monkeypatch.setattr(jev_judge, "JevSufficiencyJudge", _fake_judge(w, "jev"))
    monkeypatch.setattr(judge_keys, "JudgeKeyStore", _Store)
    monkeypatch.setattr(laya_runtime, "detect", detect)
    # Paths here are fictitious; the interpreter safety rule has its own tests.
    monkeypatch.setattr(laya_runtime, "check_interpreter",
                        lambda python, strict_location=True: "")
    monkeypatch.setattr(judge_selection, "interpreter_has_module",
                        lambda python, module: module == "laya_mlx" and python in w.has_laya)
    monkeypatch.setattr(judge_selection, "_live", None)
    # The real builder, for this module only: conftest's guard is tested below.
    monkeypatch.setattr(engine_wiring, "init_sufficiency_judge", _real_init_sufficiency_judge)
    yield w
    live = judge_selection._live
    if live is not None:
        live.judge.shutdown()


def _cfg(mode: str = "auto", **kw) -> RetrievalConfig:
    kw.setdefault("sufficiency_jev_consent", False)
    return RetrievalConfig(sufficiency_judge=mode, **kw)


def _jev_cfg(**kw) -> RetrievalConfig:
    kw.setdefault("sufficiency_jev_consent", True)
    kw.setdefault("sufficiency_jev_provider", "typesafe")
    return _cfg("jev", **kw)


def _engine(judge=None) -> SimpleNamespace:
    return SimpleNamespace(_sufficiency_judge=judge)


def _response() -> RecallResponse:
    return RecallResponse(results=[RetrievalResult(
        fact=AtomicFact(content="the answer"), score=0.7, confidence=1.0)])


# ---------------------------------------------------------------------------
# Which backend runs
# ---------------------------------------------------------------------------

def _expected(mode: str, supported: bool, installed: bool, key: bool, consent: bool) -> str:
    if mode in ("auto", "laya"):
        return "laya" if supported and installed else "off"
    if mode == "jev":
        return "jev" if key and consent else "off"
    return "off"


_MATRIX = list(itertools.product(
    ["off", "auto", "laya", "jev", "sometimes"], [True, False], [True, False],
    [True, False], [True, False]))


@pytest.mark.parametrize("mode,supported,installed,key,consent", _MATRIX)
def test_exactly_the_chosen_backend_runs(world, mode, supported, installed, key, consent) -> None:
    world.supported = supported
    world.detect = "ready" if installed else "not_installed"
    world.has_laya = {_MANAGED_PY} if installed else set()
    world.keys = {"typesafe"} if key else set()
    cfg = _cfg(mode, sufficiency_jev_consent=consent)
    expected = _expected(mode, supported, installed, key, consent)

    assert judge_selection.resolve_judge_mode(cfg) == expected
    assert world.events == [], "resolving the mode must not start anything"
    judge = engine_wiring.init_sufficiency_judge(cfg)
    assert judge_selection.backend_of(judge) == expected
    assert world.events == ([] if expected == "off" else [f"{expected}.init"])


class TestAutoNeverChoosesTheHostedCheck:
    def test_not_even_with_a_key_and_consent(self, world) -> None:
        world.supported = False
        cfg = _cfg("auto", sufficiency_jev_consent=True, sufficiency_jev_provider="typesafe")
        assert engine_wiring.init_sufficiency_judge(cfg) is None
        assert judge_selection.resolve_judge_mode(cfg) == "off"
        assert world.events == []
        assert world.stores == [], "auto must not even look at the hosted check's key"

    def test_with_laya_present_auto_is_laya(self, world) -> None:
        cfg = _cfg("auto", sufficiency_jev_consent=True)
        assert judge_selection.backend_of(engine_wiring.init_sufficiency_judge(cfg)) == "laya"
        assert world.events == ["laya.init"]

    @pytest.mark.parametrize("mode", ["A", "B"])
    def test_the_shipped_default_is_laya_in_mode_a_and_mode_b(
            self, world, tmp_path, mode) -> None:
        """4.1.25: the on-device check is on by default wherever the hardware
        runs it, whatever the mode; nothing about the mode turns it off."""
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.storage.models import Mode

        retrieval = SLMConfig.for_mode(getattr(Mode, mode), base_dir=tmp_path).retrieval
        assert retrieval.sufficiency_judge == "auto"
        assert judge_selection.backend_of(engine_wiring.init_sufficiency_judge(retrieval)) \
            == "laya"
        world.supported = False  # not Apple Silicon: off, in either mode
        assert engine_wiring.init_sufficiency_judge(retrieval) is None

    def test_attach_on_auto_never_starts_jev(self, world) -> None:
        world.supported = False
        engine = _engine()
        cfg = _cfg("auto", sufficiency_jev_consent=True)
        assert engine_wiring.attach_sufficiency_judge(engine, cfg) == "off"
        assert engine._sufficiency_judge is None and world.events == []


class TestTheHostedCheckNeedsEveryCondition:
    @pytest.mark.parametrize("consent", ["true", 1, "yes", None])
    def test_consent_must_be_the_boolean_true(self, world, consent) -> None:
        """A hand-edited config saying "true" is not someone clicking accept."""
        cfg = _cfg("jev", sufficiency_jev_consent=consent)
        assert engine_wiring.init_sufficiency_judge(cfg) is None
        assert world.events == []

    def test_an_unknown_provider_is_off(self, world) -> None:
        assert engine_wiring.init_sufficiency_judge(
            _jev_cfg(sufficiency_jev_provider="somewhere-else")) is None
        assert world.events == []

    def test_no_stored_key_is_off(self, world) -> None:
        world.keys = set()
        assert engine_wiring.init_sufficiency_judge(_jev_cfg()) is None

    @pytest.mark.parametrize("error", [NotImplementedError, OSError])
    def test_a_key_store_that_fails_is_off(self, world, error) -> None:
        world.key_store_error = error
        assert engine_wiring.init_sufficiency_judge(_jev_cfg()) is None
        assert judge_selection.resolve_judge_mode(_jev_cfg()) == "off"

    @pytest.mark.parametrize("error", [NotImplementedError, RuntimeError])
    def test_a_judge_that_cannot_start_is_off_with_one_log_line(
        self, world, error, caplog,
    ) -> None:
        world.jev_error = error
        with caplog.at_level(logging.INFO, logger=judge_selection.__name__):
            assert engine_wiring.init_sufficiency_judge(_jev_cfg()) is None
        assert len([r for r in caplog.records if "hosted" in r.getMessage()]) == 1

    def test_it_is_built_with_the_configured_provider_and_timeout(self, world) -> None:
        judge = engine_wiring.init_sufficiency_judge(_jev_cfg(sufficiency_jev_timeout_s=3.5))
        assert judge.kwargs["provider"] == "typesafe"
        assert judge.kwargs["timeout_s"] == 3.5
        assert judge.kwargs["key_store"] is world.stores[-1]


class TestWhereLayaIsFound:
    def test_an_explicit_interpreter_comes_first(self, world, tmp_path) -> None:
        world.has_laya = {_MANAGED_PY, "/opt/laya/bin/python"}
        snapshot = tmp_path / "hub" / "models--aac6fef--laya-mlx" / "snapshots" / _SHA
        snapshot.mkdir(parents=True)
        judge = engine_wiring.init_sufficiency_judge(_cfg(
            "laya", sufficiency_python="/opt/laya/bin/python", sufficiency_hf_home=str(tmp_path)))
        assert judge.kwargs["python"] == "/opt/laya/bin/python"
        assert judge.kwargs["hf_home"] == str(tmp_path)
        # The repo id is pinned to the measured revision's snapshot (M-2).
        assert judge.kwargs["model"] == str(snapshot)

    def test_an_explicit_interpreter_without_laya_is_off_not_replaced(self, world) -> None:
        """An explicit choice that is broken is reported, not silently swapped."""
        assert engine_wiring.init_sufficiency_judge(
            _cfg("laya", sufficiency_python="/opt/empty/bin/python")) is None
        assert world.events == []

    def test_a_ready_managed_runtime_supplies_interpreter_weights_and_cache(self, world) -> None:
        judge = engine_wiring.init_sufficiency_judge(_cfg("laya"))
        assert judge.kwargs == {"python": _MANAGED_PY, "model": _SNAPSHOT,
                                "hf_home": "/managed/hf", "timeout_s": 1.5, "start": False}

    @pytest.mark.parametrize("detect", [NotImplementedError, RuntimeError, "installing",
                                        "failed", "not_installed"])
    def test_otherwise_slms_own_interpreter_if_it_has_laya(self, world, detect, tmp_path) -> None:
        import sys

        world.detect = detect
        world.has_laya = {sys.executable}
        snapshot = tmp_path / "hub" / "models--aac6fef--laya-mlx" / "snapshots" / _SHA
        snapshot.mkdir(parents=True)
        judge = engine_wiring.init_sufficiency_judge(
            _cfg("laya", sufficiency_hf_home=str(tmp_path)))
        assert judge.kwargs["python"] == sys.executable
        assert judge.kwargs["model"] == str(snapshot)

    def test_nothing_found_is_off(self, world) -> None:
        world.detect = NotImplementedError
        world.has_laya = set()
        assert engine_wiring.init_sufficiency_judge(_cfg("laya")) is None

    def test_a_malformed_timeout_falls_back_instead_of_breaking_startup(self, world) -> None:
        judge = engine_wiring.init_sufficiency_judge(_cfg("laya", sufficiency_timeout_s="abc"))
        assert judge.kwargs["timeout_s"] == 1.5

    def test_an_unknown_mode_is_off_with_a_warning(self, world, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger=judge_selection.__name__):
            assert engine_wiring.init_sufficiency_judge(_cfg("sometimes")) is None
        assert any("sometimes" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Switching: the old one stops before the new one exists
# ---------------------------------------------------------------------------

class TestSwitching:
    def test_laya_to_jev_stops_laya_before_jev_is_built(self, world) -> None:
        engine = _engine()
        assert engine_wiring.attach_sufficiency_judge(engine, _cfg("laya")) == "laya"
        assert engine_wiring.attach_sufficiency_judge(engine, _jev_cfg()) == "jev"
        assert world.events == ["laya.init", "laya.shutdown", "jev.init"]
        assert judge_selection.backend_of(engine._sufficiency_judge) == "jev"

    def test_a_judge_set_on_the_engine_directly_is_also_stopped_first(self, world) -> None:
        """The engine may hold a judge that never came through the process
        registry (a plugin, an older build). The switch itself must stop it
        before building — not rely on the registry to."""
        engine = _engine(sufficiency.LayaSufficiencyJudge(python="/elsewhere/python"))
        assert engine_wiring.attach_sufficiency_judge(engine, _jev_cfg()) == "jev"
        assert world.events == ["laya.init", "laya.shutdown", "jev.init"]

    def test_jev_to_laya_stops_jev_before_laya_is_built(self, world) -> None:
        engine = _engine()
        engine_wiring.attach_sufficiency_judge(engine, _jev_cfg())
        assert engine_wiring.attach_sufficiency_judge(engine, _cfg("laya")) == "laya"
        assert world.events == ["jev.init", "jev.shutdown", "laya.init"]

    def test_switching_off_stops_it_and_leaves_no_judge(self, world) -> None:
        engine = _engine()
        engine_wiring.attach_sufficiency_judge(engine, _cfg("laya"))
        assert engine_wiring.attach_sufficiency_judge(engine, _cfg("off")) == "off"
        assert engine._sufficiency_judge is None
        assert world.events == ["laya.init", "laya.shutdown"]

    def test_no_recall_sees_a_judge_while_the_new_one_is_built(self, world, monkeypatch) -> None:
        engine = _engine()
        engine_wiring.attach_sufficiency_judge(engine, _cfg("laya"))
        seen: list = []
        fake_jev = jev_judge.JevSufficiencyJudge

        class _Watching(fake_jev):
            def __init__(self, **kwargs) -> None:
                seen.append(engine._sufficiency_judge)
                super().__init__(**kwargs)

        monkeypatch.setattr(jev_judge, "JevSufficiencyJudge", _Watching)
        engine_wiring.attach_sufficiency_judge(engine, _jev_cfg())
        assert seen == [None], "the old judge was still on the engine during the build"

    def test_an_old_judge_that_fails_to_stop_does_not_block_the_switch(self, world) -> None:
        class _Stubborn:
            backend = "laya"

            def shutdown(self) -> None:
                raise RuntimeError("would not stop")

        engine = _engine(_Stubborn())
        assert engine_wiring.attach_sufficiency_judge(engine, _jev_cfg()) == "jev"

    def test_a_closed_engine_never_gets_a_new_judge(self, world) -> None:
        """A switch that races an engine close must not start a model nobody
        will ever stop."""
        engine = _engine()
        engine_wiring.attach_sufficiency_judge(engine, _cfg("laya"))
        engine_wiring.release_sufficiency_judge(engine, final=True)
        assert engine_wiring.attach_sufficiency_judge(engine, _jev_cfg()) == "off"
        assert engine._sufficiency_judge is None
        assert world.events == ["laya.init", "laya.shutdown"]

    def test_concurrent_switches_never_run_two_models(self, world) -> None:
        world.build_delay_s = 0.002   # long enough for threads to interleave
        engine = _engine()
        configs = [_cfg("laya"), _jev_cfg(), _cfg("laya", sufficiency_timeout_s=2.0)]
        errors: list[BaseException] = []

        def flip(offset: int) -> None:
            try:
                for i in range(15):
                    engine_wiring.attach_sufficiency_judge(engine, configs[(i + offset) % 3])
            except BaseException as exc:  # noqa: BLE001 — that is the assertion
                errors.append(exc)

        threads = [threading.Thread(target=flip, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        assert errors == []
        assert world.max_live == 1, "two answer-check models were alive at once"
        assert world.live == 1 and engine._sufficiency_judge is not None


class TestOneModelPerProcess:
    """The daemon's hot reconfigure builds the new engine before it closes the
    old one, so two engines briefly coexist. They must not each run a model."""

    def test_two_engines_with_the_same_laya_share_one_worker(self, world) -> None:
        old, new = _engine(), _engine()
        old._sufficiency_judge = engine_wiring.init_sufficiency_judge(_cfg("laya"))
        new._sufficiency_judge = engine_wiring.init_sufficiency_judge(_cfg("laya"))
        assert old._sufficiency_judge is new._sufficiency_judge
        assert world.events == ["laya.init"]
        engine_wiring.release_sufficiency_judge(old, final=True)
        assert world.events == ["laya.init"], "closing the old engine stopped the new one's judge"
        engine_wiring.release_sufficiency_judge(new, final=True)
        assert world.events == ["laya.init", "laya.shutdown"]

    def test_switching_backend_across_engines_stops_the_old_one_first(self, world) -> None:
        old, new = _engine(), _engine()
        old._sufficiency_judge = engine_wiring.init_sufficiency_judge(_cfg("laya"))
        new._sufficiency_judge = engine_wiring.init_sufficiency_judge(_jev_cfg())
        assert world.events == ["laya.init", "laya.shutdown", "jev.init"]
        # A recall still on the old engine is unjudged, not broken.
        assert _judge_sufficiency(old, "q", _response()) is None
        engine_wiring.release_sufficiency_judge(old, final=True)
        assert world.events == ["laya.init", "laya.shutdown", "jev.init"]

    def test_a_different_laya_setup_replaces_the_first(self, world) -> None:
        engine_wiring.init_sufficiency_judge(_cfg("laya"))
        engine_wiring.init_sufficiency_judge(_cfg("laya", sufficiency_timeout_s=3.0))
        assert world.events == ["laya.init", "laya.shutdown", "laya.init"]

    def test_a_judge_shut_down_elsewhere_is_not_handed_out_again(self, world) -> None:
        first = engine_wiring.init_sufficiency_judge(_cfg("laya"))
        first.shutdown()
        second = engine_wiring.init_sufficiency_judge(_cfg("laya"))
        assert second is not first and not second.closed


class TestARecallInFlight:
    def test_a_recall_holding_the_old_judge_gets_none_and_completes(self, world) -> None:
        from superlocalmemory.core.score_contract import finalize_score_contract

        engine = _engine()
        engine_wiring.attach_sufficiency_judge(engine, _cfg("laya"))
        grabbed = _engine(engine._sufficiency_judge)  # read before the switch
        engine_wiring.attach_sufficiency_judge(engine, _cfg("off"))
        response = _response()
        verdict = _judge_sufficiency(grabbed, "q", response)
        assert verdict is None
        finalize_score_contract(response, verdict=verdict)
        assert response.results and response.calibration_status == "uncalibrated"
        assert response.abstained is False

    def test_a_shut_down_judge_that_raises_is_still_only_none(self, world) -> None:
        class _Raising:
            top_k = 3

            def judge(self, query, documents):
                raise RuntimeError("shut down")

        assert _judge_sufficiency(_engine(_Raising()), "q", _response()) is None


class TestTheEngineClosePath:
    def test_closing_the_engine_stops_its_judge_and_bars_a_new_one(self, world) -> None:
        from superlocalmemory.core.engine import MemoryEngine

        retrieval = SimpleNamespace(
            _reranker=None, close=lambda wait=False: None,
            _sufficiency_judge=engine_wiring.init_sufficiency_judge(_cfg("laya")))
        engine = MemoryEngine.__new__(MemoryEngine)
        engine._retrieval_engine = retrieval
        engine.close()
        assert world.events == ["laya.init", "laya.shutdown"]
        assert retrieval._sufficiency_judge is None
        assert engine_wiring.attach_sufficiency_judge(retrieval, _jev_cfg()) == "off"
        assert world.events == ["laya.init", "laya.shutdown"]

    def test_any_judge_is_stopped_not_only_laya(self, world) -> None:
        from superlocalmemory.core.engine import MemoryEngine

        retrieval = SimpleNamespace(_reranker=None, close=lambda wait=False: None,
                                    _sufficiency_judge=engine_wiring.init_sufficiency_judge(
                                        _jev_cfg()))
        engine = MemoryEngine.__new__(MemoryEngine)
        engine._retrieval_engine = retrieval
        engine.close()
        assert world.events == ["jev.init", "jev.shutdown"]


def test_the_session_guard_also_covers_switching(world, monkeypatch) -> None:
    """conftest's guard replaces the builder; attach must go through it, or a
    dashboard test that switches judges would start a real Laya worker."""
    monkeypatch.setattr(engine_wiring, "init_sufficiency_judge", lambda cfg: None)
    engine = _engine()
    assert engine_wiring.attach_sufficiency_judge(engine, _cfg("laya")) == "off"
    assert world.events == []
