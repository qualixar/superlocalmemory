"""Tests for bounded-loop value types: validation, immutability, helpers."""

import dataclasses

import pytest

from superlocalmemory.loops.models import (
    Bounds,
    LapResult,
    Outcome,
    Rung,
    Status,
    Verdict,
)


def test_bounds_requires_positive_iterations():
    with pytest.raises(ValueError, match="max_iterations must be >= 1"):
        Bounds(max_iterations=0)


def test_bounds_rejects_negative_window_and_budgets():
    with pytest.raises(ValueError):
        Bounds(max_iterations=3, no_progress_window=-1)
    with pytest.raises(ValueError):
        Bounds(max_iterations=3, max_tokens=-5)
    with pytest.raises(ValueError):
        Bounds(max_iterations=3, max_wallclock_s=-1.0)


def test_bounds_defaults():
    b = Bounds(max_iterations=4)
    assert b.no_progress_window == 3
    assert b.max_tokens is None
    assert b.max_wallclock_s is None
    assert b.require_approval is None


def test_frozen_dataclasses_reject_mutation():
    for obj, fieldname, value in [
        (Bounds(max_iterations=3), "max_iterations", 9),
        (Verdict(True, "ok"), "passed", False),
        (LapResult(changed=True), "changed", False),
        (Outcome(Status.DONE, "gate-passed", 1, "r1"), "laps", 2),
    ]:
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, fieldname, value)


def test_verdict_evidence_not_shared():
    a = Verdict(True, "a")
    b = Verdict(True, "b")
    a.evidence["k"] = 1
    assert b.evidence == {}  # separate dict per instance


def test_rung_and_status_roundtrip_strings():
    assert Rung.L2 == "L2"
    assert Rung("L3") is Rung.L3
    assert Status.HALT == "HALT"
    assert Status("DONE") is Status.DONE


def test_outcome_ok_only_for_done():
    assert Outcome(Status.DONE, "gate-passed", 1, "r").ok is True
    for s in (Status.HALT, Status.PAUSE, Status.KILLED, Status.ERROR):
        assert Outcome(s, "x", 1, "r").ok is False


def test_lap_result_defaults():
    r = LapResult(changed=True)
    assert r.agent_claimed_done is False
    assert r.tokens == 0
    assert r.log == ""
