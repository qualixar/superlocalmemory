"""Tests for run_bounded_loop — the gate-decides-not-the-agent orchestrator."""

import pytest

from superlocalmemory.loops.engine import run_bounded_loop
from superlocalmemory.loops.ledger import InMemoryLedger
from superlocalmemory.loops.models import Bounds, LapResult, Rung, Status, Verdict


def _runner(changed=True, claimed=False, tokens=0):
    def run(lap):
        return LapResult(changed=changed, agent_claimed_done=claimed, tokens=tokens)
    return run


def test_converges_when_gate_passes():
    led = InMemoryLedger()
    out = run_bounded_loop(
        "conv",
        bounds=Bounds(max_iterations=10),
        runner=_runner(),
        gate=lambda lap: Verdict(lap >= 3, f"lap {lap}"),
        ledger=led,
        run_id="r",
    )
    assert out.status is Status.DONE
    assert out.reason == "gate-passed"
    assert out.laps == 3
    assert [e.decision for e in led.laps("r")] == ["continue", "continue", "done"]


def test_agent_claimed_done_never_terminates():
    """The single most important invariant: the agent's claim is ignored."""
    out = run_bounded_loop(
        "claim",
        bounds=Bounds(max_iterations=2),
        runner=_runner(claimed=True),  # agent insists it is done every lap
        gate=lambda lap: Verdict(False, "gate says no"),
        run_id="r",
    )
    assert out.status is Status.HALT
    assert out.reason == "max-iterations"
    assert out.laps == 2


def test_no_progress_halts():
    out = run_bounded_loop(
        "np",
        bounds=Bounds(max_iterations=10, no_progress_window=2),
        runner=_runner(changed=False),
        gate=lambda lap: Verdict(False, "x"),
        run_id="r",
    )
    assert out.status is Status.HALT
    assert out.reason == "no-progress"
    assert out.laps == 2


def test_kill_switch_before_any_work():
    led = InMemoryLedger()
    out = run_bounded_loop(
        "kill",
        bounds=Bounds(max_iterations=5),
        runner=_runner(),
        gate=lambda lap: Verdict(True, "would pass"),
        killswitch=lambda: True,
        ledger=led,
        run_id="r",
    )
    assert out.status is Status.KILLED
    assert out.laps == 0
    assert [e.decision for e in led.laps("r")] == ["killed"]


def test_pause_when_approval_required_and_absent():
    out = run_bounded_loop(
        "appr",
        bounds=Bounds(max_iterations=5),
        rung=Rung.L2,
        runner=_runner(),
        gate=lambda lap: Verdict(True, "pass"),
        run_id="r",
    )
    assert out.status is Status.PAUSE
    assert out.reason == "awaiting-approval"


def test_done_when_approval_granted():
    out = run_bounded_loop(
        "appr",
        bounds=Bounds(max_iterations=5),
        rung=Rung.L2,
        runner=_runner(),
        gate=lambda lap: Verdict(True, "pass"),
        approver=lambda verdict: True,
        run_id="r",
    )
    assert out.status is Status.DONE


def test_runner_error_is_terminal():
    def boom(lap):
        raise RuntimeError("kaboom")

    out = run_bounded_loop(
        "err", bounds=Bounds(max_iterations=5), runner=boom,
        gate=lambda lap: Verdict(True, "x"), run_id="r",
    )
    assert out.status is Status.ERROR
    assert "runner error" in out.reason
    assert "kaboom" in out.reason


def test_gate_error_is_terminal():
    def bad_gate(lap):
        raise ValueError("gate broke")

    out = run_bounded_loop(
        "err", bounds=Bounds(max_iterations=5), runner=_runner(),
        gate=bad_gate, run_id="r",
    )
    assert out.status is Status.ERROR
    assert "gate error" in out.reason


def test_token_budget_halts():
    out = run_bounded_loop(
        "tok",
        bounds=Bounds(max_iterations=100, max_tokens=25),
        runner=_runner(tokens=10),  # 10 per lap
        gate=lambda lap: Verdict(False, "never"),
        run_id="r",
    )
    # laps 1,2,3 spend 10/20/30; budget checked BEFORE lap 4 (30 > 25) → halt.
    assert out.status is Status.HALT
    assert out.reason == "token-budget"


def test_wallclock_budget_halts():
    ticks = iter([0.0, 0.0, 5.0, 10.0, 20.0, 30.0, 40.0])

    def fake_monotonic():
        return next(ticks)

    out = run_bounded_loop(
        "wc",
        bounds=Bounds(max_iterations=100, max_wallclock_s=8.0),
        runner=_runner(),
        gate=lambda lap: Verdict(False, "never"),
        monotonic=fake_monotonic,
        run_id="r",
    )
    assert out.status is Status.HALT
    assert out.reason == "wallclock"


def test_deterministic_clock_stamps_ledger():
    led = InMemoryLedger()
    run_bounded_loop(
        "clock",
        bounds=Bounds(max_iterations=1),
        runner=_runner(),
        gate=lambda lap: Verdict(False, "x"),
        clock=lambda: "2026-07-23T00:00:00+00:00",
        ledger=led,
        run_id="r",
    )
    entries = led.laps("r")
    assert entries and all(e.ts == "2026-07-23T00:00:00+00:00" for e in entries)


def test_run_id_is_generated_when_absent():
    out = run_bounded_loop(
        "auto", bounds=Bounds(max_iterations=1),
        runner=_runner(), gate=lambda lap: Verdict(True, "ok"),
    )
    assert out.run_id.startswith("auto-")
    assert len(out.run_id) > len("auto-")
