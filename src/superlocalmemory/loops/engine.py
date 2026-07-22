"""The bounded-loop orchestrator.

``run_bounded_loop`` executes a loop for a single goal under a fixed set of
:class:`Bounds`. Its one non-negotiable invariant: **the independent gate
decides when the loop is finished — never the agent.** A runner's
``agent_claimed_done`` flag is written to the ledger for audit and is never
read when deciding to terminate.

Each lap, in strict order:

1. Poll the kill switch (highest priority — checked before any work).
2. Check the budget bounds (iteration cap, tokens, wall-clock).
3. Run the agent's proposer for one lap.
4. Accumulate token spend.
5. Ask the *independent* gate for a verdict.
6. Decide: a passing gate (plus any required approval) ends the run; an
   exhausted no-progress window halts it; otherwise continue.

Runner and gate are plain callables taking the 1-based lap number, so this
engine carries no subprocess, sandbox, or framework machinery — SLM loops
converge on a checkable memory/verification condition, and heavier isolation
belongs to the standalone bounded-loops engine, not here.
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

from superlocalmemory.loops.budget import BudgetMeter
from superlocalmemory.loops.ledger import InMemoryLedger, LedgerEntry, LedgerStore
from superlocalmemory.loops.models import (
    Bounds,
    LapResult,
    Outcome,
    Rung,
    Status,
    Verdict,
)
from superlocalmemory.loops.rules import (
    no_progress,
    rung_requires_approval,
    stop_condition_met,
)

RunnerFn = Callable[[int], LapResult]
GateFn = Callable[[int], Verdict]
ApproverFn = Callable[[Verdict], bool]
KillSwitchFn = Callable[[], bool]
ClockFn = Callable[[], str]

_KILL_ENV = "SLM_LOOP_KILL"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_killed() -> bool:
    return bool(os.environ.get(_KILL_ENV))


def run_bounded_loop(
    name: str,
    *,
    bounds: Bounds,
    runner: RunnerFn,
    gate: GateFn,
    rung: Rung = Rung.L1,
    ledger: Optional[LedgerStore] = None,
    approver: Optional[ApproverFn] = None,
    killswitch: Optional[KillSwitchFn] = None,
    clock: Optional[ClockFn] = None,
    monotonic: Optional[Callable[[], float]] = None,
    run_id: Optional[str] = None,
) -> Outcome:
    """Run one bounded loop and return its :class:`Outcome`.

    Only ``name``, ``bounds``, ``runner`` and ``gate`` are required. Every
    other dependency is injected for determinism in tests; sensible defaults
    (UTC clock, monotonic timer, env-var kill switch, in-memory ledger) apply
    otherwise.
    """
    run_id = run_id or f"{name}-{uuid.uuid4().hex[:8]}"
    ledger = ledger if ledger is not None else InMemoryLedger()
    clock = clock or _utc_now_iso
    monotonic = monotonic or time.monotonic
    killswitch = killswitch or _env_killed
    budget = BudgetMeter(monotonic)

    lap_changes: list[bool] = []
    lap = 0

    def emit(decision: str, verdict: Verdict) -> None:
        ledger.record(
            LedgerEntry(
                run_id=run_id,
                name=name,
                lap=lap,
                ts=clock(),
                decision=decision,
                passed=verdict.passed,
                detail=verdict.detail,
                budget=budget.snapshot(),
            )
        )

    while True:
        lap += 1

        # 1. Kill switch — before any work (so laps reports completed laps).
        if killswitch():
            emit("killed", Verdict(False, "kill switch tripped"))
            return Outcome(Status.KILLED, "killed", lap - 1, run_id)

        # 2. Budget bounds — before running the agent.
        tripped, why = budget.exceeded(lap, bounds)
        if tripped:
            emit("halt", Verdict(False, why))
            return Outcome(Status.HALT, why, lap - 1, run_id)

        # 3. Run the proposer for one lap.
        try:
            result = runner(lap)
        except Exception as exc:  # runner failure is a terminal ERROR
            detail = f"runner error: {type(exc).__name__}: {exc}"
            emit("error", Verdict(False, detail))
            return Outcome(Status.ERROR, detail, lap, run_id)

        # 4. Accumulate spend.
        budget.spend(result.tokens)
        lap_changes.append(result.changed)

        # 5. Independent gate — agent's own claim is never consulted here.
        try:
            verdict = gate(lap)
        except Exception as exc:
            detail = f"gate error: {type(exc).__name__}: {exc}"
            emit("error", Verdict(False, detail))
            return Outcome(Status.ERROR, detail, lap, run_id)

        # 6. Decide.
        if stop_condition_met(verdict):
            if rung_requires_approval(rung, bounds):
                granted = bool(approver(verdict)) if approver is not None else False
                if not granted:
                    emit("pause", verdict)
                    return Outcome(Status.PAUSE, "awaiting-approval", lap, run_id)
            emit("done", verdict)
            return Outcome(Status.DONE, "gate-passed", lap, run_id)

        if no_progress(lap_changes, bounds.no_progress_window):
            emit("halt", Verdict(False, "no-progress"))
            return Outcome(Status.HALT, "no-progress", lap, run_id)

        emit("continue", verdict)
