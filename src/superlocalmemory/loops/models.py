"""Immutable value types for SuperLocalMemory bounded loops.

A *bounded loop* is an agent loop that terminates when an independent gate
passes — never when the agent claims it is finished. This module holds the
pure data types the loop engine reasons over.

Design rules (kept deliberately strict):
  * Standard-library imports only. No I/O, no framework, no side effects.
  * Every dataclass is ``frozen=True`` — any attribute mutation raises
    ``TypeError`` at runtime, so a lap result cannot be rewritten after the
    gate has judged it.
  * Timestamps are ISO-8601 strings supplied by the engine's clock, never
    produced here with ``datetime.now()``.

This is SuperLocalMemory's own realization of the bounded-loop concept; the
loop-control discipline it encodes (gate-verified termination, enforced
bounds, an advisory-only agent claim) is a general practice, reimplemented
here against SLM's durable memory rather than a flat file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Rung(str, Enum):
    """Autonomy rung governing how much a human stays in the loop.

    ``L1`` report      — a human reads every verdict; the loop still exits on
                         a passing gate but nothing is auto-approved.
    ``L2`` assisted     — the agent acts but pauses for human approval before a
                         passing gate is accepted as DONE.
    ``L3`` unattended   — the agent acts autonomously; approval is derived from
                         the bounds alone.

    Subclassing ``(str, Enum)`` makes ``Rung.L2 == "L2"`` true and
    ``Rung("L2")`` reconstruct the member, so a rung round-trips through JSON
    or a CLI argument without a hand-written lookup table.
    """

    L1 = "L1"
    L2 = "L2"
    L3 = "L3"


class Status(str, Enum):
    """Terminal status of a bounded-loop run.

    ``DONE``   — the gate passed and approval was granted or not required.
    ``HALT``   — a safety bound tripped (iteration cap, no progress, budget).
    ``PAUSE``  — the gate passed but required approval was not granted.
    ``KILLED`` — an external kill switch tripped between laps.
    ``ERROR``  — the runner or gate raised before a verdict was produced.
    """

    DONE = "DONE"
    HALT = "HALT"
    PAUSE = "PAUSE"
    KILLED = "KILLED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class Bounds:
    """The safety envelope a loop runs inside.

    ``max_iterations``     hard cap on laps; required and must be >= 1.
    ``no_progress_window`` consecutive no-change laps that trigger a HALT.
    ``max_tokens``         cumulative token budget across laps, or ``None``.
    ``max_wallclock_s``    wall-clock ceiling in seconds, or ``None``.
    ``require_approval``    ``True``/``False`` forces the approval posture;
                           ``None`` derives it from the rung (L1 -> no
                           approval, L2/L3 -> approval required).
    """

    max_iterations: int
    no_progress_window: int = 3
    max_tokens: Optional[int] = None
    max_wallclock_s: Optional[float] = None
    require_approval: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.no_progress_window < 0:
            raise ValueError("no_progress_window must be >= 0")
        if self.max_tokens is not None and self.max_tokens < 0:
            raise ValueError("max_tokens must be >= 0 when set")
        if self.max_wallclock_s is not None and self.max_wallclock_s < 0:
            raise ValueError("max_wallclock_s must be >= 0 when set")


@dataclass(frozen=True)
class Verdict:
    """The independent gate's judgement of a single lap.

    ``passed``   True only when the gate mechanically confirmed the goal.
    ``detail``   human-readable one-line summary (required, non-empty).
    ``evidence`` structured gate output (counts, tails, diffs). Defaults to a
                 fresh dict per instance via ``default_factory`` so verdicts do
                 not share one mutable dict.

    ``passed=True`` is necessary but not sufficient for the loop to exit; the
    engine still consults the approval rung.
    """

    passed: bool
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LapResult:
    """What the runner reports after proposing one lap of work.

    ``changed``            True if the runner altered the workspace/state.
    ``agent_claimed_done`` the agent's own "I am finished" signal. Recorded for
                           audit and **never** used to terminate the loop — the
                           gate is the sole authority.
    ``tokens``             tokens spent this lap (0 when unknown).
    ``log``                short runner log for the lap.
    """

    changed: bool
    agent_claimed_done: bool = False
    tokens: int = 0
    log: str = ""


@dataclass(frozen=True)
class Outcome:
    """The final result of a bounded-loop run.

    ``status``   terminal status.
    ``reason``   short machine-friendly explanation ("gate-passed",
                 "no-progress", "max-iterations", "awaiting-approval",
                 "killed", or a gate/runner error string).
    ``laps``     number of laps executed at termination.
    ``run_id``   identifier used to locate this run's ledger in SLM memory.
    """

    status: Status
    reason: str
    laps: int
    run_id: str

    @property
    def ok(self) -> bool:
        """True only for a DONE outcome — the single success state."""
        return self.status is Status.DONE
