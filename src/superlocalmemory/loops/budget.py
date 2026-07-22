"""Cumulative budget accounting for a bounded-loop run.

The meter tracks laps, tokens, and wall-clock time and reports when any bound
in effect has been exceeded. It is intentionally tiny and side-effect-free
apart from its own internal counters, and the clock is injected so tests are
deterministic.
"""

from __future__ import annotations

from typing import Callable

from superlocalmemory.loops.models import Bounds


class BudgetMeter:
    """Accumulate spend across laps and answer "have we gone over?".

    ``now`` is a zero-argument callable returning monotonic-ish seconds
    (``time.monotonic`` in production, a fake in tests). The start time is
    captured at construction so wall-clock enforcement needs no globals.
    """

    def __init__(self, now: Callable[[], float]) -> None:
        self._now = now
        self._start = now()
        self._tokens = 0

    def spend(self, tokens: int) -> None:
        """Record token spend for a completed lap (negative values ignored)."""
        if tokens > 0:
            self._tokens += tokens

    def exceeded(self, lap: int, bounds: Bounds) -> tuple[bool, str]:
        """Return ``(tripped, reason)`` for the bounds checked before a lap runs.

        Checked in priority order: iteration cap, token budget, wall-clock.
        ``lap`` is the 1-based number of the lap about to run, so exceeding
        ``max_iterations`` is reported when the (max+1)-th lap is attempted.
        """
        if lap > bounds.max_iterations:
            return True, "max-iterations"
        if bounds.max_tokens is not None and self._tokens > bounds.max_tokens:
            return True, "token-budget"
        if bounds.max_wallclock_s is not None:
            elapsed = self._now() - self._start
            if elapsed > bounds.max_wallclock_s:
                return True, "wallclock"
        return False, ""

    def snapshot(self) -> dict:
        """Point-in-time spend, suitable for a ledger entry."""
        return {
            "tokens": self._tokens,
            "wallclock_s": round(self._now() - self._start, 3),
        }
