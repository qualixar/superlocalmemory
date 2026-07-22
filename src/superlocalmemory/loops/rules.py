"""Pure decision predicates for the bounded-loop engine.

Every function here is a pure function of its arguments: same inputs give the
same output, and nothing is mutated. The engine calls these to decide whether
to stop, halt, or keep going; adapters never call them directly.

Imports are limited to the standard library and the loop models.
"""

from __future__ import annotations

from typing import Sequence

from superlocalmemory.loops.models import Bounds, Rung, Verdict


def stop_condition_met(verdict: Verdict) -> bool:
    """Return True when the gate verdict means the loop may exit.

    The rule is deliberately conservative: a loop is eligible to finish only
    when the independent gate reports ``passed``. The agent's own opinion is
    never consulted here — it is not even an argument.
    """
    return verdict.passed


def no_progress(lap_changes: Sequence[bool], window: int) -> bool:
    """Return True when the last ``window`` laps all made no change.

    ``lap_changes`` is the ordered history of each lap's ``changed`` flag,
    most-recent last. A window of ``0`` disables the check (a spinning agent
    is then bounded only by the iteration cap). Fewer laps than the window
    means "not enough evidence yet" and returns False.
    """
    if window <= 0:
        return False
    tail = lap_changes[-window:]
    if len(tail) < window:
        return False
    return all(changed is False for changed in tail)


def rung_requires_approval(rung: Rung, bounds: Bounds) -> bool:
    """Return True when a human must approve a passing gate before DONE.

    An explicit ``bounds.require_approval`` wins outright. When it is ``None``
    the posture is derived from the rung: L1 exits without approval, while
    L2 and L3 require it.
    """
    if bounds.require_approval is not None:
        return bounds.require_approval
    return rung in (Rung.L2, Rung.L3)
