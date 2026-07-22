"""Tests for the pure loop-control predicates."""

from superlocalmemory.loops.models import Bounds, Rung, Verdict
from superlocalmemory.loops.rules import (
    no_progress,
    rung_requires_approval,
    stop_condition_met,
)


def test_stop_condition_tracks_gate_only():
    assert stop_condition_met(Verdict(True, "gate ok")) is True
    assert stop_condition_met(Verdict(False, "gate failed")) is False


def test_no_progress_triggers_after_full_window():
    assert no_progress([False, False, False], 3) is True
    assert no_progress([True, False, False], 3) is False  # a change breaks it
    assert no_progress([False, False], 3) is False  # not enough laps yet


def test_no_progress_window_zero_disabled():
    assert no_progress([False, False, False, False], 0) is False


def test_no_progress_uses_only_the_tail():
    # Early changes don't matter; only the last `window` laps count.
    assert no_progress([True, True, False, False], 2) is True


def test_explicit_require_approval_wins():
    b_true = Bounds(max_iterations=1, require_approval=True)
    b_false = Bounds(max_iterations=1, require_approval=False)
    assert rung_requires_approval(Rung.L1, b_true) is True
    assert rung_requires_approval(Rung.L3, b_false) is False


def test_approval_derived_from_rung_when_unset():
    b = Bounds(max_iterations=1)  # require_approval=None
    assert rung_requires_approval(Rung.L1, b) is False
    assert rung_requires_approval(Rung.L2, b) is True
    assert rung_requires_approval(Rung.L3, b) is True
