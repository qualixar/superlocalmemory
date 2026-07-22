# audit-10 (3.7.9): the evolve_skill MCP path must run under a budget cycle so
# the LLM-call / wall-time / per-day caps apply (previously it bypassed them).
from unittest.mock import MagicMock

import pytest

from superlocalmemory.core.config import SLMConfig
from superlocalmemory.evolution.skill_evolver import SkillEvolver


def test_charge_outside_cycle_raises():
    """The guard is real: charging a call with no open cycle raises RuntimeError."""
    evolver = SkillEvolver(db_path=":memory:", config=SLMConfig.default())
    with pytest.raises(RuntimeError):
        evolver._budget.charge_llm_call()


def test_evolve_candidate_wraps_process_in_budget_cycle():
    """evolve_candidate must run _process_candidate INSIDE a budget cycle so
    the caps apply (the MCP path previously called _process_candidate raw)."""
    from contextlib import contextmanager

    evolver = SkillEvolver(db_path=":memory:", config=SLMConfig.default())
    events = []

    @contextmanager
    def fake_cycle():
        events.append("enter")
        try:
            yield MagicMock()
        finally:
            events.append("exit")

    evolver._budget = MagicMock()
    evolver._budget.cycle = fake_cycle

    def fake_process(candidate, profile_id):
        events.append("process")
        return "evolved"

    evolver._process_candidate = fake_process
    out = evolver.evolve_candidate(MagicMock(), "default")

    assert events == ["enter", "process", "exit"], (
        "_process_candidate must run inside the budget cycle"
    )
    assert out == "evolved"
