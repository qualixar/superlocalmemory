# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.encoding.foresight.

Covers:
  - extract_foresight_signals() pattern matching
  - foresight_to_temporal_events() conversion
  - has_foresight() quick check
  - Edge cases: no signals, empty content, multiple patterns
"""

from __future__ import annotations

from superlocalmemory.encoding.foresight import (
    extract_foresight_signals,
    foresight_to_temporal_events,
    has_foresight,
)
from superlocalmemory.storage.models import AtomicFact, TemporalEvent

# ---------------------------------------------------------------------------
# extract_foresight_signals
# ---------------------------------------------------------------------------

class TestExtractForesightSignals:
    def _make_fact(self, content: str) -> AtomicFact:
        return AtomicFact(fact_id="f1", content=content)

    def test_plan_to(self) -> None:
        fact = self._make_fact("Alice plans to visit Japan next year")
        signals = extract_foresight_signals(fact)
        assert len(signals) == 1
        assert signals[0]["fact_id"] == "f1"

    def test_going_to(self) -> None:
        fact = self._make_fact("Bob is going to start a new job")
        signals = extract_foresight_signals(fact)
        assert len(signals) == 1

    def test_will(self) -> None:
        fact = self._make_fact("Alice will finish the project by Friday")
        signals = extract_foresight_signals(fact)
        assert len(signals) == 1

    def test_scheduled(self) -> None:
        fact = self._make_fact("The meeting is scheduled for next Monday")
        signals = extract_foresight_signals(fact)
        assert len(signals) == 1

    def test_appointment(self) -> None:
        fact = self._make_fact("Doctor appointment on March 20")
        signals = extract_foresight_signals(fact)
        assert len(signals) == 1

    def test_reminder(self) -> None:
        fact = self._make_fact("Don't forget to submit the report")
        signals = extract_foresight_signals(fact)
        assert len(signals) == 1

    def test_deadline(self) -> None:
        fact = self._make_fact("The deadline is March 15, 2026")
        signals = extract_foresight_signals(fact)
        assert len(signals) == 1

    def test_next_weekday(self) -> None:
        fact = self._make_fact("Starting next Friday at the new office")
        signals = extract_foresight_signals(fact)
        assert len(signals) == 1

    def test_no_foresight(self) -> None:
        fact = self._make_fact("Alice works at Google as an engineer")
        signals = extract_foresight_signals(fact)
        assert len(signals) == 0

    def test_one_signal_per_fact(self) -> None:
        # Even with multiple patterns, only one signal returned
        fact = self._make_fact(
            "Alice plans to attend the scheduled appointment"
        )
        signals = extract_foresight_signals(fact)
        assert len(signals) <= 1

    def test_empty_content(self) -> None:
        fact = self._make_fact("")
        signals = extract_foresight_signals(fact)
        assert len(signals) == 0


# ---------------------------------------------------------------------------
# foresight_to_temporal_events
# ---------------------------------------------------------------------------

class TestForesightToTemporalEvents:
    def test_creates_events_per_entity(self) -> None:
        fact = AtomicFact(
            fact_id="f1",
            content="Alice plans to visit Japan next year",
            observation_date="2026-03-11",
            referenced_date="2027-01-01",
        )
        events = foresight_to_temporal_events(
            fact, ["ent_alice", "ent_japan"], "default",
        )
        assert len(events) == 2
        assert all(isinstance(e, TemporalEvent) for e in events)
        entity_ids = {e.entity_id for e in events}
        assert "ent_alice" in entity_ids
        assert "ent_japan" in entity_ids

    def test_event_has_fact_dates(self) -> None:
        fact = AtomicFact(
            fact_id="f1",
            content="Alice will start the project next week",
            observation_date="2026-03-11",
            interval_start="2026-03-18",
        )
        events = foresight_to_temporal_events(fact, ["ent_alice"], "default")
        assert len(events) == 1
        assert events[0].observation_date == "2026-03-11"
        assert events[0].interval_start == "2026-03-18"
        assert events[0].description.startswith("Foresight:")

    def test_no_events_when_no_foresight(self) -> None:
        fact = AtomicFact(
            fact_id="f1",
            content="Alice works at Google as an engineer",
        )
        events = foresight_to_temporal_events(fact, ["ent_alice"], "default")
        assert len(events) == 0

    def test_no_events_when_no_entities(self) -> None:
        fact = AtomicFact(
            fact_id="f1",
            content="Alice plans to visit Japan",
        )
        events = foresight_to_temporal_events(fact, [], "default")
        assert len(events) == 0

    def test_profile_id_set(self) -> None:
        fact = AtomicFact(
            fact_id="f1",
            content="Will finish the deadline soon",
        )
        events = foresight_to_temporal_events(fact, ["ent_a"], "work")
        assert len(events) == 1
        assert events[0].profile_id == "work"


# ---------------------------------------------------------------------------
# has_foresight
# ---------------------------------------------------------------------------

class TestHasForesight:
    def test_true_for_plan(self) -> None:
        assert has_foresight("I plan to travel next month") is True

    def test_true_for_will(self) -> None:
        assert has_foresight("She will complete the course") is True

    def test_true_for_going_to(self) -> None:
        assert has_foresight("They are going to launch the product") is True

    def test_true_for_schedule(self) -> None:
        assert has_foresight("Meeting is scheduled for Monday") is True

    def test_true_for_deadline(self) -> None:
        assert has_foresight("The deadline is next week") is True

    def test_true_for_next(self) -> None:
        assert has_foresight("Starting next Tuesday") is True

    def test_false_for_past(self) -> None:
        assert has_foresight("Alice went to the store") is False

    def test_false_for_factual(self) -> None:
        assert has_foresight("Paris is the capital of France") is False

    def test_false_for_empty(self) -> None:
        assert has_foresight("") is False
