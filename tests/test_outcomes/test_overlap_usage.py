# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W3-2b: content overlap of a turn's reply with the memories delivered in the session.

Opt-in (``outcomes.usage``), local, ids and scores only: the reply is scored in
the Stop hook's own process and never leaves it.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from superlocalmemory.core.feature_flags import FeatureFlags, FlagState, write_flag
from superlocalmemory.hooks import event_delivery as ed
from superlocalmemory.hooks import stop_outcome_hook as stop
from superlocalmemory.outcomes import usage
from superlocalmemory.server import hook_event_writer as hew
from tests.test_outcomes.test_injection_receipts import _fact, _rows
from tests.test_outcomes.test_usage_join import T0, _deliver, _drain, _usage

REPLY = "Sure: the invoice run happens on the last working day, so book it for the 28th."


@pytest.fixture
def store(tmp_path, monkeypatch):
    from tests.test_core.test_coverage_backfill import _db

    db = _db(tmp_path / "memory.db")
    monkeypatch.setenv("SLM_HOME", str(tmp_path))  # the hook finds this store
    return db


# -- the score ------------------------------------------------------------------------------------

def test_the_score_is_the_share_of_the_memorys_distinctive_tokens_in_the_reply() -> None:
    memory = "The invoice run happens on the last working day of the month."
    # distinctive: invoice, happens, last, working, month -> the reply has 4 of 5
    assert usage.content_overlap(memory, REPLY) == pytest.approx(4 / 5)
    assert usage.content_overlap(memory, "I like pizza on Fridays") == 0.0


def test_numbers_count_and_case_does_not() -> None:
    assert usage.content_overlap("Port 8443 serves the STAGING proxy", "8443 serves the staging "
                                 "proxy port") == 1.0


def test_a_short_number_counts_like_any_token() -> None:
    # gate, code, moved, 41: the reply misses only the number
    assert usage.content_overlap("The gate code moved to 41", "the gate code moved to 77") == 0.75


def test_a_memory_with_two_distinctive_tokens_is_too_short_to_score() -> None:
    assert usage.content_overlap("Alex likes Vim", "Alex likes Vim") is None


def test_function_words_and_short_words_never_count() -> None:
    assert usage.content_overlap("there would have been about these", "x") is None
    assert usage.content_overlap("ok yes run", "ok yes run") is None


# -- what is compared -----------------------------------------------------------------------------

def _scores(store, session="conv-1", reply=REPLY):
    conn = sqlite3.connect(str(store.db_path))
    try:
        return dict(usage.overlap_scores(conn, "default", session, reply))
    finally:
        conn.close()


def test_only_memories_delivered_in_this_session_are_compared(store, tmp_path) -> None:
    here = _fact(store, "The invoice run happens on the last working day of the month.")
    there = _fact(store, "The invoice run happens on the last working day of the year.")
    _deliver(store, tmp_path, [here])
    _deliver(store, tmp_path, [there], session="conv-2")
    assert set(_scores(store)) == {here}


def test_a_withheld_memory_is_not_compared(store, tmp_path) -> None:
    a = _fact(store, "The invoice run happens on the last working day of the month.")
    did = _deliver(store, tmp_path, [a])
    conn = sqlite3.connect(str(store.db_path))
    with conn:
        conn.execute("UPDATE injection_receipts SET included = 0, delivered = 0, eligible = 1, "
                     "exclusion = NULL, propensity = 0.9 WHERE delivery_id = ?", (did,))
    conn.close()
    assert _scores(store) == {}


def test_at_most_the_latest_delivered_are_compared(store, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(usage, "MAX_COMPARED", 2)
    ids = [_fact(store, f"Server number {i} runs the nightly backup job") for i in range(4)]
    for i, fid in enumerate(ids):
        _deliver(store, tmp_path, [fid], at=T0 + timedelta(minutes=i))
    conn = sqlite3.connect(str(store.db_path))
    try:
        got = [f for f, _ in usage.delivered_in_session(conn, "default", "conv-1")]
    finally:
        conn.close()
    assert got == [ids[3], ids[2]]


def test_an_invented_session_or_an_empty_reply_compares_nothing(store, tmp_path) -> None:
    a = _fact(store, "The invoice run happens on the last working day of the month.")
    _deliver(store, tmp_path, [a], session="http:1")
    _deliver(store, tmp_path, [a])
    assert _scores(store, session="http:1") == {}
    assert _scores(store, reply="") == {}


# -- storing the score ----------------------------------------------------------------------------

def _overlap(store, scores, at, session="conv-1") -> int:
    conn = sqlite3.connect(str(store.db_path))
    try:
        with conn:
            return usage.join_overlap(conn, profile_id="default", session_id=session,
                                      scores=scores, observed_at=usage.stamp(at))
    finally:
        conn.close()


def test_a_delivery_keeps_its_highest_overlap_over_the_turns(store, tmp_path) -> None:
    a = _fact(store, "fact a")
    did = _deliver(store, tmp_path, [a])
    later = T0 + timedelta(minutes=2)
    assert _overlap(store, [(a, 0.2)], T0 + timedelta(minutes=1)) == 1
    _overlap(store, [(a, 0.7)], later)
    _overlap(store, [(a, 0.4)], T0 + timedelta(minutes=3))
    assert _usage(store.db_path) == [(did, a, "overlap", 0.7)]
    assert _rows(store.db_path, "SELECT observed_at FROM injection_usage")[0][0] == \
        usage.stamp(later)


def test_a_score_out_of_range_or_an_invented_session_is_not_stored(store, tmp_path) -> None:
    a = _fact(store, "fact a")
    _deliver(store, tmp_path, [a])
    assert _overlap(store, [(a, 1.5), (a, -0.1)], T0 + timedelta(minutes=1)) == 0
    _deliver(store, tmp_path, [a], session="view:x")  # a run that recorded its invented id
    assert _overlap(store, [(a, 0.5)], T0 + timedelta(minutes=1), session="view:x") == 0


# -- the event and the writer ---------------------------------------------------------------------

def test_the_event_carries_ids_and_numbers_and_refuses_anything_else() -> None:
    event = ed.overlap_signal(session_id="conv-1", scores=[("a", 0.5)], profile_id="default")
    hew._EVENT_ADAPTER.validate_python(event)
    with pytest.raises(ValidationError):
        hew._EVENT_ADAPTER.validate_python({**event, "reply": REPLY})
    with pytest.raises(ValidationError):
        hew._EVENT_ADAPTER.validate_python({**event, "scores": [{"fact_id": "a", "score": 2}]})
    with pytest.raises(ValidationError):
        hew._EVENT_ADAPTER.validate_python({**event, "scores": [
            {"fact_id": "a", "score": 0.5, "text": "the memory"}]})


def test_the_writer_stores_overlap_and_settles_no_reward(store, tmp_path) -> None:
    a = _fact(store, "fact a")
    did = _deliver(store, tmp_path, [a])
    conn = sqlite3.connect(str(store.db_path))
    with conn:
        conn.execute("INSERT INTO pending_outcomes (outcome_id, profile_id, session_id, "
                     "recall_query_id, fact_ids_json, query_text_hash, created_at_ms, "
                     "expires_at_ms) VALUES ('oc1', 'default', 'conv-1', 'rq', ?, 'h', 0, ?)",
                     (json.dumps([a]), 10 ** 13))
    conn.close()
    event = ed.overlap_signal(session_id="conv-1", scores=[(a, 0.6)], profile_id="default")
    event["observed_ms"] = int((T0 + timedelta(minutes=1)).timestamp() * 1000)
    _drain(store, tmp_path, [event])
    assert _usage(store.db_path) == [(did, a, "overlap", 0.6)]
    assert _rows(store.db_path, "SELECT signals_json, status FROM pending_outcomes") == [
        ("{}", "pending")]


# -- the Stop hook --------------------------------------------------------------------------------

def _delivered_now(store, tmp_path) -> str:
    a = _fact(store, "The invoice run happens on the last working day of the month.")
    _deliver(store, tmp_path, [a], at=datetime.now(UTC) - timedelta(minutes=1))
    return a


def test_usage_detection_is_on_by_default(tmp_path) -> None:
    assert FeatureFlags(tmp_path, environ={}).enabled("outcomes.usage")


def test_the_hook_sends_nothing_while_usage_detection_is_off(store, tmp_path) -> None:
    write_flag(tmp_path, "outcomes.usage", FlagState.OFF)
    _delivered_now(store, tmp_path)
    assert stop.overlap_events({"last_assistant_message": REPLY}, "conv-1", "default") == []


@pytest.mark.parametrize("state", [FlagState.ON, FlagState.SHADOW])
def test_the_hook_sends_ids_and_scores_never_the_reply(store, tmp_path, state) -> None:
    write_flag(tmp_path, "outcomes.usage", state)
    a = _delivered_now(store, tmp_path)
    (event,) = stop.overlap_events({"last_assistant_message": REPLY}, "conv-1", "default")
    assert event["scores"] == [{"fact_id": a, "score": 0.8}]
    wire = json.dumps(event)
    assert "invoice" not in wire and "28th" not in wire


@pytest.mark.parametrize("payload", [{}, {"last_assistant_message": ""},
                                     {"last_assistant_message": "  \n "},
                                     {"last_assistant_message": ["not text"]}])
def test_no_reply_means_no_event(store, tmp_path, payload) -> None:
    write_flag(tmp_path, "outcomes.usage", FlagState.ON)
    _delivered_now(store, tmp_path)
    assert stop.overlap_events(payload, "conv-1", "default") == []


def test_a_missing_store_never_breaks_the_hook(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SLM_HOME", str(tmp_path))
    write_flag(tmp_path, "outcomes.usage", FlagState.ON)
    assert stop.overlap_events({"last_assistant_message": REPLY}, "conv-1", "default") == []


def test_the_overlap_is_queued_before_the_finalize(store, tmp_path, monkeypatch) -> None:
    write_flag(tmp_path, "outcomes.usage", FlagState.ON)
    _delivered_now(store, tmp_path)
    sent: list = []
    monkeypatch.setattr(stop, "read_stdin_json", lambda: {
        "session_id": "conv-1", "last_assistant_message": REPLY})
    monkeypatch.setattr(ed, "active_profile", lambda: "default")
    monkeypatch.setattr(ed, "deliver", lambda events, **k: sent.extend(events) or "ack")
    stop._inner_main(0.0)
    assert [e["kind"] for e in sent] == ["overlap_signal", "session_finalize"]
