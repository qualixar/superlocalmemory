"""An answer carries a name, and a report about it can quote that name.

Closing the loop needs three things that were each missing. A recall had to
leave a record — it did not, on 34 of the 35 paths that call it, because the
record is discarded when the caller cannot name a session and almost no caller
could. The record had to be findable — the column meant to find it by was never
written. And a client had to be able to quote the name — the answer never
contained one, so nothing could.

With all three, a report joins to the exact recall it is about. Without them,
the best available link is which memories happen to overlap, inside a window.
"""

from __future__ import annotations

import os


class _Engine:
    """Just enough of an engine to exercise the naming rule."""

    _last_session_id = ""
    _ambient_session_id = f"engine:{os.getpid()}"


def _resolve(engine, session_id):
    from superlocalmemory.core.engine import MemoryEngine

    return MemoryEngine._session_for_signals(engine, session_id)


def test_a_caller_that_names_no_session_still_gets_a_name():
    engine = _Engine()
    assert _resolve(engine, None)
    assert _resolve(engine, "") == _resolve(engine, None)


def test_the_name_the_caller_gives_is_the_one_used():
    engine = _Engine()
    assert _resolve(engine, "agent-42") == "agent-42"


def test_a_later_unnamed_call_stays_in_the_session_it_was_told_about():
    """One conversation, one name, even when only its first turn said so."""
    engine = _Engine()
    _resolve(engine, "agent-42")
    assert _resolve(engine, None) == "agent-42"
    assert _resolve(engine, "   ") == "agent-42"


def test_the_queue_refuses_a_name_only_the_engine_uses():
    """The rule produces a name; that is not the same as producing a join key.

    This asserted acceptance until the store was measured. ``engine:<pid>``
    names the daemon, so every client that reaches one daemon shares it and a
    restart replaces it. Nothing else writes it, so no tool event ever carried
    a matching id and no signal was ever registered against one. Accepting it
    produced a row that could never be
    matched, which the proxy settler then defaulted to a neutral 0.5 — and a
    neutral update tightens a Beta posterior around its prior rather than
    leaving it free to move.

    The drop is counted, not silent: recording nothing invisibly is the failure
    this file was written to prevent, and it still is.
    """
    from superlocalmemory.learning.outcome_queue import (
        RecallEvent,
        enqueue_recall,
        get_counters,
    )

    before = get_counters()
    enqueue_recall(RecallEvent(
        session_id=_resolve(_Engine(), None), profile_id="default",
        query="anything", fact_ids=("f1",), query_id="qid",
    ))
    after = get_counters()

    assert after.get("recall_enqueued", 0) == before.get("recall_enqueued", 0)
    assert after.get("recall_dropped_synthetic_session", 0) == (
        before.get("recall_dropped_synthetic_session", 0) + 1
    )


def test_the_queue_accepts_a_name_the_caller_actually_gave():
    """The counterpart: a real conversation id is still recorded."""
    from superlocalmemory.learning.outcome_queue import (
        RecallEvent,
        enqueue_recall,
        get_counters,
    )

    before = get_counters().get("recall_enqueued", 0)
    enqueue_recall(RecallEvent(
        session_id="01a01bba-5f09-7a61-86f5-d67511c89283", profile_id="default",
        query="anything", fact_ids=("f1",), query_id="qid",
    ))
    assert get_counters().get("recall_enqueued", 0) == before + 1


def test_an_unnamed_record_is_still_refused():
    """The control. The rule exists to produce a name, not to remove the need
    for one — a record with an empty join key can never be matched."""
    from superlocalmemory.learning.outcome_queue import (
        RecallEvent,
        enqueue_recall,
        get_counters,
    )

    before = get_counters()["recall_enqueued"]
    enqueue_recall(RecallEvent(
        session_id="", profile_id="default", query="q", fact_ids=(), query_id="x",
    ))
    assert get_counters()["recall_enqueued"] == before


def test_naming_the_session_costs_nothing_measurable():
    """Recall must not get slower in order to record that it happened."""
    import time

    engine = _Engine()
    iterations = 50_000
    started = time.perf_counter()
    for _ in range(iterations):
        _resolve(engine, None)
    nanoseconds = (time.perf_counter() - started) / iterations * 1e9
    assert nanoseconds < 20_000, (
        f"naming the session costs {nanoseconds:.0f} ns per recall"
    )


def test_an_answer_carries_a_name_a_client_can_quote():
    from superlocalmemory.storage.models import RecallResponse

    response = RecallResponse(query="q")
    assert hasattr(response, "query_id")
    response.query_id = "abc123"

    from superlocalmemory.server.recall_serializer import recall_response_metadata

    assert recall_response_metadata(response).get("query_id") == "abc123"


def test_a_report_records_which_answer_it_is_about(tmp_path, monkeypatch):
    import pathlib

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.infra.data_root import state_path
    from superlocalmemory.learning.outcomes import OutcomeTracker
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage.migration_runner import apply_all, apply_deferred

    config = SLMConfig.load()
    db = DatabaseManager(config.db_path)
    db.initialize(schema)
    learning_db = pathlib.Path(state_path("learning.db"))
    apply_all(learning_db, pathlib.Path(config.db_path))
    # The column arrives after the runtime tables exist, which is why a store
    # gains it on first open rather than at install.
    apply_deferred(learning_db, pathlib.Path(config.db_path))

    tracker = OutcomeTracker(db)
    outcome = tracker.record_outcome(
        query="[test]", fact_ids=["f1", "f2"], outcome="success",
        profile_id="default", recall_query_id="the-answer-id",
    )
    stored = db.execute(
        "SELECT recall_query_id FROM action_outcomes WHERE outcome_id = ?",
        (outcome.outcome_id,),
    )
    assert dict(stored[0])["recall_query_id"] == "the-answer-id"


def test_a_report_with_no_answer_named_is_still_recorded(tmp_path, monkeypatch):
    """The control. Quoting the name is an improvement, not a requirement."""
    import pathlib

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.infra.data_root import state_path
    from superlocalmemory.learning.outcomes import OutcomeTracker
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage.migration_runner import apply_all

    config = SLMConfig.load()
    db = DatabaseManager(config.db_path)
    db.initialize(schema)
    apply_all(pathlib.Path(state_path("learning.db")), pathlib.Path(config.db_path))

    tracker = OutcomeTracker(db)
    outcome = tracker.record_outcome(
        query="[test]", fact_ids=["f1"], outcome="partial", profile_id="default",
    )
    assert outcome.outcome_id
