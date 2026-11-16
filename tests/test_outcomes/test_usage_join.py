# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W3-2: every usage signal is tied to the delivery it came from (D-14, M-12 input)."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from superlocalmemory.hooks import event_delivery as ed
from superlocalmemory.hooks import event_spool
from superlocalmemory.outcomes import receipts, usage
from superlocalmemory.server import hook_event_writer as hew
from superlocalmemory.storage.migrations import M075_injection_receipts as M075
from tests.test_outcomes.test_injection_receipts import _fact, _rows

T0 = datetime(2026, 11, 16, 20, 0, tzinfo=UTC)


@pytest.fixture
def store(tmp_path):
    from tests.test_core.test_coverage_backfill import _db

    return _db(tmp_path / "memory.db")


def _deliver(store, tmp_path, facts, *, session="conv-1", at=T0, query="where is the key",
             path="mcp_recall", profile="default") -> str:
    pull = receipts.Pull(tmp_path, store.db_path, profile, session, path, query,
                         tuple((f, 0.5) for f in facts), at=at)
    assert receipts.record_pull(pull)
    return _rows(store.db_path, "SELECT delivery_id FROM injection_deliveries "
                                "ORDER BY rowid DESC LIMIT 1")[0][0]


def _usage(db_path: Path) -> list[tuple]:
    return _rows(db_path, "SELECT delivery_id, fact_id, signal, value FROM injection_usage "
                          "ORDER BY observed_at, fact_id")


def _join(store, facts, *, session="conv-1", at=T0 + timedelta(minutes=1), signal="marker_tool",
          profile="default") -> int:
    conn = sqlite3.connect(str(store.db_path))
    try:
        with conn:
            return usage.join_session(conn, profile_id=profile, session_id=session,
                                      fact_ids=facts, signal=signal,
                                      observed_at=usage.stamp(at), value=None)
    finally:
        conn.close()


# -- the schema -----------------------------------------------------------------------------------

def test_a_store_from_before_w3_2_is_repaired_with_the_usage_table_and_its_erase() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY)")
    M075.apply(conn)
    conn.execute("DROP TABLE injection_usage")
    conn.execute("DROP TRIGGER trg_injection_receipts_fact_erased")
    conn.execute("CREATE TRIGGER trg_injection_receipts_fact_erased AFTER DELETE ON atomic_facts "
                 "BEGIN DELETE FROM injection_receipts WHERE fact_id = OLD.fact_id; END")
    assert not M075.verify(conn)
    M075.repair(conn)
    assert M075.verify(conn)


def test_erasing_a_fact_erases_its_usage(store, tmp_path) -> None:
    a = _fact(store, "the spare key is under the mat")
    _deliver(store, tmp_path, [a])
    assert _join(store, [a]) == 1
    conn = sqlite3.connect(str(store.db_path))
    with conn:
        conn.execute("DELETE FROM atomic_facts WHERE fact_id = ?", (a,))
    conn.close()
    assert _usage(store.db_path) == []


# -- joining by session (marker echo) -------------------------------------------------------------

def test_a_marker_joins_the_latest_delivery_of_that_fact_in_that_session(store, tmp_path) -> None:
    a = _fact(store, "the gate code is 41")
    _deliver(store, tmp_path, [a], at=T0)
    second = _deliver(store, tmp_path, [a], at=T0 + timedelta(seconds=30))
    _deliver(store, tmp_path, [a], at=T0 + timedelta(minutes=5))  # after the signal
    assert _join(store, [a]) == 1
    assert _usage(store.db_path) == [(second, a, "marker_tool", None)]


@pytest.mark.parametrize("kw", [{"session": "conv-other"}, {"profile": "work"},
                                {"at": T0 - timedelta(seconds=1)}])
def test_no_join_across_sessions_profiles_or_before_the_delivery(store, tmp_path, kw) -> None:
    a = _fact(store, "the boiler is serviced in March")
    _deliver(store, tmp_path, [a])
    assert _join(store, [a], **kw) == 0 and _usage(store.db_path) == []


@pytest.mark.parametrize("sid", ["", "http:1700000000000", "view:x"])
def test_an_invented_session_joins_nothing(store, tmp_path, sid) -> None:
    a = _fact(store, "the van is parked on level 2")
    _deliver(store, tmp_path, [a], session=sid)
    assert _join(store, [a], session=sid) == 0


def test_a_fact_that_was_not_delivered_is_never_used(store, tmp_path) -> None:
    a, b = _fact(store, "fact a"), _fact(store, "fact b")
    _deliver(store, tmp_path, [a])
    assert _join(store, [a, b]) == 1
    assert [u[1] for u in _usage(store.db_path)] == [a]


def test_a_withheld_memory_cannot_be_used(store, tmp_path) -> None:
    a = _fact(store, "fact a")
    did = _deliver(store, tmp_path, [a])
    conn = sqlite3.connect(str(store.db_path))
    with conn:
        conn.execute("UPDATE injection_receipts SET included = 0, delivered = 0, eligible = 1, "
                     "exclusion = NULL, propensity = 0.9 WHERE delivery_id = ?", (did,))
    conn.close()
    assert _join(store, [a]) == 0


def test_a_replayed_signal_adds_nothing(store, tmp_path) -> None:
    a = _fact(store, "fact a")
    _deliver(store, tmp_path, [a])
    assert _join(store, [a]) == 1 and _join(store, [a]) == 0
    assert len(_usage(store.db_path)) == 1


def test_a_store_without_the_tables_joins_nothing_and_does_not_raise() -> None:
    conn = sqlite3.connect(":memory:")
    assert usage.join_session(conn, profile_id="p", session_id="conv", fact_ids=["a"],
                              signal="marker_tool", observed_at=usage.stamp(T0)) == 0


def test_a_late_write_keeps_the_moment_the_answer_left(store, tmp_path) -> None:
    a = _fact(store, "fact a")
    _deliver(store, tmp_path, [a], at=T0)
    got = _rows(store.db_path, "SELECT delivered_at FROM injection_deliveries")[0][0]
    assert got == usage.stamp(T0)


# -- joining by question (explicit outcome) -------------------------------------------------------

def test_an_explicit_outcome_joins_the_delivery_that_answered_its_question(store,
                                                                           tmp_path) -> None:
    from superlocalmemory.learning.outcome_events import OutcomeReport, write_outcome

    a, b = _fact(store, "the invoice run is on the last working day"), _fact(store, "other")
    did = _deliver(store, tmp_path, [a], query="when is the invoice run",
                   at=datetime.now(UTC) - timedelta(minutes=1))
    _deliver(store, tmp_path, [a, b], query="something else",
             at=datetime.now(UTC) - timedelta(minutes=1))
    write_outcome(OutcomeReport(profile_id="default", fact_ids=(a,), outcome="success",
                                surface="mcp", query="when is the invoice run"),
                  memory_db_path=store.db_path, bridge_signals=False)
    assert _usage(store.db_path) == [(did, a, "explicit_success", 1.0)]


def test_an_outcome_without_its_question_joins_nothing(store, tmp_path) -> None:
    from superlocalmemory.learning.outcome_events import OutcomeReport, write_outcome

    a = _fact(store, "fact a")
    _deliver(store, tmp_path, [a], at=datetime.now(UTC) - timedelta(minutes=1))
    write_outcome(OutcomeReport(profile_id="default", fact_ids=(a,), outcome="failure",
                                surface="mcp", query=""),
                  memory_db_path=store.db_path, bridge_signals=False)
    assert _usage(store.db_path) == []


def test_no_key_means_no_digest_and_the_key_is_never_created(tmp_path) -> None:
    assert usage.query_digest(tmp_path, "q") is None
    assert not (tmp_path / "keys").exists()
    assert usage.query_digest(tmp_path, "") is None


# -- the hook writer: marker echo and requery -----------------------------------------------------

def _drain(store, tmp_path, events) -> None:
    spool = tmp_path / "spool"
    assert event_spool.append_events(events, profile="default", root=spool).dropped == 0
    hew.HookEventWriter(store.db_path, root=spool).drain_once()


def _ms(at: datetime) -> int:
    return int(at.timestamp() * 1000)


def test_a_marker_in_an_edit_and_in_another_tool_are_told_apart(store, tmp_path) -> None:
    a, b = _fact(store, "fact a"), _fact(store, "fact b")
    did = _deliver(store, tmp_path, [a, b])
    seen = T0 + timedelta(minutes=2)
    edit = ed.fact_signal(session_id="conv-1", fact_ids=[a], signal_name="edit",
                          signal_value=True, profile_id="default")
    read = ed.fact_signal(session_id="conv-1", fact_ids=[b], signal_name="dwell_ms",
                          signal_value=3000, profile_id="default")
    edit["observed_ms"] = read["observed_ms"] = _ms(seen)
    _drain(store, tmp_path, [edit, read])
    assert set(_usage(store.db_path)) == {(did, a, "marker_edit", None),
                                          (did, b, "marker_tool", 3000.0)}


def test_a_requery_joins_through_the_recall_it_was_about(store, tmp_path) -> None:
    a = _fact(store, "fact a")
    did = _deliver(store, tmp_path, [a])
    conn = sqlite3.connect(str(store.db_path))
    with conn:
        conn.execute("INSERT INTO pending_outcomes (outcome_id, profile_id, session_id, "
                     "recall_query_id, fact_ids_json, query_text_hash, created_at_ms, "
                     "expires_at_ms) VALUES ('oc1', 'default', 'conv-1', 'rq', ?, 'h', ?, ?)",
                     (json.dumps([a]), _ms(T0), _ms(T0 + timedelta(hours=1))))
    conn.close()
    signal = ed.outcome_signal(outcome_id="oc1", signal_name="requery", signal_value=True,
                               profile_id="default")
    signal["observed_ms"] = _ms(T0 + timedelta(minutes=3))
    _drain(store, tmp_path, [signal])
    assert _usage(store.db_path) == [(did, a, "requery", None)]


def test_a_requery_about_an_unknown_outcome_joins_nothing(store, tmp_path) -> None:
    signal = ed.outcome_signal(outcome_id="nope", signal_name="requery", signal_value=True,
                               profile_id="default")
    _drain(store, tmp_path, [signal])
    assert _usage(store.db_path) == []


def test_a_truncated_key_file_gives_no_digest(tmp_path) -> None:
    from superlocalmemory.core.receipt_key import KEY_FILE

    (tmp_path / "keys").mkdir()
    (tmp_path / "keys" / KEY_FILE).write_bytes(b"short")
    assert usage.query_digest(tmp_path, "q") is None


# -- end to end on a real engine ------------------------------------------------------------------

from tests.test_modes.test_mode_a_never_fails import mode_a_engine  # noqa: E402,F401


def test_a_real_recall_then_its_echoed_marker_is_usage_of_that_delivery(
        mode_a_engine, tmp_path) -> None:  # noqa: F811
    from superlocalmemory.hooks import post_tool_outcome_hook as hook
    from superlocalmemory.server.unified_daemon import EngineRecallAdapter

    mode_a_engine.store("The invoice run happens on the last working day of the month.")
    out = EngineRecallAdapter(mode_a_engine).recall("invoice run", 5, "conv-real")
    assert receipts.PULL_WRITER.drain()
    first = out["results"][0]
    assert first.get("marker"), "a session-bearing recall carries markers"
    tool_output = f"Read notes ... {first['marker']} ... done"
    hits = hook._validated_fact_ids(tool_output)
    assert hits == [first["fact_id"]]
    db = mode_a_engine.db.db_path
    spool = tmp_path / "spool"
    event_spool.append_events([ed.fact_signal(session_id="conv-real", fact_ids=hits,
                                              signal_name="edit", signal_value=True,
                                              profile_id=mode_a_engine.profile_id)],
                              profile=mode_a_engine.profile_id, root=spool)
    hew.HookEventWriter(db, root=spool).drain_once()
    (did,) = _rows(db, "SELECT delivery_id FROM injection_deliveries")[0]
    assert _usage(db) == [(did, first["fact_id"], "marker_edit", None)]
