# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The daemon's hook event writer: the only thing that writes hook events.

It drains the spool into memory.db exactly once per event: a batch's rows and
the drain cursor commit in one transaction, duplicate event ids (a hook that
gave up on a slow daemon and spooled the same events itself) are skipped, and
a failing database leaves the spool untouched for the next pass.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from superlocalmemory.hooks import event_delivery as ed
from superlocalmemory.hooks import event_spool
from superlocalmemory.server import hook_event_writer as hew
from tests.helpers.hook_event_db import bootstrap, query, seed_pending


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    return bootstrap(tmp_path / "memory.db")


@pytest.fixture()
def spool(tmp_path: Path) -> Path:
    return tmp_path / "hook_spool"


def _tool(n: int, session: str = "s1") -> dict:
    return ed.tool_event(session_id=session, tool_name=f"T{n}", profile_id="default",
                         project_path="/p", tool_input={"n": n}, response_text="ok")


def _spool(spool: Path, events: list[dict], profile: str = "default") -> None:
    assert event_spool.append_events(events, profile=profile, root=spool).dropped == 0


def _rows(db: Path) -> list[dict]:
    return query(db, "SELECT * FROM tool_events ORDER BY id")


def test_drain_writes_each_event_once_and_empties_the_spool(db: Path, spool: Path) -> None:
    events = [_tool(i) for i in range(5)]
    _spool(spool, events)
    writer = hew.HookEventWriter(db, root=spool)
    report = writer.drain_once()
    assert report.applied == 5
    rows = _rows(db)
    assert [r["tool_name"] for r in rows] == [f"T{i}" for i in range(5)]
    assert {json.loads(r["metadata"])["event_id"] for r in rows} == {
        e["event_id"] for e in events}
    assert rows[0]["profile_id"] == "default" and rows[0]["project_path"] == "/p"
    assert event_spool.read_stats(spool).pending_events == 0
    assert writer.drain_once().applied == 0
    assert len(_rows(db)) == 5
    assert not query(db, "SELECT * FROM config WHERE key LIKE 'hook_spool_cursor:%'")


def test_duplicate_event_ids_are_written_once(db: Path, spool: Path) -> None:
    events = [_tool(i) for i in range(3)]
    _spool(spool, events)
    writer = hew.HookEventWriter(db, root=spool)
    writer.drain_once()
    _spool(spool, events[1:])  # the hook's own copy after a late daemon ack
    writer.drain_once()
    assert len(_rows(db)) == 3


def test_a_restarted_writer_still_skips_events_already_written(db: Path, spool: Path) -> None:
    events = [_tool(i) for i in range(3)]
    _spool(spool, events)
    hew.HookEventWriter(db, root=spool).drain_once()
    _spool(spool, events)
    hew.HookEventWriter(db, root=spool).drain_once()  # fresh process memory
    assert len(_rows(db)) == 3


def test_a_drain_that_dies_mid_file_resumes_without_loss_or_duplicates(
    db: Path, spool: Path, monkeypatch,
) -> None:
    events = [_tool(i) for i in range(7)]
    _spool(spool, events)
    writer = hew.HookEventWriter(db, root=spool, batch_size=3)
    real = hew._insert_tool_events
    calls = {"n": 0}

    def dies_on_second_batch(conn, batch):
        calls["n"] += 1
        if calls["n"] == 2:
            raise sqlite3.OperationalError("database is locked")
        return real(conn, batch)

    monkeypatch.setattr(hew, "_insert_tool_events", dies_on_second_batch)
    with pytest.raises(sqlite3.OperationalError):
        writer.drain_once()
    assert len(_rows(db)) == 3  # first batch committed with its cursor
    monkeypatch.setattr(hew, "_insert_tool_events", real)
    hew.HookEventWriter(db, root=spool, batch_size=3).drain_once()
    assert [r["tool_name"] for r in _rows(db)] == [f"T{i}" for i in range(7)]
    assert event_spool.read_stats(spool).pending_events == 0


def test_a_missing_table_keeps_every_event_for_the_next_pass(tmp_path: Path, spool: Path) -> None:
    db = tmp_path / "empty.db"
    sqlite3.connect(db).close()
    _spool(spool, [_tool(1), _tool(2)])
    writer = hew.HookEventWriter(db, root=spool)
    with pytest.raises(sqlite3.OperationalError):
        writer.drain_once()
    assert event_spool.read_stats(spool).pending_events == 2
    bootstrap(db)
    assert writer.drain_once().applied == 2


def test_unreadable_lines_are_counted_and_the_rest_applied(db: Path, spool: Path) -> None:
    _spool(spool, [_tool(1)])
    with (spool / "default.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("not json\n")
        fh.write(json.dumps({"kind": "nope", "event_id": "a" * 32}) + "\n")
        fh.write(json.dumps({**_tool(9), "session_id": ""}) + "\n")
    _spool(spool, [_tool(2)])
    report = hew.HookEventWriter(db, root=spool).drain_once()
    assert (report.applied, report.invalid) == (2, 3)
    assert event_spool.read_stats(spool).invalid == 3


def test_fact_signal_lands_on_the_matching_pending_outcome(db: Path, spool: Path) -> None:
    seed_pending(db, outcome_id="o1", session_id="s1", fact_ids=["f1", "f2"])
    seed_pending(db, outcome_id="o2", session_id="s1", fact_ids=["f9"])
    _spool(spool, [ed.fact_signal(session_id="s1", fact_ids=["f2"], signal_name="edit",
                                  signal_value=True, profile_id="default")])
    hew.HookEventWriter(db, root=spool).drain_once()
    signals = {r["outcome_id"]: json.loads(r["signals_json"])
               for r in query(db, "SELECT * FROM pending_outcomes")}
    assert signals == {"o1": {"edit": True}, "o2": {}}


def test_a_signal_is_judged_by_when_it_happened_not_when_it_was_drained(
    db: Path, spool: Path,
) -> None:
    # The window closed while the event waited in the spool (daemon down).
    seed_pending(db, outcome_id="o1", session_id="s1", fact_ids=["f1"], expires_in_ms=-1000)
    event = ed.outcome_signal(outcome_id="o1", signal_name="requery", signal_value=True,
                              profile_id="default")
    _spool(spool, [{**event, "observed_ms": int(time.time() * 1000) - 5000}])
    hew.HookEventWriter(db, root=spool).drain_once()
    (row,) = query(db, "SELECT signals_json FROM pending_outcomes")
    assert json.loads(row["signals_json"]) == {"requery": True}


def test_session_finalize_settles_after_the_signals_before_it(db: Path, spool: Path) -> None:
    seed_pending(db, outcome_id="o1", session_id="s1", fact_ids=["f1"])
    _spool(spool, [
        ed.fact_signal(session_id="s1", fact_ids=["f1"], signal_name="edit",
                       signal_value=True, profile_id="default"),
        ed.session_finalize(session_id="s1", profile_id="default"),
    ])
    hew.HookEventWriter(db, root=spool).drain_once()
    (pending,) = query(db, "SELECT status FROM pending_outcomes")
    (action,) = query(db, "SELECT profile_id, reward FROM action_outcomes")
    assert pending["status"] == "settled"
    assert action["profile_id"] == "default" and action["reward"] is not None


def test_background_thread_drains_on_wake_and_stops_cleanly(db: Path, spool: Path) -> None:
    writer = hew.HookEventWriter(db, root=spool, interval_s=30.0)
    writer.start()
    try:
        _spool(spool, [_tool(1)])
        writer.wake()
        deadline = time.monotonic() + 10
        while not _rows(db) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert len(_rows(db)) == 1
    finally:
        assert writer.stop(timeout_s=5.0)


def test_stop_drains_what_is_left(db: Path, spool: Path) -> None:
    writer = hew.HookEventWriter(db, root=spool, interval_s=30.0)
    writer.start()
    _spool(spool, [_tool(1), _tool(2)])
    assert writer.stop(timeout_s=5.0)
    assert len(_rows(db)) == 2


def test_erasing_a_profile_removes_its_spooled_events_only(db: Path, spool: Path) -> None:
    """GDPR erase: a profile's events waiting in the spool (tool summaries) are
    deleted before the row sweep, so the writer cannot bring them back."""
    _spool(spool, [_tool(1), _tool(2)], profile="work")
    event_spool.claim_ready(spool)  # one claimed file, as mid-drain
    _spool(spool, [_tool(3)], profile="work")
    _spool(spool, [_tool(4)], profile="default")
    removed = hew.erase_profile_spool("work", db_path=db, root=spool)
    assert removed == 3
    assert not list(spool.glob("work.*"))
    hew.HookEventWriter(db, root=spool).drain_once()
    assert [r["tool_name"] for r in _rows(db)] == ["T4"]


def test_gdpr_sidecar_erasure_reaches_the_spool(db: Path, tmp_path: Path, monkeypatch) -> None:
    from superlocalmemory.compliance.erasure_sidecars import erase_profile_sidecars

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    _spool(tmp_path / "hook_spool", [_tool(1)], profile="work")
    counts: dict = {}
    erase_profile_sidecars(tmp_path, db, "work", counts)
    assert counts.get("hook_spool") == 1 and "hook_spool_failed" not in counts
    assert not list((tmp_path / "hook_spool").glob("work.*"))
