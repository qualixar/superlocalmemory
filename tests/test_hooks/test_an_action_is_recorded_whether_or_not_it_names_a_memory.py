# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""``tool_events`` gets written by the hook that watches tools finish.

Three separate readers -- assertion mining, skill-performance mining, and the
engagement features that settle a recall against what the agent did next -- all
select from ``tool_events``. None of them writes it. The writers were an
explicit ``log_tool_event`` call and a bulk importer, neither of which runs on
an ordinary install, so the table stopped receiving invocations and every reader
downstream of it quietly went blind.

The hook already fires on the right edge with the right session, and until now
it returned at the first line of its marker check -- the branch that is taken
for the great majority of tool calls, since nothing obliges an agent to repeat a
marker back. So the action itself, which is what those readers want, was thrown
away precisely when there was no marker to distract from it.

These tests hold the separation: recording that a tool ran and deciding what a
recall was worth are different questions, and the first does not wait on the
second.

4.2.8: the hook hands the row to the daemon's writer (spooled when no daemon
answers, as here); ``drain`` plays the daemon's part.
"""

from __future__ import annotations

import io
import sqlite3
import sys

import pytest

from superlocalmemory.hooks import post_tool_outcome_hook as h
from superlocalmemory.server.hook_event_writer import HookEventWriter
from tests.helpers.hook_event_db import bootstrap


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A memory.db in an isolated data root, wired to the hook and the writer."""
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    return bootstrap(tmp_path / "memory.db")


@pytest.fixture()
def feed(monkeypatch):
    """Feed a payload to the hook in place of Claude Code's stdin JSON.

    Scoped to the test: ``read_stdin_json`` is imported into the hook's own
    namespace, so patching it there without restoring would leak into every
    test that ran afterwards.
    """
    def _feed(payload: dict) -> None:
        monkeypatch.setattr(h, "read_stdin_json", lambda: payload)
    return _feed


def _rows(db):
    HookEventWriter(db, root=db.parent / "hook_spool").drain_once()
    conn = sqlite3.connect(db)
    try:
        conn.row_factory = sqlite3.Row
        return conn.execute("SELECT * FROM tool_events").fetchall()
    finally:
        conn.close()


def test_an_action_is_recorded_even_when_no_memory_was_named(store, feed) -> None:
    """The regression: the no-marker branch is the common one, not a dead end."""
    payload = {
        "session_id": "sess-1",
        "tool_name": "Bash",
        "tool_input": {"command": "ls -la"},
        "tool_response": "total 0\ndrwxr-xr-x  2 user staff  64 Jan  1 00:00 .",
        "cwd": "/tmp/project",
    }
    feed(payload)

    outcome = h._inner_main()  # noqa: SLF001

    assert outcome == "spooled", "no marker, no daemon: the action is spooled"
    rows = _rows(store)
    assert len(rows) == 1, "the action still happened and must be recorded"
    assert rows[0]["tool_name"] == "Bash"
    assert rows[0]["session_id"] == "sess-1"
    assert rows[0]["event_type"] == "complete"
    assert rows[0]["project_path"] == "/tmp/project"
    assert "ls -la" in rows[0]["input_summary"]


def test_a_tool_that_returned_nothing_is_still_an_action(store, feed) -> None:
    """An empty response ends the marker path early; the invocation is real."""
    feed({
        "session_id": "sess-2",
        "tool_name": "Read",
        "tool_response": "",
    })

    assert h._inner_main() == "spooled"  # noqa: SLF001
    assert len(_rows(store)) == 1


def test_an_unnamed_tool_is_not_recorded(store, feed) -> None:
    """Without a tool name there is no action to attribute; write nothing."""
    feed({"session_id": "sess-3", "tool_name": "", "tool_response": "x"})

    assert h._inner_main() == "nothing"  # noqa: SLF001

    assert _rows(store) == []


def test_credentials_do_not_reach_the_stored_summary(store, feed) -> None:
    """Telemetry is durable, so a secret written into it stays written."""
    feed({
        "session_id": "sess-4", "tool_name": "Bash",
        "tool_input": {"command": "export API_KEY=sk-abcdefghijklmnop1234"},
        "tool_response": "password = hunter2seventeen",
    })
    h._inner_main()  # noqa: SLF001

    row = _rows(store)[0]
    assert "sk-abcdefghijklmnop1234" not in row["input_summary"]
    assert "[REDACTED]" in row["input_summary"]
    assert "hunter2seventeen" not in row["output_summary"]
    spooled = (store.parent / "hook_spool").glob("*")
    assert not any("hunter2seventeen" in p.read_text(errors="ignore") for p in spooled
                   if p.is_file())


def test_a_summary_cannot_grow_without_bound(store, feed) -> None:
    """A tool can return megabytes; a telemetry column should not carry them."""
    from superlocalmemory.hooks.event_delivery import MAX_SUMMARY_LEN

    feed({"session_id": "sess-5", "tool_name": "Bash",
          "tool_input": "x " * 50_000, "tool_response": "y " * 50_000})
    h._inner_main()  # noqa: SLF001

    row = _rows(store)[0]
    assert len(row["input_summary"]) <= MAX_SUMMARY_LEN
    assert len(row["output_summary"]) <= MAX_SUMMARY_LEN


def test_telemetry_failure_is_never_the_tool_call_s_problem(
    tmp_path, monkeypatch,
) -> None:
    """No daemon and an unusable spool -- and still exit 0, with a breadcrumb.

    The hook's contract is to exit 0 whatever happens. A tool call must not
    report a problem to the user because a telemetry row could not be stored,
    but the failure is said out loud (stderr + perf log), never swallowed.
    """
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    (tmp_path / "hook_spool").write_text("a file where the folder should be")
    payload = {"session_id": "sess-6", "tool_name": "Bash", "tool_response": "out"}
    monkeypatch.setattr(sys, "stdin", io.StringIO(__import__("json").dumps(payload)))
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err)
    seen: list[str] = []
    monkeypatch.setattr(h, "log_perf", lambda name, ms, outcome: seen.append(outcome))

    assert h.main() == 0
    assert seen == ["spool_failed"]
    assert "not saved" in err.getvalue()
