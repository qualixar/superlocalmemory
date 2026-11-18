# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The shell door (WS-5, package B): a save with no agent and no terminal is not the user's.

Package A left a known gap: an agent running ``slm remember`` through a shell has no agent
id and was classed ``user_direct``. The client now says whether it sat at a terminal; the
daemon believes that only from a capability holder and records it as the raw signal
``agent_id_source``; the mapping reads it.
"""

from __future__ import annotations

import io

import pytest

from superlocalmemory.core import agent_attribution as aa
from superlocalmemory.trust import fact_trust
from tests.test_server.test_canonical_remember_route import _client

CAP = "daemon-capability:abc"


class _Stream(io.StringIO):
    def __init__(self, tty: bool) -> None:
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def _streams(monkeypatch, stdin: bool, stdout: bool) -> None:
    monkeypatch.setattr("sys.stdin", _Stream(stdin))
    monkeypatch.setattr("sys.stdout", _Stream(stdout))


# -- the client says whether it had a terminal ----------------------------------------------------

def test_a_terminal_on_both_ends_is_tty(monkeypatch) -> None:
    _streams(monkeypatch, True, True)
    assert aa.shell_headers("cap") == {aa.SHELL_HEADER: "tty"}


@pytest.mark.parametrize("stdin,stdout", [(False, True), (True, False), (False, False)])
def test_anything_less_is_no_tty(monkeypatch, stdin, stdout) -> None:
    _streams(monkeypatch, stdin, stdout)
    assert aa.shell_headers("cap") == {aa.SHELL_HEADER: "no_tty"}


def test_a_closed_stream_is_no_tty_not_a_crash(monkeypatch) -> None:
    closed = _Stream(True)
    closed.close()
    monkeypatch.setattr("sys.stdin", closed)  # isatty() still answers; make it raise
    monkeypatch.setattr(closed, "isatty", lambda: (_ for _ in ()).throw(ValueError("closed")))
    assert aa.shell_headers("cap") == {aa.SHELL_HEADER: "no_tty"}


def test_no_capability_sends_nothing() -> None:
    assert aa.shell_headers(None) == {}


# -- the daemon believes it only from a capability holder -----------------------------------------

@pytest.mark.parametrize("value,expected", [("tty", "shell:tty"), ("no_tty", "shell:no_tty"),
                                            (" NO_TTY ", "shell:no_tty"), ("", None),
                                            ("person", None)])
def test_a_capability_holders_header_is_read(value, expected) -> None:
    assert aa.shell_signal(value, CAP) == expected


@pytest.mark.parametrize("actor", ["", "dashboard:x", "local-capability:cli:1", None])
def test_anyone_else_says_nothing(actor) -> None:
    assert aa.shell_signal("tty", actor) is None


def test_the_signal_is_recorded_only_when_the_save_names_no_agent() -> None:
    assert aa.bind({}, "", "shell:no_tty")["agent_id_source"] == "shell:no_tty"
    assert aa.bind({"agent_id": "claude"}, "", "shell:no_tty")["agent_id_source"] == "claimed"
    assert aa.bind({}, "codex", "shell:no_tty")["agent_id_source"] == "connection"
    assert "agent_id_source" not in aa.bind({}, "", None)


def test_a_caller_cannot_set_the_signal_through_metadata() -> None:
    assert "agent_id_source" not in aa.bind({"agent_id_source": "shell:tty"}, "")


# -- end to end through POST /remember ------------------------------------------------------------

def _saved(engine, headers, text, key):
    with _client(engine) as client:
        r = client.post("/remember?wait=true", headers=headers,
                        json={"content": text, "idempotency_key": key})
        assert r.status_code == 200, r.text[:300]
    # The memory's own whole-text fact carries the request's signals; the facts derived from it
    # by the extractor record the channel and the shell signal but no role (so they never read
    # as the user's own: package A, fail closed).
    rows = engine._db.execute(
        "SELECT o.fact_id, o.agent_id_source, o.agent_id, o.role FROM fact_origin o "
        "WHERE o.transform IN ('verbatim', 'excerpt') AND o.fact_id IN "
        "(SELECT fact_id FROM atomic_facts WHERE memory_id = (SELECT memory_id FROM atomic_facts "
        "WHERE fact_id = ?))", (r.json()["fact_ids"][0],))
    assert rows, "the whole-text fact was not recorded"
    return rows[0]["fact_id"], dict(rows[0])


def test_a_shell_without_a_terminal_is_not_the_user(engine_with_mock_deps) -> None:
    engine = engine_with_mock_deps
    fid, row = _saved(engine, {aa.SHELL_HEADER: "no_tty"},
                      "The staging database moved to the new cluster.", "shell-no-tty")
    assert row["agent_id_source"] == "shell:no_tty" and not row["agent_id"]
    assert fact_trust.origin_of(_conn(engine), fid)["class"] == "agent_inferred"


def _conn(engine):
    import sqlite3

    conn = sqlite3.connect(str(engine._db.db_path))
    conn.row_factory = sqlite3.Row
    return conn


def test_a_terminal_and_the_dashboard_stay_the_users_own(engine_with_mock_deps) -> None:
    engine = engine_with_mock_deps
    at_terminal, row = _saved(engine, {aa.SHELL_HEADER: "tty"},
                              "My editor of choice is Helix for everything.", "shell-tty")
    assert row["agent_id_source"] == "shell:tty"
    silent, row2 = _saved(engine, {}, "I prefer the dark theme in every tool.", "no-signal")
    assert row2["agent_id_source"] is None
    conn = _conn(engine)
    assert fact_trust.origin_of(conn, at_terminal)["class"] == "user_direct"
    assert fact_trust.origin_of(conn, silent)["class"] == "user_direct"


def test_the_idempotency_key_tells_the_two_apart(monkeypatch) -> None:
    from superlocalmemory.cli.commands import _cli_remember_idempotency_key as key

    body = {"content": "same words", "tags": ""}
    _streams(monkeypatch, True, True)
    at_terminal = key(body)
    _streams(monkeypatch, False, False)
    assert key(body) != at_terminal and key(body).startswith("cli:")
    _streams(monkeypatch, True, True)
    assert key(body) == at_terminal  # a terminal keeps the key it always had
