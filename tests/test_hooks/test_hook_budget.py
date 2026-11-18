# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A-15 (Fable audit): the per-prompt hook keeps ONE total budget under its host limit.

When the daemon's socket did not answer in five seconds, the hook used to enqueue a second recall
and wait five more, then try HTTP for two and a half: 12.5 s against a 7 s host limit, and a
duplicate recall piled onto a daemon that was already stuck. Now a socket that exists but timed
out ends the hook with ``{}``; the fallback runs only when the socket is absent (or failed at
once), and only inside what is left of the one budget.
"""

from __future__ import annotations

import io
import json
import sys

import pytest

from superlocalmemory.hooks import auto_recall_hook as hook
from superlocalmemory.hooks import hook_daemon

HOST_LIMIT_S = 7.0  # hooks.json: the UserPromptSubmit hook is killed here


class World:
    """Fakes for the socket, the queue and HTTP, on a clock that only the fakes advance."""

    def __init__(self, monkeypatch, socket_outcome, *, queue_answers=False, http_answers=False):
        self.now = 0.0
        self.calls: list[tuple[str, float]] = []
        self.enqueued: list[dict] = []
        self.socket_outcome = socket_outcome
        self.queue_answers, self.http_answers = queue_answers, http_answers
        self.reports_timeout_early = False  # "timeout" with budget still left
        self.queue_fast_fail = False  # the queue file cannot be opened: no waiting
        monkeypatch.setattr(hook, "_clock", lambda: self.now)
        monkeypatch.setattr(hook, "_switched_off", lambda: False)
        self.startup_s = 0.0  # time the hook has already used before it reaches the socket

        def per_prompt_off():
            self.now += self.startup_s
            return False

        monkeypatch.setattr(hook, "_per_prompt_off", per_prompt_off)
        monkeypatch.setattr(hook, "_detect_mode", lambda: "A")
        monkeypatch.setattr(hook, "_format_envelope", lambda results, response=None: {
            "hookSpecificOutput": {"additionalContext": f"{len(results)} memories"}})
        monkeypatch.setattr(hook_daemon, "socket_recall", self._socket)
        monkeypatch.setattr("superlocalmemory.core.recall_queue.RecallQueue", self._queue_class())
        monkeypatch.setattr("superlocalmemory.cli.daemon.daemon_request", self._http)

    def _socket(self, sock_path=None, prompt="", session_id="", timeout=15.0):
        outcome = self.socket_outcome
        self.calls.append(("socket", timeout))
        if outcome == hook_daemon.SOCKET_TIMEOUT and not self.reports_timeout_early:
            self.now += timeout
        return None, outcome

    def _http(self, method, path, timeout_seconds=None, **kw):
        self.calls.append(("http", timeout_seconds))
        self.now += timeout_seconds
        return {"results": [{"fact_id": "f", "content": "x", "score": 1.0}]} if self.http_answers else None

    def _queue_class(self):
        world = self

        class Queue:
            def __init__(self, *a, **k): pass

            def enqueue(self, **kw):
                world.enqueued.append(kw)
                world.calls.append(("queue enqueue", kw["stall_timeout_s"]))
                return "r"

            def poll_result(self, request_id, timeout_s):
                world.calls.append(("queue wait", timeout_s))
                if world.queue_fast_fail:
                    raise RuntimeError("queue unavailable")
                world.now += timeout_s
                if world.queue_answers:
                    return {"ok": True, "results": [{"fact_id": "f", "content": "x", "score": 1.0}]}
                raise RuntimeError("timed out")

            def close(self): pass

        return Queue

    def run(self, monkeypatch) -> dict:
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(
            {"prompt": "what did we decide about the release date?", "session_id": "s"})))
        out = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        hook.main()
        return json.loads(out.getvalue() or "{}")

    def waited(self) -> float:
        """Seconds of simulated time the hook spent waiting (the clock only the fakes advance)."""
        return self.now - self.startup_s

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]


def test_the_probe_a_socket_that_timed_out_ends_the_hook(monkeypatch) -> None:
    w = World(monkeypatch, hook_daemon.SOCKET_TIMEOUT)
    assert w.run(monkeypatch) == {}
    assert w.names() == ["socket"]  # no second recall, no HTTP: the daemon is busy
    assert w.waited() < HOST_LIMIT_S and w.waited() == hook.CLIENT_BUDGET_S


def test_an_absent_socket_uses_the_fallback_inside_one_budget(monkeypatch) -> None:
    w = World(monkeypatch, hook_daemon.SOCKET_ABSENT, queue_answers=True)
    out = w.run(monkeypatch)
    assert w.names() == ["socket", "queue enqueue", "queue wait"]
    assert "hookSpecificOutput" in out and w.waited() < HOST_LIMIT_S


def test_the_worst_case_with_every_fake_slow_stays_under_the_host_limit(monkeypatch) -> None:
    w = World(monkeypatch, hook_daemon.SOCKET_ABSENT)  # queue times out; HTTP would be tried
    assert w.run(monkeypatch) == {}
    assert w.waited() <= hook.CLIENT_BUDGET_S < HOST_LIMIT_S
    assert w.names().count("queue enqueue") == 1


def test_a_socket_that_failed_at_once_still_gets_the_fallback_in_what_is_left(monkeypatch) -> None:
    w = World(monkeypatch, hook_daemon.SOCKET_FAILED, http_answers=True)
    w.queue_fast_fail = True
    out = w.run(monkeypatch)
    assert "hookSpecificOutput" in out and w.names()[-1] == "http" and w.waited() < HOST_LIMIT_S


def test_each_wait_is_cut_to_the_remaining_budget(monkeypatch) -> None:
    w = World(monkeypatch, hook_daemon.SOCKET_ABSENT)
    w.startup_s = 3.5  # a slow start: only 1.5 s of the 5 s remain
    w.run(monkeypatch)
    assert max(t for name, t in w.calls if name != "queue enqueue") <= 1.5 + 1e-9


def test_nothing_is_tried_once_the_budget_is_spent(monkeypatch) -> None:
    w = World(monkeypatch, hook_daemon.SOCKET_ABSENT)
    w.startup_s = hook.CLIENT_BUDGET_S  # already used up
    assert w.run(monkeypatch) == {}
    assert w.names() == []  # not even the socket: there is no time to wait for it


def test_the_queue_request_carries_the_clients_deadline(monkeypatch) -> None:
    """A-14: what the hook stops waiting for is never recorded as delivered."""
    w = World(monkeypatch, hook_daemon.SOCKET_ABSENT, queue_answers=True)
    w.run(monkeypatch)
    (request,) = w.enqueued
    assert request["deadline_s"] == request["stall_timeout_s"] > 0


def test_a_missing_socket_file_is_absent_and_a_dead_one_fails_at_once(tmp_path) -> None:
    assert hook_daemon.socket_recall(sock_path=tmp_path / "none.sock", prompt="p")[1] == (
        hook_daemon.SOCKET_ABSENT)


def test_a_listener_that_never_answers_is_a_timeout(tmp_path, monkeypatch) -> None:
    import socket as socketlib
    from pathlib import Path

    monkeypatch.chdir(tmp_path)  # a short relative name: AF_UNIX paths are capped near 100 bytes
    path = Path("busy.sock")
    server = socketlib.socket(socketlib.AF_UNIX, socketlib.SOCK_STREAM)
    server.bind(str(path))
    server.listen(1)  # accepts the connection (backlog) and never replies
    try:
        resp, outcome = hook_daemon._socket_request_ex(path, {"prompt": "p"}, 0.2)
    finally:
        server.close()
    assert resp is None and outcome == hook_daemon.SOCKET_TIMEOUT


def test_a_refused_connection_is_a_quick_failure_not_a_timeout(tmp_path, monkeypatch) -> None:
    import socket as socketlib
    from pathlib import Path

    monkeypatch.chdir(tmp_path)
    path = Path("stale.sock")
    server = socketlib.socket(socketlib.AF_UNIX, socketlib.SOCK_STREAM)
    server.bind(str(path))
    server.close()  # the file stays, nothing listens
    resp, outcome = hook_daemon._socket_request_ex(path, {"prompt": "p"}, 0.2)
    assert resp is None and outcome == hook_daemon.SOCKET_FAILED


def test_a_timeout_reported_early_still_ends_the_hook(monkeypatch) -> None:
    """The socket said "timeout" with budget to spare (a slow connect or read): still busy."""
    w = World(monkeypatch, hook_daemon.SOCKET_TIMEOUT)
    w.reports_timeout_early = True
    assert w.run(monkeypatch) == {} and w.names() == ["socket"]


def test_the_http_wait_is_cut_to_what_is_left(monkeypatch) -> None:
    w = World(monkeypatch, hook_daemon.SOCKET_ABSENT)
    w.queue_fast_fail = True
    w.startup_s = 3.5
    w.run(monkeypatch)
    assert ("http", pytest.approx(1.5)) in w.calls


def test_the_queue_path_does_nothing_with_no_budget_left(monkeypatch) -> None:
    w = World(monkeypatch, hook_daemon.SOCKET_ABSENT)
    spent = hook._Budget(0.0)
    assert hook._do_recall_response("what did we decide", 5, "s", budget=spent) is None
    assert w.names() == []
