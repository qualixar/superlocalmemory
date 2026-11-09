# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Regression tests for the "Daemon was not running" false report (v4.1.4).

The bug, confirmed live against the real running daemon on this machine
(pid, ``daemon.pid``, and ``lsof -i :8765`` all agreed the process was up and
listening the whole time):

  ``slm serve stop`` and Step 1 of ``slm restart`` intermittently reported the
  daemon as not running while ``ps``/``lsof`` proved it was alive and
  listening. A raw ``kill -TERM <pid>`` always worked within ~5s, which rules
  out a wedged process or an ignored signal — the bug was in the CLI's own
  "is a daemon running, and can I stop it" detection, not in the daemon.

Root cause: ``/maintenance/run`` and ``/consolidate/cognitive`` run
multi-second (sometimes multi-minute) synchronous work directly inline in
their ``async def`` handlers, with no ``asyncio.to_thread``/executor offload
— unlike ``/recall`` (fixed for exactly this class of bug in v3.4.52, per its
own inline comment) and unlike the periodic full-consolidation timer loop
(``server/consolidation_runner.py``, which does use ``asyncio.to_thread``).
FastAPI/uvicorn runs a single-threaded event loop by default, so while one of
those two handlers is running, *no* request on that loop can be serviced,
``GET /health`` included.

``daemon.py``'s ``_fetch_health()`` makes exactly one HTTP attempt with a 2s
timeout and no retry, and both ``is_daemon_running()`` and
``daemon_request()`` (which ``stop_daemon()`` used unconditionally before
this fix) treated that single timeout as proof the daemon was down — even
though ``_resolve_descriptor_liveness()`` (PID + clock-independent start
token, no HTTP involved) had *already*, independently, proven the recorded
process was alive on that call path. Reproduced live: while a genuine
``POST /maintenance/run`` call was in flight against the production daemon,
15 out of 15 health polls from a separate process timed out at exactly the
2.00s cap, and ``is_daemon_running()`` returned False for the entire busy
window, while ``ps``/``lsof`` never stopped showing the process listening.

The fix, exercised below:
  - ``daemon_request(..., verify_health=False)`` — new opt-in that skips the
    ``GET /health`` identity preflight. The daemon still authenticates the
    request by its capability header on arrival (``_require_daemon_actor``),
    so this drops a redundant, stall-prone round trip, not a security check.
  - ``stop_daemon()`` now checks ``_descriptor_process_is_alive(descriptor)``
    directly (the same cheap, non-HTTP proof ``is_daemon_running()`` already
    computes) and, once that is proven, sends ``POST /stop`` with
    ``verify_health=False`` instead of letting a stalled health preflight
    veto a stop request to a process it has already proven it owns.
  - ``owned_daemon_process_alive()`` — a new, HTTP-free liveness check for
    "is there a process I need to stop," used by ``cmd_restart``'s Step 1
    instead of ``is_daemon_running()`` (which answers a different question:
    "is it healthy enough to serve"). Before this fix, a busy-but-alive
    daemon made Step 1 skip the stop entirely ("already stopped"), then Step
    3 refused to start a second daemon on the still-occupied port and the
    whole restart failed.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from unittest.mock import MagicMock, patch

from superlocalmemory.infra.daemon_identity import build_descriptor, write_descriptor
from tests._urlopen_fake import patch_urlopen


def _owned_descriptor(port: int = 43923):
    """A descriptor whose recorded PID is this very test process.

    Matches the pattern used in test_daemon_owned_discovery.py: using
    ``os.getpid()`` lets ``_descriptor_process_is_alive()`` prove liveness
    for real (PID + start token), with no psutil/process mocking needed.
    """
    root = Path(os.environ["SLM_DATA_DIR"])
    descriptor = build_descriptor(
        data_root=root,
        port=port,
        version="4.1.4",
        pid=os.getpid(),
        instance_id="owned-instance",
        capability="owned-capability",
        state="ready",
    )
    write_descriptor(descriptor, data_root=root)
    return descriptor


class _AlwaysTimesOut:
    """Stand-in for a busy event loop: every GET /health attempt times out."""

    def __call__(self, *args, **kwargs):
        raise socket.timeout("timed out")


class TestBusyHealthIsNotDeath:
    """The two liveness questions the bug conflated, and their new split."""

    def test_is_daemon_running_false_positive_while_busy(self) -> None:
        """Characterizes the bug: unchanged is_daemon_running() still says
        False the moment /health can't be reached, even though the process
        (proven below) never stopped. This function intentionally keeps that
        HTTP-readiness contract — callers that need "should I try to stop
        this" must use owned_daemon_process_alive() instead (next test)."""
        from superlocalmemory.cli import daemon

        _owned_descriptor()
        with patch_urlopen(side_effect=_AlwaysTimesOut()):
            assert daemon.is_daemon_running() is False

    def test_owned_daemon_process_alive_survives_busy_health(self) -> None:
        """The fix: process liveness does not depend on /health answering."""
        from superlocalmemory.cli import daemon

        _owned_descriptor()
        with patch_urlopen(side_effect=_AlwaysTimesOut()) as request:
            assert daemon.owned_daemon_process_alive() is True
        # Never even tried HTTP -- liveness came from the PID/start-token
        # check alone, so it cannot be starved by a busy event loop.
        request.assert_not_called()

    def test_owned_daemon_process_alive_false_when_process_is_gone(self) -> None:
        """Negative control: a genuinely dead process must still read False."""
        from superlocalmemory.cli import daemon

        _owned_descriptor()
        with patch.object(daemon, "_descriptor_process_is_alive", return_value=False):
            assert daemon.owned_daemon_process_alive() is False


class TestDaemonRequestVerifyHealthFlag:
    def test_default_behavior_is_unchanged(self) -> None:
        """verify_health defaults True: every existing caller keeps preflighting."""
        from superlocalmemory.cli import daemon

        _owned_descriptor()
        with patch_urlopen(side_effect=_AlwaysTimesOut()) as request:
            assert daemon.daemon_request("POST", "/stop") is None
        assert request.call_count == 1  # the (failed) health preflight only

    def test_verify_health_false_skips_preflight_and_sends_the_request(self) -> None:
        from superlocalmemory.cli import daemon

        descriptor = _owned_descriptor()
        # daemon_request() swallows any exception raised while sending the
        # request (including one raised by a broken test double), so record
        # what was actually sent and assert on it AFTER the call rather than
        # asserting inside the side_effect itself -- an assertion in there
        # would silently turn into "result is None" instead of a clear
        # failure.
        seen_urls: list[str] = []
        seen_capability: list[str | None] = []

        class _StopResponse:
            status = 200

            def read(self) -> bytes:
                import json
                return json.dumps({"status": "stopping"}).encode()

        def _urlopen(req, timeout=None):
            seen_urls.append(req.full_url)
            # Request.add_header() stores keys via str.capitalize() (see
            # cpython urllib.request.Request), and get_header() does not
            # re-normalize its argument, so the lookup key must already be
            # in that stored form.
            seen_capability.append(req.get_header("X-slm-daemon-capability"))
            return _StopResponse()

        with patch_urlopen(side_effect=_urlopen):
            result = daemon.daemon_request(
                "POST", "/stop",
                expected_descriptor=descriptor,
                verify_health=False,
            )
        assert result == {"status": "stopping"}
        # Exactly one HTTP call: the real request. No /health round trip.
        assert seen_urls == [f"http://127.0.0.1:{descriptor.port}/stop"]
        assert seen_capability == [descriptor.capability]


class TestStopDaemonSurvivesBusyHealth:
    """End-to-end for stop_daemon() itself -- the code slm serve stop calls."""

    def test_stop_succeeds_when_health_preflight_would_have_timed_out(self) -> None:
        """Reproduces the field bug in miniature: /health never answers, but
        the daemon is alive (real PID) and /stop itself succeeds. Before this
        fix, stop_daemon() called daemon_request() with its default
        verify_health=True, so the /health timeout alone made it return None
        and stop_daemon() reported failure -- "Daemon was not running" --
        without ever sending /stop."""
        from superlocalmemory.cli import daemon

        descriptor = _owned_descriptor()

        class _StopResponse:
            status = 200

            def read(self) -> bytes:
                import json
                return json.dumps({"status": "stopping"}).encode()

        def _urlopen(req_or_url, timeout=None):
            # _fetch_health() calls urlopen(url: str, ...); the real mutating
            # request calls urlopen(Request, ...). If stop_daemon() regresses
            # to preflighting health again, either shape must be hit here for
            # GET /health and must fail loudly instead of silently degrading
            # back to the old broken behavior.
            url = req_or_url if isinstance(req_or_url, str) else req_or_url.full_url
            if url.endswith("/health"):
                raise socket.timeout("timed out")
            assert not isinstance(req_or_url, str), "expected a POST Request, got a bare URL"
            assert url.endswith("/stop")
            return _StopResponse()

        with patch_urlopen(side_effect=_urlopen) as request, \
             patch.object(daemon, "wait_for_owned_daemon_shutdown", return_value=True) as wait:
            assert daemon.stop_daemon() is True

        wait.assert_called_once_with(descriptor)
        # The only HTTP call made was the POST /stop -- no /health preflight.
        assert request.call_count == 1
        assert request.call_args.args[0].get_method() == "POST"

    def test_stop_returns_false_without_a_network_call_when_process_is_dead(self) -> None:
        """A descriptor whose process has actually exited must not attempt a
        stop at all -- that would send a live capability token toward
        whatever, if anything, now holds the port."""
        from superlocalmemory.cli import daemon

        _owned_descriptor()
        with patch.object(daemon, "_descriptor_process_is_alive", return_value=False), \
             patch_urlopen() as request:
            assert daemon.stop_daemon() is False
        request.assert_not_called()


class TestRestartStep1UsesProcessLivenessNotHealthReadiness:
    """cmd_restart's Step 1 must not skip stopping a busy-but-alive daemon."""

    def test_step1_stops_a_busy_daemon_instead_of_reporting_already_stopped(self) -> None:
        from argparse import Namespace

        from superlocalmemory.cli import commands, daemon

        descriptor = _owned_descriptor()
        stop_calls = MagicMock(return_value=True)

        with patch.object(daemon, "owned_daemon_process_alive", return_value=True), \
             patch.object(daemon, "read_descriptor", return_value=descriptor), \
             patch.object(daemon, "stop_daemon", stop_calls), \
             patch.object(daemon, "wait_for_owned_daemon_shutdown", return_value=True), \
             patch.object(daemon, "_start_daemon_subprocess", return_value=True), \
             patch.object(daemon, "daemon_request", return_value={
                 "engine": "initialized", "version": "4.1.4", "pid": os.getpid(),
             }), \
             patch("time.sleep"):
            commands.cmd_restart(Namespace(json=True, dashboard=False))

        # Before the fix this was never called: owned_daemon_process_alive()
        # is what decides "was_running" now, not the HTTP-readiness check.
        stop_calls.assert_called_once()

    def test_step1_skips_stop_cleanly_when_nothing_is_running(self) -> None:
        """No process at all: Step 1 must still report success, not a
        failure, matching the documented "already stopped" happy path."""
        from argparse import Namespace

        from superlocalmemory.cli import commands, daemon

        stop_calls = MagicMock(return_value=True)

        with patch.object(daemon, "owned_daemon_process_alive", return_value=False), \
             patch.object(daemon, "stop_daemon", stop_calls), \
             patch.object(daemon, "_start_daemon_subprocess", return_value=True), \
             patch.object(daemon, "daemon_request", return_value={
                 "engine": "initialized", "version": "4.1.4", "pid": os.getpid(),
             }), \
             patch("time.sleep"):
            commands.cmd_restart(Namespace(json=True, dashboard=False))

        stop_calls.assert_not_called()
