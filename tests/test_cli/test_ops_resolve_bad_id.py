# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Issue #148: ``slm ops resolve`` with a non-ASCII / unsafe ID crashed.

``_cmd_ops_resolve`` interpolated the raw CLI argument into
``f"/operations/{operation_id}/resolve"``; ``_daemon_post`` only caught
``HTTPError``/``URLError``, so a literal U+2026 placeholder (or any
ASCII-unsafe-but-printable ID) reached ``http.client``'s request-line
encoding unescaped and raised an uncaught ``UnicodeEncodeError`` -- a raw
traceback instead of a CLI error.

Every test here runs the real CLI entry point
(``python -m superlocalmemory.cli``) as a subprocess against a local fake
daemon (random port, never 8765), exactly the way a user's shell invokes it.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import superlocalmemory
from tests._portable import child_env_base

REPO_SRC = str(Path(superlocalmemory.__file__).resolve().parents[1])


class _FakeDaemon(BaseHTTPRequestHandler):
    """Records every path it is asked to resolve and answers canned JSON."""

    seen_paths: list[str] = []
    response_body: bytes = b'{"success": true, "message": "cancelled"}'
    response_status: int = 200
    #: The identity this account's daemon proves on /health (4.1.21: the CLI
    #: sends nothing to a port until the daemon there proves it is this
    #: account's own).
    health_body: bytes = b"{}"

    def _answer(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        if self.path == "/health":
            body, status = type(self).health_body, 200
        else:
            type(self).seen_paths.append(self.path)
            body, status = type(self).response_body, type(self).response_status
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = _answer

    def log_message(self, format: str, *args: object) -> None:  # quiet
        pass


def _ephemeral_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@pytest.fixture()
def fake_daemon(tmp_path):
    from superlocalmemory.infra.daemon_identity import build_descriptor, write_descriptor

    _FakeDaemon.seen_paths = []
    _FakeDaemon.response_body = b'{"success": true, "message": "cancelled"}'
    _FakeDaemon.response_status = 200
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeDaemon)
    server.daemon_threads = True
    port = int(server.server_address[1])
    assert port != 8765, "must never bind the live daemon port"
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    descriptor = build_descriptor(data_root=data, port=port, version="test",
                                  pid=os.getpid(), state="ready")
    write_descriptor(descriptor, data_root=data)
    _FakeDaemon.health_body = json.dumps(
        {"status": "ok", **descriptor.public_health_fields()}).encode()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield port, _FakeDaemon
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _run_cli(tmp_path: Path, port: int, *args: str) -> subprocess.CompletedProcess:
    env = dict(
        os.environ,
        **child_env_base(tmp_path / "home"),
        SLM_DATA_DIR=str(tmp_path / "data"),
        SLM_DAEMON_PORT=str(port),
        PYTHONPATH=REPO_SRC,
        # The CLI auto-starts a real daemon for any command not on its
        # no-daemon allowlist; this is pytest's existing escape hatch (see
        # cli/daemon.py::ensure_daemon) so the fake daemon above is the only
        # thing on the port -- never a second, real one.
        SLM_TEST_ISOLATION="1",
    )
    env.pop("SLM_TEST_ALLOW_DAEMON_SPAWN", None)
    (tmp_path / "home").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        [sys.executable, "-m", "superlocalmemory.cli", *args],
        capture_output=True, text=True, env=env, timeout=120,
    )


def _assert_no_traceback(proc: subprocess.CompletedProcess) -> None:
    combined = proc.stdout + proc.stderr
    for line in combined.splitlines():
        assert not line.startswith("Traceback"), f"raw traceback leaked:\n{combined}"
    assert "UnicodeEncodeError" not in combined, f"raw UnicodeEncodeError leaked:\n{combined}"


class TestLiteralEllipsisId:
    """AC1: a literal U+2026 ID must fail cleanly, never crash."""

    def test_exits_non_zero_with_friendly_error(self, tmp_path, fake_daemon):
        port, handler = fake_daemon
        proc = _run_cli(tmp_path, port, "ops", "resolve", "…", "--action", "cancel")

        assert proc.returncode != 0
        _assert_no_traceback(proc)
        assert "error: invalid operation ID" in proc.stderr, proc.stderr

    def test_never_sends_the_bad_id_to_the_daemon(self, tmp_path, fake_daemon):
        port, handler = fake_daemon
        _run_cli(tmp_path, port, "ops", "resolve", "…", "--action", "cancel")

        assert handler.seen_paths == [], (
            "an invalid ID must be rejected before any request is sent"
        )


class TestAsciiUnsafeId:
    """AC2: ASCII but URL-unsafe (``'a/b c?'``) must also fail cleanly."""

    def test_exits_non_zero_with_friendly_error(self, tmp_path, fake_daemon):
        port, handler = fake_daemon
        proc = _run_cli(tmp_path, port, "ops", "resolve", "a/b c?", "--action", "cancel")

        assert proc.returncode != 0
        _assert_no_traceback(proc)
        assert "error: invalid operation ID" in proc.stderr, proc.stderr

    def test_same_exit_code_as_the_ellipsis_case(self, tmp_path, fake_daemon):
        port, _ = fake_daemon
        ellipsis_proc = _run_cli(tmp_path, port, "ops", "resolve", "…",
                                  "--action", "cancel")
        unsafe_proc = _run_cli(tmp_path, port, "ops", "resolve", "a/b c?",
                                "--action", "cancel")

        assert ellipsis_proc.returncode == unsafe_proc.returncode


class TestValidIdUnaffected:
    """AC3: a real, daemon-shaped ID must behave exactly as before the fix."""

    def test_request_path_is_byte_identical(self, tmp_path, fake_daemon):
        port, handler = fake_daemon
        valid_id = "abc123DEF_-45"  # synthetic [A-Za-z0-9_-]+ id

        proc = _run_cli(tmp_path, port, "ops", "resolve", valid_id, "--action", "cancel")

        assert proc.returncode == 0, proc.stderr
        assert handler.seen_paths == [f"/operations/{valid_id}/resolve"]

    def test_daemon_success_surfaces_as_before(self, tmp_path, fake_daemon):
        port, handler = fake_daemon
        handler.response_body = json.dumps(
            {"success": True, "message": "resolved via cancel"}
        ).encode()
        valid_id = "real-op-id-001"

        proc = _run_cli(tmp_path, port, "ops", "resolve", valid_id, "--action", "cancel")

        assert proc.returncode == 0
        assert "OK: operation" in proc.stdout
        assert "resolved via cancel" in proc.stdout

    def test_daemon_failure_surfaces_as_before(self, tmp_path, fake_daemon):
        port, handler = fake_daemon
        handler.response_body = json.dumps(
            {"success": False, "reason": "operation already resolved"}
        ).encode()
        valid_id = "real-op-id-002"

        proc = _run_cli(tmp_path, port, "ops", "resolve", valid_id, "--action", "cancel")

        assert proc.returncode != 0
        assert "Resolve failed: operation already resolved" in proc.stderr


class TestListProfileFilterBadValue:
    """``slm ops list --profile <id>`` interpolates --profile into a query
    value the same unsafe way -- found while auditing every call site for
    issue #148, not part of the original report."""

    def test_unicode_profile_filter_fails_clean(self, tmp_path, fake_daemon):
        port, handler = fake_daemon
        proc = _run_cli(tmp_path, port, "ops", "list", "--profile", "…")

        assert proc.returncode != 0
        _assert_no_traceback(proc)
        assert "error: invalid --profile value" in proc.stderr
        assert handler.seen_paths == []

    def test_valid_profile_filter_unaffected(self, tmp_path, fake_daemon):
        port, handler = fake_daemon
        handler.response_body = json.dumps({"total": 0}).encode()

        proc = _run_cli(tmp_path, port, "ops", "list", "--profile", "work-team_1")

        assert proc.returncode == 0, proc.stderr
        assert handler.seen_paths == ["/operations/failed?profile=work-team_1"]


class TestMalformedResponseNotMisreportedAsBadPath:
    """CRIT self-check: the new defense-in-depth catch in _daemon_get/
    _daemon_post must only wrap building and sending the request, never
    reading or decoding the daemon's response -- otherwise a response the
    daemon sends back broken would be misreported as "invalid request
    path", which is a different bug and not one this issue is about."""

    def test_non_utf8_response_body_is_not_reported_as_a_bad_path(
        self, tmp_path, fake_daemon
    ):
        port, handler = fake_daemon
        handler.response_body = b"\xff\xfe not valid utf-8"
        valid_id = "real-op-id-003"

        proc = _run_cli(tmp_path, port, "ops", "resolve", valid_id, "--action", "cancel")

        assert "invalid request path" not in (proc.stdout + proc.stderr)
