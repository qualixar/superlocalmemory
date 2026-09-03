# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Port-conflict contracts for issue #132's false-ready daemon.

A foreign service answering HTTP on the daemon port must fail fast and
loud — never a 30 s silent wait, never a false success. A silent port
(raw squat, indistinguishable from a slow starter) keeps the legacy
wait. All ports are kernel-assigned ephemeral; production ports are
never bound (the root conftest denies them by audit hook).
"""
from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from superlocalmemory.cli import daemon as daemon_mod


@pytest.fixture
def isolated_root(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "slm-data"))
    monkeypatch.setenv("SLM_TEST_ALLOW_DAEMON_SPAWN", "1")
    monkeypatch.delenv("SLM_TEST_ISOLATION", raising=False)
    return tmp_path


def _ephemeral_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


class _ForeignHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - stdlib handler naming
        body = json.dumps({"ok": True, "service": "not-slm"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # silence test output
        pass


@pytest.fixture
def foreign_http_port():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ForeignHandler)
    server.daemon_threads = True
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield port
    server.shutdown()


def test_foreign_http_occupant_fails_fast_and_loud(
    isolated_root, foreign_http_port, caplog
) -> None:
    """#132: answering squatter — fast False plus a loud error, no 30 s wait."""
    started = time.monotonic()
    with caplog.at_level("ERROR", logger="superlocalmemory.cli.daemon"):
        assert daemon_mod.ensure_daemon(port=foreign_http_port) is False
    elapsed = time.monotonic() - started
    assert elapsed < 12.0, f"foreign occupant burned {elapsed:.1f}s"
    assert "foreign service" in caplog.text
    assert str(foreign_http_port) in caplog.text


def test_configured_port_is_probed_not_the_default(
    isolated_root, monkeypatch
) -> None:
    """#132: the TCP probe follows the configured port (was _DEFAULT_PORT)."""
    probed: list[int] = []
    real_has_listener = daemon_mod._has_tcp_listener

    def recording_probe(port: int) -> bool:
        probed.append(port)
        return False

    monkeypatch.setattr(daemon_mod, "_has_tcp_listener", recording_probe)
    monkeypatch.setattr(daemon_mod, "_start_daemon_subprocess", lambda: True)

    custom = _ephemeral_port()
    assert daemon_mod.ensure_daemon(port=custom) is True
    assert probed == [custom], probed
    assert real_has_listener  # silence linters about unused import shape


def test_silent_port_keeps_legacy_wait(isolated_root, monkeypatch) -> None:
    """#132: nothing answering stays ambiguous — the systemd wait is kept."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = int(listener.getsockname()[1])
    waited: list[int] = []
    monkeypatch.setattr(
        daemon_mod, "_wait_for_daemon", lambda timeout=60: waited.append(timeout) or True
    )
    try:
        assert daemon_mod.ensure_daemon(port=port) is True
    finally:
        listener.close()
    assert waited == [30], waited
