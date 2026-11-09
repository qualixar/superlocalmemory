# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for hooks.hook_daemon — persistent socket-based recall server.

The hook daemon eliminates Python subprocess startup (~300-500ms) by
keeping a long-lived process that Claude Code talks to via Unix socket.

HARD RULES:
  - Daemon NEVER imports MemoryEngine (memory blast risk).
  - If daemon is dead, hook falls back to subprocess (v3.4.35 behavior).
  - Daemon auto-restarts via watchdog if SLM daemon is alive.
  - Claude Code performance is NEVER impacted by daemon failure.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

#: The hook daemon serves a Unix socket. CPython on Windows has no AF_UNIX, so
#: there the daemon never starts and hooks take the socket-free path, which
#: test_without_unix_sockets_the_hook_falls_back checks on every platform.
needs_unix_socket = pytest.mark.skipif(
    getattr(socket, "AF_UNIX", None) is None,
    reason="no Unix sockets in this Python (Windows); the hook uses its socket-free path",
)


@pytest.fixture
def short_tmp(tmp_path: Path):
    """Short temp dir for Unix socket (macOS 104-char path limit)."""
    short_root = "/tmp" if os.path.isdir("/tmp") else None  # None: the OS temp dir
    d = Path(tempfile.mkdtemp(prefix="slm", dir=short_root))
    yield d
    import shutil
    shutil.rmtree(d, ignore_errors=True)


# -----------------------------------------------------------------------
# Daemon lifecycle
# -----------------------------------------------------------------------

@needs_unix_socket
def test_hook_daemon_starts_and_stops(short_tmp: Path) -> None:
    from superlocalmemory.hooks.hook_daemon import HookDaemon
    sock_path = short_tmp / "hook.sock"
    daemon = HookDaemon(sock_path=sock_path, queue_db_path=short_tmp / "q.db")
    daemon.start()
    assert daemon.running
    assert sock_path.exists()
    daemon.stop()
    assert not daemon.running
    assert not sock_path.exists()


@needs_unix_socket
def test_hook_daemon_stop_is_idempotent(short_tmp: Path) -> None:
    from superlocalmemory.hooks.hook_daemon import HookDaemon
    sock_path = short_tmp / "hook.sock"
    daemon = HookDaemon(sock_path=sock_path, queue_db_path=short_tmp / "q.db")
    daemon.start()
    daemon.stop()
    daemon.stop()


@needs_unix_socket
def test_hook_daemon_cleans_stale_socket(short_tmp: Path) -> None:
    """If a stale socket file exists from a crashed daemon, new daemon cleans it."""
    from superlocalmemory.hooks.hook_daemon import HookDaemon
    sock_path = short_tmp / "hook.sock"
    sock_path.touch()
    daemon = HookDaemon(sock_path=sock_path, queue_db_path=short_tmp / "q.db")
    daemon.start()
    assert daemon.running
    daemon.stop()


# -----------------------------------------------------------------------
# Socket communication
# -----------------------------------------------------------------------

@needs_unix_socket
def test_hook_daemon_responds_to_recall_request(short_tmp: Path) -> None:
    """Client sends recall request via socket, gets response (even empty)."""
    from superlocalmemory.hooks.hook_daemon import HookDaemon
    sock_path = short_tmp / "hook.sock"
    queue_db = short_tmp / "q.db"

    daemon = HookDaemon(sock_path=sock_path, queue_db_path=queue_db)
    daemon.start()
    time.sleep(0.1)

    try:
        # Patch mode timeout to 1s so test doesn't wait 25s for Mode B
        with patch("superlocalmemory.hooks.auto_recall_hook._get_mode_timeout", return_value=1.0), \
             patch("superlocalmemory.hooks.auto_recall_hook._detect_mode", return_value="A"):
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.settimeout(5.0)
            client.connect(str(sock_path))

            request = json.dumps({
                "prompt": "What is the recall queue?",
                "session_id": "test-sock",
            }) + "\n"
            client.sendall(request.encode("utf-8"))

            data = b""
            while b"\n" not in data:
                chunk = client.recv(4096)
                if not chunk:
                    break
                data += chunk

            response = json.loads(data.decode("utf-8").strip())
            # No QueueConsumer in test, so response is {} (timeout → empty)
            assert isinstance(response, dict)
            client.close()
    finally:
        daemon.stop()


@needs_unix_socket
def test_hook_daemon_ack_returns_empty(short_tmp: Path) -> None:
    """Ack prompts through socket return empty dict."""
    from superlocalmemory.hooks.hook_daemon import HookDaemon
    sock_path = short_tmp / "hook.sock"

    daemon = HookDaemon(sock_path=sock_path, queue_db_path=short_tmp / "q.db")
    daemon.start()
    time.sleep(0.1)

    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(5.0)
        client.connect(str(sock_path))

        request = json.dumps({"prompt": "yes", "session_id": "test"}) + "\n"
        client.sendall(request.encode("utf-8"))

        data = b""
        while b"\n" not in data:
            chunk = client.recv(4096)
            if not chunk:
                break
            data += chunk

        response = json.loads(data.decode("utf-8").strip())
        assert response == {}
        client.close()
    finally:
        daemon.stop()


# -----------------------------------------------------------------------
# Watchdog: auto-restart
# -----------------------------------------------------------------------

@needs_unix_socket
def test_ensure_hook_daemon_starts_if_not_running(short_tmp: Path) -> None:
    """ensure_hook_daemon() starts daemon if socket doesn't exist."""
    from superlocalmemory.hooks.hook_daemon import ensure_hook_daemon
    sock_path = short_tmp / "hook.sock"
    queue_db = short_tmp / "q.db"

    daemon = ensure_hook_daemon(sock_path=sock_path, queue_db_path=queue_db)
    assert daemon is not None
    assert daemon.running
    daemon.stop()


@needs_unix_socket
def test_ensure_hook_daemon_reuses_existing(short_tmp: Path) -> None:
    """ensure_hook_daemon() reuses existing running daemon."""
    from superlocalmemory.hooks.hook_daemon import HookDaemon, ensure_hook_daemon
    sock_path = short_tmp / "hook.sock"
    queue_db = short_tmp / "q.db"

    d1 = HookDaemon(sock_path=sock_path, queue_db_path=queue_db)
    d1.start()
    time.sleep(0.1)

    d2 = ensure_hook_daemon(sock_path=sock_path, queue_db_path=queue_db)
    assert d2 is None  # didn't create new — existing is fine
    d1.stop()


# -----------------------------------------------------------------------
# Fallback: subprocess when daemon is dead
# -----------------------------------------------------------------------

def test_socket_connect_to_dead_daemon_returns_none(short_tmp: Path) -> None:
    """Connecting to non-existent socket returns None (triggers fallback)."""
    from superlocalmemory.hooks.hook_daemon import try_socket_recall
    sock_path = short_tmp / "hook.sock"
    result = try_socket_recall(
        sock_path=sock_path,
        prompt="test",
        session_id="s1",
        timeout=1.0,
    )
    assert result is None


# -----------------------------------------------------------------------
# Memory safety
# -----------------------------------------------------------------------

def test_hook_daemon_never_imports_engine() -> None:
    """hook_daemon must not import MemoryEngine at module level."""
    import importlib
    import sys
    mod_name = "superlocalmemory.hooks.hook_daemon"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    mod = importlib.import_module(mod_name)
    assert not hasattr(mod, "MemoryEngine"), \
        "hook_daemon must NOT import MemoryEngine"


def test_without_unix_sockets_the_hook_falls_back(short_tmp: Path, monkeypatch) -> None:
    """Where AF_UNIX does not exist (Windows), no daemon is started and the
    socket client reports "unavailable", so the hook runs its fallback."""
    from superlocalmemory.hooks import hook_daemon

    monkeypatch.setattr(hook_daemon, "_AF_UNIX", None)
    sock_path = short_tmp / "hook.sock"
    assert hook_daemon.ensure_hook_daemon(sock_path=sock_path,
                                          queue_db_path=short_tmp / "q.db") is None
    assert hook_daemon.try_socket_recall(sock_path=sock_path, prompt="p",
                                         session_id="s", timeout=1.0) is None
    assert not sock_path.exists()
