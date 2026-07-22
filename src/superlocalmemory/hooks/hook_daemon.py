# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Persistent hook daemon — Unix socket server for sub-200ms recall.

Eliminates Python subprocess startup (~300-500ms) by keeping a long-lived
process that Claude Code hooks talk to via Unix domain socket.

Protocol (newline-delimited JSON):
  Client → {"prompt": "...", "session_id": "..."}\n
  Server → {"hookSpecificOutput": {...}}\n  (or {}\n for ack/empty)

MEMORY SAFETY: This module NEVER imports MemoryEngine. All recall goes
through recall_queue.db → QueueConsumer → pool.recall(). The hook daemon
stays at ~15-20MB RSS.

Lifecycle: started by unified_daemon.py alongside QueueConsumer. If it
crashes, auto_recall_hook.py falls back to subprocess (v3.4.35 path).
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
import threading
import time
from pathlib import Path

from superlocalmemory.infra.data_root import state_path

logger = logging.getLogger(__name__)

_DEFAULT_SOCK_NAME = "hook_daemon.sock"

# AF_UNIX is absent on Windows builds < 10.0.17063 and on Python < 3.9. When it
# is unavailable the hook daemon does not start and callers fall back to the
# subprocess recall path. Detect it explicitly (once, with a log) instead of
# relying on an AttributeError being swallowed by a broad except.
_AF_UNIX = getattr(socket, "AF_UNIX", None)


def _default_sock_path() -> Path:
    return state_path(_DEFAULT_SOCK_NAME)


def _default_queue_db_path() -> Path:
    return state_path("recall_queue.db")


class HookDaemon:
    """Unix socket server for persistent auto-recall.

    Accepts newline-delimited JSON requests, runs the same logic as
    auto_recall_hook.main() but without Python startup cost.
    """

    def __init__(
        self,
        sock_path: Path | None = None,
        queue_db_path: Path | None = None,
    ) -> None:
        self._sock_path = sock_path or _default_sock_path()
        self._queue_db_path = queue_db_path or _default_queue_db_path()
        self._running = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._server_sock: socket.socket | None = None
        self._queue = None

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        if self._sock_path.exists():
            self._sock_path.unlink()

        from superlocalmemory.core.recall_queue import RecallQueue
        self._queue = RecallQueue(self._queue_db_path)

        if _AF_UNIX is None:
            logger.info(
                "HookDaemon: AF_UNIX unavailable on %s; hook recall uses the "
                "subprocess fallback", sys.platform,
            )
            raise RuntimeError("AF_UNIX unavailable on this platform")
        self._server_sock = socket.socket(_AF_UNIX, socket.SOCK_STREAM)
        self._server_sock.bind(str(self._sock_path))
        self._server_sock.listen(8)
        self._server_sock.settimeout(1.0)

        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._accept_loop,
            daemon=True,
            name="slm-hook-daemon",
        )
        self._thread.start()
        logger.info("HookDaemon started on %s", self._sock_path)

    def stop(self) -> None:
        if not self._running:
            return
        self._stop_event.set()
        self._running = False
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except Exception:
                pass
            self._server_sock = None
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._sock_path.exists():
            try:
                self._sock_path.unlink()
            except Exception:
                pass
        if self._queue is not None:
            try:
                self._queue.close()
            except Exception:
                pass
            self._queue = None
        logger.info("HookDaemon stopped")

    def _accept_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                client, _ = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stop_event.is_set():
                    break
                continue
            threading.Thread(
                target=self._handle_client,
                args=(client,),
                daemon=True,
                name="slm-hook-client",
            ).start()

    def _handle_client(self, client: socket.socket) -> None:
        try:
            client.settimeout(30.0)
            data = b""
            while b"\n" not in data:
                chunk = client.recv(4096)
                if not chunk:
                    return
                data += chunk

            line = data.decode("utf-8").strip()
            if not line:
                client.sendall(b"{}\n")
                return

            try:
                payload = json.loads(line)
            except Exception:
                client.sendall(b"{}\n")
                return

            response = self._process_request(payload)
            client.sendall((json.dumps(response) + "\n").encode("utf-8"))
        except Exception:
            try:
                client.sendall(b"{}\n")
            except Exception:
                pass
        finally:
            try:
                client.close()
            except Exception:
                pass

    def _process_request(self, payload: dict) -> dict:
        from superlocalmemory.hooks.auto_recall_hook import (
            _is_ack, _get_mode_timeout, _detect_mode, _format_envelope,
            _DEFAULT_LIMIT,
        )
        from superlocalmemory.core.recall_queue import QueueTimeoutError

        prompt = payload.get("prompt", "")
        session_id = payload.get("session_id", "")

        if not prompt or not isinstance(prompt, str):
            return {}

        if _is_ack(prompt):
            return {}

        try:
            mode = _detect_mode()
            timeout = _get_mode_timeout(mode)
            stall_timeout = max(timeout - 5.0, 5.0)

            request_id = self._queue.enqueue(
                query=prompt,
                limit_n=_DEFAULT_LIMIT,
                mode=mode,
                agent_id="hook_daemon",
                session_id=session_id,
                priority="high",
                stall_timeout_s=stall_timeout,
            )

            result = self._queue.poll_result(request_id, timeout_s=timeout)

            if isinstance(result, dict) and result.get("ok") is not False:
                results = result.get("results", [])
                if results:
                    return _format_envelope(results)
            return {}
        except (QueueTimeoutError, Exception):
            return {}


def try_socket_recall(
    sock_path: Path | None = None,
    prompt: str = "",
    session_id: str = "",
    timeout: float = 15.0,
) -> dict | None:
    """Try to get recall result via the persistent hook daemon socket.

    Returns the hook envelope dict on success, or None if the daemon
    is unavailable (triggers subprocess fallback in auto_recall_hook).
    """
    path = sock_path or _default_sock_path()
    if not path.exists():
        return None
    if _AF_UNIX is None:
        return None

    try:
        client = socket.socket(_AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(timeout)
        client.connect(str(path))

        request = json.dumps({"prompt": prompt, "session_id": session_id}) + "\n"
        client.sendall(request.encode("utf-8"))

        data = b""
        while b"\n" not in data:
            chunk = client.recv(8192)
            if not chunk:
                break
            data += chunk

        client.close()

        if not data.strip():
            return None

        response = json.loads(data.decode("utf-8").strip())
        return response if isinstance(response, dict) else None
    except Exception:
        return None


def ensure_hook_daemon(
    sock_path: Path | None = None,
    queue_db_path: Path | None = None,
) -> HookDaemon | None:
    """Start hook daemon if not already running. Returns daemon or None."""
    path = sock_path or _default_sock_path()
    if _AF_UNIX is None:
        return None

    if path.exists():
        try:
            test = socket.socket(_AF_UNIX, socket.SOCK_STREAM)
            test.settimeout(1.0)
            test.connect(str(path))
            test.close()
            return None
        except Exception:
            pass

    daemon = HookDaemon(
        sock_path=path,
        queue_db_path=queue_db_path or _default_queue_db_path(),
    )
    daemon.start()
    return daemon
