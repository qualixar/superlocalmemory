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

import hashlib
import json
import logging
import os
import socket
import stat
import sys
import threading
from pathlib import Path

from superlocalmemory.infra.data_root import state_path

logger = logging.getLogger(__name__)

_DEFAULT_SOCK_NAME = "hook_daemon.sock"

# AF_UNIX is absent on Windows builds < 10.0.17063 and on Python < 3.9. When it
# is unavailable the hook daemon does not start and callers fall back to the
# subprocess recall path. Detect it explicitly (once, with a log) instead of
# relying on an AttributeError being swallowed by a broad except.
_AF_UNIX = getattr(socket, "AF_UNIX", None)


#: Longest socket path the OS accepts (``sun_path`` minus its terminator).
_SUN_PATH_MAX = 107 if sys.platform.startswith("linux") else 103


#: A request is one JSON line: a prompt, or a batch of hook events (4.2.8).
_MAX_REQUEST_BYTES = 1024 * 1024
#: Request type for hook events handed to the daemon's single writer.
HOOK_EVENTS_TYPE = "hook_events"


def _default_sock_path() -> Path:
    return state_path(_DEFAULT_SOCK_NAME)


def _short_socket_base() -> Path:
    """A short, per-user folder for socket paths that do not fit.

    Fixed, not taken from the environment, so the daemon (often started by
    the OS service manager) and the hook (started by an editor) agree on it.
    """
    uid = os.getuid() if hasattr(os, "getuid") else 0
    return Path("/tmp") / f"slm-{uid}"


def _private_dir(base: Path, *, create: bool) -> Path | None:
    """``base`` if it is a real folder owned by this user and closed to others.

    None for a symlink, a file, or a folder another account owns — someone
    who pre-created it could swap our socket for theirs, read prompts and
    answer with their own "memories". Our own folder is tightened to 0700.
    """
    if create:
        try:
            os.mkdir(base, 0o700)
        except FileExistsError:
            pass
        except OSError as exc:
            logger.warning("HookDaemon: cannot create %s: %s", base, type(exc).__name__)
            return None
    try:
        st = os.lstat(base)
    except OSError:
        return None
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        return None
    if hasattr(os, "getuid") and st.st_uid != os.getuid():
        return None
    if stat.S_IMODE(st.st_mode) & 0o077:
        if not create:
            return None
        os.chmod(base, 0o700)
    return base


def resolve_sock_path(path: Path, *, create: bool = False) -> Path | None:
    """Where the socket for ``path`` actually lives.

    ``path`` itself when it fits the OS limit — unchanged for most installs.
    Otherwise a name derived from ``path`` inside the short per-user folder,
    so two data folders never share a socket and the hook computes the same
    place the daemon bound. None when that folder is not safe to use (the
    hook then takes its slower, socket-free path).
    """
    if len(os.fsencode(str(path))) <= _SUN_PATH_MAX:
        return path
    folder = _private_dir(_short_socket_base(), create=create)
    if folder is None:
        return None
    digest = hashlib.sha256(os.fsencode(str(path))).hexdigest()[:16]
    return folder / f"hook-{digest}.sock"


def _owned_by_me(path: Path) -> bool:
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return not hasattr(os, "getuid") or st.st_uid == os.getuid()


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
        #: Where the socket was actually bound (see ``resolve_sock_path``).
        self._bound_path: Path | None = None

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        if _AF_UNIX is None:
            logger.info(
                "HookDaemon: AF_UNIX unavailable on %s; hook recall uses the "
                "subprocess fallback", sys.platform,
            )
            raise RuntimeError("AF_UNIX unavailable on this platform")
        bind_path = resolve_sock_path(self._sock_path, create=True)
        if bind_path is None:
            raise RuntimeError(
                "no private short folder for the hook socket; hook recall uses "
                "the subprocess fallback"
            )
        if bind_path.exists() or bind_path.is_symlink():
            bind_path.unlink()

        from superlocalmemory.core.recall_queue import RecallQueue
        self._queue = RecallQueue(self._queue_db_path)

        self._server_sock = socket.socket(_AF_UNIX, socket.SOCK_STREAM)
        self._server_sock.bind(str(bind_path))
        # Owner-only: the socket answers with memories.
        os.chmod(bind_path, 0o600)
        self._server_sock.listen(8)
        self._server_sock.settimeout(1.0)
        self._bound_path = bind_path

        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._accept_loop,
            daemon=True,
            name="slm-hook-daemon",
        )
        self._thread.start()
        logger.info("HookDaemon started on %s", bind_path)

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
        bound = self._bound_path or self._sock_path
        if bound.exists():
            try:
                bound.unlink()
            except Exception:
                pass
        self._bound_path = None
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
                chunk = client.recv(65536)
                if not chunk or len(data) + len(chunk) > _MAX_REQUEST_BYTES:
                    return  # the caller reads no ack and keeps its request
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
        if payload.get("type") == HOOK_EVENTS_TYPE:
            return _accept_hook_events(payload)
        from superlocalmemory.hooks.auto_recall_hook import (
            _is_ack, _get_mode_timeout, _detect_mode, _format_envelope,
            _DEFAULT_LIMIT, _switched_off,
        )
        from superlocalmemory.core.recall_queue import QueueTimeoutError

        prompt = payload.get("prompt", "")
        session_id = payload.get("session_id", "")

        if not prompt or not isinstance(prompt, str):
            return {}

        if _is_ack(prompt) or _switched_off():
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
                    return _format_envelope(results, response=result)
            return {}
        except (QueueTimeoutError, Exception):
            return {}


def _accept_hook_events(payload: dict) -> dict:
    """Validate a hook event batch, spool it and wake the writer (4.2.8).

    The ack (``ok: true``) is sent only after the events are on disk. A
    malformed batch or an unusable spool is ``ok: false``: the hook then
    spools the events itself, so nothing is acked that was not kept.
    """
    try:
        from superlocalmemory.server.hook_event_writer import HookEventBatch, accept_batch

        batch = HookEventBatch.model_validate({"events": payload.get("events")})
    except Exception:
        return {"ok": False, "error": "invalid_events"}
    try:
        written, dropped = accept_batch(batch)
    except Exception as exc:
        logger.warning("HookDaemon: hook events not spooled: %s", type(exc).__name__)
        return {"ok": False, "error": "spool_failed"}
    return {"ok": True, "accepted": written, "dropped": dropped}


def owned_socket_path(sock_path: Path | None = None) -> Path | None:
    """The hook socket when it exists and belongs to this account, else None."""
    if _AF_UNIX is None:
        return None
    path = resolve_sock_path(sock_path or _default_sock_path())
    if path is None or not path.exists() or not _owned_by_me(path):
        return None
    return path


def _socket_request(path: Path, request: dict, timeout: float) -> dict | None:
    """One newline-delimited JSON exchange. None on any failure."""
    try:
        client = socket.socket(_AF_UNIX, socket.SOCK_STREAM)
        try:
            client.settimeout(timeout)
            client.connect(str(path))
            client.sendall((json.dumps(request) + "\n").encode("utf-8"))
            data = b""
            while b"\n" not in data:
                chunk = client.recv(8192)
                if not chunk:
                    break
                data += chunk
        finally:
            client.close()
        if not data.strip():
            return None
        response = json.loads(data.decode("utf-8").strip())
        return response if isinstance(response, dict) else None
    except Exception:
        return None


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
    path = owned_socket_path(sock_path)
    if path is None:
        return None
    return _socket_request(path, {"prompt": prompt, "session_id": session_id}, timeout)


def try_socket_events(
    events: list, *, timeout: float, sock_path: Path | None = None,
) -> dict | None:
    """Hand hook events to the daemon over the hook socket (4.2.8).

    The daemon's answer (``{"ok": true, ...}`` once spooled), or None when
    the socket is missing or did not answer within ``timeout``.
    """
    path = owned_socket_path(sock_path)
    if path is None:
        return None
    return _socket_request(path, {"type": HOOK_EVENTS_TYPE, "events": list(events)}, timeout)


def ensure_hook_daemon(
    sock_path: Path | None = None,
    queue_db_path: Path | None = None,
) -> HookDaemon | None:
    """Start hook daemon if not already running. Returns daemon or None."""
    path = sock_path or _default_sock_path()
    if _AF_UNIX is None:
        return None

    bound = resolve_sock_path(path)
    if bound is not None and bound.exists():
        try:
            test = socket.socket(_AF_UNIX, socket.SOCK_STREAM)
            test.settimeout(1.0)
            test.connect(str(bound))
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
