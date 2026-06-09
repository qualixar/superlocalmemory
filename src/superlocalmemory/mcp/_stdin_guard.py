# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Pure decision logic for the MCP stdin-EOF self-termination monitor.

Isolated here (no imports, no side effects) so the kqueue ``EV_EOF``
handling contract is unit-testable without importing the FastMCP server
module, which starts daemon threads and auto-starts the SLM daemon at
import time.

Background (v3.6.4 fix)
-----------------------
The stdin-EOF monitor (``superlocalmemory.mcp.server._stdin_eof_monitor``)
exists to reap orphaned ``slm mcp`` processes when an IDE/agent abandons the
stdio pipe without quitting. It registers ``EVFILT_READ | EV_EOF`` on stdin
and terminates the process when the write-end closes.

On macOS, ``EVFILT_READ`` reports ``EV_EOF`` *together with* still-readable
bytes (``ev.data > 0``) when the write-end is closed while a final request
is still buffered in the pipe. The original monitor exited on the EOF flag
alone — dropping that buffered request and tearing down a session that
still had a pending in-flight call. Under strict MCP hosts whose transport
half-closes stdin around reconnect/teardown, this surfaced as the server
self-terminating mid-request, which the host then logged as a keepalive
failure and respawned (observed against the Hermes agent).

The guard defers termination until the buffer is genuinely drained
(``ev.data <= 0``), letting the FastMCP reader consume the last request
first. Genuine disconnects (EOF with an empty buffer) still terminate
immediately — behaviour identical to before for the common case.
"""

from __future__ import annotations

__all__ = ["eof_action"]


def eof_action(flags: int, data: int, eof_flag: int) -> str:
    """Decide how the stdin-EOF monitor should react to one kqueue event.

    Args:
        flags: ``kevent.flags`` bitmask returned by ``kqueue.control``.
        data: ``kevent.data`` — for ``EVFILT_READ`` this is the number of
            bytes still readable on the descriptor.
        eof_flag: the platform ``select.KQ_EV_EOF`` constant (injected so
            this function stays import-free and trivially testable).

    Returns:
        - ``"exit"``   — genuine end-of-stream: write-end closed and the
          buffer is drained (``data <= 0``). Safe to self-terminate.
        - ``"drain"``  — write-end closed but unread bytes remain
          (``data > 0``). Must NOT terminate yet; let the reader consume
          the buffered request first, otherwise it is silently dropped.
        - ``"ignore"`` — no EOF on this event (ordinary readability or a
          spurious wake); nothing to do.
    """
    if not (flags & eof_flag):
        return "ignore"
    return "drain" if data > 0 else "exit"
