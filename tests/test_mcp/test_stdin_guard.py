# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for the MCP stdin-EOF self-termination guard.

Regression coverage for the kqueue ``EV_EOF`` + buffered-data race
(v3.6.4): on macOS ``EVFILT_READ`` reports ``EV_EOF`` *together with*
unread bytes (``ev.data > 0``) when the write-end closes while a final
request is still buffered. The old monitor exited immediately on the EOF
flag, dropping that in-flight request and self-terminating a session that
still had work to deliver. The guard must defer termination until the
buffer is drained (``ev.data == 0``).

These are pure-logic tests — no FastMCP server import, no daemon threads.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from superlocalmemory.mcp._stdin_guard import eof_action

# A stand-in for select.KQ_EV_EOF (real value is platform-defined; the
# guard only does a bitwise-AND, so any distinct bit works for the contract).
EOF = 0x8000


def test_genuine_drained_eof_exits():
    """Write-end closed and buffer empty → terminate (unchanged behaviour)."""
    assert eof_action(EOF, 0, EOF) == "exit"


def test_eof_with_buffered_bytes_does_not_exit():
    """THE BUG: EOF reported with unread bytes must NOT terminate — the
    buffered request has to be drained first, or it is lost."""
    assert eof_action(EOF, 42, EOF) == "drain"


def test_no_eof_is_ignored():
    """Readable data without EOF is a normal read, not a session end."""
    assert eof_action(0, 100, EOF) == "ignore"


def test_no_eof_no_data_is_ignored():
    """Spurious wake with neither EOF nor data → ignore."""
    assert eof_action(0, 0, EOF) == "ignore"


def test_negative_data_treated_as_drained():
    """Defensive: some kernels report ev.data <= 0 at EOF; treat as drained."""
    assert eof_action(EOF, -1, EOF) == "exit"


def test_other_flags_alongside_eof_still_drain_when_buffered():
    """Extra flags (e.g. EV_ADD residue) must not defeat the data check."""
    assert eof_action(EOF | 0x1, 7, EOF) == "drain"
