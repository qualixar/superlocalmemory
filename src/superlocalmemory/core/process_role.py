# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Is this process the SLM daemon?

Only the daemon may run the managed model worker: every other process that
embeds text (a command, the MCP server, the recall worker) asks the daemon,
so the model is loaded once per computer. The daemon marks itself here once it
owns the data folder, before it builds its engine. The mark is a plain module
flag, deliberately not an environment variable: child processes of the daemon
must not inherit it.
"""

from __future__ import annotations

_daemon = False


def mark_daemon_process() -> None:
    global _daemon
    _daemon = True


def clear_daemon_process() -> None:
    global _daemon
    _daemon = False


def is_daemon_process() -> bool:
    return _daemon


__all__ = ["clear_daemon_process", "is_daemon_process", "mark_daemon_process"]
