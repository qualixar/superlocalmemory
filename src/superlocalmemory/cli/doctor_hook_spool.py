# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""The ``slm doctor`` row for hook events (4.2.8).

Hooks hand tool events and reward signals to the daemon's single writer, or
leave them in ``<data root>/hook_spool`` when the daemon does not ack. This row
says how many wait there, and how many were ever refused because the spool was
full or unreadable when drained: the one place a lost hook event shows up.
Read-only: doctor creates nothing.
"""

from __future__ import annotations

from pathlib import Path

FIX_DROPPED = ("Keep the daemon running (`slm serve start`; `slm serve status` shows it) so it drains "
               "the spool; raise SLM_HOOK_SPOOL_MAX_BYTES if it was stopped for days")
FIX_INVALID = "See logs/daemon.log in the SLM data folder for the unreadable spool file"


def hook_spool_finding(root: str | Path) -> tuple[str, str, str]:
    """``(status, detail, fix)`` for the hook event spool under data root ``root``."""
    from superlocalmemory.hooks.event_spool import SPOOL_DIR_NAME, read_stats

    stats = read_stats(Path(root) / SPOOL_DIR_NAME)
    waiting = (f"{stats.pending_events} hook event(s) waiting for the daemon"
               if stats.pending_events else "no hook events waiting")
    if stats.dropped:
        last = f", last at {stats.last_drop_at[:19]}" if stats.last_drop_at else ""
        return ("WARN", f"{stats.dropped} hook event(s) dropped because the spool was "
                        f"full{last}; {waiting}", FIX_DROPPED)
    if stats.invalid:
        return ("WARN", f"{stats.invalid} unreadable hook event line(s) skipped; {waiting}",
                FIX_INVALID)
    return "PASS", waiting, ""


__all__ = ["FIX_DROPPED", "FIX_INVALID", "hook_spool_finding"]
