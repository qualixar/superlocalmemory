# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Clock-independent identity for one running local process.

Why this module exists
----------------------
A process's *wall-clock* creation time is not a stable identifier on every
platform.  On Linux -- and therefore inside WSL2 -- psutil derives it as::

    create_time = /proc/<pid>/stat:starttime / CLOCK_TICKS + /proc/stat:btime

``starttime`` is boot-relative and never changes for the life of the process.
``btime`` is the kernel's *estimate* of the boot instant, recomputed from the
current wall clock, so any clock step moves ``btime`` and moves every process's
computed ``create_time`` with it -- retroactively.  WSL2 periodically
resynchronises its VM clock against the Windows host, so a ``create_time``
recorded when the daemon started stops matching the ``create_time`` computed
for that very same process minutes later.  Issue #104 measured a ~35 second
divergence after roughly four minutes of uptime.

Any *constant* tolerance on that comparison is a delay, not a fix: the
divergence is unbounded and keeps growing.  The correct identifier is one the
wall clock cannot move at all, which is what this module produces.

What a start token is
---------------------
``process_start_token_for(pid)`` returns an opaque string identifying one
process *instance*, derived without reference to wall-clock time, or ``None``
when the platform cannot supply one.  Tokens are only comparable when they use
the same scheme, so :func:`compare_start_tokens` is deliberately tri-state --
callers fall back to a weaker signal instead of guessing.

Schemes
-------
``lx1``
    Linux/WSL2: ``lx1:<boot_id>:<starttime_ticks>``.  Both halves come straight
    from procfs and are an opaque UUID and an integer tick count rather than
    timestamps, so a clock adjustment cannot rewrite either.  ``boot_id``
    differs across reboots, so a post-reboot PID collision can never look like
    a match.
``mn1``
    Platforms where psutil exposes a monotonic creation time (macOS, NetBSD,
    and Linux if procfs is unreadable): ``mn1:<value>``.  psutil builds its own
    PID-reuse identity from exactly this value for exactly this reason.

Windows has no monotonic variant, and needs none: its creation time comes from
``GetProcessTimes``, a kernel timestamp that a clock adjustment does not
rewrite.  ``process_start_token_for`` returns ``None`` there and the caller's
creation-time comparison stays correct.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

LINUX_SCHEME = "lx1"
MONOTONIC_SCHEME = "mn1"

_PROCFS = Path("/proc")
# "man proc" numbers /proc/<pid>/stat fields from 1 and starttime is field 22.
# The comm field can contain spaces and parentheses, so parsing starts after
# the last ')', which drops fields 1 and 2 -- hence 22 - 3 == 19.
_STARTTIME_INDEX = 19


def _boot_id() -> str | None:
    """Return this boot's opaque kernel identifier, or None when unavailable.

    Deliberately strict: without a boot id, two processes from different boots
    could share a PID *and* a tick count, so the token would be unsound. A
    missing boot id therefore means "no token" rather than a weaker token.
    """
    try:
        value = (_PROCFS / "sys" / "kernel" / "random" / "boot_id").read_text(
            encoding="utf-8",
        ).strip()
    except OSError:
        return None
    return value or None


def _linux_start_ticks(pid: int) -> int | None:
    """Return boot-relative start ticks for a PID from procfs, or None."""
    try:
        data = (_PROCFS / str(pid) / "stat").read_bytes()
    except OSError:
        return None
    closing = data.rfind(b")")
    if closing < 0:
        return None
    fields = data[closing + 2:].split()
    if len(fields) <= _STARTTIME_INDEX:
        return None
    try:
        return int(fields[_STARTTIME_INDEX])
    except ValueError:
        return None


def _linux_start_token(pid: int) -> str | None:
    boot_id = _boot_id()
    if boot_id is None:
        return None
    ticks = _linux_start_ticks(pid)
    if ticks is None:
        return None
    return f"{LINUX_SCHEME}:{boot_id}:{ticks}"


def _monotonic_start_token(pid: int) -> str | None:
    """Return psutil's monotonic creation time as a token, or None.

    ``Process._proc.create_time(monotonic=True)`` is the same private accessor
    psutil uses internally to build ``Process._ident``.  It is guarded on every
    axis -- missing psutil, missing attribute, platforms whose implementation
    takes no ``monotonic`` keyword (Windows) -- and degrades to ``None``.
    """
    try:
        import psutil

        platform_process = getattr(psutil.Process(pid), "_proc", None)
    except Exception:  # noqa: BLE001 - identity probing must never raise
        return None
    create_time = getattr(platform_process, "create_time", None)
    if create_time is None:
        return None
    try:
        raw = create_time(monotonic=True)
    except TypeError:
        # No monotonic variant on this platform (Windows).
        return None
    except Exception:  # noqa: BLE001
        return None
    try:
        return f"{MONOTONIC_SCHEME}:{float(raw)!r}"
    except (TypeError, ValueError):
        return None


def process_start_token_for(pid: int) -> str | None:
    """Return a clock-independent identity token for ``pid``, or None.

    ``None`` is a normal answer, not an error: it means "this platform cannot
    prove process identity without the wall clock", and the caller should fall
    back to comparing creation times.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None
    if sys.platform.startswith("linux"):
        token = _linux_start_token(pid)
        if token is not None:
            return token
    return _monotonic_start_token(pid)


def compare_start_tokens(recorded: str | None, observed: str | None) -> bool | None:
    """Tri-state comparison of two start tokens.

    ``True``  -- same scheme, identical value: proven the same process instance.
    ``False`` -- same scheme, different value: proven a different instance.
    ``None``  -- not comparable (either side missing, or different schemes);
                 the caller must fall back rather than assume either way.
    """
    if not recorded or not observed:
        return None
    if not isinstance(recorded, str) or not isinstance(observed, str):
        return None
    if recorded.split(":", 1)[0] != observed.split(":", 1)[0]:
        return None
    return recorded == observed
