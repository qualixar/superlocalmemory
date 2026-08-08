# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Does this process still match what is installed on disk?  (issue #107)

Why this module exists
----------------------
Python imports a module once.  ``superlocalmemory.__version__`` is therefore
frozen at the instant a process started, and upgrading the package underneath a
long-lived ``slm mcp`` server changes nothing for that server -- it keeps
serving the code it read at startup, forever.

Nothing detected that.  The stale process did not error; it returned confident,
plausible, *wrong* answers, and reported a ``serverInfo.version`` matching the
code it had loaded, which is self-consistent and therefore useless as a
staleness signal.  During the v3.8.12 work this machine had eighteen ``slm mcp``
processes alive at once, spanning four days and two releases.  One of them made
issue #106 look unfixed across two debugging sessions and contributed to v3.8.11
shipping a wrong fix, because the "evidence" that the fix had failed was really
a four-day-old process.

The asymmetry that makes this dangerous
---------------------------------------
A *loud* failure costs a user one confused minute.  A *silent* one costs
whoever debugs it their entire session, because it actively argues that correct
code is broken.  Everything here is therefore built so that no failure mode can
produce a false :data:`STATE_CURRENT`.  Unreadable metadata, a hostile reader,
a non-string return -- all resolve to :data:`STATE_UNKNOWN`, which reports
"I could not tell" rather than "all is well".

Why ``importlib.metadata`` is the right source
----------------------------------------------
It reads the ``*.dist-info`` directory from disk on each call rather than
returning a value captured at import.  Verified empirically: rewriting a
distribution's metadata underneath a live process and re-reading returns the
*new* version, with no ``importlib.invalidate_caches()`` needed.  That is the
one property this whole module rests on, so it is pinned by a test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from superlocalmemory import __version__

__all__ = (
    "STATE_AHEAD",
    "STATE_CURRENT",
    "STATE_MISMATCH",
    "STATE_STALE",
    "STATE_UNKNOWN",
    "VersionIntegrity",
    "check_version_integrity",
    "installed_distribution_version",
)

#: Imported code matches the installed distribution.
STATE_CURRENT = "current"
#: Imported code is *older* than what is installed -- the #107 failure.
STATE_STALE = "stale"
#: Imported code is *newer* than the installed distribution.  Normal for an
#: editable checkout; deliberately not reported as a problem, because a warning
#: that fires on every maintainer's machine is a warning everyone learns to
#: ignore, and then it will not be read on the day it matters.
STATE_AHEAD = "ahead"
#: The two differ but cannot be ordered (local labels, unexpected formats).
#: Still surfaced -- a difference we cannot rank is not a difference we hide.
STATE_MISMATCH = "mismatch"
#: The installed version could not be determined at all.
STATE_UNKNOWN = "unknown"

_DISTRIBUTION_NAME = "superlocalmemory"

_RESTART_HINT = (
    "Restart this process to load the installed code "
    "(`slm restart` for the daemon; restart your MCP client for `slm mcp`)."
)


@dataclass(frozen=True)
class VersionIntegrity:
    """The outcome of comparing imported code against the installed dist."""

    running: str
    installed: Optional[str]
    state: str
    detail: str
    hint: str = ""

    @property
    def is_stale(self) -> bool:
        """True only for the #107 failure: running behind what is installed.

        Deliberately narrow.  Callers gate warnings on this, and widening it to
        mean "anything unusual" would make an editable checkout look broken.
        """
        return self.state == STATE_STALE

    @property
    def differs(self) -> bool:
        """True whenever imported and installed are known to be different."""
        return self.state in (STATE_STALE, STATE_AHEAD, STATE_MISMATCH)

    def as_dict(self) -> dict:
        """JSON-safe payload for ``/health``, ``slm status --json``, doctor."""
        return {
            "running": self.running,
            "installed": self.installed,
            "state": self.state,
            "detail": self.detail,
            "hint": self.hint,
            "is_stale": self.is_stale,
        }


def installed_distribution_version() -> str:
    """Return the on-disk version of the installed distribution.

    Raises whatever ``importlib.metadata`` raises; :func:`check_version_integrity`
    is the layer that turns failure into :data:`STATE_UNKNOWN`.  Keeping the
    raise here means a caller that genuinely wants the error can have it.
    """
    from importlib.metadata import version as _version

    return _version(_DISTRIBUTION_NAME)


def _version_tuple(raw: str) -> Optional[tuple[int, ...]]:
    """Parse a plain dotted release into ints, or ``None`` if it is not one.

    Intentionally strict and dependency-free: anything carrying a local label,
    pre-release marker, or non-numeric field returns ``None`` and is reported as
    :data:`STATE_MISMATCH`.  Guessing an order for such versions could mask a
    real drift behind a confident-looking "current".
    """
    parts = raw.strip().split(".")
    if not parts or any(not p.isdigit() for p in parts):
        return None
    return tuple(int(p) for p in parts)


def check_version_integrity(
    *,
    running: Optional[str] = None,
    installed_reader: Optional[Callable[[], str]] = None,
) -> VersionIntegrity:
    """Compare imported code against the installed distribution.

    Never raises.  Both sides are injectable so tests can drive every branch
    without touching the real environment.

    Args:
        running: Version of the *imported* code.  Defaults to
            ``superlocalmemory.__version__``, which is frozen at import.
        installed_reader: Callable returning the on-disk version.  Defaults to
            reading the installed distribution metadata.
    """
    running_version = running if running is not None else __version__

    reader = installed_reader or installed_distribution_version
    installed: Optional[str] = None
    try:
        candidate = reader()
    except BaseException:  # noqa: BLE001 - staleness reporting must never raise
        # BaseException, not Exception: this runs on daemon and MCP startup
        # paths, and a diagnostic must never be the reason a process dies.
        candidate = None

    if isinstance(candidate, str) and candidate.strip():
        installed = candidate.strip()

    if installed is None:
        return VersionIntegrity(
            running=running_version,
            installed=None,
            state=STATE_UNKNOWN,
            detail=(
                f"running {running_version}; could not read the installed "
                f"distribution version, so staleness is undetermined"
            ),
        )

    if running_version == installed:
        return VersionIntegrity(
            running=running_version,
            installed=installed,
            state=STATE_CURRENT,
            detail=f"running {running_version}, matching the installed distribution",
        )

    running_parts = _version_tuple(running_version)
    installed_parts = _version_tuple(installed)

    if running_parts is None or installed_parts is None:
        return VersionIntegrity(
            running=running_version,
            installed=installed,
            state=STATE_MISMATCH,
            detail=(
                f"running {running_version} but {installed} is installed; "
                f"the two cannot be ordered"
            ),
            hint=_RESTART_HINT,
        )

    if running_parts < installed_parts:
        return VersionIntegrity(
            running=running_version,
            installed=installed,
            state=STATE_STALE,
            detail=(
                f"running {running_version} but {installed} is installed — this "
                f"process loaded its code before the upgrade and will keep "
                f"serving {running_version} until it restarts"
            ),
            hint=_RESTART_HINT,
        )

    return VersionIntegrity(
        running=running_version,
        installed=installed,
        state=STATE_AHEAD,
        detail=(
            f"running {running_version}, ahead of the installed {installed} "
            f"(normal for an editable or source checkout)"
        ),
    )
