# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""What folder sources need from the daemon, passed in so this package never imports the server."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

#: Tombstoned files and replaced versions are erased this long after they were hidden.
PURGE_AFTER_S = 7 * 86400.0


@dataclass
class SourceHost:
    #: True while remote access is set up. A check that raises counts as True.
    remote_check: Callable[[], bool]
    #: The memory writer (``remember``, ``archive_fact``), or None while it is not ready.
    runtime: Callable[[], Any]
    config: Callable[[], Any]
    #: ``eraser(profile_id, fact_ids, subject_id)`` hard-erases facts and returns the receipt counts.
    eraser: Callable[[str, list[str], str], dict] | None
    profile: Callable[[], str]
    actor_id: Callable[[], str]
    data_root: Path | None = None
    sleep: Callable[[float], None] = time.sleep
    #: Called after work is queued so the scan service can pick it up.
    wake: Callable[[], None] = field(default=lambda: None)
    purge_after_s: float = PURGE_AFTER_S
    #: The second look at a changed file waits this long (once per pass).
    stability_s: float = 2.0

    def remote_on(self) -> bool:
        try:
            return bool(self.remote_check())
        except Exception:  # noqa: BLE001 - a check that cannot answer is not "off"
            return True


_host: SourceHost | None = None


def configure(host: SourceHost | None) -> None:
    """Install (or, with None, remove) the daemon's collaborators."""
    global _host
    _host = host


def current_host() -> SourceHost:
    if _host is None:
        raise RuntimeError("folder sources are not set up in this process")
    return _host
