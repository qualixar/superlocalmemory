# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Is the machine short of memory? One answer for every model SLM holds.

SLM must never be the reason a laptop hangs. Its models (embedder, reranker,
picture worker) are large, so when the machine runs short SLM gives memory back
first: the health monitor releases loaded models, and a released model is not
loaded again until the machine recovers.

"Short" uses two signals and no new threshold:

* the operating system's own judgement where it publishes one (macOS
  ``kern.memorystatus_vm_pressure_level``: 1 normal, 2 warning, 4 critical);
  warning or worse counts;
* memory a new allocation could get without paging (``embedding_memory.
  available_mb``) below the floor the embedder already uses
  (``SLM_MIN_AVAILABLE_MEMORY_GB``, 2 GB by default).

A reading that cannot be taken is "not pressed": this module only ever
releases memory, so an unknown must not make SLM stop working.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import sys
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: macOS kernel levels (``sys/kern_memorystatus.h``).
LEVEL_NORMAL = 1
LEVEL_WARNING = 2
LEVEL_CRITICAL = 4

_libc = None


@dataclass(frozen=True)
class Reading:
    pressed: bool
    available_mb: float | None
    floor_mb: float
    kernel_level: int | None
    reason: str


def kernel_level() -> int | None:
    """The OS pressure level, or None where none is published or readable."""
    global _libc
    if sys.platform != "darwin":
        return None
    try:
        if _libc is None:
            _libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        value = ctypes.c_int(0)
        size = ctypes.c_size_t(ctypes.sizeof(value))
        rc = _libc.sysctlbyname(b"kern.memorystatus_vm_pressure_level", ctypes.byref(value),
                                ctypes.byref(size), None, ctypes.c_size_t(0))
        return int(value.value) if rc == 0 else None
    except (OSError, AttributeError, TypeError):
        return None


def reading() -> Reading:
    """Whether the machine is short of memory now, and why."""
    from superlocalmemory.core.embedding_memory import available_mb, load_floor_mb

    level = kernel_level()
    available, floor = available_mb(), load_floor_mb()
    if level is not None and level >= LEVEL_WARNING:
        name = "critical" if level >= LEVEL_CRITICAL else "warning"
        return Reading(True, available, floor, level, f"the system reports memory pressure "
                                                      f"({name})")
    if available is not None and floor > 0 and available < floor:
        return Reading(True, available, floor, level,
                       f"{available:.0f} MB available, under the {floor:.0f} MB floor")
    return Reading(False, available, floor, level, "")


def pressed() -> bool:
    try:
        return reading().pressed
    except Exception:  # noqa: BLE001 - an unreadable machine never stops SLM
        logger.debug("memory pressure unreadable", exc_info=True)
        return False


__all__ = ["LEVEL_CRITICAL", "LEVEL_NORMAL", "LEVEL_WARNING", "Reading", "kernel_level",
           "pressed", "reading"]
