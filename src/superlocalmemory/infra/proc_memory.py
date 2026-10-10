# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""How much memory a process really holds -- the one reader every memory guard uses.

On macOS the resident size (RSS) under-reports badly once memory is compressed: a
picture worker showed 215 MB RSS against a 3,650 MB physical footprint, so any guard
built on RSS never fired on a busy Mac. The physical footprint is the figure
``footprint`` and Activity Monitor show, so macOS reads that through libproc. Elsewhere,
and whenever libproc cannot answer, the answer is the RSS from psutil.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)

_MB = 1024 * 1024
_LIBPROC_PATH = "/usr/lib/libproc.dylib"
_RUSAGE_INFO_V2 = 2
#: ``struct rusage_info_v2`` viewed as uint64[]: a 16-byte uuid (two slots), then the
#: counters; ``ri_phys_footprint`` is the eighth counter after the uuid.
_FOOTPRINT_SLOT = 9
_BUFFER_SLOTS = 64  # larger than any rusage_info version, so libproc never overruns it
_LIBPROC: Any = None


def _libproc() -> Any:
    """The loaded libproc, cached after the first load. Raises OSError when it is missing."""
    global _LIBPROC
    if _LIBPROC is None:
        _LIBPROC = ctypes.CDLL(_LIBPROC_PATH)
    return _LIBPROC


def _footprint_mb(pid: int) -> float:
    """Physical footprint in MB (macOS). Raises when libproc cannot answer."""
    buf = (ctypes.c_uint64 * _BUFFER_SLOTS)()
    if _libproc().proc_pid_rusage(pid, _RUSAGE_INFO_V2, buf) != 0:
        raise OSError(f"proc_pid_rusage failed for pid {pid}")
    return buf[_FOOTPRINT_SLOT] / _MB


def _rss_mb(pid: int) -> float:
    import psutil

    return psutil.Process(pid).memory_info().rss / _MB


def process_memory_mb(pid: int) -> float:
    """Memory held by ``pid`` in MB; 0.0 when it is unknown or already gone."""
    if pid <= 0:
        return 0.0
    if sys.platform == "darwin":
        try:
            return _footprint_mb(pid)
        except Exception as exc:  # noqa: BLE001 - RSS is the safe fallback
            logger.debug("physical footprint unavailable for pid %d (%s); using RSS", pid, exc)
    try:
        return _rss_mb(pid)
    except Exception:  # noqa: BLE001 - a process that is gone holds nothing
        return 0.0


def tree_memory_mb(pid: int) -> float:
    """Memory held by ``pid`` plus all its descendants in MB; 0.0 when unknown."""
    if pid <= 0:
        return 0.0
    try:
        import psutil

        children = psutil.Process(pid).children(recursive=True)
    except Exception:  # noqa: BLE001 - nothing readable means nothing counted
        return process_memory_mb(pid)
    return process_memory_mb(pid) + sum(process_memory_mb(c.pid) for c in children)


__all__ = ["process_memory_mb", "tree_memory_mb"]
