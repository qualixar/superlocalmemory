# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Open a folder file for reading without following a link, without blocking, and only if it is the file seen."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import BinaryIO

_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)


def open_regular(path: Path | str, file_id: str | None = None) -> BinaryIO:
    """The open file, or OSError when it is a link, a pipe, a device, or not the file ``file_id`` names.

    ``file_id`` is ``"<st_dev>:<st_ino>"`` as the walk recorded it.
    """
    fd = os.open(path, _FLAGS)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise OSError("not a regular file")
        if file_id is not None and f"{info.st_dev}:{info.st_ino}" != file_id:
            raise OSError("not the file that was listed")
        return os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise


def read_bounded(path: Path | str, limit: int, file_id: str | None = None) -> bytes:
    """Up to ``limit`` bytes from the start of a regular file."""
    with open_regular(path, file_id) as fh:
        return fh.read(limit)


__all__ = ["open_regular", "read_bounded"]
