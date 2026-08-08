# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""F-14 invariant: fileno-less timeout path must not leak reader threads."""

from __future__ import annotations

import os
import threading
import time

import pytest

from superlocalmemory.core.embeddings import EmbeddingService


class _FilenoLessBlockingStream:
    """Stream without a usable fileno that blocks in readline until closed.

    Forces the thread fallback path (Windows pipes / mocks without fileno).
    """

    def __init__(self) -> None:
        self._r_fd, self._w_fd = os.pipe()
        self._stream = os.fdopen(self._r_fd, "r")
        self._closed = False

    def fileno(self):  # noqa: ANN201
        raise OSError("no fileno — force thread path")

    def readline(self) -> str:
        return self._stream.readline()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            os.close(self._w_fd)
        except Exception:
            pass
        try:
            self._stream.close()
        except Exception:
            pass


def test_fileno_less_timeouts_leave_zero_read_threads() -> None:
    """N timeouts on a fileno-less stream → zero surviving *_read* threads."""
    n = 5
    held: list[_FilenoLessBlockingStream] = []
    before_ids = {t.ident for t in threading.enumerate()}
    try:
        for _ in range(n):
            stream = _FilenoLessBlockingStream()
            held.append(stream)
            result = EmbeddingService._readline_with_timeout(stream, 0.05)
            assert result == ""

        time.sleep(0.15)
        survivors = [
            t
            for t in threading.enumerate()
            if t.is_alive()
            and t.ident not in before_ids
            and "_read" in (t.name or "")
        ]
        assert survivors == [], (
            f"leaked {len(survivors)} reader thread(s) after {n} fileno-less "
            f"timeouts: {[t.name for t in survivors]}"
        )
    finally:
        for stream in held:
            try:
                stream.close()
            except Exception:
                pass
        # Give any correctly-unblocked readers a moment to exit.
        time.sleep(0.05)
