# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Carry the picture worker's stderr into the daemon log, bounded.

A worker that crashes or runs out of memory explains itself on stderr, and that used
to be thrown away. The drain reads every line (so a chatty worker can never fill the
pipe and stall), logs the first few per minute at WARNING, cuts long lines, and ends
on its own when the pipe closes.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import IO

logger = logging.getLogger("superlocalmemory.runtimes.worker_client")

PREFIX = "image worker: "
THREAD_NAME = "media-worker-stderr"
MAX_LINE_CHARS = 500
MAX_LINES_PER_WINDOW = 50
WINDOW_S = 60.0
_READ_CHUNK = 4096  # bounds memory for a line that never ends


class StderrDrain:
    def __init__(self, stream: IO[str], *, clock: Callable[[], float] = time.monotonic,
                 max_lines: int = MAX_LINES_PER_WINDOW, window_s: float = WINDOW_S) -> None:
        self._stream, self._clock = stream, clock
        self._max_lines, self._window_s = max_lines, window_s
        self._window_start, self._logged, self._held_back = None, 0, 0

    def _emit(self, text: str) -> None:
        logger.warning("%s%s", PREFIX, text)

    def _report_held_back(self) -> None:
        if self._held_back:
            logger.info("%s%d more line(s) not logged (rate limit)", PREFIX, self._held_back)
        self._held_back = 0

    def _admit(self) -> bool:
        now = self._clock()
        if self._window_start is None or now - self._window_start >= self._window_s:
            self._report_held_back()
            self._window_start, self._logged = now, 0
        if self._logged >= self._max_lines:
            self._held_back += 1
            return False
        self._logged += 1
        return True

    def run(self) -> None:
        """Read until the pipe closes. Never raises: a drain that stops would stall the worker."""
        continuation = False  # inside the tail of a line already logged (cut) or counted
        try:
            for chunk in iter(lambda: self._stream.readline(_READ_CHUNK), ""):
                was_continuation, continuation = continuation, not chunk.endswith("\n")
                text = chunk.rstrip("\r\n")
                if was_continuation or not text.strip():
                    continue
                if self._admit():
                    self._emit(text[:MAX_LINE_CHARS])
        except (OSError, ValueError):
            pass  # the pipe was closed under us: the worker is gone
        finally:
            self._report_held_back()


def start_drain(stream: IO[str]) -> threading.Thread:
    """Drain ``stream`` on a daemon thread that ends when the stream does."""
    thread = threading.Thread(target=StderrDrain(stream).run, daemon=True, name=THREAD_NAME)
    thread.start()
    return thread


def stop_drain(thread: threading.Thread | None, stream: IO[str] | None, *, wait_s: float = 2.0) -> None:
    """After the worker process is dead: let the drain reach EOF, then close the pipe.

    A grandchild that inherited the pipe could keep it open; the stream is then left
    for the daemon thread to finish rather than closed under a blocked read.
    """
    if thread is not None:
        thread.join(wait_s)
        if thread.is_alive():
            logger.debug("%sstderr reader is still draining after the process ended", PREFIX)
            return
    if stream is not None:
        try:
            stream.close()
        except Exception:  # noqa: BLE001 - already closed
            pass
