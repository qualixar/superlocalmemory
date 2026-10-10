# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Run the PDF parse script as a child process and hold it to its limits.

The child reads one page at a time and waits for a go-ahead before the next, so this
side decides the pace. Limits are enforced by watching, not by asking the child:
a page that takes too long, a job that runs too long, or memory above the cap ends
the child with a kill. Nothing here logs what the child sent.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ParseLimit(Exception):
    """A limit was hit; ``reason`` is a short code (page_timeout, time_limit, memory_limit)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ParseFailed(Exception):
    """The script reported an error or ended early; ``reason`` is its short code."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ParseStopped(Exception):
    """The caller asked to stop."""


def _process_mb(pid: int) -> float:
    try:
        import psutil

        proc = psutil.Process(pid)
        total = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                continue
        return total / (1024 * 1024)
    except Exception:  # noqa: BLE001 - an unreadable size counts as fine
        return 0.0


class ParseSession:
    """One child process; use as a context manager, call ``next_event`` until a ``done`` event."""

    def __init__(self, python: Path, script: Path, request: dict[str, Any], *, cwd: Path,
                 page_timeout_s: float, job_deadline: float, rss_limit_mb: int, poll_s: float,
                 should_stop: Callable[[], bool], on_tick: Callable[[], None]) -> None:
        self._argv = [str(python), "-I", str(script)]
        self._request, self._cwd = request, cwd
        self._page_timeout_s, self._job_deadline = page_timeout_s, job_deadline
        self._rss_limit_mb, self._poll_s = rss_limit_mb, max(0.01, poll_s)
        self._should_stop, self._on_tick = should_stop, on_tick
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue = queue.Queue()
        self._owes_go = False

    def __enter__(self) -> "ParseSession":
        env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
        self._proc = subprocess.Popen(
            self._argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            encoding="utf-8", cwd=str(self._cwd), env=env)
        threading.Thread(target=self._pump, args=(self._proc, self._lines), daemon=True,
                         name="document-parse-reader").start()
        self._send(json.dumps(self._request) + "\n")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @staticmethod
    def _pump(proc: subprocess.Popen, lines: queue.Queue) -> None:
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                lines.put(line)
        except (OSError, ValueError):
            pass
        lines.put(None)

    def _send(self, text: str) -> None:
        try:
            self._proc.stdin.write(text)  # type: ignore[union-attr]
            self._proc.stdin.flush()  # type: ignore[union-attr]
        except (OSError, ValueError):
            pass  # the child ended; the reader sees that

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.kill()
        except OSError:
            pass
        for stream in (proc.stdin, proc.stdout):
            try:
                stream.close()  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001 - already closed
                pass
        try:
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001 - nothing more can be done
            pass

    def _check_limits(self, page_deadline: float) -> None:
        now = time.monotonic()
        if now > self._job_deadline:
            raise ParseLimit("time_limit")
        if now > page_deadline:
            raise ParseLimit("page_timeout")
        if self._rss_limit_mb > 0 and self._proc is not None and _process_mb(self._proc.pid) > self._rss_limit_mb:
            logger.warning("document parse is over its memory limit; stopping it")
            raise ParseLimit("memory_limit")

    def next_event(self) -> dict[str, Any]:
        """The next line from the child; raises ParseLimit, ParseFailed or ParseStopped."""
        if self._owes_go:
            self._send("\n")
        page_deadline = time.monotonic() + self._page_timeout_s
        while True:
            if self._should_stop():
                raise ParseStopped()
            try:
                line = self._lines.get(timeout=self._poll_s)
            except queue.Empty:
                self._check_limits(page_deadline)
                self._on_tick()
                continue
            if line is None:
                raise ParseFailed("parse_ended")
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            if "error" in event:
                raise ParseFailed(str(event["error"])[:40])
            self._owes_go = "page_no" in event
            return event
