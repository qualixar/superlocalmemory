"""Peak resident memory sampler (process plus children), 100 ms by default."""
from __future__ import annotations

import threading

import psutil


def _rss_mb(pid: int) -> float:
    try:
        proc = psutil.Process(pid)
        total = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                pass
        return total / 1e6
    except psutil.Error:
        return 0.0


class RssSampler:
    """Context manager: samples RSS of pid (+children) in a background thread."""

    def __init__(self, pid: int, interval: float = 0.1):
        self.pid, self.interval, self.peak_mb = pid, interval, 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, _rss_mb(self.pid))
            self._stop.wait(self.interval)

    def __enter__(self) -> "RssSampler":
        self.peak_mb = _rss_mb(self.pid)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_mb = max(self.peak_mb, _rss_mb(self.pid))
