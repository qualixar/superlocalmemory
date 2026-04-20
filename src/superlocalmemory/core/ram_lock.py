# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-00 §7 + P0.5

"""Process-wide RAM semaphore for heavyweight consolidation subsystems.

LLD-00 §7: hnswlib index builds (LLD-12) and trigram full rebuilds
(LLD-13) both spike RAM into the hundreds of MB. Running them
concurrently in the same consolidation tick can blow past the I2 RAM
ceiling on Light/Minimal profiles. This semaphore serialises them.

Design notes:
- One flock per process pair (exclusive, non-blocking with poll loop).
- Fast-fail if psutil reports less than ``MIN_FREE_MB + required_mb``
  available before we even try — better to defer than thrash.
- Lock file lives at ``~/.superlocalmemory/ram_lock.sem`` by default,
  tests may monkeypatch ``RAM_LOCK_PATH``.
- POSIX only; Windows build is deferred per LLD-00 §7 and MASTER-PLAN H-01.
"""

from __future__ import annotations

import fcntl
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import psutil


RAM_LOCK_PATH: Path = Path.home() / ".superlocalmemory" / "ram_lock.sem"
MIN_FREE_MB: int = 400


@contextmanager
def ram_reservation(
    name: str,
    *,
    timeout_s: float = 60.0,
    required_mb: int = 200,
) -> Iterator[None]:
    """Process-wide RAM semaphore. Acquire before heavy subsystems.

    Usage::

        with ram_reservation('hnswlib', required_mb=200):
            build_index(...)

    Guarantees:

    - Fast-fails if ``psutil.virtual_memory().available < (MIN_FREE_MB +
      required_mb) * 1024 * 1024``. No body execution in that case.
    - Acquires the exclusive flock on ``RAM_LOCK_PATH`` with a polling
      wait. Raises ``RuntimeError`` after ``timeout_s``.
    - Releases the lock on normal exit AND on exception propagation.
    - Writes a short audit line ``<pid>:<name>\\n`` on acquire so
      an operator can see which subsystem holds it.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")
    if not isinstance(required_mb, int) or required_mb < 0:
        raise ValueError(f"required_mb must be non-negative int, got {required_mb!r}")

    vm = psutil.virtual_memory()
    free_mb = vm.available / (1024 * 1024)
    floor_mb = MIN_FREE_MB + required_mb
    if free_mb < floor_mb:
        raise RuntimeError(
            f"ram_reservation({name}): free {free_mb:.0f}MB < required "
            f"{floor_mb}MB"
        )

    RAM_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    # SEC-M6 — tighten the parent dir so the audit marker (``{pid}:{name}``)
    # is not readable by other UIDs on shared hosts. Idempotent; POSIX-only.
    try:
        if os.name == "posix":
            os.chmod(RAM_LOCK_PATH.parent, 0o700)
    except OSError:  # pragma: no cover — perms race
        pass
    fd = os.open(str(RAM_LOCK_PATH), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        deadline = time.time() + timeout_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() >= deadline:
                    raise RuntimeError(
                        f"ram_reservation({name}) timeout after {timeout_s:.1f}s"
                    )
                time.sleep(0.05)
        try:
            # Truncate + write a fresh audit marker.
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, f"{os.getpid()}:{name}\n".encode("utf-8"))
        except OSError:  # pragma: no cover — marker write is best-effort
            pass
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


__all__ = ("RAM_LOCK_PATH", "MIN_FREE_MB", "ram_reservation")
