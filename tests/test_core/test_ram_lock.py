# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-00 §7 + P0.5

"""Tests for ``ram_reservation`` — the process-wide RAM semaphore.

LLD-00 §7 requires heavyweight subsystems (hnswlib index build in LLD-12,
trigram rebuild in LLD-13) to coordinate so they don't collide in the
same consolidation tick and blow past the I2 RAM ceiling. The semaphore
lives in ``~/.superlocalmemory/ram_lock.sem`` and is acquired via
``fcntl.flock``.

Covers IMPLEMENTATION-MANIFEST-v3.4.22-FINAL.md §P0.5.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest


# POSIX-only — fcntl is Linux/macOS. Windows gets its own path later.
if sys.platform == "win32":
    pytest.skip("ram_reservation is POSIX-only", allow_module_level=True)


@pytest.fixture
def tmp_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the lock file into the test tmp_path."""
    from superlocalmemory.core import ram_lock
    lock_path = tmp_path / "ram_lock.sem"
    monkeypatch.setattr(ram_lock, "RAM_LOCK_PATH", lock_path)
    return lock_path


class _FakeVMem:
    def __init__(self, available_bytes: int) -> None:
        self.available = available_bytes


def test_ram_reservation_acquires_when_free(tmp_lock: Path) -> None:
    from superlocalmemory.core import ram_lock
    with ram_lock.ram_reservation("unit", required_mb=1):
        assert tmp_lock.exists()


def test_ram_reservation_fails_fast_below_threshold(
    tmp_lock: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from superlocalmemory.core import ram_lock
    # Pretend only 50 MB free.
    monkeypatch.setattr(
        ram_lock.psutil, "virtual_memory",
        lambda: _FakeVMem(50 * 1024 * 1024),
    )
    with pytest.raises(RuntimeError, match="free"):
        with ram_lock.ram_reservation("heavy", required_mb=200):
            pytest.fail("should not enter body")


def test_ram_reservation_timeout_raises(tmp_lock: Path) -> None:
    """Second acquisition while the first is held must time out."""
    from superlocalmemory.core import ram_lock

    ready_flag = tmp_lock.parent / "holder_ready"
    release_flag = tmp_lock.parent / "holder_release"
    holder_script = textwrap.dedent(f"""
        import sys, time
        from pathlib import Path
        sys.path.insert(0, {str(Path(__file__).resolve().parents[2] / 'src')!r})
        from superlocalmemory.core import ram_lock as rl
        rl.RAM_LOCK_PATH = Path({str(tmp_lock)!r})
        with rl.ram_reservation('holder', required_mb=1, timeout_s=10.0):
            Path({str(ready_flag)!r}).write_text('ok')
            deadline = time.time() + 15
            while time.time() < deadline:
                if Path({str(release_flag)!r}).exists():
                    break
                time.sleep(0.05)
    """)
    proc = subprocess.Popen([sys.executable, "-c", holder_script])
    try:
        # Wait for the holder to signal it acquired the lock.
        t0 = time.time()
        while not ready_flag.exists() and time.time() - t0 < 10:
            time.sleep(0.05)
        assert ready_flag.exists(), "holder never acquired"

        with pytest.raises(RuntimeError, match="timeout"):
            with ram_lock.ram_reservation("waiter", required_mb=1,
                                           timeout_s=0.5):
                pytest.fail("should not enter body")
    finally:
        release_flag.write_text("go")
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover — defensive
            proc.kill()
            proc.wait(timeout=5)


def test_ram_reservation_releases_on_exception(tmp_lock: Path) -> None:
    """Lock is released when the body raises."""
    from superlocalmemory.core import ram_lock

    class _Kaboom(Exception): ...

    with pytest.raises(_Kaboom):
        with ram_lock.ram_reservation("kaboom", required_mb=1):
            raise _Kaboom()

    # Re-acquiring immediately must succeed.
    with ram_lock.ram_reservation("after", required_mb=1, timeout_s=1.0):
        pass


def test_ram_reservation_releases_on_normal_exit(tmp_lock: Path) -> None:
    from superlocalmemory.core import ram_lock
    with ram_lock.ram_reservation("a", required_mb=1):
        pass
    # Second acquire must succeed immediately.
    t0 = time.time()
    with ram_lock.ram_reservation("b", required_mb=1, timeout_s=1.0):
        pass
    assert (time.time() - t0) < 1.0


def test_ram_reservation_creates_parent_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from superlocalmemory.core import ram_lock
    deep = tmp_path / "nested" / "dir" / "ram_lock.sem"
    monkeypatch.setattr(ram_lock, "RAM_LOCK_PATH", deep)
    with ram_lock.ram_reservation("mk", required_mb=1):
        assert deep.parent.is_dir()


def test_ram_reservation_rejects_empty_name(tmp_lock: Path) -> None:
    from superlocalmemory.core import ram_lock
    with pytest.raises(ValueError):
        with ram_lock.ram_reservation("", required_mb=1):
            pass


def test_ram_reservation_rejects_negative_required_mb(tmp_lock: Path) -> None:
    from superlocalmemory.core import ram_lock
    with pytest.raises(ValueError):
        with ram_lock.ram_reservation("neg", required_mb=-1):
            pass
