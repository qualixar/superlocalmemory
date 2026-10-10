"""One process-memory reader: physical footprint on macOS, RSS elsewhere."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from superlocalmemory.infra import proc_memory

MB = 1024 * 1024


class _FakeLibproc:
    """Fills the rusage buffer the way libproc does: uuid (2 slots), 7 counters, footprint."""

    def __init__(self, footprint_bytes: int, rc: int = 0) -> None:
        self.footprint_bytes, self.rc, self.calls = footprint_bytes, rc, []

    def proc_pid_rusage(self, pid, flavor, buf):
        self.calls.append((pid, flavor))
        buf[8] = 215 * MB  # ri_resident_size: what RSS-based guards used to see
        buf[9] = self.footprint_bytes  # ri_phys_footprint
        return self.rc


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    monkeypatch.setattr(proc_memory, "_LIBPROC", None)


def test_own_process_is_positive():
    assert proc_memory.process_memory_mb(os.getpid()) > 0


def test_unknown_or_gone_process_is_zero():
    assert proc_memory.process_memory_mb(0) == 0.0
    assert proc_memory.process_memory_mb(-5) == 0.0
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    assert proc_memory.process_memory_mb(child.pid) == 0.0


def test_macos_reads_the_physical_footprint(monkeypatch):
    fake = _FakeLibproc(3650 * MB)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(proc_memory, "_libproc", lambda: fake)
    assert proc_memory.process_memory_mb(os.getpid()) == pytest.approx(3650.0)
    assert fake.calls == [(os.getpid(), 2)]  # RUSAGE_INFO_V2


def test_macos_falls_back_to_rss_when_the_reader_raises(monkeypatch):
    def boom():
        raise OSError("no libproc")

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(proc_memory, "_libproc", boom)
    assert proc_memory.process_memory_mb(os.getpid()) > 0


def test_macos_falls_back_to_rss_when_libproc_reports_failure(monkeypatch):
    fake = _FakeLibproc(3650 * MB, rc=-1)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(proc_memory, "_libproc", lambda: fake)
    value = proc_memory.process_memory_mb(os.getpid())
    assert 0 < value < 3650


def test_other_platforms_use_rss_and_never_touch_libproc(monkeypatch):
    def boom():
        raise AssertionError("libproc must not load off macOS")

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(proc_memory, "_libproc", boom)
    assert proc_memory.process_memory_mb(os.getpid()) > 0


def test_libproc_is_loaded_once(monkeypatch):
    loads = []
    monkeypatch.setattr(proc_memory.ctypes, "CDLL", lambda path: loads.append(path) or object())
    assert proc_memory._libproc() is proc_memory._libproc()
    assert len(loads) == 1


def test_tree_adds_the_children():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        own = proc_memory.process_memory_mb(os.getpid())
        tree = proc_memory.tree_memory_mb(os.getpid())
        assert tree >= own + proc_memory.process_memory_mb(child.pid) * 0.5
    finally:
        child.kill()
        child.wait()


def test_tree_of_a_gone_process_is_zero():
    assert proc_memory.tree_memory_mb(0) == 0.0
