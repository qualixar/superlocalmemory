# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""L-CONC-1: WorkerPool._kill must deterministically close the worker pipes.

Leaving the pipe fds to GC let an orphaned _readline_with_timeout reader thread
keep the stdout file object alive, so repeated request timeouts could leak
threads + file descriptors. _kill() now closes stdin/stdout/stderr explicitly.
"""

from __future__ import annotations

from superlocalmemory.core.worker_pool import WorkerPool


class _FakeStream:
    def __init__(self, on_write=None) -> None:
        self.closed = False
        self._on_write = on_write

    def write(self, _data) -> None:
        if self._on_write is not None:
            self._on_write()

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _FakeProc:
    def __init__(self, *, write_raises: bool = False) -> None:
        self.pid = 4321
        self.killed = False
        self.waits = 0

        def _boom() -> None:
            raise BrokenPipeError("worker gone")

        self.stdin = _FakeStream(on_write=_boom if write_raises else None)
        self.stdout = _FakeStream()
        self.stderr = None  # matches DEVNULL stderr (no pipe)

    def wait(self, timeout=None):  # noqa: ANN001
        self.waits += 1
        return 0

    def kill(self) -> None:
        self.killed = True


def _pool_with(proc: _FakeProc) -> WorkerPool:
    pool = WorkerPool.__new__(WorkerPool)  # bypass singleton __init__
    pool._idle_timer = None
    pool._proc = proc
    return pool


def test_kill_closes_pipes_on_graceful_quit() -> None:
    proc = _FakeProc(write_raises=False)
    pool = _pool_with(proc)

    pool._kill()

    assert proc.stdin.closed is True
    assert proc.stdout.closed is True
    assert pool._proc is None


def test_kill_closes_pipes_even_when_quit_write_fails() -> None:
    proc = _FakeProc(write_raises=True)
    pool = _pool_with(proc)

    pool._kill()

    assert proc.killed is True  # fell through to hard kill
    assert proc.stdin.closed is True
    assert proc.stdout.closed is True
    assert pool._proc is None
