# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""One daemon per data folder, decided by an operating-system file lock.

The process that holds this lock is the daemon of its data folder, and the only
process that may write the daemon record (``daemon.json`` and its mirrors). The
operating system drops the lock when the holder dies, however it dies, so a
crashed daemon never leaves a stale owner behind.

The lock file is never deleted: removing a locked file would let a second
process lock a brand-new file of the same name and believe it owns the folder.
"""

from __future__ import annotations

import errno
import logging
import os
import sys
import threading
import time
from pathlib import Path

from superlocalmemory.infra.daemon_identity import descriptor_path

INSTANCE_LOCK_NAME = "daemon.instance.lock"

logger = logging.getLogger(__name__)

_BUSY_ERRNOS = (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES, errno.EDEADLK)


def instance_lock_path(data_root: str | Path | None = None) -> Path:
    """Return the instance-lock path inside the selected data folder."""
    return descriptor_path(data_root).with_name(INSTANCE_LOCK_NAME)


class InstanceLock:
    """A non-blocking, process-held exclusive lock on one file."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._fd: int | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def held(self) -> bool:
        return self._fd is not None

    def try_acquire(self) -> bool:
        """Take the lock without waiting. True also when this object holds it."""
        if self._fd is not None:
            return True
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o600)
        except OSError as exc:
            logger.warning("could not open the instance lock %s: %s", self._path, exc)
            return False
        if not self._lock_fd(fd):
            os.close(fd)
            return False
        self._fd = fd
        if sys.platform != "win32":
            try:
                os.chmod(self._path, 0o600)
            except OSError:
                pass
        return True

    def still_owns_file(self) -> bool:
        """True while the locked file is still the file at the lock path.

        If someone removes or replaces the file, a second process could lock
        the new one and also believe it owns the folder; the holder must stop
        writing the record when that happens.
        """
        if self._fd is None:
            return False
        try:
            held, named = os.fstat(self._fd), os.stat(self._path)
        except OSError:
            return False
        return (held.st_dev, held.st_ino) == (named.st_dev, named.st_ino)

    def _lock_fd(self, fd: int) -> bool:
        try:
            if sys.platform == "win32":
                import msvcrt

                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError as exc:
            if exc.errno not in _BUSY_ERRNOS:
                logger.warning("instance lock %s failed: %s", self._path, exc)
            return False

    def release(self) -> None:
        """Unlock and close. The file stays on disk. Safe to call twice."""
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt

                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

    def __enter__(self) -> "InstanceLock":
        self.try_acquire()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()


_REGISTRY: dict[Path, InstanceLock] = {}
_REGISTRY_GUARD = threading.Lock()


def get_instance_lock(data_root: str | Path | None = None) -> InstanceLock:
    """Return this process's one lock object for the data folder.

    A second ``open`` of the same file in one process is a separate open file
    description, so it would contend with the first. Everything in the daemon
    process therefore shares this single object.
    """
    path = instance_lock_path(data_root)
    with _REGISTRY_GUARD:
        lock = _REGISTRY.get(path)
        if lock is None:
            lock = _REGISTRY[path] = InstanceLock(path)
        return lock


def _probe_held(path: Path) -> bool:
    """Look at the lock without ever taking it exclusively.

    POSIX uses a shared, non-blocking flock released at once, so the probe can
    never make a real owner's exclusive acquire succeed or hold it for long.
    Windows has no shared mode here: it locks byte 0 and unlocks at once, and
    the daemon's retry window absorbs a collision with that instant.
    """
    if not path.exists():
        return False
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return False
    try:
        if sys.platform == "win32":
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            except OSError:
                return True
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            return False
        import fcntl

        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError as exc:
            return exc.errno in _BUSY_ERRNOS
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except OSError:
        return False
    finally:
        os.close(fd)


def instance_lock_is_held(data_root: str | Path | None = None) -> bool:
    """Probe: True when some process (this one included) owns the data folder."""
    return _probe_held(instance_lock_path(data_root))


def acquire_with_backoff(
    lock: InstanceLock, wait_s: float, *, first_s: float = 0.1, max_s: float = 1.0,
) -> bool:
    """Try to take the lock, retrying (0.1 s doubling to 1 s) for ``wait_s``.

    A restart starts the new daemon while the old one is still letting go of
    the folder; this window turns that overlap into a short wait instead of a
    refusal.
    """
    deadline = time.monotonic() + max(0.0, wait_s)
    delay = first_s
    while not lock.try_acquire():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(delay, remaining))
        delay = min(delay * 2, max_s)
    return True


def wait_until_instance_lock_free(
    timeout_s: float = 30.0,
    poll_s: float = 0.1,
    data_root: str | Path | None = None,
) -> bool:
    """Wait for the previous daemon to let go of the data folder."""
    deadline = time.monotonic() + timeout_s
    while True:
        if not instance_lock_is_held(data_root):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll_s)
