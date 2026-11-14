"""Service definitions share the registry's pinned, no-follow atomic IO."""

from __future__ import annotations

import contextlib
import hashlib
import os
from collections.abc import Iterator
from pathlib import Path

from superlocalmemory.infra.mcp_registry_files import (
    RegistryFiles,
    _open_root,
    _owned,
    _windows_pin,
)


class ServiceFiles(RegistryFiles):
    def __init__(self, directory: Path, *, create: bool = False):
        super().__init__(directory.parent, create=create)
        self.path = directory

    def __enter__(self):
        try:
            if self.path.resolve() != self.path.absolute():
                raise OSError("Service definition directory is redirected")
            if os.name == "nt":
                if self.create:
                    self.path.mkdir(parents=True, exist_ok=True)
                self._pins.enter_context(_windows_pin(self.path))
            else:
                self.fd = _open_root(self.path, create=self.create)
                _owned(os.fstat(self.fd), directory=True)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def snapshot(self, name: str) -> tuple[str, tuple]:
        with self.verified(name) as bound:
            return bound.content, bound.revision

    @contextlib.contextmanager
    def verified(
        self, name: str, content: str | None = None, revision: tuple | None = None,
    ) -> Iterator[BoundDefinition]:
        """Keep the checked file open; recheck its name, bytes and directory."""
        before = self._check(name)
        fd = _definition_fd(self, name)
        try:
            _owned(os.fstat(fd))
            if not os.path.samestat(before, os.fstat(fd)):
                raise OSError("Service definition changed while opening")
            bound = BoundDefinition(self, name, fd)
            if content is not None and bound.content != content:
                raise OSError("Service definition content changed")
            if revision is not None and bound.revision != revision:
                raise OSError("Service definition identity changed")
            bound.check()
            yield bound
        finally:
            os.close(fd)


def _definition_fd(files: ServiceFiles, name: str) -> int:
    if os.name != "nt":
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        return os.open(files._target(name), flags, **files._kwargs())
    import ctypes
    import msvcrt
    from ctypes import wintypes

    from superlocalmemory.infra.mcp_registry_files import _windows_owned

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                      wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    # Read only; deny write AND delete while validating/mutating the OS task.
    handle = create(str(files.path / name), 0x80030000, 1, None, 3, 0x00200000, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY)
    except BaseException:
        close = kernel.CloseHandle
        close.argtypes = [wintypes.HANDLE]
        close.restype = wintypes.BOOL
        close(handle)
        raise
    try:
        _windows_owned(handle)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _revision(info: os.stat_result) -> tuple:
    """The fields that must not change between authorizing and acting.

    On Windows ``st_ctime_ns`` is left out (issue #319): CPython's path
    ``stat`` and handle ``fstat`` can report different values for one
    unchanged file, so the guard refused every freshly written wrapper.
    Nothing is lost: the bytes are pinned by the SHA-256 digest, and the
    handle is opened deny-write/deny-delete while it is checked and used.
    """
    revision = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    return revision if os.name == "nt" else (*revision, info.st_ctime_ns)


class BoundDefinition:
    """The file descriptor and exact byte digest that authorized removal."""

    def __init__(self, files: ServiceFiles, name: str, fd: int) -> None:
        self.files, self.name, self.fd = files, name, fd
        self.bound_stat = os.fstat(fd)
        data = self._bytes()
        self.digest = hashlib.sha256(data).digest()
        self.content = data.decode("utf-8").replace("\r\n", "\n")
        self.directory = files.path.stat()

    @property
    def revision(self) -> tuple:
        return _revision(self.bound_stat)

    def _bytes(self) -> bytes:
        os.lseek(self.fd, 0, os.SEEK_SET)
        chunks = []
        while chunk := os.read(self.fd, 65536):
            chunks.append(chunk)
        return b"".join(chunks)

    def check(self) -> None:
        current = self.files._check(self.name)
        if not os.path.samestat(self.directory, self.files.path.stat()):
            raise OSError("Service directory changed")
        if self.files.fd is not None and not os.path.samestat(
            self.directory, os.fstat(self.files.fd)
        ):
            raise OSError("Service directory was redirected")
        _owned(os.fstat(self.fd))
        bound = self.revision
        if (_revision(current) != bound
                or _revision(os.fstat(self.fd)) != bound
                or hashlib.sha256(self._bytes()).digest() != self.digest):
            raise OSError("Verified service definition changed")

    def remove(self) -> None:
        """Delete the verified handle on Windows, checked directory entry on POSIX."""
        self.check()
        if os.name != "nt":
            os.unlink(self.files._target(self.name), **self.files._kwargs())
            return
        import ctypes
        import msvcrt
        from ctypes import wintypes

        class Disposition(ctypes.Structure):
            _fields_ = [("DeleteFile", ctypes.c_ubyte)]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        dispose = kernel.SetFileInformationByHandle
        dispose.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
        dispose.restype = wintypes.BOOL
        info = Disposition(True)
        if not dispose(msvcrt.get_osfhandle(self.fd), 4, ctypes.byref(info), ctypes.sizeof(info)):
            raise ctypes.WinError(ctypes.get_last_error())
