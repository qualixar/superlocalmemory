"""Windows handle contracts can be checked without calling the live OS manager."""

import ctypes
import os
import sys
from types import SimpleNamespace

import pytest

from superlocalmemory.infra import mcp_registry_files, service_files


class Call:
    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def __call__(self, *args):
        self.calls.append(args)
        return self.result


def windows_api(monkeypatch):
    kernel = SimpleNamespace(CreateFileW=Call(900), CloseHandle=Call(1),
                             SetFileInformationByHandle=Call(1))
    monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **kw: kernel, raising=False)
    proxy = SimpleNamespace(**vars(os))
    proxy.name = "nt"
    monkeypatch.setattr(service_files, "os", proxy)
    return kernel


def test_failed_fd_conversion_releases_raw_deny_write_delete_handle(monkeypatch, tmp_path):
    kernel = windows_api(monkeypatch)

    def cannot_convert(*args):
        raise OSError("synthetic CRT descriptor exhaustion")

    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(open_osfhandle=cannot_convert))
    files = service_files.ServiceFiles(tmp_path)
    with pytest.raises(OSError, match="descriptor exhaustion"):
        service_files._definition_fd(files, "owned.vbs")
    assert kernel.CloseHandle.calls == [(900,)]
    args = kernel.CreateFileW.calls[0]
    assert args[1:3] == (0x80030000, 1)  # read + delete, share read only


def test_failed_owner_check_closes_converted_fd(monkeypatch, tmp_path):
    descriptor = os.open(tmp_path / "wrapper", os.O_CREAT | os.O_RDONLY, 0o600)
    kernel = windows_api(monkeypatch)
    monkeypatch.setitem(sys.modules, "msvcrt",
                        SimpleNamespace(open_osfhandle=lambda *a: descriptor))

    def foreign_owner(*args):
        raise OSError("foreign owner")

    monkeypatch.setattr(mcp_registry_files, "_windows_owned", foreign_owner)
    files = service_files.ServiceFiles(tmp_path)
    with pytest.raises(OSError, match="foreign owner"):
        service_files._definition_fd(files, "wrapper")
    with pytest.raises(OSError):
        os.fstat(descriptor)
    assert kernel.CloseHandle.calls == []  # CRT descriptor owns the raw handle


def test_windows_removal_targets_bound_handle_and_byte_sized_disposition(monkeypatch, tmp_path):
    (tmp_path / "owned.vbs").write_text("owned content")
    with service_files.ServiceFiles(tmp_path) as files:
        with files.verified("owned.vbs") as bound:
            kernel = windows_api(monkeypatch)
            monkeypatch.setitem(sys.modules, "msvcrt",
                                SimpleNamespace(get_osfhandle=lambda fd: 900))
            bound.remove()
            handle, kind, pointer, size = kernel.SetFileInformationByHandle.calls[0]
            assert handle == 900
            assert kind == 4  # FileDispositionInfo
            assert size == 1  # FILE_DISPOSITION_INFO.DeleteFile is BOOLEAN, not BOOL
            assert ctypes.cast(pointer, ctypes.POINTER(ctypes.c_ubyte)).contents.value == 1
    # Read after the guard handle closes: on real Windows it denies other
    # opens while held. The file is still there, so nothing unlinked a path.
    assert (tmp_path / "owned.vbs").read_text() == "owned content"


def _fake_stat(*, dev=7, ino=11, size=64, mtime_ns=2_000_000_000, ctime_ns):
    return SimpleNamespace(st_dev=dev, st_ino=ino, st_size=size,
                           st_mtime_ns=mtime_ns, st_ctime_ns=ctime_ns)


def test_windows_revision_ignores_ctime_that_stat_and_fstat_disagree_on(monkeypatch):
    """Issue #319: on Windows, CPython's path stat and handle fstat can report
    different st_ctime_ns for one unchanged file (identity, size and mtime
    agree). The guard compared that field, so every install refused its own
    freshly written wrapper. Content is still pinned by the SHA-256 digest
    and the deny-write handle, so dropping ctime there weakens nothing."""
    proxy = SimpleNamespace(**vars(os))
    proxy.name = "nt"
    monkeypatch.setattr(service_files, "os", proxy)
    by_path, by_handle = _fake_stat(ctime_ns=1), _fake_stat(ctime_ns=999)
    assert service_files._revision(by_path) == service_files._revision(by_handle)
    # Identity, size and mtime still count.
    assert service_files._revision(by_path) != service_files._revision(
        _fake_stat(ctime_ns=1, size=65))
    assert service_files._revision(by_path) != service_files._revision(
        _fake_stat(ctime_ns=1, mtime_ns=3))


def test_posix_revision_still_includes_ctime(monkeypatch):
    proxy = SimpleNamespace(**vars(os))
    proxy.name = "posix"
    monkeypatch.setattr(service_files, "os", proxy)
    assert service_files._revision(_fake_stat(ctime_ns=1)) != service_files._revision(
        _fake_stat(ctime_ns=2))


@pytest.mark.skipif(sys.platform != "win32", reason="the real Windows stat/fstat pair")
def test_real_windows_wrapper_written_then_verified_passes_its_own_check(tmp_path):
    """Issue #319 end to end on a real Windows filesystem: write the wrapper
    the way `slm serve install` does, then bind and re-check it."""
    directory = tmp_path / "service"
    with service_files.ServiceFiles(directory, create=True) as files:
        files.write("start-daemon.vbs", 'WScript.Echo "slm"\n')
    with service_files.ServiceFiles(directory) as files:
        with files.verified("start-daemon.vbs") as bound:
            bound.check()
        content, revision = files.snapshot("start-daemon.vbs")
        with files.verified("start-daemon.vbs", content, revision) as bound:
            bound.check()
