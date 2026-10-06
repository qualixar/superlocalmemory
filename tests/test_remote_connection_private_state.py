"""Windows ACL adapter probes are mocked; native Windows proof is separate."""
import os
import sys
from types import SimpleNamespace

import pytest

from superlocalmemory.remote_connections import private_state
from superlocalmemory.remote_connections.journal import EnrollmentJournal


def test_directory_failure_precedes_database_creation(tmp_path, monkeypatch):
    root = tmp_path / "remote"
    def refuse(path):
        assert path == root
        assert not (path / "enrollment.sqlite3").exists()
        raise OSError("synthetic ACL refusal")
    monkeypatch.setattr(private_state, "protect_directory", refuse)
    with pytest.raises(OSError):
        EnrollmentJournal(root)
    assert not (root / "enrollment.sqlite3").exists()


@pytest.mark.parametrize("verified,returned_flags", [(True, 3), (False, 3), (True, 0)])
def test_windows_directory_protection_verified_and_inherited(tmp_path, monkeypatch, verified, returned_flags):
    from superlocalmemory.optimize.proxy import capture
    calls = []
    class ACL:
        def AddAccessAllowedAceEx(self, revision, flags, mask, owner):
            calls.append(("ace", flags, mask, owner))
        def GetAce(self, index):
            return ((0, returned_flags), 0x1F01FF, "owner")
    descriptor = SimpleNamespace(GetSecurityDescriptorDacl=lambda: ACL())
    security = SimpleNamespace(ACL=ACL, ACL_REVISION=2, SE_FILE_OBJECT=1,
        OWNER_SECURITY_INFORMATION=1, DACL_SECURITY_INFORMATION=4,
        PROTECTED_DACL_SECURITY_INFORMATION=0x80000000,
        SetNamedSecurityInfo=lambda *args: calls.append(("set", args[0])),
        GetNamedSecurityInfo=lambda *args: descriptor)
    for name, module in {"ntsecuritycon": SimpleNamespace(FILE_ALL_ACCESS=0x1F01FF),
                         "win32api": SimpleNamespace(),
                         "win32con": SimpleNamespace(OBJECT_INHERIT_ACE=1, CONTAINER_INHERIT_ACE=2),
                         "win32security": security}.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(private_state, "os", SimpleNamespace(name="nt", fspath=os.fspath))
    monkeypatch.setattr(capture, "_windows_owner_dacl", lambda *args: ("owner", ACL()))
    monkeypatch.setattr(capture, "_windows_dacl_is_owner_only", lambda *args: verified)
    if verified and returned_flags == 3:
        private_state.protect_directory(tmp_path)
    else:
        with pytest.raises(OSError, match="verification failed"):
            private_state.protect_directory(tmp_path)
    assert calls[0] == ("ace", 3, 0x1F01FF, "owner")
    assert calls[1] == ("set", str(tmp_path))
