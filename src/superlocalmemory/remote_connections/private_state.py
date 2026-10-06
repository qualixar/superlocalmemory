"""Owner-only journal directory, including newly created SQLite sidecars."""
from __future__ import annotations

import os
from pathlib import Path


def protect_directory(root: Path) -> None:
    """Set and verify the owner boundary before opening any journal file."""
    if os.name == "posix":
        if root.stat().st_uid != os.getuid():
            raise ValueError("unsafe_journal_path")
        os.chmod(root, 0o700)
        return
    try:
        import ntsecuritycon
        import win32api
        import win32con
        import win32security
    except ImportError as exc:
        raise OSError("Windows owner-only directory support unavailable") from exc
    from superlocalmemory.optimize.proxy.capture import (
        _windows_owner_dacl, _windows_dacl_is_owner_only,
    )

    owner, _ = _windows_owner_dacl(win32api, win32con, win32security)
    # Both flags are needed: new SQLite journal/WAL/SHM files must inherit the
    # same owner boundary, not the process/default security descriptor.
    inherit = (getattr(win32con, "OBJECT_INHERIT_ACE", 1)
               | getattr(win32con, "CONTAINER_INHERIT_ACE", 2))
    acl = win32security.ACL()
    acl.AddAccessAllowedAceEx(win32security.ACL_REVISION, inherit,
                             ntsecuritycon.FILE_ALL_ACCESS, owner)
    win32security.SetNamedSecurityInfo(
        os.fspath(root), win32security.SE_FILE_OBJECT,
        win32security.OWNER_SECURITY_INFORMATION | win32security.DACL_SECURITY_INFORMATION
        | win32security.PROTECTED_DACL_SECURITY_INFORMATION, owner, None, acl, None)
    descriptor = win32security.GetNamedSecurityInfo(
        os.fspath(root), win32security.SE_FILE_OBJECT,
        win32security.OWNER_SECURITY_INFORMATION | win32security.DACL_SECURITY_INFORMATION)
    if (not _windows_dacl_is_owner_only(descriptor, owner, ntsecuritycon, win32security)
            or descriptor.GetSecurityDescriptorDacl().GetAce(0)[0][1] != inherit):
        raise OSError("owner-only directory verification failed")
