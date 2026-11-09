# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Owner-only permissions for a secret file, on every platform.

On macOS and Linux a secret is created ``0600`` and that is the whole story.
On Windows the mode bits do nothing: a new file inherits its folder's access
list, which normally lets Administrators and SYSTEM read it and can let more
in. The remote TLS private key and the remote key store were written that way
(audit 4.1.20 L12).

This applies the same protected, owner-only access list the proxy capture file
already uses -- built and checked by the helpers in
``optimize/proxy/capture.py`` -- so there is one Windows ACL policy in the
product, not two.
"""

from __future__ import annotations

import os
from pathlib import Path


def restrict_to_owner(path: Path) -> None:
    """Make ``path`` readable and writable by the current user only.

    POSIX: ``chmod 600``. Windows: replace the access list with one protected
    entry for the current user (no inheritance), then read it back and raise
    ``OSError`` unless it is exactly that. Call it on a new file BEFORE any
    secret is written into it.
    """
    path = Path(path)
    if os.name != "nt":
        os.chmod(path, 0o600)
        return
    try:
        import ntsecuritycon
        import win32api
        import win32con
        import win32security
    except ImportError as exc:  # pywin32 is a declared Windows dependency
        raise OSError("Windows permission support (pywin32) is unavailable") from exc
    from superlocalmemory.optimize.proxy.capture import (
        _windows_dacl_is_owner_only,
        _windows_owner_dacl,
    )

    owner_sid, dacl = _windows_owner_dacl(win32api, win32con, win32security)
    # The owner is set too: under an elevated prompt Windows makes the
    # Administrators group the owner of a new file, and the policy is "this
    # user", not "whoever administers the machine".
    win32security.SetNamedSecurityInfo(
        os.fspath(path), win32security.SE_FILE_OBJECT,
        win32security.OWNER_SECURITY_INFORMATION
        | win32security.DACL_SECURITY_INFORMATION
        | win32security.PROTECTED_DACL_SECURITY_INFORMATION,
        owner_sid, None, dacl, None)
    descriptor = win32security.GetNamedSecurityInfo(
        os.fspath(path), win32security.SE_FILE_OBJECT,
        win32security.OWNER_SECURITY_INFORMATION
        | win32security.DACL_SECURITY_INFORMATION)
    if not _windows_dacl_is_owner_only(descriptor, owner_sid, ntsecuritycon,
                                       win32security):
        raise OSError(f"could not make {path.name} readable only by you")


def is_owner_only(path: Path) -> bool:
    """Whether only the current user can read ``path``.

    POSIX: no group or other permission bits. Windows: the access list is the
    protected owner-only one :func:`restrict_to_owner` sets.
    """
    path = Path(path)
    if os.name != "nt":
        return not (os.stat(path).st_mode & 0o077)
    try:
        import ntsecuritycon
        import win32api
        import win32con
        import win32security
    except ImportError as exc:  # pywin32 is a declared Windows dependency
        raise OSError("Windows permission support (pywin32) is unavailable") from exc
    from superlocalmemory.optimize.proxy.capture import (
        _windows_dacl_is_owner_only,
        _windows_owner_dacl,
    )

    owner_sid, _dacl = _windows_owner_dacl(win32api, win32con, win32security)
    descriptor = win32security.GetNamedSecurityInfo(
        os.fspath(path), win32security.SE_FILE_OBJECT,
        win32security.OWNER_SECURITY_INFORMATION
        | win32security.DACL_SECURITY_INFORMATION)
    return bool(_windows_dacl_is_owner_only(descriptor, owner_sid, ntsecuritycon,
                                            win32security))


__all__ = ["is_owner_only", "restrict_to_owner"]
