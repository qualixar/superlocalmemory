# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Platform safety tests for the bounded-loops MCP bridge.

Verifies that the bridge's executable trust checks behave correctly under
simulated Windows conditions.

Windows trust-check behaviour (documented limitation):
    On Windows, only the S_ISREG check applies.  The group/other-writable
    mode-bit check and the uid/ownership check are both skipped because:
      - Python's os.stat() on Windows returns emulated Unix mode bits that
        do not accurately reflect real ACL permissions; the check would be
        meaningless.
      - os.geteuid() does not exist on Windows; referencing it raises
        AttributeError.  Additionally, st_uid is always 0 on Windows, so
        the uid check could never identify a non-root owner anyway.

    Proper Windows ownership verification requires Win32 ACL APIs (advapi32)
    or the optional pywin32 package.  That is scoped for a future release.
    This file documents the current contract and ensures no regression causes
    the code to crash with AttributeError on the Windows code path.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
import unittest.mock
from pathlib import Path

import pytest

from superlocalmemory.integrations.bounded_loops_mcp import BridgeUnavailable


def _purge_module(dotted_name: str) -> None:
    """Remove a module and all sub-modules from sys.modules for a fresh import."""
    prefix = dotted_name + "."
    for key in list(sys.modules.keys()):
        if key == dotted_name or key.startswith(prefix):
            sys.modules.pop(key, None)


class TestBoundedLoopsMcpWindowsTrustChecks:
    """The bounded-loops bridge must not crash on Windows.

    The trust check section in observe_from_stdio guards its POSIX-specific
    calls with `if os.name != "nt":`.  This class verifies that guard is in
    place and holds under simulated Windows conditions.
    """

    def test_bounded_loops_mcp_imports_without_geteuid(self) -> None:
        """Importing the bridge module must not call os.geteuid at module level.

        Any reference to os.geteuid must be inside a function body, behind
        an os.name != 'nt' guard.  Patching geteuid to raise AttributeError
        and then importing the module is the mechanical proof.
        """
        _purge_module("superlocalmemory.integrations.bounded_loops_mcp")
        with unittest.mock.patch.object(
            os, "geteuid", side_effect=AttributeError("simulated: Windows has no geteuid"),
            create=True,  # on Windows there is no geteuid to replace
        ):
            importlib.import_module("superlocalmemory.integrations.bounded_loops_mcp")

    def test_trust_check_does_not_call_geteuid_on_windows(self, tmp_path: Path) -> None:
        """With os.name == 'nt', the owner check must short-circuit before calling geteuid.

        Setup:
          - A world-writable regular file is provided as the bridge executable.
            On POSIX this would fail the group/other-writable check.
          - os.name is patched to 'nt' (simulating Windows).
          - os.geteuid is patched to raise AttributeError (simulating its absence
            on Windows).
          - The mcp library is blocked so the function fails fast after the trust
            checks pass, without needing a real MCP server.

        Expected: the exception raised is NOT AttributeError.  If the guard is
        missing, os.geteuid() would be called before the mcp import, surfacing as
        AttributeError rather than the ModuleNotFoundError from the blocked mcp.

        Documented limitation recorded here: on Windows only S_ISREG applies;
        mode-bit and uid trust checks are skipped.
        """
        import superlocalmemory.integrations.bounded_loops_mcp as bl_mcp

        # A world-writable regular file — POSIX uid/permission checks would
        # reject this, but on simulated Windows those checks are skipped.
        exe = tmp_path / "fake_bridge.exe"
        exe.write_bytes(b"")
        exe.chmod(0o777)  # group- and other-writable

        with (
            # Simulate Windows: os.name == 'nt'
            unittest.mock.patch.object(os, "name", "nt"),
            # Simulate Windows: geteuid absent — raises AttributeError if called
            unittest.mock.patch.object(
                os,
                "geteuid",
                side_effect=AttributeError("simulated: Windows has no geteuid"),
                create=True,
            ),
            # Block mcp so the coroutine fails fast after the trust-check section
            # without needing a real bounded-loops server running.
            unittest.mock.patch.dict(sys.modules, {"mcp": None}),
        ):
            with pytest.raises(Exception) as exc_info:
                asyncio.run(
                    bl_mcp.observe_from_stdio(
                        command=str(exe.resolve()),
                        cwd=str(tmp_path.resolve()),
                        profile_id="test",
                    )
                )

        # The error must NOT be AttributeError — that would mean os.geteuid() was
        # called without the os.name != "nt" guard.
        assert not isinstance(exc_info.value, AttributeError), (
            f"AttributeError from os.geteuid leaked through the Windows guard: "
            f"{exc_info.value!r}"
        )

        # If BridgeUnavailable, it must NOT cite 'trusted regular file'.
        # A world-writable file must not trigger the POSIX mode-bit check on Windows.
        if isinstance(exc_info.value, BridgeUnavailable):
            assert "trusted regular file" not in str(exc_info.value), (
                "On Windows the mode-bit check should be skipped for world-writable files; "
                "got BridgeUnavailable with unexpected message"
            )


class TestAssertTrustedExecutable:
    """Direct unit tests for the extracted trust-check helper.

    Testing _assert_trusted_executable in isolation removes the dependency on
    the async MCP machinery in observe_from_stdio, so each assertion targets
    exactly one trust-check condition.
    """

    def test_rejects_non_regular_file(self, tmp_path: Path) -> None:
        """A directory is not a regular file; the check must reject it."""
        from superlocalmemory.integrations.bounded_loops_mcp import (
            BridgeUnavailable as _BU,
        )
        from superlocalmemory.integrations.bounded_loops_mcp import (
            _assert_trusted_executable,
        )
        with pytest.raises(_BU, match="trusted regular file"):
            _assert_trusted_executable(tmp_path)  # tmp_path is a directory

    def test_posix_rejects_group_writable(self, tmp_path: Path) -> None:
        """On POSIX, group-writable executable must be rejected."""
        from superlocalmemory.integrations.bounded_loops_mcp import (
            BridgeUnavailable as _BU,
        )
        from superlocalmemory.integrations.bounded_loops_mcp import (
            _assert_trusted_executable,
        )
        exe = tmp_path / "exe"
        exe.write_bytes(b"")
        exe.chmod(0o775)  # group-writable
        with (
            unittest.mock.patch.object(os, "name", "posix"),
            # geteuid returns the actual uid so the owner check passes
            unittest.mock.patch.object(os, "geteuid", return_value=exe.stat().st_uid,
                                       create=True),  # absent on Windows
        ):
            with pytest.raises(_BU, match="trusted regular file"):
                _assert_trusted_executable(exe)

    def test_windows_allows_group_writable(self, tmp_path: Path) -> None:
        """On Windows the mode-bit check is skipped; world-writable file must pass."""
        from superlocalmemory.integrations.bounded_loops_mcp import _assert_trusted_executable
        exe = tmp_path / "exe"
        exe.write_bytes(b"")
        exe.chmod(0o777)  # would be rejected on POSIX
        with (
            unittest.mock.patch.object(os, "name", "nt"),
            # Simulate Windows: geteuid absent
            unittest.mock.patch.object(
                os, "geteuid", side_effect=AttributeError("no geteuid on Windows"), create=True
            ),
        ):
            _assert_trusted_executable(exe)  # must not raise

    def test_windows_does_not_call_geteuid(self, tmp_path: Path) -> None:
        """With os.name == 'nt', geteuid must never be called.

        If the guard is missing, geteuid raises AttributeError — that is the
        failure mode this test catches.
        """
        from superlocalmemory.integrations.bounded_loops_mcp import _assert_trusted_executable
        exe = tmp_path / "exe"
        exe.write_bytes(b"")
        with (
            unittest.mock.patch.object(os, "name", "nt"),
            unittest.mock.patch.object(
                os, "geteuid", side_effect=AttributeError("simulated: no geteuid"), create=True
            ),
        ):
            _assert_trusted_executable(exe)  # must not raise AttributeError
