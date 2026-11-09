# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Platform-import safety tests.

Verifies that modules with platform-specific dependencies can be imported
on any supported OS, including Windows, without crashing.

The technique: remove the module under test from sys.modules, then block
the platform-specific stdlib module (by setting sys.modules[name] = None,
which the Python import machinery treats as "unavailable"). A fresh import
of the module under test must succeed — any platform-specific stdlib usage
must be deferred to call time inside the appropriate platform branch, not
executed at module import time.
"""

from __future__ import annotations

import importlib
import sys
import unittest.mock
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


@contextmanager
def _blocked_module(name: str) -> Generator[None, None, None]:
    """Temporarily make ``import <name>`` raise ImportError.

    Works by placing None in sys.modules[name], which the CPython import
    machinery treats as a blocked entry.  Restores the original state on
    exit whether or not the body raises.
    """
    previously_present = name in sys.modules
    original = sys.modules.get(name)
    sys.modules[name] = None  # type: ignore[assignment]
    try:
        yield
    finally:
        if previously_present:
            sys.modules[name] = original  # type: ignore[assignment]
        else:
            sys.modules.pop(name, None)


def _purge_module(dotted_name: str) -> None:
    """Remove a module and all of its sub-modules from sys.modules.

    Required before testing a fresh import, because Python caches
    already-imported modules and would skip the import machinery entirely.
    """
    prefix = dotted_name + "."
    for key in list(sys.modules.keys()):
        if key == dotted_name or key.startswith(prefix):
            sys.modules.pop(key, None)


class TestPlistlibImportGuard:
    """service_installer must import cleanly even when plistlib is absent.

    plistlib is a macOS-only stdlib module.  On Windows it does not exist.
    Placing any import of it at module level causes an ImportError before
    any user code can run, making the entire CLI unusable on Windows.
    """

    def test_service_installer_imports_without_plistlib(self) -> None:
        """Importing service_installer must not require plistlib.

        plistlib must not be imported at module level.  It is only needed
        inside the macOS branch, so the import must be deferred to that
        branch's call site.
        """
        _purge_module("superlocalmemory.cli.service_installer")
        with _blocked_module("plistlib"):
            # This must succeed.  If plistlib is still imported at module
            # level, the import machinery raises ImportError here and the
            # test fails with a clear traceback pointing at the offending
            # import statement.
            importlib.import_module("superlocalmemory.cli.service_installer")

    def test_plistlib_only_used_inside_macos_functions(self) -> None:
        """plistlib must only be reachable through the macOS code path.

        Cross-platform dispatchers (install_service, uninstall_service,
        service_status) must not touch plistlib on non-macOS platforms.
        Calling the Linux or Windows branches with plistlib blocked must
        succeed end-to-end.
        """
        _purge_module("superlocalmemory.cli.service_installer")
        with _blocked_module("plistlib"):
            import superlocalmemory.cli.service_installer as si

            # Linux branch must not touch plistlib at all.
            content = si._linux_service_content()
            assert "--start" in content

            # Windows VBS branch must not touch plistlib at all.
            vbs = si._windows_vbs_content()
            assert "SLM_DATA_DIR" in vbs


class TestWindowsPlatformDispatch:
    """install_service / uninstall_service / service_status must never
    enter the macOS branch when sys.platform is 'win32'.

    plistlib is blocked so any accidental call into _macos_plist_content
    would surface immediately as an ImportError rather than silently
    succeeding on a macOS developer machine.
    """

    def test_dispatchers_skip_macos_branch_on_windows(self) -> None:
        """With sys.platform == 'win32', the macOS plist branch is never entered.

        We block plistlib and spy on _macos_plist_content; if the Windows
        dispatch accidentally calls the macOS path, the test fails on either
        the ImportError (from the blocked module) or the mock call assertion.
        """
        _purge_module("superlocalmemory.cli.service_installer")
        with _blocked_module("plistlib"):
            import superlocalmemory.cli.service_installer as si

            with (
                unittest.mock.patch.object(sys, "platform", "win32"),
                unittest.mock.patch.object(
                    si, "_macos_plist_content", wraps=si._macos_plist_content
                ) as mock_plist,
                # Prevent actual schtasks / subprocess calls on non-Windows hosts.
                unittest.mock.patch(
                    "superlocalmemory.cli.service_installer.subprocess.run",
                    return_value=unittest.mock.Mock(returncode=0, stdout="", stderr=""),
                ),
                # Prevent file writes (vbs_path.write_text) in install_windows.
                unittest.mock.patch(
                    "superlocalmemory.cli.service_installer.state_path",
                    return_value=unittest.mock.MagicMock(
                        exists=lambda: False,
                        write_text=lambda *a, **kw: None,
                        unlink=lambda: None,
                    ),
                ),
            ):
                si.install_service()
                si.uninstall_service()
                si.service_status()

            # The macOS branch must never have been entered.
            mock_plist.assert_not_called()


class TestFcntlMigrationGuard:
    """Migration module must import cleanly when fcntl is absent (Windows).

    fcntl is POSIX-only.  The migration serialises concurrent callers with
    a platform-dispatched lock: Windows uses msvcrt, POSIX uses fcntl.
    The fcntl import must be inside the POSIX branch, never at module level.
    """

    def test_migration_v3_4_25_to_v3_4_26_imports_without_fcntl(
        self, monkeypatch
    ) -> None:
        """Importing the migration module must not require fcntl.

        Saves and restores BOTH sys.modules[mod_name] AND the parent package
        attribute (superlocalmemory.migrations.v3_4_25_to_v3_4_26) so that
        downstream tests whose module-level imports bind to the original
        module object are unaffected by this re-import inside fcntl-blocked
        context.
        """
        mod_name = "superlocalmemory.migrations.v3_4_25_to_v3_4_26"
        parent_name = "superlocalmemory.migrations"
        attr = "v3_4_25_to_v3_4_26"

        # Save sys.modules entry for restoration.
        if mod_name in sys.modules:
            monkeypatch.setitem(sys.modules, mod_name, sys.modules[mod_name])
        # Save parent-package attribute for restoration (Python also sets this
        # during import, so monkeypatch must restore it too).
        parent_pkg = sys.modules.get(parent_name)
        if parent_pkg is not None and hasattr(parent_pkg, attr):
            monkeypatch.setattr(parent_pkg, attr, getattr(parent_pkg, attr))

        _purge_module(mod_name)
        with _blocked_module("fcntl"):
            importlib.import_module(mod_name)
        _purge_module(mod_name)  # Remove stub so monkeypatch restores the real entry.

    def test_ram_lock_imports_without_fcntl(self, monkeypatch) -> None:
        """ram_lock uses try/except ImportError for fcntl — must always import.

        The module guards its fcntl usage through a try/except ImportError
        block and checks whether the module object is None before each use.
        This test confirms that guarantee remains intact.

        Saves and restores BOTH sys.modules[mod_name] AND the parent package
        attribute so that ram_lock tests running after this test always get
        the real module (with _fcntl set to the fcntl module) and never fall
        through to the Windows msvcrt branch.
        """
        mod_name = "superlocalmemory.core.ram_lock"
        parent_name = "superlocalmemory.core"
        attr = "ram_lock"

        # Save sys.modules entry for restoration.
        if mod_name in sys.modules:
            monkeypatch.setitem(sys.modules, mod_name, sys.modules[mod_name])
        # Save parent-package attribute for restoration.
        parent_pkg = sys.modules.get(parent_name)
        if parent_pkg is not None and hasattr(parent_pkg, attr):
            monkeypatch.setattr(parent_pkg, attr, getattr(parent_pkg, attr))

        _purge_module(mod_name)
        with _blocked_module("fcntl"):
            mod = importlib.import_module(mod_name)
            # The internal alias must be None when fcntl is unavailable.
            assert mod._fcntl is None, (  # type: ignore[attr-defined]
                "ram_lock._fcntl should be None when fcntl is absent"
            )
        _purge_module(mod_name)  # Remove stub so monkeypatch restores the real entry.


class TestFcntlMigrationCallPath:
    """The migration's Windows locking path must call msvcrt, never fcntl.

    The module-import test in TestFcntlMigrationGuard passes even if the
    call-time guard were removed, because fcntl is never imported at module
    level.  This class tests the actual runtime dispatch: that with
    sys.platform == 'win32', the locking code takes the msvcrt branch and
    never touches the (blocked) fcntl module.
    """

    def test_migrate_if_safe_uses_msvcrt_on_windows(self, tmp_path: Path) -> None:
        """sys.platform == 'win32' + fcntl blocked => msvcrt.locking is called once.

        Without the msvcrt mock this test fails with ModuleNotFoundError on macOS:
        the Windows branch executes `import msvcrt`, which is absent on POSIX, and
        that ImportError propagates because the guard only catches IOError/OSError.
        That is the RED state before the msvcrt mock is provided.
        """
        import types

        import superlocalmemory.migrations.v3_4_25_to_v3_4_26 as mig

        # Minimal msvcrt stand-in with the two names the migration uses.
        locking_calls: list = []
        fake_msvcrt = types.ModuleType("msvcrt")
        fake_msvcrt.LK_LOCK = 2  # type: ignore[attr-defined]
        fake_msvcrt.locking = lambda fd, mode, n: locking_calls.append((fd, mode, n))  # type: ignore[attr-defined]

        with (
            unittest.mock.patch.object(sys, "platform", "win32"),
            _blocked_module("fcntl"),
            unittest.mock.patch.dict(sys.modules, {"msvcrt": fake_msvcrt}),
            # Avoid hitting the real daemon and real RecallQueue.
            unittest.mock.patch.object(mig, "_daemon_running", return_value=False),
            unittest.mock.patch.object(
                mig, "migrate", return_value={"data_dir": str(tmp_path)}
            ),
        ):
            result = mig.migrate_if_safe(tmp_path)

        assert result["status"] == "applied", f"unexpected status: {result}"
        # msvcrt.locking must have been called exactly once — proving the
        # Windows branch ran — and fcntl was never imported (it stayed blocked).
        assert len(locking_calls) == 1, (
            f"expected msvcrt.locking called once; got {locking_calls}"
        )
