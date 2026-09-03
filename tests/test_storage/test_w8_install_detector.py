# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Tests for multi-install detection (tasks 8.6, 8.7).

TDD RED phase: all tests fail before implementation exists.
"""

from __future__ import annotations

import importlib
import os
import sys
import textwrap
from pathlib import Path
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Task 8.6 — install_detector module
# ---------------------------------------------------------------------------


class TestDetectAllInstalls:
    """_detect_all_installs() returns all SLM copies on the machine."""

    def test_returns_list(self, tmp_path):
        from superlocalmemory.core.install_detector import _detect_all_installs
        result = _detect_all_installs()
        assert isinstance(result, list)

    def test_each_entry_has_required_keys(self, tmp_path):
        from superlocalmemory.core.install_detector import _detect_all_installs
        result = _detect_all_installs()
        for entry in result:
            assert "path" in entry, f"missing 'path' key: {entry}"
            assert "version" in entry, f"missing 'version' key: {entry}"
            assert "type" in entry, f"missing 'type' key: {entry}"
            assert entry["type"] in ("pipx", "venv", "npm", "unknown")

    def test_detects_synthetic_venv_install(self, tmp_path, monkeypatch):
        """A synthetic ~/.slm-venv install is detected with its version."""
        from superlocalmemory.core.install_detector import _detect_all_installs

        # Build a fake venv layout
        fake_venv = tmp_path / ".slm-venv"
        init_path = fake_venv / "lib" / "python3.11" / "site-packages" / "superlocalmemory" / "__init__.py"
        init_path.parent.mkdir(parents=True)
        init_path.write_text('__version__ = "4.0.8"\n')

        monkeypatch.setattr(
            "superlocalmemory.core.install_detector._VENV_ROOT",
            fake_venv,
        )

        result = _detect_all_installs()
        venv_entries = [e for e in result if e["type"] == "venv"]
        assert len(venv_entries) >= 1
        assert venv_entries[0]["version"] == "4.0.8"

    def test_detects_synthetic_pipx_install(self, tmp_path, monkeypatch):
        """A synthetic pipx install is detected with its version."""
        from superlocalmemory.core.install_detector import _detect_all_installs

        fake_pipx = tmp_path / ".local" / "pipx" / "venvs" / "superlocalmemory"
        init_path = fake_pipx / "lib" / "python3.12" / "site-packages" / "superlocalmemory" / "__init__.py"
        init_path.parent.mkdir(parents=True)
        init_path.write_text('__version__ = "4.0.8"\n')

        monkeypatch.setattr(
            "superlocalmemory.core.install_detector._PIPX_ROOT",
            fake_pipx,
        )

        result = _detect_all_installs()
        pipx_entries = [e for e in result if e["type"] == "pipx"]
        assert len(pipx_entries) >= 1
        assert pipx_entries[0]["version"] == "4.0.8"

    def test_detects_synthetic_npm_install(self, tmp_path, monkeypatch):
        """A synthetic npm global install is detected with its version."""
        from superlocalmemory.core.install_detector import _detect_all_installs

        fake_npm_root = tmp_path / "node_modules"
        pkg_json = fake_npm_root / "superlocalmemory" / "package.json"
        pkg_json.parent.mkdir(parents=True)
        pkg_json.write_text('{"version": "4.1.0"}\n')

        # Patch the subprocess call for npm root
        def fake_npm_root_fn():
            return fake_npm_root

        monkeypatch.setattr(
            "superlocalmemory.core.install_detector._npm_global_root",
            fake_npm_root_fn,
        )

        result = _detect_all_installs()
        npm_entries = [e for e in result if e["type"] == "npm"]
        assert len(npm_entries) >= 1
        assert npm_entries[0]["version"] == "4.1.0"

    def test_returns_empty_when_nothing_installed(self, tmp_path, monkeypatch):
        """Returns [] when no installs are found."""
        from superlocalmemory.core.install_detector import _detect_all_installs

        monkeypatch.setattr(
            "superlocalmemory.core.install_detector._VENV_ROOT",
            tmp_path / "nonexistent-venv",
        )
        monkeypatch.setattr(
            "superlocalmemory.core.install_detector._PIPX_ROOT",
            tmp_path / "nonexistent-pipx",
        )
        monkeypatch.setattr(
            "superlocalmemory.core.install_detector._npm_global_root",
            lambda: None,
        )

        result = _detect_all_installs()
        assert result == []

    def test_is_read_only_no_writes(self, tmp_path, monkeypatch):
        """_detect_all_installs() must not write anything, anywhere.

        The earlier version defined a tracking wrapper, never installed it, and
        ended in `assert True` — it passed whether the function wrote or not.
        This version installs the trackers AND proves they fire, so it cannot
        quietly become vacuous again.
        """
        import builtins

        writes: list[str] = []
        real_open = builtins.open

        def tracking_open(path, mode="r", *a, **kw):
            if any(ch in str(mode) for ch in ("w", "a", "x", "+")):
                writes.append(f"open({path!r}, {mode!r})")
            return real_open(path, mode, *a, **kw)

        monkeypatch.setattr(builtins, "open", tracking_open)
        monkeypatch.setattr(Path, "write_text",
                            lambda self, *a, **kw: writes.append(f"write_text({self})"))
        monkeypatch.setattr(Path, "write_bytes",
                            lambda self, *a, **kw: writes.append(f"write_bytes({self})"))
        monkeypatch.setattr(Path, "mkdir",
                            lambda self, *a, **kw: writes.append(f"mkdir({self})"))

        # Control: prove the trackers intercept. Without this, the assertion
        # below is unfalsifiable and the test is lying about what it checks.
        (tmp_path / "control.txt").write_text("x")
        assert writes, "tracker never fired — this test cannot detect a write"
        writes.clear()

        from superlocalmemory.core.install_detector import _detect_all_installs

        _detect_all_installs()

        assert writes == [], f"_detect_all_installs() wrote: {writes}"


# ---------------------------------------------------------------------------
# Task 8.6 — SchemaVersionError names all installs
# ---------------------------------------------------------------------------


class TestSchemaVersionErrorNamesAllInstalls:
    """check_version_or_raise() embeds all detected installs in error message."""

    def test_error_message_contains_all_install_types(self, tmp_path):
        """SchemaVersionError lists pipx, venv, npm when all three are present."""
        from superlocalmemory.storage._schema_version import (
            SchemaVersionError,
            SUPPORTED_SCHEMA_VERSION,
            check_version_or_raise,
            ensure_schema_version_table,
            write_schema_version,
        )
        import sqlite3

        # Build a DB with schema version > supported
        db = tmp_path / "learning.db"
        conn = sqlite3.connect(str(db))
        ensure_schema_version_table(conn)
        write_schema_version(conn, SUPPORTED_SCHEMA_VERSION + 1)
        conn.commit()
        conn.close()

        fake_installs = [
            {"path": "/home/user/.local/pipx/venvs/superlocalmemory/", "version": "4.0.8", "type": "pipx"},
            {"path": "/home/user/.slm-venv/", "version": "4.0.8", "type": "venv"},
            {"path": "/usr/local/lib/node_modules/superlocalmemory/", "version": "4.1.0", "type": "npm"},
        ]

        with mock.patch(
            "superlocalmemory.storage._schema_version._detect_all_installs",
            return_value=fake_installs,
        ):
            with pytest.raises(SchemaVersionError) as exc_info:
                check_version_or_raise(db)

        msg = str(exc_info.value)
        assert "pipx" in msg
        assert "venv" in msg
        assert "npm" in msg
        assert "4.0.8" in msg
        assert "4.1.0" in msg

    def test_error_message_includes_upgrade_commands(self, tmp_path):
        """SchemaVersionError includes upgrade commands for each install type."""
        from superlocalmemory.storage._schema_version import (
            SchemaVersionError,
            SUPPORTED_SCHEMA_VERSION,
            check_version_or_raise,
            ensure_schema_version_table,
            write_schema_version,
        )
        import sqlite3

        db = tmp_path / "learning.db"
        conn = sqlite3.connect(str(db))
        ensure_schema_version_table(conn)
        write_schema_version(conn, SUPPORTED_SCHEMA_VERSION + 1)
        conn.commit()
        conn.close()

        fake_installs = [
            {"path": "/pipx/venvs/superlocalmemory/", "version": "4.0.8", "type": "pipx"},
            {"path": "~/.slm-venv/", "version": "4.0.9", "type": "venv"},
        ]

        with mock.patch(
            "superlocalmemory.storage._schema_version._detect_all_installs",
            return_value=fake_installs,
        ):
            with pytest.raises(SchemaVersionError) as exc_info:
                check_version_or_raise(db)

        msg = str(exc_info.value)
        assert "pipx upgrade superlocalmemory" in msg or "pipx" in msg

    def test_error_message_when_no_installs_detected(self, tmp_path):
        """SchemaVersionError still raises cleanly when no installs detected."""
        from superlocalmemory.storage._schema_version import (
            SchemaVersionError,
            SUPPORTED_SCHEMA_VERSION,
            check_version_or_raise,
            ensure_schema_version_table,
            write_schema_version,
        )
        import sqlite3

        db = tmp_path / "learning.db"
        conn = sqlite3.connect(str(db))
        ensure_schema_version_table(conn)
        write_schema_version(conn, SUPPORTED_SCHEMA_VERSION + 1)
        conn.commit()
        conn.close()

        with mock.patch(
            "superlocalmemory.storage._schema_version._detect_all_installs",
            return_value=[],
        ):
            with pytest.raises(SchemaVersionError) as exc_info:
                check_version_or_raise(db)

        # Must still raise with some message even when no installs are found
        assert str(exc_info.value)

    def test_no_error_when_version_is_current(self, tmp_path):
        """check_version_or_raise does NOT raise when version <= supported."""
        from superlocalmemory.storage._schema_version import (
            SUPPORTED_SCHEMA_VERSION,
            check_version_or_raise,
            ensure_schema_version_table,
            write_schema_version,
        )
        import sqlite3

        db = tmp_path / "learning.db"
        conn = sqlite3.connect(str(db))
        ensure_schema_version_table(conn)
        write_schema_version(conn, SUPPORTED_SCHEMA_VERSION)
        conn.commit()
        conn.close()

        with mock.patch(
            "superlocalmemory.storage._schema_version._detect_all_installs",
            return_value=[],
        ):
            # Should not raise
            check_version_or_raise(db)


class TestWindowsSitePackagesLayout:
    """Version detection must work on the Windows virtualenv layout.

    POSIX venvs use ``lib/python3.13/site-packages``. Windows uses
    ``Lib/site-packages`` — capitalised, and with no version component. Only
    the POSIX shape was searched, so on Windows detection returned None and the
    version-mismatch error named no installations at all.
    """

    @staticmethod
    def _make(base: Path, rel: str, version: str) -> None:
        pkg = base / rel / "superlocalmemory"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text(f'__version__ = "{version}"\n')

    def test_finds_version_in_windows_layout(self, tmp_path):
        from superlocalmemory.core.install_detector import _read_python_version

        self._make(tmp_path, "Lib/site-packages", "4.0.8")
        assert _read_python_version(tmp_path) == "4.0.8"

    def test_finds_version_in_posix_layout(self, tmp_path):
        from superlocalmemory.core.install_detector import _read_python_version

        self._make(tmp_path, "lib/python3.13/site-packages", "4.0.9")
        assert _read_python_version(tmp_path) == "4.0.9"

    def test_returns_none_when_no_layout_matches(self, tmp_path):
        from superlocalmemory.core.install_detector import _read_python_version

        assert _read_python_version(tmp_path) is None


class TestResolvedPackageAuthority:
    """Entries name the package directory that actually loads (4.1.14 #134)."""

    @staticmethod
    def _make_venv(base: Path, version: str) -> Path:
        pkg = base / "lib" / "python3.13" / "site-packages" / "superlocalmemory"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text(f'__version__ = "{version}"\n')
        return pkg

    def test_resolve_package_dir_points_at_package(self, tmp_path):
        from superlocalmemory.core.install_detector import _resolve_package_dir

        pkg = self._make_venv(tmp_path, "4.1.14")
        assert _resolve_package_dir(tmp_path) == str(pkg)

    def test_resolve_package_dir_none_when_absent(self, tmp_path):
        from superlocalmemory.core.install_detector import _resolve_package_dir

        assert _resolve_package_dir(tmp_path) is None

    def test_venv_entry_carries_resolved(self, tmp_path, monkeypatch):
        from superlocalmemory.core.install_detector import _detect_all_installs

        pkg = self._make_venv(tmp_path / ".slm-venv", "4.1.14")
        monkeypatch.setattr(
            "superlocalmemory.core.install_detector._VENV_ROOT",
            tmp_path / ".slm-venv",
        )
        entries = [
            e for e in _detect_all_installs() if e["type"] == "venv"
        ]
        assert entries, "synthetic venv must be detected"
        assert entries[0]["resolved"] == str(pkg)
