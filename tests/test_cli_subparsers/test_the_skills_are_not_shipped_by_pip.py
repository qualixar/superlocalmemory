# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Upgrading the package does not upgrade the skills, and that has to be said.

The skills, agents and commands live in the editor's plugin channel — the
``plugin/`` tree, distributed through the editor's own marketplace — not in the
Python distribution. Confirmed against the built wheel: it contains **zero**
files from ``plugin/``, ``codex-plugin/`` or ``copilot-plugin/``.

So ``pip install --upgrade superlocalmemory`` moves the package and leaves every
skill exactly where it was. 4.1 changed 76 files and 862 lines across those
trees. A user who upgraded, read a clean ``slm doctor``, and reasonably concluded
they had the whole release, had no way to discover otherwise — the word "plugin"
appeared nowhere in the output.

This does not change how they are distributed. It makes the gap visible.
"""

from __future__ import annotations

import json

from superlocalmemory.cli.commands import (
    _installed_plugin_versions,
    _slm_version,
)


class TestItFindsAnInstalledPlugin:
    def _plugin_at(self, root, name="superlocalmemory", version="4.1.2"):
        d = root / ".claude" / "plugins" / "superlocalmemory" / ".claude-plugin"
        d.mkdir(parents=True, exist_ok=True)
        (d / "plugin.json").write_text(
            json.dumps({"name": name, "version": version}), encoding="utf-8"
        )

    def test_it_reads_the_version_from_the_manifest(
        self, tmp_path, monkeypatch,
    ) -> None:
        self._plugin_at(tmp_path, version="4.0.8")
        monkeypatch.setenv("HOME", str(tmp_path))

        found = _installed_plugin_versions()

        assert found, "an installed plugin was not detected"
        assert "4.0.8" in found.values()

    def test_nothing_installed_reports_nothing(self, tmp_path, monkeypatch) -> None:
        """The case that matters most, and the one nobody was told about: pip is
        the only thing being upgraded."""
        monkeypatch.setenv("HOME", str(tmp_path))

        assert _installed_plugin_versions() == {}

    def test_another_vendors_plugin_is_not_ours(self, tmp_path, monkeypatch) -> None:
        """A machine with twenty plugins installed must not have one of them
        reported as this one."""
        self._plugin_at(tmp_path, name="somebody-elses-plugin", version="9.9.9")
        monkeypatch.setenv("HOME", str(tmp_path))

        assert _installed_plugin_versions() == {}

    def test_a_malformed_manifest_does_not_break_the_probe(
        self, tmp_path, monkeypatch,
    ) -> None:
        """Someone else's broken json is not a reason for our doctor to fail."""
        d = tmp_path / ".claude" / "plugins" / "broken" / ".claude-plugin"
        d.mkdir(parents=True)
        (d / "plugin.json").write_text("{not json", encoding="utf-8")
        self._plugin_at(tmp_path, version="4.1.2")
        monkeypatch.setenv("HOME", str(tmp_path))

        assert "4.1.2" in _installed_plugin_versions().values()

    def test_no_plugin_directory_at_all_is_fine(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path / "nothing-here"))

        assert _installed_plugin_versions() == {}


class TestTheVersionItComparesAgainst:
    def test_the_package_version_is_readable(self) -> None:
        version = _slm_version()

        assert version and version != "unknown"
        assert version[0].isdigit(), f"not a version: {version!r}"


class TestTheWheelReallyHasNoSkills:
    """The premise. If the package ever did start shipping them, this test
    should fail so the doctor warning can be removed rather than left lying."""

    def test_the_plugin_trees_are_not_in_the_installed_package(self) -> None:
        import pathlib

        import superlocalmemory

        installed = pathlib.Path(superlocalmemory.__file__).parent
        for tree in ("plugin", "codex-plugin", "copilot-plugin"):
            assert not (installed / tree).is_dir(), (
                f"{tree}/ is now inside the package — the doctor warning about "
                f"pip not shipping skills is out of date"
            )
