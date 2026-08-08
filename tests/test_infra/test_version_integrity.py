# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com

"""Issue #107 -- a running process must be able to notice it is stale.

Python imports a module once.  A long-lived ``slm mcp`` server therefore keeps
serving the code it read at startup, and upgrading the package on disk does
nothing for it.  Before this module there was no way for such a process to
notice, so it answered with confident, plausible, *wrong* results -- and made a
correct fix look broken.  That cost a full release cycle on #106.

These tests pin the two properties that matter:

* the check compares *imported* against *on-disk* state and never raises, and
* every failure mode degrades to a reported "unknown", never to a false
  "current".  A false "current" is the whole bug: silence that reads as health.
"""

from __future__ import annotations

import pytest

from superlocalmemory.infra.version_integrity import (
    STATE_AHEAD,
    STATE_CURRENT,
    STATE_MISMATCH,
    STATE_STALE,
    STATE_UNKNOWN,
    VersionIntegrity,
    check_version_integrity,
)


def _check(running: str, installed):
    """Run the check with both sides injected, no real environment."""
    def reader() -> str:
        if isinstance(installed, BaseException):
            raise installed
        if installed is None:
            raise LookupError("not installed")
        return installed

    return check_version_integrity(running=running, installed_reader=reader)


class TestDriftDetection:
    def test_matching_versions_are_current(self) -> None:
        result = _check("3.8.12", "3.8.12")
        assert result.state == STATE_CURRENT
        assert result.is_stale is False
        assert result.running == "3.8.12"
        assert result.installed == "3.8.12"

    def test_running_older_than_installed_is_stale(self) -> None:
        """The #107 case: upgraded on disk, process still on old code."""
        result = _check("3.8.11", "3.8.12")
        assert result.state == STATE_STALE
        assert result.is_stale is True
        # The message must name both versions -- a user reading a log line
        # needs to know what to restart and what they are running.
        assert "3.8.11" in result.detail
        assert "3.8.12" in result.detail

    def test_two_releases_behind_is_stale(self) -> None:
        assert _check("3.8.10", "3.8.12").state == STATE_STALE

    def test_running_newer_than_installed_is_ahead_not_stale(self) -> None:
        """An editable/dev checkout is ahead of the installed dist.

        This is normal during development and must not be reported as the
        #107 failure, or the warning becomes noise every maintainer ignores.
        """
        result = _check("3.9.0", "3.8.12")
        assert result.state == STATE_AHEAD
        assert result.is_stale is False

    @pytest.mark.parametrize(
        "running,installed",
        [
            ("3.8.12", "3.8.12+local"),
            ("weird", "3.8.12"),
            ("3.8.12", "also-weird"),
        ],
    )
    def test_unparseable_but_differing_versions_report_mismatch(
        self, running: str, installed: str
    ) -> None:
        """Unknown ordering still surfaces a difference rather than hiding it."""
        result = _check(running, installed)
        assert result.state == STATE_MISMATCH
        assert result.is_stale is False
        assert result.differs is True


class TestNeverLies:
    """Every failure path must reach 'unknown', never a false 'current'."""

    def test_missing_distribution_metadata_is_unknown(self) -> None:
        result = _check("3.8.12", None)
        assert result.state == STATE_UNKNOWN
        assert result.installed is None
        assert result.is_stale is False

    def test_arbitrary_reader_exception_is_unknown_not_raised(self) -> None:
        result = _check("3.8.12", RuntimeError("metadata exploded"))
        assert result.state == STATE_UNKNOWN

    def test_reader_returning_a_non_string_is_unknown(self) -> None:
        result = check_version_integrity(
            running="3.8.12", installed_reader=lambda: 3.8  # type: ignore[arg-type,return-value]
        )
        assert result.state == STATE_UNKNOWN

    def test_reader_returning_blank_is_unknown(self) -> None:
        assert check_version_integrity(
            running="3.8.12", installed_reader=lambda: "   "
        ).state == STATE_UNKNOWN

    def test_check_never_raises_even_with_a_hostile_reader(self) -> None:
        class Hostile:
            def __call__(self):
                raise BaseException("not even an Exception")  # noqa: TRY002

        # BaseException is deliberately not caught by bare `except Exception`.
        # Identity reporting must still not take the process down.
        result = check_version_integrity(running="3.8.12", installed_reader=Hostile())
        assert result.state == STATE_UNKNOWN


class TestRealEnvironment:
    def test_default_call_reads_the_real_environment_without_raising(self) -> None:
        result = check_version_integrity()
        assert isinstance(result, VersionIntegrity)
        assert result.state in {
            STATE_CURRENT, STATE_STALE, STATE_AHEAD, STATE_MISMATCH, STATE_UNKNOWN,
        }
        assert isinstance(result.detail, str) and result.detail

    def test_running_version_defaults_to_the_imported_package_version(self) -> None:
        from superlocalmemory import __version__

        assert check_version_integrity().running == __version__

    def test_installed_version_is_read_from_disk_not_from_the_import(self) -> None:
        """The check must consult disk, or it can only ever compare a value
        to itself -- which is exactly the blind spot #107 describes."""
        observed: list[bool] = []

        def reader() -> str:
            observed.append(True)
            return "3.8.12"

        check_version_integrity(running="3.8.12", installed_reader=reader)
        assert observed, "installed version was never read from the reader"


class TestImportlibMetadataReadsDisk:
    """The one external assumption this module rests on.

    If ``importlib.metadata`` ever returned a value captured at import time
    rather than read from disk, the whole check would silently compare a value
    to itself and always report "current" -- reintroducing exactly the blind
    spot of #107, but now wearing a green checkmark.  That failure would be
    invisible, so it gets a real test against a real filesystem rather than a
    comment asserting it.
    """

    def test_metadata_version_reflects_an_upgrade_under_a_live_process(
        self, tmp_path, monkeypatch
    ) -> None:
        import importlib.metadata as md
        import shutil
        import sys

        dist = tmp_path / "fakepkg-1.0.0.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: fakepkg\nVersion: 1.0.0\n", encoding="utf-8"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        sys.path_importer_cache.pop(str(tmp_path), None)

        assert md.version("fakepkg") == "1.0.0"

        # Simulate `pip install --upgrade` landing underneath us.
        shutil.rmtree(dist)
        upgraded = tmp_path / "fakepkg-2.0.0.dist-info"
        upgraded.mkdir()
        (upgraded / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: fakepkg\nVersion: 2.0.0\n", encoding="utf-8"
        )

        assert md.version("fakepkg") == "2.0.0", (
            "importlib.metadata returned a cached version. The staleness check "
            "in version_integrity.py depends on this reading from disk; if this "
            "assumption breaks, the check silently always reports 'current'."
        )


class TestReporting:
    def test_stale_result_carries_a_restart_hint(self) -> None:
        result = _check("3.8.11", "3.8.12")
        assert result.hint
        assert "restart" in result.hint.lower()

    def test_current_result_has_no_hint(self) -> None:
        assert not _check("3.8.12", "3.8.12").hint

    def test_result_is_json_serialisable_for_the_health_endpoint(self) -> None:
        import json

        payload = _check("3.8.11", "3.8.12").as_dict()
        assert json.loads(json.dumps(payload))["state"] == STATE_STALE
        assert payload["running"] == "3.8.11"
        assert payload["installed"] == "3.8.12"
        assert payload["is_stale"] is True
