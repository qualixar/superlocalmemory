"""Version banner — runs once per pip/npm upgrade.

End-user contract:
- First invocation after ``pip install -U`` prints a short factual banner
  describing what changed, then never again until the next upgrade.
- Fresh install (no prior marker + no existing memory.db) prints nothing
  from the upgrade path — the welcome flow is handled by the setup wizard.
- Same-version invocations are no-ops (no banner, no disk write).
"""
from __future__ import annotations

import os
import sys

import pytest

from superlocalmemory.cli.version_banner import (
    _banner,
    check_and_emit_upgrade_banner,
    read_marker_version,
    write_marker_version,
)


@pytest.fixture
def slm_home(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    return tmp_path


class TestMarkerFile:
    def test_marker_absent_returns_none(self, slm_home):
        assert read_marker_version() is None

    def test_write_then_read_roundtrips(self, slm_home):
        write_marker_version("3.4.26")
        assert read_marker_version() == "3.4.26"

    def test_write_is_atomic_and_overwrites(self, slm_home):
        write_marker_version("3.4.25")
        write_marker_version("3.4.26")
        assert read_marker_version() == "3.4.26"


class TestUpgradeBannerEmission:
    def test_v381_banner_describes_actual_stability_fixes(self):
        banner = _banner(prior="3.8.0", current="3.8.1")

        assert "Large existing databases no longer delay daemon readiness" in banner
        assert "Failed enrichment retries stop after ten automatic attempts" in banner
        assert "Dashboard panes stay mounted between navigation changes" in banner
        assert "Local model workers stay warm for a 30-minute working session" in banner
        assert "Multi-IDE MCP processes now share a worker" not in banner
        assert "Silent data migration complete" not in banner

    def test_fresh_install_no_banner(self, slm_home, capsys):
        """No prior marker AND no existing memory.db = fresh install.
        Setup wizard owns the welcome; this helper stays silent."""
        emitted = check_and_emit_upgrade_banner(current="3.4.26")
        captured = capsys.readouterr()
        assert emitted is False
        assert captured.out == ""
        # Marker is still written so subsequent calls are no-ops.
        assert read_marker_version() == "3.4.26"

    def test_pre_v3426_install_detected_via_memory_db(self, slm_home, capsys):
        """marker absent but memory.db present = pre-v3.4.26 user upgrading.
        Banner should emit with 'from an earlier version'."""
        (slm_home / "memory.db").write_bytes(b"SQLite format 3\x00")

        emitted = check_and_emit_upgrade_banner(current="3.4.26")
        captured = capsys.readouterr()
        assert emitted is True
        # v3.6.13: banner emits to STDERR (stdout is reserved for MCP JSON-RPC).
        assert "3.4.26" in captured.err
        # Idempotent — the marker is now at current version.
        assert read_marker_version() == "3.4.26"

        # Second call is silent (no banner on stderr).
        emitted2 = check_and_emit_upgrade_banner(current="3.4.26")
        captured2 = capsys.readouterr()
        assert emitted2 is False
        assert captured2.err == ""

    def test_explicit_version_upgrade_shows_from_to(self, slm_home, capsys):
        write_marker_version("3.4.25")
        emitted = check_and_emit_upgrade_banner(current="3.4.26")
        captured = capsys.readouterr()
        assert emitted is True
        assert "3.4.25" in captured.err
        assert "3.4.26" in captured.err
        assert read_marker_version() == "3.4.26"

    def test_same_version_no_banner(self, slm_home, capsys):
        write_marker_version("3.4.26")
        emitted = check_and_emit_upgrade_banner(current="3.4.26")
        captured = capsys.readouterr()
        assert emitted is False
        assert captured.out == ""

    def test_banner_mentions_slm_doctor(self, slm_home, capsys):
        write_marker_version("3.4.25")
        check_and_emit_upgrade_banner(current="3.4.26")
        captured = capsys.readouterr()
        assert "slm doctor" in captured.err

    def test_banner_does_not_leak_internal_jargon(self, slm_home, capsys):
        """Lean copy — no Stage-N IDs, no plan-doc references,
        no competitor names. This is user-visible text."""
        write_marker_version("3.4.25")
        check_and_emit_upgrade_banner(current="3.4.26")
        # Banner is on stderr (v3.6.13) — read it there, not the empty stdout.
        captured = capsys.readouterr().err.lower()
        for forbidden in ("stage 8", "stage-8", "path a", "path b",
                          "qdrant", "mem0", "supermemory"):
            assert forbidden not in captured


class TestSecurityHardening:
    """v3.4.26 Stage 9 security fixes."""

    def test_marker_written_at_0600_perms(self, slm_home):
        if sys.platform == "win32":
            pytest.skip("POSIX perm check")
        write_marker_version("3.4.26")
        mode = os.stat(slm_home / ".version").st_mode & 0o777
        assert mode == 0o600, f"marker mode {oct(mode)} leaks upgrade timing"

    def test_read_refuses_symlinked_marker(self, slm_home, tmp_path):
        if sys.platform == "win32":
            pytest.skip("symlink perms differ on Windows")
        target = tmp_path / "elsewhere.txt"
        target.write_text("3.4.99")
        marker = slm_home / ".version"
        marker.symlink_to(target)
        assert read_marker_version() is None

    def test_read_refuses_oversized_marker(self, slm_home):
        marker = slm_home / ".version"
        marker.write_bytes(b"3.4.26" + b"x" * 200)
        assert read_marker_version() is None

    def test_read_refuses_garbage_version(self, slm_home):
        marker = slm_home / ".version"
        marker.write_text("not a version string!!!!")
        assert read_marker_version() is None

    def test_write_rejects_non_version_input(self, slm_home):
        assert write_marker_version("not a version!") is False
        assert not (slm_home / ".version").exists()

    def test_write_rejects_control_chars(self, slm_home):
        assert write_marker_version("3.4.26\x1b[31m") is False


class TestBannerOutput:
    """Banner output goes to stderr, NOT stdout (v3.6.13).

    On the `slm mcp` stdio transport, any non-JSON-RPC byte on stdout corrupts
    the stream and the MCP client rejects it. The human upgrade notice must go
    to stderr only. See version_banner.py:170-175.
    """

    def test_banner_goes_to_stderr_not_stdout(self, slm_home, capsys):
        (slm_home / "memory.db").write_bytes(b"SQLite format 3\x00")
        check_and_emit_upgrade_banner(current="3.4.30")
        captured = capsys.readouterr()
        assert captured.err, "banner must write to stderr"
        assert captured.out == "", "banner must NOT write to stdout (MCP stdio safety)"


class TestIdempotencyAndFailureMode:
    def test_read_tolerant_of_corrupt_marker(self, slm_home):
        marker = slm_home / ".version"
        marker.write_bytes(b"\x00\x00garbage")
        # Must not raise — treat as "unknown", never crash user's CLI.
        assert read_marker_version() is None

    def test_write_creates_parent_dir(self, tmp_path, monkeypatch):
        # SLM_DATA_DIR points at a path whose parent exists but self doesn't
        target = tmp_path / "fresh"
        monkeypatch.setenv("SLM_DATA_DIR", str(target))
        write_marker_version("3.4.26")
        assert (target / ".version").read_text().strip() == "3.4.26"

    def test_check_never_raises_on_io_error(self, tmp_path, monkeypatch):
        # Point at a read-only parent to force a write failure — must not
        # propagate. The banner is advisory; a write failure must never
        # block the user's CLI invocation.
        ro = tmp_path / "ro"
        ro.mkdir()
        ro.chmod(0o555)
        monkeypatch.setenv("SLM_DATA_DIR", str(ro / "inner"))
        # Should not raise.
        check_and_emit_upgrade_banner(current="3.4.26")
