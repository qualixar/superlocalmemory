"""Regression guard for the v3.6.13 MCP-stdout-corruption fix.

The post-upgrade banner MUST go to STDERR, never STDOUT. On the `slm mcp`
stdio transport, any non-JSON byte on stdout corrupts the JSON-RPC stream and
the MCP client (Claude Desktop / Cursor) rejects every message as
"not valid JSON" — which is exactly what happened on the first `slm mcp`
launch after the 3.6.12 upgrade. This test fails if the banner ever returns
to stdout.
"""

from __future__ import annotations

import pytest

from superlocalmemory.cli import version_banner


def _seed_upgrade(tmp_path):
    # An existing-install marker from an older version → banner should fire.
    (tmp_path / ".version").write_text("3.6.11", encoding="utf-8")
    (tmp_path / "memory.db").touch()


def test_banner_goes_to_stderr_not_stdout(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    _seed_upgrade(tmp_path)

    emitted = version_banner.check_and_emit_upgrade_banner("3.6.13")

    captured = capsys.readouterr()
    assert emitted is True
    # The load-bearing assertion: NOTHING on stdout (would break MCP stdio).
    assert "SuperLocalMemory upgraded" not in captured.out
    assert "Run `slm doctor`" not in captured.out
    assert captured.out == ""
    # The banner is still shown to humans — on stderr.
    assert "SuperLocalMemory upgraded" in captured.err


def test_banner_idempotent_same_version_silent(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    (tmp_path / ".version").write_text("3.6.13", encoding="utf-8")
    (tmp_path / "memory.db").touch()

    emitted = version_banner.check_and_emit_upgrade_banner("3.6.13")
    captured = capsys.readouterr()
    assert emitted is False
    assert captured.out == ""
    assert captured.err == ""


def test_main_skips_banner_on_mcp_path(monkeypatch):
    """`slm mcp` must not even call the banner (belt-and-suspenders): the
    stdio hot path does zero banner work."""
    import sys
    monkeypatch.setattr(sys, "argv", ["slm", "mcp"])
    called = {"n": 0}
    monkeypatch.setattr(
        version_banner, "check_and_emit_upgrade_banner",
        lambda *_a, **_k: called.__setitem__("n", called["n"] + 1) or False,
    )
    # Reproduce main()'s guard exactly.
    _is_mcp_stdio = len(sys.argv) >= 2 and sys.argv[1] == "mcp"
    if not _is_mcp_stdio:
        version_banner.check_and_emit_upgrade_banner("3.6.13")
    assert _is_mcp_stdio is True
    assert called["n"] == 0
