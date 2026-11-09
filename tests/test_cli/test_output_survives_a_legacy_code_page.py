"""CLI output never crashes on a stream whose encoding lacks its symbols.

On Windows, when stdout or stderr is a pipe or a file, Python encodes with the
locale code page (usually cp1252), which has no ``✓``, ``✗`` or ``→``. Before
this fix ``slm loop demo > out.txt`` died with ``UnicodeEncodeError``. These
tests reproduce that on any OS: ``PYTHONIOENCODING=cp1252`` for an explicit
choice, and a cp1252 ``TextIOWrapper`` over the real pipe for the Windows
locale default, which is exactly how Python builds ``sys.stdout`` there.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# Rebuild stdout and stderr the way Python does on Windows for a pipe: the
# locale code page, strict errors. Then run the CLI exactly as ``slm`` does.
_LOCALE_CP1252_LAUNCHER = """\
import io, runpy, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="cp1252", line_buffering=False)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="cp1252", line_buffering=True)
sys.argv = ["slm"] + sys.argv[1:]
runpy.run_module("superlocalmemory.cli.main", run_name="__main__")
"""


def _env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONIOENCODING", None)
    env["PYTHONUTF8"] = "0"
    env["PYTHONPATH"] = str(ROOT / "src")
    env["SLM_DATA_DIR"] = str(tmp_path / "data")
    env["HOME"] = str(tmp_path / "home")
    env["USERPROFILE"] = str(tmp_path / "home")
    (tmp_path / "home").mkdir()
    return env


def _run(argv: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv, cwd=ROOT, env=env, capture_output=True, timeout=120, check=False,
    )


def test_explicit_cp1252_replaces_symbols_instead_of_crashing(tmp_path: Path) -> None:
    env = _env(tmp_path)
    env["PYTHONIOENCODING"] = "cp1252"

    result = _run(
        [sys.executable, "-m", "superlocalmemory.cli.main", "loop", "demo"], env,
    )

    assert b"UnicodeEncodeError" not in result.stderr, result.stderr.decode("cp1252")
    assert result.returncode == 0, result.stderr.decode("cp1252")
    out = result.stdout.decode("cp1252")
    # An explicit PYTHONIOENCODING is honoured: cp1252 bytes, "?" for the tick.
    assert out.startswith("? [DONE]"), out
    assert "run_id:" in out


def test_locale_cp1252_pipe_is_written_as_utf8(tmp_path: Path) -> None:
    result = _run(
        [sys.executable, "-c", _LOCALE_CP1252_LAUNCHER, "loop", "demo"],
        _env(tmp_path),
    )

    assert b"UnicodeEncodeError" not in result.stderr, result.stderr.decode("utf-8", "replace")
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    out = result.stdout.decode("utf-8")
    assert out.startswith("✓ [DONE]"), out


def test_json_output_is_unchanged_ascii_on_a_cp1252_pipe(tmp_path: Path) -> None:
    result = _run(
        [sys.executable, "-c", _LOCALE_CP1252_LAUNCHER, "loop", "demo", "--json"],
        _env(tmp_path),
    )

    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert result.stdout.isascii()
    assert json.loads(result.stdout)["status"] == "DONE"


class _Stream(io.TextIOWrapper):
    def __init__(self, encoding: str, *, tty: bool, errors: str = "strict") -> None:
        super().__init__(io.BytesIO(), encoding=encoding, errors=errors)
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def _with_streams(monkeypatch: pytest.MonkeyPatch, stdout, stderr) -> None:
    from superlocalmemory.cli.stdio_encoding import make_stdio_safe

    monkeypatch.delenv("PYTHONIOENCODING", raising=False)
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    make_stdio_safe()


def test_pipe_on_locale_code_page_switches_to_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    out, err = _Stream("cp1252", tty=False), _Stream("cp1252", tty=False)
    _with_streams(monkeypatch, out, err)
    assert (out.encoding, out.errors) == ("utf-8", "strict")
    assert (err.encoding, err.errors) == ("utf-8", "strict")


def test_terminal_keeps_its_encoding_and_replaces(monkeypatch: pytest.MonkeyPatch) -> None:
    # A terminal renders its own encoding; UTF-8 bytes there would be mojibake.
    out = _Stream("cp1252", tty=True)
    _with_streams(monkeypatch, out, None)
    assert (out.encoding, out.errors) == ("cp1252", "replace")
    out.write("✓ done")  # must not raise


def test_streams_that_can_already_encode_are_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = _Stream("utf-8", tty=False)
    err = _Stream("cp1252", tty=False, errors="backslashreplace")
    _with_streams(monkeypatch, out, err)
    assert (out.encoding, out.errors) == ("utf-8", "strict")
    # backslashreplace cannot raise, so the caller's choice stands.
    assert (err.encoding, err.errors) == ("cp1252", "backslashreplace")


def test_streams_without_reconfigure_are_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = io.StringIO()
    _with_streams(monkeypatch, out, None)
    out.write("✓")
