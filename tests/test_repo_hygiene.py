"""Repo-hygiene invariants for a repository that is published publicly.

F-12: ``.pytest_tmp_data/config.json`` was tracked and embedded the author's
absolute home path, disclosing the username and working-tree layout. The
``.gitignore`` rule for that directory already existed -- but **gitignore never
untracks a file that is already tracked**, so the rule was inert and the leak
survived every commit after it. Assert on the index, never on ``.gitignore``.

These gates query git directly rather than reading the working tree. That is
deliberate. The first version of this file walked ``git ls-files`` and called
``Path.read_text()`` with ``except OSError: continue``. This repo's conftest
redirects ``HOME`` into a per-run sandbox; under it the read raised, the
``continue`` swallowed it, every file was skipped, and the gate reported PASS
while the tracked offender sat right there. Verified 2026-08-08.

A gate that cannot inspect its inputs must FAIL, never silently skip. There is
no ``continue`` in this file for exactly that reason.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Assembled at runtime so this file does not itself contain the literal it bans.
_HOME_PREFIX = "/Users/" + "varunpratapbhardwaj"

# Machine-local scratch and code-intelligence output. None of it may be tracked.
_FORBIDDEN_TRACKED_DIRS = (
    ".pytest_tmp_data/",
    "graphify-out/",
    ".gitnexus/",
)


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _require_git() -> None:
    if _git("rev-parse", "--git-dir").returncode != 0:
        pytest.skip("not a git checkout")


def test_no_tracked_file_contains_an_absolute_home_path() -> None:
    """No tracked blob may carry anyone's home directory.

    ``git grep`` searches tracked content through git itself, so a sandboxed
    HOME or an unreadable working tree cannot turn this gate into a no-op.
    """
    _require_git()
    # -I skips binary, -F fixed string, -l names files only.
    proc = _git("grep", "-I", "-F", "-l", "-e", _HOME_PREFIX, "--", ".")
    # git grep: 0 = matches found, 1 = none found, >1 = real error.
    assert proc.returncode in (0, 1), (
        f"git grep failed ({proc.returncode}) — the gate could not run, which "
        f"must never be reported as success: {proc.stderr.strip()}"
    )
    offenders = [line for line in proc.stdout.splitlines() if line.strip()]
    assert not offenders, (
        "Tracked files contain an absolute home path — this repository is "
        f"published: {offenders}"
    )


def test_machine_local_scratch_is_not_tracked() -> None:
    """An ignore rule governs future files only; the index decides what ships."""
    _require_git()
    proc = _git("ls-files", "-z")
    assert proc.returncode == 0, (
        f"git ls-files failed — gate could not run: {proc.stderr.strip()}"
    )
    tracked = [p for p in proc.stdout.split("\0") if p]
    assert tracked, "git ls-files returned nothing; the gate cannot verify anything"

    offenders = [
        rel for rel in tracked
        if any(rel.startswith(d) for d in _FORBIDDEN_TRACKED_DIRS)
    ]
    assert not offenders, (
        "Machine-local scratch is tracked. Adding it to .gitignore does NOT "
        f"untrack it — use `git rm --cached`: {offenders}"
    )
