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


# Home-directory shapes, including the flattened form tools embed in scratch
# paths and session identifiers.
# POSIX ERE — `git grep -E` rejects Python's non-capturing `(?:...)` with
# "repetition-operator operand invalid". Keep this dialect-portable.
_USER_SEGMENT_RE = r"(/Users/|/home/|-Users-)[A-Za-z0-9._-]+"

# Names that are documentation placeholders, not anyone's identity. Docs and
# fixtures legitimately show `/Users/yourusername/...`; flagging those would
# make the gate cry wolf, and a gate that cries wolf gets deleted by whoever
# hits it next.
_PLACEHOLDER_SEGMENTS = frozenset({
    "you", "your", "yourname", "yourusername", "your_username",
    "user", "username", "name", "me", "someone", "somebody",
    "alice", "bob", "carol", "foo", "bar", "baz", "example", "test",
    "runner", "ci", "root", "home", "u", "x", "...",
})


def test_no_tracked_file_leaks_a_real_identity_in_a_path() -> None:
    """No tracked blob may carry a real person's home-directory path.

    Two failure modes are being avoided at once, and the balance is the point.

    Too narrow: the first version grepped the literal ``/Users/<one-name>`` and
    missed ``benchmark/results/bench_perf.json``, which recorded the SAME
    identity hyphenated by a different tool as ``-Users-<author>-``
    (F-23).

    Too broad: matching every home-directory shape flagged 23 files, nearly all
    of them documentation showing ``/Users/yourusername/`` — legitimate content.

    So the gate matches the shape and then judges the *segment*: a name outside
    the placeholder set is treated as a real identity. Verified 2026-08-08.
    """
    _require_git()
    proc = _git("grep", "-I", "-h", "-o", "-E", _USER_SEGMENT_RE, "--", ".")
    assert proc.returncode in (0, 1), (
        f"git grep failed ({proc.returncode}) — the gate could not run, which "
        f"must never be reported as success: {proc.stderr.strip()}"
    )

    offenders: set[str] = set()
    for line in proc.stdout.splitlines():
        match = line.strip()
        if not match:
            continue
        # `-o` yields e.g. "/Users/alice" or the flattened "-Users-someone".
        for prefix in ("/Users/", "/home/", "-Users-"):
            if match.startswith(prefix):
                seg = match[len(prefix):]
                if seg and seg.lower() not in _PLACEHOLDER_SEGMENTS:
                    offenders.add(seg)
                break

    assert not offenders, (
        "Tracked files embed what look like real home-directory identities — "
        f"this repository is published: {sorted(offenders)}. Use a placeholder "
        f"(one of: yourusername, user, alice) or redact the path."
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
