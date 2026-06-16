# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com

"""CI guard: no stray <MagicMock *> files in the repo root.

WP-16 identified that tests/test_core/test_engine_recall_path.py::
TestRecallAdaptiveRanking::test_recall_phase1_no_reranking was leaking
artifact files named "<MagicMock id='...'>" into the repo root. The
root cause was patch("pathlib.Path.home") without pinning all path
attributes to real tmp_path locations, causing os.open(str(mock)) to
create files in the current working directory.

AC-1 (WP-16): git status --porcelain --ignored | grep '<MagicMock' = ZERO.
AC-2 (WP-16): this guard PASSES post-fix, would have FAILED pre-fix.

Uses Path.glob (sees gitignored files) + --ignored flag because
<MagicMock*> is in .gitignore; bare git status omits gitignored files.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_no_magicmock_files_tracked_or_untracked() -> None:
    """Fail if any stray <MagicMock *> artifact files exist in the repo root.

    This guard is always-on in CI. It catches the class of bug where
    a test patches pathlib.Path.home (or any home-derived path) with an
    unconfigured MagicMock whose str() representation becomes a file path,
    and production code calls os.open() / open() / np.save() on that
    str()-coerced path, writing into the working directory.

    Pre-fix: would find artifacts from test_recall_phase1_no_reranking.
    Post-fix (WP-16): the test uses a real tmp_path/fake_home, so all
    path operations go to a real temp directory.
    """
    # Path.glob sees gitignored files; ** recurse is limited to root-level
    # to avoid scanning node_modules/.venv/build (those subdirs have their
    # own .gitignore entries and will not contain <MagicMock*> files from
    # this bug class).
    leaked = [p.name for p in REPO_ROOT.glob("<MagicMock*")]

    assert leaked == [], (
        f"Stray MagicMock artifact files found in repo root ({REPO_ROOT}): "
        f"{leaked}. "
        "A test is patching pathlib.Path.home with an unconfigured MagicMock "
        "and production code is writing to str(mock) as a file path. "
        "Fix: use a real tmp_path/fake_home in the patch so writes go to "
        "a temp directory (see WP-16 fix in test_engine_recall_path.py)."
    )

    # Also check git's ignored-file status (belt + suspenders).
    out = subprocess.run(
        ["git", "status", "--porcelain", "--ignored"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    assert "<MagicMock" not in out, (
        f"git status --porcelain --ignored shows stray MagicMock files:\n{out}"
    )


# ---------------------------------------------------------------------------
# OQ-1 reproduction harness (flip skip=True to False only while reproducing)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(True, reason="OQ-1 reproduction harness — enable only to confirm culprit")
def test_oq1_magicmock_reproduction_harness() -> None:  # pragma: no cover
    """Verify the exact culprit test node produces a leak when un-fixed.

    CULPRIT: tests/test_core/test_engine_recall_path.py::
        TestRecallAdaptiveRanking::test_recall_phase1_no_reranking

    SHAPE used: SHAPE 1 (pin mock home to real tmp_path/fake_home directory
    so all Path.home()-derived paths, including _install_token_path() in
    security_primitives.py and learning_db in recall_pipeline.py, resolve to
    real filesystem locations instead of str(MagicMock()) filenames).

    Root cause: patch("pathlib.Path.home") intercepted
    security_primitives._install_token_path() → ensure_install_token() which
    is called from recall_pipeline.py:55,74 during engine.recall(). With an
    unconfigured mock_home chain, token_path.exists() returned the configured
    False but os.open(str(token_path)) where str(mock) = "<MagicMock id='...'>"
    created a real file in the test CWD (repo root).
    """
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest",
         "tests/test_core/test_engine_recall_path.py::"
         "TestRecallAdaptiveRanking::test_recall_phase1_no_reranking",
         "-v", "--tb=short"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout
