"""A session summary that carries only git bookkeeping is a stub, not an answer.

Found on the author's store (9 Nov 2026): 100 Stop-hook summaries read
``[Agentic_official] session ended 2026-10-31 19:02 | branch: main |
uncommitted: 2 files changed, 0 insertions(+), 0 deletions(-)`` and nothing
else, and three of them were injected at one session start as "relevant
memories". The rule only knew the bare line and its ``| files:`` tail, so a
summary that also named the branch counted as real content. A summary with
``recent:`` commits still is: that is what "what did I ship last session?"
needs.

The first fixtures are produced by running the real Stop hook against real git
repositories, so a later change to the hook's format breaks this file instead
of quietly un-flagging the rows.
"""

from __future__ import annotations

import os
import subprocess
from unittest.mock import patch

import pytest

from superlocalmemory.encoding.non_answer import is_stub
from superlocalmemory.hooks.hook_handlers import handle_hook

_OLD = "2026-01-01T09:00:00"


def _git(repo, *args, when: str | None = None) -> None:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}
    if when:
        env.update(GIT_AUTHOR_DATE=when, GIT_COMMITTER_DATE=when)
    subprocess.run(["git", "-C", str(repo), *args], check=True, env=env,
                   capture_output=True)


def _repo(path, *, fresh_commit: bool):
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    (path / "a.txt").write_text("one\n")
    (path / "b.txt").write_text("one\n")
    _git(path, "add", ".")
    _git(path, "commit", "-q", "-m", "first", when=_OLD)
    if fresh_commit:
        (path / "c.txt").write_text("new\n")
        _git(path, "add", ".")
        _git(path, "commit", "-q", "-m", "ship the loader retry")
    (path / "a.txt").write_text("two\n")
    (path / "b.txt").write_text("two\n")
    return path


def _stop_summary(project_dir: str, monkeypatch) -> str:
    import tempfile

    for name in os.listdir(tempfile.gettempdir()):
        if name.startswith("slm-obs-"):
            os.remove(os.path.join(tempfile.gettempdir(), name))
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", project_dir)
    with patch("superlocalmemory.hooks.hook_handlers._daemon_post") as post, \
            patch("superlocalmemory.hooks.hook_handlers._maybe_consolidate"), \
            patch("superlocalmemory.hooks.hook_handlers._captures_session_summary",
                  return_value=True):
        with pytest.raises(SystemExit):
            handle_hook("stop")
    [content] = [c.args[1]["content"] for c in post.call_args_list
                 if c.args and c.args[0] == "/remember"]
    return content


def test_the_stop_hook_line_with_only_branch_and_uncommitted_is_a_stub(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "Agentic_official", fresh_commit=False)
    text = _stop_summary(str(repo), monkeypatch)
    assert text.startswith("[Agentic_official] session ended ")
    assert "| branch: main" in text and "| uncommitted: 2 files changed" in text
    assert "recent:" not in text
    assert is_stub(text), text


def test_an_unnamed_project_writes_empty_brackets_and_is_still_a_stub(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "proj", fresh_commit=False)
    text = _stop_summary(str(repo) + os.sep, monkeypatch)  # basename is ""
    assert text.startswith("[] session ended "), text
    assert is_stub(text), text


def test_a_summary_with_recent_commits_stays_an_answer(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "Agentic_official", fresh_commit=True)
    text = _stop_summary(str(repo), monkeypatch)
    assert "| recent: " in text and "ship the loader retry" in text
    assert not is_stub(text), text


@pytest.mark.parametrize("text", [
    "[Agentic_official] session ended 2026-10-31 19:02 | branch: main"
    " | uncommitted: 2 files changed, 0 insertions(+), 0 deletions(-)",
    "[Agentic_official] session ended 2026-10-31 18:38 | branch: main"
    " | uncommitted: 2 files changed, 0 insertions(+), 0 deletions(-)"
    " | files: assay-viz.tsx, sf-viz.tsx",
    "[qualixar-portal] session ended 2026-10-29 10:12 | branch: fix/hero-copy",
    "[] session ended 2026-10-29 10:12 | branch: main | files: index.html",
    # ingestion derives these from the summary (counts from the same store)
    "session ended 2026-10-31 19:02 | branch: main"
    " | uncommitted: 1 file changed, 3 insertions(+)",
    "the user's session ended 2026-10-31 19:02",
    "The user's session ended 2026-10-31 19:02.",
    "2 files changed, 0 insertions(+), 0 deletions(-)",
    "1 file changed, 12 insertions(+)",
    "3 files changed, 4 deletions(-)",
    "uncommitted: 2 files changed, 0 insertions(+), 0 deletions(-)",
])
def test_git_only_shapes_are_stubs(text):
    assert is_stub(text)


@pytest.mark.parametrize("text", [
    # Copies the extractor paraphrases from the Stop/agent lines; every count is
    # unflagged rows on the same store after the first M068 pass.
    "session ended on 2026-10-11 16:02",                                  # 194
    "The session ended at 2026-11-07 13:49 in config",                    # 149
    "The session ended on 2026-10-11 16:00",                              # 89
    "The session ended at 2026-11-07 13:49 in superlocalmemory.",         # 63
    "A session ended on 2026-10-11 16:19",                                # 30
    "Varun's session ended on 2026-10-11 16:19",                          # 12
    "Varunpratapbhardwaj's session ended on 2026-10-11",                  # 5
    "The Agentic_official's session ended on 2026-10-11 16:19",           # 4
    "Claude\u2019s session ended on 2026-10-11 16:19",
    "the audit-pack session ended 2026-10-11 16:19",
    "session ended at 23:53",
    "2026-10-11 is the date when Varun's session ended",                  # 7
    "2026-10-11 is the date when the session ended",
    "2026-10-11 is the date the session ended.",
])
def test_paraphrased_session_end_stamps_are_stubs(text):
    assert is_stub(text)


@pytest.mark.parametrize("text", [
    "[Agentic_official] session ended 2026-10-13 11:01 | branch: fix/4.2.2"
    " | recent: 0545bc9a fix: calibration reads the workspace",
    "[Agentic_official] session ended 2026-10-13 11:01 | branch: main"
    " | uncommitted: 1 file changed | recent: 0545bc9a fix: retry | files: a.py",
    "[Agentic_official] session ended 2026-10-13 11:01 | branch: main"
    " | next: rerun the gate before tagging",
    "The user's session ended early because the build broke.",
    "the user's session ended with a decision to ship on Monday",
    "2 files changed because I renamed the loader",
    "uncommitted: the migration plan still needs a rollback step",
    "branch: main is protected, so every change goes through a pull request",
    "There were 2 files changed in the main branch",
    "The session ended at 2026-11-01 10:00 in a deadlock we never reproduced",
    "The session ended at 2026-11-01 10:00 in the evening",
    "Varun's session ended on a Thursday",
    "The session ended yesterday",
    "2026-10-11 is the date when the billing migration went live",
    "The session ended on 2026-10-11 16:00 because the laptop died",
])
def test_lines_with_real_content_stay_answers(text):
    assert not is_stub(text)
