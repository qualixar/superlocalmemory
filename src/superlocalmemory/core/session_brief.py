# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""What a session start shows, chosen for the project the agent is in (WS-4, W3-7c).

The SessionStart hook used to show the profile's most important recent facts,
whatever the project: the same list in every repository. This module picks by
where the agent is, from the store and two bounded git calls, with no engine
(the hook's fast path):

* **the workspace**: project name, branch and the files touched lately
  (uncommitted, then the last commits of the past week);
* **project memories**: memories saved under this project (``memories``
  metadata ``project``, compared by ``project_key``) and memories whose text
  names the project, the branch's words or a touched file (full-text search),
  never a memory saved under *another* project;
* **the last session here**: the newest session-end summary of this project
  that says more than its date (``[name] session ended <date> | ...``);
* **what needs attention here**: open disputes with a side saved under this
  project, and memories whose source file inside this project changed or went
  missing (``truth.leases``).

Where a part has nothing, the start falls back to what it showed before (the
profile's recent important facts), so a new project still gets context.
Read-only; every part fails open to "nothing".
"""

from __future__ import annotations

import os
import re
import sqlite3
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from superlocalmemory.core.project_identity import (
    project_key,
    project_name,
    session_end_project,
)

#: Each git call's own limit; the hook's whole start has 15 s, the fast path 0.5 s.
GIT_TIMEOUT_S = 0.4
#: Files named in the workspace line and searched for.
TOUCHED_MAX = 8
#: Search terms (project name, branch words, touched files).
TERMS_MAX = 12
#: Disputes and changed sources listed by name (a count covers the rest).
NOTES_MAX = 3
#: Characters of each side of a dispute shown.
SIDE_CHARS = 160
#: Session summaries shown (this project's newest first).
SUMMARIES_MAX = 2

_STOP = frozenset({
    "feat", "feature", "fix", "bugfix", "hotfix", "chore", "docs", "test", "tests",
    "main", "master", "develop", "dev", "release", "wip", "refactor", "init",
    "index", "readme", "setup", "conftest", "license", "changelog", "package", "lock",
    "skill", "agents", "claude", "prompt", "config", "utils", "types", "constants",
})
_END = re.compile(r"\[[^\]\n]{1,200}\] session ended ")
_STAMP_ONLY = re.compile(r"\A\[[^\]\n]{1,200}\] session ended [0-9: -]+\Z")
_PROJECT = ("CASE WHEN json_valid(m.metadata_json) "
            "THEN json_extract(m.metadata_json, '$.project') END")
_COLUMNS = ("f.fact_id, f.content, f.importance, f.access_count, f.created_at, "
            f"{_PROJECT} AS project")


@dataclass(frozen=True)
class Workspace:
    root: str = ""
    #: Shown name: the repository's, so a worktree is its repository.
    name: str = ""
    #: Every project key this workspace answers to (repository, then folder).
    keys: tuple[str, ...] = ()
    folder: str = ""
    branch: str = ""
    touched: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(n for n in (self.name, self.folder) if n))


@dataclass(frozen=True)
class Brief:
    notes: tuple[str, ...] = ()
    #: (role, row) with role ``recall`` (chosen for this project) or
    #: ``session_summary``; the caller fills ``recent`` when this is short.
    facts: tuple[tuple[str, Mapping], ...] = field(default_factory=tuple)


# ---- the workspace ---------------------------------------------------------

def _git(root: str, args: Sequence[str], run: Callable, timeout: float) -> str:
    try:
        done = run(["git", "-C", root, *args], capture_output=True, text=True,
                   timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError, ValueError):
        return ""
    return done.stdout if done.returncode == 0 and isinstance(done.stdout, str) else ""


def _branch(header: str) -> str:
    head = header[3:].split("...", 1)[0].strip()
    if head.startswith("No commits yet on "):
        return head.removeprefix("No commits yet on ").strip()
    return "" if head.startswith("HEAD") else head


def _repository(root: str, run: Callable, timeout: float) -> str:
    """The main checkout's folder (a worktree's ``.git`` is a file pointing there)."""
    common = _git(root, ("rev-parse", "--git-common-dir"), run, timeout).strip()
    if not common:
        return ""
    path = common if os.path.isabs(common) else os.path.join(root, common)
    return os.path.dirname(os.path.normpath(path))


def read_workspace(project_dir: str, *, run: Callable = subprocess.run,
                   timeout_s: float = GIT_TIMEOUT_S) -> Workspace:
    """Where the agent is. A home directory or filesystem root is no project."""
    root = str(project_dir or "").strip()
    if not root or os.path.realpath(root) in {os.path.realpath(os.path.expanduser("~")),
                                              os.path.sep}:
        return Workspace()
    folder = project_name(root) or ""
    if not os.path.isdir(root):
        return Workspace(root, folder, _keys(folder), folder)
    repo = project_name(_repository(root, run, timeout_s)) or folder
    status = _git(root, ("status", "--porcelain=v1", "--branch"), run, timeout_s)
    branch, files = "", []
    for line in status.splitlines():
        if line.startswith("## "):
            branch = _branch(line)
        elif len(line) > 3:
            files.append(line[3:].split(" -> ")[-1].strip().strip('"'))
    log = _git(root, ("log", "-n", "5", "--since=7.days", "--name-only", "--format="),
               run, timeout_s)
    files += [x.strip() for x in log.splitlines() if x.strip()]
    touched = tuple(dict.fromkeys(f for f in files if f))[:TOUCHED_MAX]
    return Workspace(root, repo, _keys(repo, folder), folder, branch, touched)


def _keys(*names: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(k for k in (project_key(n) for n in names) if k))


#: A file stem worth searching names something (``recall_core``, ``SessionBrief``,
#: ``v2-plan``); a plain word (``commands``, ``main``) matches unrelated memories.
_COMPOUND = re.compile(r"[a-z0-9][_.-][A-Za-z0-9]|[a-z][A-Z]|[A-Za-z]\d")
_TERM = re.compile(r"\A(?:[A-Za-z][\w.-]{2,}|\d+(?:\.\d+)+)\Z")


def terms(ws: Workspace, *, names: bool = True) -> tuple[str, ...]:
    """Search terms: the project's names (when ``names``), the branch's parts,
    touched files' compound stems. A project's name mostly matches paths pasted into
    memories, so it is searched only when nothing is saved under the project."""
    out: list[str] = []
    words = [*(ws.names if names else ()), *ws.branch.split("/"),
             *(st for st in (PurePosixPath(p).stem for p in ws.touched) if _COMPOUND.search(st))]
    for w in (x.strip() for x in words):
        low = w.lower()
        if not _TERM.match(w) or low in _STOP or low.startswith("test_"):
            continue
        if low not in {x.lower() for x in out}:
            out.append(w)
    return tuple(out[:TERMS_MAX])


def fts_query(words: Sequence[str]) -> str:
    """Each term as a quoted phrase (the store's tokenizer splits it), OR-ed."""
    return " OR ".join('"' + w.replace('"', '""') + '"' for w in words)


# ---- memories --------------------------------------------------------------

def relation(raw: object, ws: Workspace) -> bool | None:
    """True: saved under this project; False: under another; None: no project,
    or a folder this project sits inside (a session run from a parent folder)."""
    theirs = project_key(raw)
    if theirs is None:
        return None
    if theirs in ws.keys:
        return True
    text = str(raw).strip()
    if ws.root and os.path.isabs(text):
        try:
            base = os.path.normpath(text)
            if os.path.commonpath([base, os.path.normpath(ws.root)]) == base:
                return None
        except ValueError:
            pass
    return False


#: The project's newest memories read for the start (an index walk that stops
#: early); their facts are ranked by importance.
HERE_MEMORIES = 300

_HERE = f"""
WITH here AS MATERIALIZED (
  SELECT m.memory_id, {_PROJECT} AS project FROM memories m
   WHERE m.profile_id = ? AND ({{likes}}) ORDER BY m.created_at DESC LIMIT ?)
SELECT f.rowid AS rid, here.project, f.fact_id, f.content, f.importance, f.access_count,
       f.created_at, COALESCE(f.non_answer, 0) AS non_answer
  FROM here CROSS JOIN atomic_facts f ON f.memory_id = here.memory_id
 WHERE +f.profile_id = ? AND +f.lifecycle = 'active'
"""


def _kept(conn, rows: list, clause: str, params: tuple) -> list:
    """``rows`` (with ``rid``) that pass ``clause`` over ``atomic_facts f``, order kept."""
    if not rows:
        return []
    marks = ",".join("?" for _ in rows)
    ok = {r[0] for r in conn.execute(
        f"SELECT f.rowid FROM atomic_facts f WHERE f.rowid IN ({marks}){clause}",
        (*(r["rid"] for r in rows), *params)).fetchall()}
    return [r for r in rows if r["rid"] in ok]


def saved_here(conn, pid: str, ws: Workspace, source: tuple[str, tuple], current: str,
               n: int) -> tuple[list, list]:
    """(this project's facts by importance, its session-end summaries newest first),
    from its ``HERE_MEMORIES`` newest memories."""
    if not ws.keys:
        return [], []
    likes = " OR ".join(f"{_PROJECT} LIKE ?" for _ in ws.names)
    patterns = tuple("%" + x.replace("%", "") + "%" for x in ws.names)
    rows = conn.execute(_HERE.format(likes=likes),
                        (pid, *patterns, HERE_MEMORIES, pid)).fetchall()
    rows = [r for r in rows if relation(r["project"], ws) is True]
    ends = sorted((r for r in rows if _END.match(str(r["content"]))),
                  key=lambda r: str(r["created_at"] or ""), reverse=True)
    # Most important first, newest first among equals (two stable sorts).
    top = sorted((r for r in rows if not r["non_answer"]),
                 key=lambda r: str(r["created_at"] or ""), reverse=True)
    top = sorted(top, key=lambda r: -(r["importance"] or 0.0))
    return _kept(conn, top[:n], *source), _kept(conn, ends[:50], current, ())


def _found(conn, pid: str, ws: Workspace, source: tuple[str, tuple], n: int, *,
           names: bool = True) -> list:
    """Memories whose text names the branch or a touched file (and the project
    itself when ``names``: a store with no project tags). Never another
    project's memory."""
    words = terms(ws, names=names)
    if not words:
        return []
    sql, params = source
    try:
        rows = conn.execute(
            f"SELECT {_COLUMNS} FROM atomic_facts_fts "
            "JOIN atomic_facts f ON f.rowid = atomic_facts_fts.rowid "
            "LEFT JOIN memories m ON m.memory_id = f.memory_id "
            f"WHERE atomic_facts_fts MATCH ? AND f.profile_id = ? AND f.lifecycle = 'active'{sql} "
            "ORDER BY bm25(atomic_facts_fts) LIMIT ?",
            (fts_query(words), pid, *params, n)).fetchall()
    except sqlite3.OperationalError:
        return []
    return [r for r in rows if relation(r["project"], ws) is not False]


def mix(tagged: list, found: list, limit: int) -> list:
    """Saved here and named by the search first, then the two lists in turn."""
    here = {r["fact_id"] for r in tagged}
    both = [r for r in found if r["fact_id"] in here]
    taken = {r["fact_id"] for r in both}
    a = [r for r in tagged if r["fact_id"] not in taken]
    b = [r for r in found if r["fact_id"] not in here]
    out = list(both)
    for i in range(max(len(a), len(b))):
        out.extend(x[i] for x in (a, b) if i < len(x))
    return distinct(out)[:limit]


def distinct(rows: list) -> list:
    """One row per text, order kept: a text inside one already kept (the same
    sentence saved twice, or a fragment of a longer memory) is left out."""
    kept: list[str] = []
    out = []
    for r in rows:
        text = " ".join(str(r["content"]).split()).lower()
        if any(text in k or k in text for k in kept):
            continue
        kept.append(text)
        out.append(r)
    return out


def last_sessions(ends: list, ws: Workspace) -> list:
    """This project's newest session summaries that say more than their date."""
    return [r for r in ends
            if project_key(session_end_project(r["content"])) in ws.keys
            and not _STAMP_ONLY.match(str(r["content"]).strip())][:SUMMARIES_MAX]


# ---- attention -------------------------------------------------------------

def _inside(locator: str, root: str) -> str | None:
    """``locator`` relative to ``root`` when the file is inside the project."""
    if not locator or not root:
        return None
    base = os.path.realpath(root)
    for path in {os.path.normpath(locator), os.path.realpath(locator)}:
        try:
            if os.path.commonpath([base, path]) == base:
                return os.path.relpath(path, base)
        except ValueError:
            continue
    return None


def stale_here(conn, pid: str, ws: Workspace) -> tuple[list[str], int]:
    from superlocalmemory.truth import leases

    lines = []
    for item in leases.stale(conn, pid, limit=200):
        rel = _inside(str(item.get("file") or item.get("locator") or ""), ws.root)
        if rel is None:
            continue
        lines.append(f"[source {item.get('state', 'changed')}] {rel}: a memory saved from it "
                     "may be out of date; re-read the file before relying on it")
    return lines[:NOTES_MAX], len(lines)


def _projects_of(conn, fact_ids: Sequence[str]) -> dict[str, str | None]:
    if not fact_ids:
        return {}
    marks = ",".join("?" for _ in fact_ids)
    rows = conn.execute(
        f"SELECT f.fact_id, {_PROJECT} AS project FROM atomic_facts f "
        f"LEFT JOIN memories m ON m.memory_id = f.memory_id WHERE f.fact_id IN ({marks})",
        tuple(fact_ids)).fetchall()
    return {r["fact_id"]: project_key(r["project"]) for r in rows}


def disputes_here(conn, pid: str, ws: Workspace) -> tuple[list[str], int, int]:
    """(lines for this project's open disputes, how many here, how many in all)."""
    from superlocalmemory.core.injection import sanitize_untrusted_content
    from superlocalmemory.truth import disputes

    found = disputes.listing(conn, pid, limit=disputes.MAX_LISTING)
    keys = _projects_of(conn, sorted({s["fact_id"] for d in found for s in d["sides"]}))
    here = [d for d in found if any(keys.get(s["fact_id"]) in ws.keys for s in d["sides"])]
    lines = []
    for d in here[:NOTES_MAX]:
        said = [f'"{sanitize_untrusted_content((s.get("content") or "")[:SIDE_CHARS])}"'
                for s in d["sides"] if s.get("visible")]
        lines.append(f"[open dispute] {' vs '.join(said) or 'two memories disagree'}: ask "
                     f"which is true now, then settle_dispute(\"{d['dispute_id']}\", keep=...)")
    return lines, len(here), disputes.open_count(conn, pid)


def _workspace_line(ws: Workspace) -> str:
    shown = ws.name if ws.folder in ("", ws.name) else f"{ws.name} (worktree {ws.folder})"
    parts = [f"[project] {shown}"]
    if ws.branch:
        parts.append(f"on branch {ws.branch}")
    line = " ".join(parts)
    if ws.touched:
        line += "; recently touched: " + ", ".join(ws.touched)
    return line


def notes(conn, pid: str, ws: Workspace) -> tuple[str, ...]:
    """The workspace line, then what needs attention in this project."""
    if not ws.keys:
        return ()
    out = [_workspace_line(ws)]
    for part in (_dispute_notes, _stale_notes):
        try:
            out.extend(part(conn, pid, ws))
        except (sqlite3.Error, KeyError, TypeError, ValueError, AttributeError):
            continue
    return tuple(out)


def _dispute_notes(conn, pid: str, ws: Workspace) -> list[str]:
    lines, here, total = disputes_here(conn, pid, ws)
    if here > len(lines):
        lines.append(f"[open dispute] {here - len(lines)} more in this project: slm disputes")
    elif total > here:
        lines.append(f"[open dispute] {total - here} open elsewhere: slm disputes")
    return lines


def _stale_notes(conn, pid: str, ws: Workspace) -> list[str]:
    lines, total = stale_here(conn, pid, ws)
    if total > len(lines):
        lines.append(f"[source changed] {total - len(lines)} more files in this project")
    return lines


def build(conn, pid: str, ws: Workspace, source: tuple[str, tuple], current: str, *,
          limit: int, keep: Callable[[list, int], list]) -> Brief:
    """Notes, this project's memories and its last sessions. Empty parts are left
    out (the caller fills the rest with the profile-wide selection)."""
    facts: list[tuple[str, Mapping]] = []
    try:
        tagged, ends = saved_here(conn, pid, ws, source, current, limit * 10)
        found = keep(_found(conn, pid, ws, source, limit * 10, names=not tagged),
                     limit * 10)
        facts += [("recall", r) for r in mix(keep(tagged, limit * 10), found, limit)]
        facts += [("session_summary", r) for r in last_sessions(ends, ws)]
    except sqlite3.Error:
        facts = []
    return Brief(notes(conn, pid, ws), tuple(facts))


__all__ = ["Brief", "GIT_TIMEOUT_S", "HERE_MEMORIES", "NOTES_MAX", "SUMMARIES_MAX", "TERMS_MAX", "TOUCHED_MAX",
           "Workspace", "build", "disputes_here", "fts_query", "last_sessions", "mix", "notes",
           "distinct", "read_workspace", "relation", "saved_here", "stale_here", "terms"]
