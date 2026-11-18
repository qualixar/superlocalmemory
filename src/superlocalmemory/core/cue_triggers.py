# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Cue triggers (4.2.30, W3-8, WS-4): show a memory when its cue matches the step.

A cue is ``kind:pattern`` attached to a memory (``remember(cues=[...])``):

    path:src/**/schema.py        a file glob, matched on edits and reads
    tool:Bash:git push           a tool name, then an optional part of the Bash command
    error:database is locked     a phrase in what a tool just printed
    repo:superlocalmemory        a folder name in the working directory's path
    date:2026-11-20              from this UTC date on

Matching is deterministic and visible: ``fnmatch`` for paths, case-insensitive
substrings for tools and errors, equality for repository names, a date compare.
There is no regular expression anywhere in a cue (no ReDoS), and the patterns and
the text they are tried on are length-bounded.

A second trigger rides on the same events (W3 lease re-check): when an agent edits
a file, the memories saved from that file (``memory_leases``) may now be out of
date, and the agent is told.

Pure functions plus one read-only query. The hook that delivers is
``hooks/cue_hook.py``; binding cues to facts is ``core/cue_binding.py``.
"""

from __future__ import annotations

import fnmatch
import os
import re
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath
from typing import Any

PATH, TOOL, ERROR, REPO, DATE = "path", "tool", "error", "repo", "date"
KINDS = (PATH, TOOL, ERROR, REPO, DATE)
EXPLICIT, EXTRACTED = "explicit", "extracted"

#: Cues on one memory, and the length of one pattern (routing-only caps).
MAX_CUES = 8
MAX_PATTERN = 200
#: ``*`` marks in one path glob: fnmatch compiles each to a regex group, so the
#: count is capped to keep matching linear in practice (routing-only).
MAX_STARS = 8
#: How much of a tool's output an error cue reads (design: the first 8 KB).
ERROR_WINDOW = 8192
#: Cues read per event and memories shown per event (routing-only).
MAX_SCAN = 2000
MAX_FIRED = 5

EDIT_TOOLS = frozenset({"edit", "write", "multiedit", "notebookedit"})
READ_TOOLS = frozenset({"read"})
_PATH_KEYS = ("file_path", "notebook_path", "path")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TOOL_NAME = re.compile(r"^[A-Za-z0-9_.\-]+$")


class CueError(ValueError):
    """A cue the door refuses; the message says what to send instead."""


@dataclass(frozen=True, slots=True)
class Cue:
    kind: str
    pattern: str

    @property
    def spec(self) -> str:
        return f"{self.kind}:{self.pattern}"


def parse_cue(spec: object) -> Cue:
    """``kind:pattern`` as a checked :class:`Cue`, or :class:`CueError`."""
    if not isinstance(spec, str) or ":" not in spec:
        raise CueError("a cue is KIND:PATTERN, for example path:src/**/schema.py "
                       f"(kinds: {', '.join(KINDS)})")
    kind, _, pattern = spec.partition(":")
    kind, pattern = kind.strip().lower(), pattern.strip()
    if kind not in KINDS:
        raise CueError(f"unknown cue kind {kind!r}; use one of {', '.join(KINDS)}")
    if not pattern:
        raise CueError(f"the {kind} cue needs a pattern after the colon")
    if len(pattern) > MAX_PATTERN or any(c in pattern for c in "\x00\r\n"):
        raise CueError(f"the {kind} cue must be one line of at most {MAX_PATTERN} characters")
    _CHECKS[kind](pattern)
    return Cue(kind, pattern)


def parse_cues(specs: object) -> tuple[Cue, ...]:
    """Every cue checked, duplicates dropped; ``()`` for None. Raises before any is used."""
    if specs is None:
        return ()
    if isinstance(specs, str) or not isinstance(specs, Iterable):
        raise CueError("cues must be a list of KIND:PATTERN strings")
    out: dict[str, Cue] = {}
    for spec in specs:
        cue = parse_cue(spec)
        out.setdefault(cue.spec, cue)
    if len(out) > MAX_CUES:
        raise CueError(f"at most {MAX_CUES} cues per memory")
    return tuple(out.values())


def _check_path(pattern: str) -> None:
    if pattern.count("*") > MAX_STARS:
        raise CueError(f"a path cue may have at most {MAX_STARS} '*' marks")


def _check_tool(pattern: str) -> None:
    name = pattern.partition(":")[0].strip()
    if not _TOOL_NAME.match(name):
        raise CueError("a tool cue starts with the tool name (letters, digits, _ . -), "
                       "for example tool:Bash:git push")
    if ":" in pattern and not pattern.partition(":")[2].strip():
        raise CueError("a tool cue with a second colon needs the command part after it")


def _check_repo(pattern: str) -> None:
    if "/" in pattern or "\\" in pattern:
        raise CueError("a repo cue is a folder name, not a path (repo:superlocalmemory)")


def _check_date(pattern: str) -> None:
    try:
        if not _DATE.match(pattern):
            raise ValueError(pattern)
        date.fromisoformat(pattern)
    except ValueError:
        raise CueError("a date cue is a real calendar date, YYYY-MM-DD") from None


def _check_error(pattern: str) -> None:
    del pattern  # any non-empty phrase is a valid substring


_CHECKS = {PATH: _check_path, TOOL: _check_tool, ERROR: _check_error,
           REPO: _check_repo, DATE: _check_date}


# -- the event --------------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Event:
    """One tool event from PreToolUse or PostToolUse."""

    name: str
    tool_name: str
    tool_input: Mapping[str, Any] = field(default_factory=dict)
    tool_response: str = ""
    cwd: str = ""
    session_id: str = ""

    @property
    def post(self) -> bool:
        return self.name == "PostToolUse"

    @property
    def path(self) -> str:
        """The absolute path an edit or read tool names, else ``""``."""
        if self.tool_name.lower() not in EDIT_TOOLS | READ_TOOLS:
            return ""
        for key in _PATH_KEYS:
            value = self.tool_input.get(key)
            if isinstance(value, str) and value.strip():
                raw = os.path.expanduser(value.strip())
                if not os.path.isabs(raw) and self.cwd:
                    raw = os.path.join(self.cwd, raw)
                return os.path.normpath(raw) if os.path.isabs(raw) else ""
        return ""

    @property
    def command(self) -> str:
        value = self.tool_input.get("command")
        return value if isinstance(value, str) else ""


def event_from_payload(payload: object) -> Event | None:
    """The :class:`Event` of a hook payload, or None when it is not a tool event."""
    if not isinstance(payload, dict):
        return None
    name = payload.get("hook_event_name") or payload.get("event") or ""
    tool = payload.get("tool_name")
    if name not in ("PreToolUse", "PostToolUse") or not isinstance(tool, str) or not tool:
        return None
    tool_input = payload.get("tool_input")
    sid = payload.get("session_id")
    cwd = payload.get("cwd")
    return Event(name, tool, tool_input if isinstance(tool_input, dict) else {},
                 _flatten(payload.get("tool_response")) if name == "PostToolUse" else "",
                 cwd if isinstance(cwd, str) else "", sid.strip() if isinstance(sid, str) else "")


def _flatten(value: object, *, depth: int = 0, budget: int = ERROR_WINDOW) -> str:
    """The text leaves of a tool response, joined, cut to ``budget`` characters."""
    parts: list[str] = []
    used = 0
    stack: list[tuple[object, int]] = [(value, depth)]
    while stack and used < budget:
        item, level = stack.pop(0)
        if isinstance(item, str):
            parts.append(item)
            used += len(item) + 1
        elif level < 4 and isinstance(item, dict):
            stack.extend((v, level + 1) for v in item.values())
        elif level < 4 and isinstance(item, list):
            stack.extend((v, level + 1) for v in item)
    return "\n".join(parts)[:budget]


# -- matching ---------------------------------------------------------------------------------------

def path_matches(pattern: str, abs_path: str, cwd: str = "") -> bool:
    """Whether the glob ``pattern`` names ``abs_path``.

    Tried on the absolute path and on the path relative to ``cwd``. A relative
    pattern also matches the tail of the absolute path (so ``src/**/schema.py``
    works from any folder), and ``**/`` may match no folder at all.
    """
    if not abs_path:
        return False
    pat = os.path.expanduser(pattern) if pattern.startswith("~") else pattern
    pats = {pat, pat.replace("**/", "")}
    subjects = [abs_path]
    if cwd:
        try:
            subjects.append(str(PurePosixPath(abs_path).relative_to(cwd)))
        except ValueError:
            pass
    for p in pats:
        candidates = [p] if p.startswith("/") else [p, "*/" + p]
        for subject in subjects:
            if any(fnmatch.fnmatchcase(subject, c) for c in candidates):
                return True
    return False


def _tool_matches(pattern: str, event: Event) -> bool:
    name, _, part = pattern.partition(":")
    if event.tool_name.lower() != name.strip().lower():
        return False
    part = part.strip()
    return not part or part.lower() in event.command.lower()


def _repo_matches(name: str, cwd: str) -> bool:
    return bool(cwd) and name in PurePosixPath(cwd).parts


def matches(cue: Cue, event: Event, today: date) -> bool:
    """Whether one cue fires on this event."""
    if cue.kind == PATH:
        return path_matches(cue.pattern, event.path, event.cwd)
    if cue.kind == TOOL:
        return _tool_matches(cue.pattern, event)
    if cue.kind == ERROR:
        return event.post and cue.pattern.lower() in event.tool_response[:ERROR_WINDOW].lower()
    if cue.kind == REPO:
        return _repo_matches(cue.pattern, event.cwd)
    if cue.kind == DATE:
        try:
            return today >= date.fromisoformat(cue.pattern)
        except ValueError:  # a stored row that no door would have accepted
            return False
    return False


# -- what the store says ----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Hit:
    """A memory that should be shown on this event, and why."""

    fact_id: str
    content: str
    via: str  # "cue" or "lease"
    reason: str  # the cue spec, or the edited file for a lease
    memory_kind: str = ""
    created_at: str = ""


#: A fact may be shown only while it is live: not retired, archived or quarantined.
_LIVE = """f.quarantined = 0 AND f.non_answer = 0 AND f.lifecycle != 'archived'
       AND f.merged_into IS NULL AND v.system_expired_at IS NULL"""
_CUES = f"""
SELECT c.kind, c.pattern, f.fact_id, f.content, f.memory_kind, f.created_at
  FROM memory_cues c
  JOIN atomic_facts f ON f.fact_id = c.fact_id
  LEFT JOIN fact_temporal_validity v ON v.fact_id = f.fact_id
 WHERE c.profile_id = ? AND f.profile_id = c.profile_id AND c.kind IN ({{kinds}}) AND {_LIVE}{{fw}}
 ORDER BY c.created_at, c.cue_id
 LIMIT ?
"""
_LEASES = f"""
SELECT f.fact_id, f.content, f.memory_kind, f.created_at
  FROM memory_leases l
  JOIN atomic_facts f ON f.fact_id = l.fact_id
  LEFT JOIN fact_temporal_validity v ON v.fact_id = f.fact_id
 WHERE l.profile_id = ? AND l.source_kind = 'file' AND l.locator IN ({{locators}}) AND {_LIVE}{{fw}}
 ORDER BY f.created_at, f.fact_id
 LIMIT ?
"""


def _firewall(conn: sqlite3.Connection) -> str:
    """A cue is a push channel: a memory the instruction firewall holds (taint below
    ``system``, not approved by a person) is never pushed by one (D-48, Fable A-10).
    Empty unless ``trust.taint`` is on; a fact with no stored class is held (closed)."""
    from superlocalmemory.trust import firewall

    return firewall.clause(conn)


def _kinds_for(event: Event) -> tuple[str, ...]:
    kinds = [TOOL, REPO, DATE]
    if event.path:
        kinds.append(PATH)
    if event.post:
        kinds.append(ERROR)
    return tuple(kinds)


def find(conn: sqlite3.Connection, profile_id: str, event: Event, *,
         today: date | None = None, shown: Iterable[str] = (),
         limit: int = MAX_FIRED) -> list[Hit]:
    """The memories to show on ``event``: cue matches, then lease warnings.

    ``conn`` is only read. ``shown`` holds fact ids this conversation already
    got (``need_gate.shown_in_session``); a cue or lease warning fires at most
    once per conversation per memory. A store without the cue or lease tables
    answers ``[]``.
    """
    today = today or datetime.now(UTC).date()
    seen = set(shown)
    hits: list[Hit] = []
    for hit in (*_cue_hits(conn, profile_id, event, today), *_lease_hits(conn, profile_id, event)):
        if hit.fact_id not in seen:
            seen.add(hit.fact_id)
            hits.append(hit)
        if len(hits) >= limit:
            break
    return hits


def _cue_hits(conn: sqlite3.Connection, profile_id: str, event: Event,
              today: date) -> list[Hit]:
    kinds = _kinds_for(event)
    try:
        rows = conn.execute(_CUES.format(kinds=",".join("?" * len(kinds)), fw=_firewall(conn)),
                            (profile_id, *kinds, MAX_SCAN)).fetchall()
    except sqlite3.OperationalError:  # a store before M078
        return []
    return [Hit(fid, content, "cue", Cue(kind, pattern).spec, mk or "", at or "")
            for kind, pattern, fid, content, mk, at in rows
            if matches(Cue(kind, pattern), event, today)]


def lease_locators(event: Event) -> tuple[str, ...]:
    """The stored spellings of the file an edit just changed (the lease keeps the
    resolved path), only for a finished edit."""
    if not event.post or event.tool_name.lower() not in EDIT_TOOLS or not event.path:
        return ()
    try:
        real = str(Path(event.path).resolve())
    except (OSError, RuntimeError):
        real = event.path
    return tuple(dict.fromkeys((event.path, real)))


def _lease_hits(conn: sqlite3.Connection, profile_id: str, event: Event) -> list[Hit]:
    locators = lease_locators(event)
    if not locators:
        return []
    try:
        rows = conn.execute(_LEASES.format(locators=",".join("?" * len(locators)),
                                           fw=_firewall(conn)),
                            (profile_id, *locators, MAX_FIRED)).fetchall()
    except sqlite3.OperationalError:  # a store before M070
        return []
    return [Hit(fid, content, "lease", locators[0], mk or "", at or "")
            for fid, content, mk, at in rows]


def lease_note(path: str) -> str:
    """The one line the agent gets about memories saved from a file it just edited."""
    return (f"You just edited {path}. The memories below were saved from that file and may "
            "be out of date: check them against the file before relying on them.")


__all__ = ["CueError", "Cue", "Event", "Hit", "KINDS", "EXPLICIT", "EXTRACTED", "MAX_CUES",
           "ERROR_WINDOW", "MAX_FIRED", "event_from_payload", "find", "lease_locators",
           "lease_note", "matches", "parse_cue", "parse_cues", "path_matches"]
