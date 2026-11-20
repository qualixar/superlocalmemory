# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Upgrade SLM's own skills, subagents and command in a non-plugin Claude Code install.

When SLM is wired into Claude Code by hand (its MCP server and hooks in ``~/.claude/settings.json``,
skills in ``~/.claude/skills``, advisors in ``~/.claude/agents``, ``/slm-loop`` in
``~/.claude/commands``), nothing refreshed those files after the first install: a release added
skills and rewrote the advisors, and ``slm upgrade-hosts`` still said "already current" because it
looked only at the hook entries. This module closes that gap with the same care as the hook merge.

What it owns: every ``slm-*`` skill, subagent and command that the release ships, and nothing else.
Another skill, another agent, a file with a name SLM does not ship, or anything below those paths is
never read for change and never written.

Who wrote a file is decided from a digest manifest (``~/.claude/.slm-managed.json``) of what SLM
last wrote, the way the Codex installer does it:

* absent and never recorded: added.
* identical to this release: current (and recorded).
* byte-identical to what SLM last wrote: refreshed in place.
* changed since SLM wrote it: the person's edit; left alone and reported.
* absent but recorded: the person removed it; not put back.
* different but never recorded (an install older than this manifest): refreshed only when the
  caller allows it, after the file is copied aside as ``<name>.bak-slm-<version>``. The automatic,
  once-per-version step does not allow it, so it never replaces a file nobody asked it to.
* a file that cannot be read, or a folder or broken link where a file belongs: reported, never
  written. An unreadable file is an error that is tried again; a blocked path needs the person.

The decision is made once, then every write checks again, immediately before the rename, that the
file is still what the decision saw. A person who saves an edit while an upgrade runs keeps it.

A path that resolves outside ``~/.claude`` (a skill folder linked to a checkout, or a whole skills
folder kept in a dotfiles repository) is reported and left alone. When SLM's Claude Code plugin is
installed the plugin owns all of this and ``~/.claude`` is not touched.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Result = dict[str, Any]

MANIFEST_NAME = ".slm-managed.json"
#: Only names with this prefix are ever shipped or touched.
_OWNED_PREFIX = "slm-"
_KINDS = ("skills", "agents", "commands")
#: How a Claude Code that reaches SLM as a plain MCP server (no plugin) spells an SLM tool.
DIRECT_TOOL_PREFIX = "mcp__superlocalmemory__"


# -- what this release ships ----------------------------------------------------------------

def source_root() -> Path | None:
    """The folder holding ``skills/``, ``agents/`` and ``commands/``: a checkout, else the wheel."""
    candidates = (
        Path(__file__).resolve().parents[3] / "plugin-src",
        Path(sysconfig.get_path("data")) / "share" / "superlocalmemory" / "claude-code",
    )
    for root in candidates:
        if (root / "skills").is_dir():
            return root
    return None


def _agents_dir(root: Path) -> Path:
    """Where this layout keeps the sub-agents Claude Code can resolve.

    ``plugin-src/agents`` names SLM tools bare (other hosts read it) and is never installed into
    ``~/.claude``; a checkout reads the generated ``claude-code-assets/agents`` beside it, and the
    wheel's ``agents`` folder is that same generated folder.
    """
    if root.name == "plugin-src":
        return root.parent / "claude-code-assets" / "agents"
    return root / "agents"


class BareToolNames(ValueError):
    """A sub-agent names an SLM tool bare; Claude Code would resolve it to nothing."""


def _check_direct_form(rel: str, data: bytes) -> None:
    """Refuse an agent whose SLM tools are not ``mcp__superlocalmemory__<tool>`` (a stale build)."""
    head = data.decode("utf-8", errors="replace").split("\n---\n", 1)[0]
    for line in head.splitlines():
        if not line.startswith("tools:"):
            continue
        for tool in (t.strip() for t in line[len("tools:"):].split(",")):
            if tool and not tool[0].isupper() and not tool.startswith(DIRECT_TOOL_PREFIX):
                raise BareToolNames(f"{rel} names the tool {tool!r} bare, so Claude Code cannot "
                                    "resolve it; rebuild with scripts/build-claude-code-assets.mjs")


def planned_files(root: Path) -> dict[str, bytes]:
    """Relative POSIX path (``skills/slm-recall/SKILL.md``) to the bytes this release ships."""
    planned: dict[str, bytes] = {}
    for skill in sorted((root / "skills").glob(f"{_OWNED_PREFIX}*")):
        if (skill / "SKILL.md").is_file():
            planned[f"skills/{skill.name}/SKILL.md"] = (skill / "SKILL.md").read_bytes()
    for kind, folder in (("agents", _agents_dir(root)), ("commands", root / "commands")):
        for entry in sorted(folder.glob(f"{_OWNED_PREFIX}*.md")) if folder.is_dir() else ():
            data = entry.read_bytes()
            if kind == "agents":
                _check_direct_form(f"agents/{entry.name}", data)
            planned[f"{kind}/{entry.name}"] = data
    return planned


# -- the manifest ---------------------------------------------------------------------------

def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_manifest(claude_dir: Path) -> dict[str, str]:
    try:
        loaded = json.loads((claude_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    files = loaded.get("files") if isinstance(loaded, dict) else None
    if not isinstance(files, dict):
        return {}
    return {str(k): str(v) for k, v in files.items()}


def _write_manifest(claude_dir: Path, files: dict[str, str], version: str) -> None:
    body = json.dumps({"version": version, "files": dict(sorted(files.items()))}, indent=2) + "\n"
    _replace_bytes(claude_dir / MANIFEST_NAME, body.encode("utf-8"))


# -- files ----------------------------------------------------------------------------------

def _inside(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
    except (OSError, ValueError, RuntimeError):
        return False
    return True


class _ChangedMeanwhile(Exception):
    """The file is no longer what the decision saw; the person edited or created it meanwhile."""


#: ``expect`` for a file that was absent when the decision was made.
_ABSENT = "absent"


def _still_as_decided(target: Path, expect: str, wanted: bytes) -> bool:
    try:
        have = target.read_bytes()
    except FileNotFoundError:
        return expect == _ABSENT  # a file the person removed since is not put back
    if expect == _ABSENT:
        return have == wanted  # created meanwhile: only harmless when it is already this release's
    return _digest(have) == expect


def _replace_bytes(target: Path, data: bytes, expect: str | None = None) -> None:
    """Write ``data`` to ``target`` through a temp file and a rename; keep an existing file's mode.

    With ``expect`` (the digest the file had when it was judged, or ``_ABSENT``) the file is read
    again just before the rename and ``_ChangedMeanwhile`` is raised if it differs, so an edit saved
    while the upgrade ran is never overwritten. Only the instant between that read and the rename
    remains, which a rename-based write cannot close without a lock the person's editor would ignore.
    """
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".slm_tmp")
    try:
        tmp.write_bytes(data)
        try:
            os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
        except OSError:
            pass
        if expect is not None and not _still_as_decided(target, expect, data):
            raise _ChangedMeanwhile(str(target))
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def _identical_backup(target: Path, version: str) -> Path | None:
    """An earlier ``.bak-slm-<version>`` copy holding exactly ``target``'s bytes, if there is one."""
    base = target.with_name(f"{target.name}.bak-slm-{version}")
    have, n, candidate = target.read_bytes(), 0, base
    while candidate.exists():
        if candidate.is_file() and candidate.read_bytes() == have:
            return candidate
        n += 1
        candidate = base.with_name(f"{base.name}.{n}")
    return None


def _backup(target: Path, version: str) -> Path:
    """Copy ``target`` aside; reuse an identical copy, never overwrite a different one."""
    reused = _identical_backup(target, version)
    if reused is not None:
        return reused
    base = target.with_name(f"{target.name}.bak-slm-{version}")
    backup, n = base, 0
    while backup.exists():
        n += 1
        backup = base.with_name(f"{base.name}.{n}")
    shutil.copy2(target, backup)
    return backup


# -- the decision ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Entry:
    rel: str
    action: str  # add | refresh | adopt | held | current | preserved | outside | removed | blocked | unreadable
    note: str = ""


def _decide(claude_dir: Path, rel: str, wanted: bytes, recorded: dict[str, str],
            adopt_unrecorded: bool) -> Entry:
    target = claude_dir / rel
    if not _inside(target, claude_dir):
        return Entry(rel, "outside")
    if os.path.lexists(target) and not target.is_file():
        return Entry(rel, "blocked")  # a folder or a broken link where a file belongs
    if not target.is_file():
        return Entry(rel, "removed" if rel in recorded else "add")
    try:
        have = target.read_bytes()
    except OSError as exc:
        return Entry(rel, "unreadable", str(exc))
    if have == wanted:
        return Entry(rel, "current")
    if rel in recorded:
        return Entry(rel, "refresh" if _digest(have) == recorded[rel] else "preserved")
    return Entry(rel, "adopt" if adopt_unrecorded else "held")


def _has_slm_hooks(settings_path: Path) -> bool:
    from superlocalmemory.hooks import claude_code_hooks as claude

    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and claude.settings_have_slm_hooks(data)


def _integrated(settings_path: Path, plan: dict[str, bytes], recorded: dict[str, str]) -> bool:
    """SLM is already in this Claude Code: its hooks, or any asset it ships, are there."""
    claude_dir = settings_path.parent
    return (_has_slm_hooks(settings_path) or bool(recorded)
            or any(os.path.lexists(claude_dir / rel) for rel in plan))


def is_integrated(settings_path: Path | None = None) -> bool:
    """SLM is in this Claude Code by hooks, a recorded manifest or any shipped asset (no plugin test)."""
    from superlocalmemory.hooks import claude_code_hooks as claude

    settings_path = settings_path or claude.CLAUDE_SETTINGS
    claude_dir = settings_path.parent
    try:
        if not claude_dir.is_dir():
            return False
        root = source_root()
        plan = planned_files(root) if root is not None else {}
        return _integrated(settings_path, plan, _read_manifest(claude_dir))
    except (OSError, BareToolNames):
        return False


def _result(status: str, detail: str, claude_dir: Path, entries: list[Entry] = (),
            **extra: Any) -> Result:
    def names(*actions: str) -> list[str]:
        return [e.rel for e in entries if e.action in actions]

    return {"status": status, "detail": detail, "path": str(claude_dir),
            "added": names("add"), "refreshed": names("refresh"), "adopted": names("adopt"),
            "held": names("held"), "preserved": names("preserved"), "outside": names("outside"),
            "removed": names("removed"), "current": names("current"),
            "blocked": names("blocked"), "unreadable": names("unreadable"), **extra}


def _relative(path: Path, claude_dir: Path) -> str:
    try:
        return path.relative_to(claude_dir.resolve()).as_posix()
    except ValueError:
        return str(path)


def _write_one(claude_dir: Path, entry: Entry, data: bytes, recorded: dict[str, str],
               version: str) -> str | None:
    """Write one file the decision allows; returns the backup's path under ``~/.claude``, if one was made.

    The write only goes through if the file is still what the decision saw: absent for an add, what
    SLM last wrote (the recorded digest) for a refresh, what was just copied aside for an adoption.
    """
    target = claude_dir / entry.rel
    if entry.action == "add":
        _replace_bytes(target, data, expect=_ABSENT)
        return None
    if entry.action == "refresh":
        _replace_bytes(target, data, expect=recorded[entry.rel])
        return None
    real = target.resolve()
    reused = _identical_backup(real, version)
    backup = _backup(real, version)
    try:
        _replace_bytes(target, data, expect=_digest(backup.read_bytes()))
    except BaseException:
        if reused is None:
            backup.unlink(missing_ok=True)  # nothing was replaced, so no orphan copy
        raise
    return _relative(backup, claude_dir)


def _apply(claude_dir: Path, entries: list[Entry], plan: dict[str, bytes],
           recorded: dict[str, str], version: str) -> tuple[dict[str, str], list[str], list[str], list[Entry]]:
    """Write what the decision allows. Returns the new manifest, backups made, errors, final entries."""
    files, backups, errors, final = dict(recorded), [], [], []
    for entry in entries:
        try:
            if entry.action in ("add", "refresh", "adopt"):
                made = _write_one(claude_dir, entry, plan[entry.rel], recorded, version)
                if made:
                    backups.append(made)
            if entry.action in ("add", "refresh", "adopt", "current"):
                files[entry.rel] = _digest(plan[entry.rel])
        except _ChangedMeanwhile:
            entry = Entry(entry.rel, "preserved")  # the person's edit: kept, and not recorded as ours
        except OSError as exc:
            errors.append(f"{entry.rel}: {exc}")
        final.append(entry)
    return files, backups, errors, final


def upgrade_claude_assets(settings_path: Path | None = None, *, version: str, apply: bool = True,
                          adopt_unrecorded: bool = True, source: Path | None = None) -> Result:
    """Skills, subagents and ``/slm-loop`` of a non-plugin Claude Code, refreshed by the rules above."""
    from superlocalmemory.hooks import claude_code_hooks as claude
    from superlocalmemory.hooks.host_hook_upgrade import claude_plugin_present

    settings_path = settings_path or claude.CLAUDE_SETTINGS
    claude_dir = settings_path.parent
    if claude_plugin_present(settings_path):
        return _result("plugin-managed", "SLM's skills, agents and command come from the Claude "
                       "Code plugin; ~/.claude left untouched", claude_dir)
    if not claude_dir.is_dir():
        return _result("not-integrated", "no Claude Code folder; none created", claude_dir)
    root = source or source_root()
    if root is None:
        return _result("error", "this installation ships no Claude Code skills", claude_dir)
    try:
        plan = planned_files(root)
    except BareToolNames as exc:
        return _result("error", f"not installed: {exc}", claude_dir)
    except OSError as exc:
        return _result("error", f"could not update Claude Code assets: {exc}", claude_dir)
    try:
        recorded = _read_manifest(claude_dir)
        if not _integrated(settings_path, plan, recorded):
            return _result("not-integrated", "no SLM skills or hooks there, so none added", claude_dir)
        entries = [_decide(claude_dir, rel, data, recorded, adopt_unrecorded)
                   for rel, data in plan.items()]
        if not apply:
            todo = [e for e in entries if e.action in _WRITES]
            return _finish("would-update" if todo else None, entries, claude_dir)
        files, backups, errors, entries = _apply(claude_dir, entries, plan, recorded, version)
        if files != recorded:
            _write_manifest(claude_dir, files, version)
    except OSError as exc:
        return _result("error", f"could not update Claude Code assets: {exc}", claude_dir)
    errors += [_unreadable_words(e) for e in entries if e.action == "unreadable"]
    if errors:
        return _result("error", "could not update " + "; ".join(errors), claude_dir, entries,
                       backups=backups)
    todo = [e for e in entries if e.action in _WRITES]
    return _finish("updated" if todo else None, entries, claude_dir, backups=backups)


_WRITES = ("add", "refresh", "adopt")


def _unreadable_words(entry: Entry) -> str:
    return f"{entry.rel}: cannot read it ({entry.note}); left untouched, tried again at the next start"


def _finish(status: str | None, entries: list[Entry], claude_dir: Path, **extra: Any) -> Result:
    """``status`` when something was (or would be) written; otherwise current or needs-review.

    A file that cannot be read is an error (so it is retried), not "the person changed it".
    """
    unreadable = [e for e in entries if e.action == "unreadable"]
    if unreadable:
        return _result("error", "could not check " + "; ".join(_unreadable_words(e) for e in unreadable),
                       claude_dir, entries, **extra)
    held_back = any(e.action in ("held", "preserved", "outside", "blocked") for e in entries)
    final = status or ("needs-review" if held_back else "current")
    counts = {a: sum(e.action == a for e in entries) for a in ("add", "refresh", "adopt")}
    detail = ", ".join(f"{n} to {a}" for a, n in counts.items() if n) or final
    return _result(final, detail, claude_dir, entries, **extra)


__all__ = ("MANIFEST_NAME", "is_integrated", "planned_files", "source_root", "upgrade_claude_assets")
