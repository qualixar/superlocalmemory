#!/usr/bin/env python3
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Set the release version in every place that declares it.

WHY THIS EXISTS
---------------
``tests/test_version_consistency.py`` knows all eleven places a version is
declared and fails when they disagree. Nothing UPDATED them, so every release
was a manual sweep — and 4.0.6 shipped with eight of them still reading 4.0.5:

    plugin-src/manifest.json           4.0.5
    plugin-src/requirements.txt        4.0.5
    package-lock.json (x2)             4.0.5
    plugin/.claude-plugin/plugin.json  4.0.5
    plugin/requirements.txt            4.0.5
    CITATION.cff                       4.0.5
    uv.lock                            4.0.5
    plugin-src/AGENTS.md               4.0.4

That is not cosmetic. ``requirements.txt`` installs the wrong release,
``plugin.json`` makes the editor plugin advertise the wrong version, and
CITATION.cff is the metadata academic citations resolve against.

USAGE
    python3 scripts/bump_version.py 4.0.7          # write
    python3 scripts/bump_version.py 4.0.7 --check  # report, change nothing

``--check`` is the CI-friendly mode: it exits non-zero when anything disagrees
with the target, without touching the tree.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

#: Pending rewrites, keyed by absolute path. Every writer stages its result
#: here instead of touching the disk, and readers see staged text first, so two
#: entries that edit the same file (package-lock.json, CITATION.cff) compose.
#: Nothing reaches the disk until every entry has validated -- a release bump
#: that fails half way once left sixteen files at the new version and three at
#: the old one.
_STAGED: dict[Path, str] = {}


def _load(path: Path) -> str:
    """Current text of ``path``: the staged rewrite if any, else the disk."""
    staged = _STAGED.get(path)
    return staged if staged is not None else path.read_text(encoding="utf-8")


def _store(path: Path, text: str) -> None:
    """Stage a rewrite. :func:`_commit` writes it; nothing else does."""
    _STAGED[path] = text


def _commit() -> None:
    """Write every staged rewrite to disk, then clear the stage."""
    for path, text in _STAGED.items():
        path.write_text(text, encoding="utf-8")
    _STAGED.clear()


def _sub_json(rel: str, version: str, *, package_root: bool = False) -> tuple[str, str]:
    """Rewrite a JSON ``version`` field, preserving formatting where possible."""
    path = _ROOT / rel
    data = json.loads(_load(path))
    if package_root:
        old = data["packages"][""]["version"]
        data["packages"][""]["version"] = version
    else:
        old = data["version"]
        data["version"] = version
    _store(path, json.dumps(data, indent=2) + "\n")
    return old, version


def _sub_regex(rel: str, pattern: str, replacement: str, version: str,
               reported: str | None = None) -> tuple[str, str]:
    """Rewrite the first regex match, returning (old, new) for reporting.

    ``reported`` overrides what gets printed as the new value. Needed for
    date-released, whose new value is a date, not the version — without it the
    log claimed ``2026-08-17 -> 4.0.7``, which reads like the date was replaced
    by a version string.
    """
    path = _ROOT / rel
    text = _load(path)
    m = re.search(pattern, text, re.MULTILINE)
    if not m:
        raise SystemExit(f"{rel}: pattern not found: {pattern}")
    old = m.group(1)
    _store(path, re.sub(pattern, replacement.format(v=version), text,
                        count=1, flags=re.MULTILINE))
    return old, (reported if reported is not None else version)


def _today() -> str:
    from datetime import date
    return date.today().isoformat()


def _sub_all(rel: str, pattern: str, replacement: str, version: str) -> tuple[str, str]:
    """Rewrite EVERY match. For files that stamp the version more than once."""
    path = _ROOT / rel
    text = _load(path)
    m = re.search(pattern, text, re.MULTILINE)
    if not m:
        raise SystemExit(f"{rel}: pattern not found: {pattern}")
    old = m.group(1)
    _store(path, re.sub(pattern, replacement.format(v=version), text,
                        flags=re.MULTILINE))
    return old, version


def _sub_glob(pattern_glob: str, pattern: str, replacement: str,
              version: str) -> tuple[str, str]:
    """Rewrite EVERY match in EVERY file matching a glob.

    ``plugin-src`` carries a ``SuperLocalMemory vX.Y.Z`` footer in sixteen skill
    and agent files, and packaging copies them verbatim, so a stale footer ships.
    Listing them one by one is how the next new skill gets missed; the glob is
    the same set ``tests/test_packaging`` walks.
    """
    old = "<none found>"
    for path in sorted(_ROOT.glob(pattern_glob)):
        text = _load(path)
        m = re.search(pattern, text, re.MULTILINE)
        if not m:
            continue
        old = m.group(1)
        _store(path, re.sub(pattern, replacement.format(v=version), text,
                            flags=re.MULTILINE))
    if old == "<none found>":
        raise SystemExit(f"{pattern_glob}: pattern not found in any file: {pattern}")
    return old, version


def _read_json_at(rel: str, path: tuple):
    """Read a nested JSON value by a path of keys and indexes."""
    data = json.loads(_load(_ROOT / rel))
    for step in path:
        data = data[step]
    return str(data)


def _sub_json_at(rel: str, path: tuple, version: str) -> tuple[str, str]:
    """Set a nested JSON value, preserving the rest of the document."""
    p = _ROOT / rel
    data = json.loads(_load(p))
    node = data
    for step in path[:-1]:
        node = node[step]
    old = str(node.get(path[-1]) if isinstance(node, dict) else node[path[-1]])
    node[path[-1]] = version
    _store(p, json.dumps(data, indent=2, sort_keys=True) + "\n")
    return old, version


def _read_glob(pattern_glob: str, pattern: str) -> str:
    """The one version every matching file claims, or an explicit mixed set.

    Returning the lexicographically first value hid drift once ``4.1.10`` and
    ``4.1.9`` coexisted: ``4.1.10`` sorts first and made ``--check`` report the
    target even though sixteen files were stale.
    """
    found = set()
    for path in sorted(_ROOT.glob(pattern_glob)):
        m = re.search(pattern, _load(path), re.MULTILINE)
        if m:
            found.add(m.group(1))
    if not found:
        return "<not found>"
    if len(found) == 1:
        return next(iter(found))
    return f"<mixed: {', '.join(sorted(found))}>"


def _read(rel: str, pattern: str) -> str:
    m = re.search(pattern, _load(_ROOT / rel), re.MULTILINE)
    return m.group(1) if m else "<not found>"


def _read_json(rel: str, *, package_root: bool = False) -> str:
    data = json.loads(_load(_ROOT / rel))
    return data["packages"][""]["version"] if package_root else data["version"]


#: (label, reader, writer). Kept deliberately parallel to the readers in
#: tests/test_version_consistency.py — if that file gains a source, add it here
#: too, or the next release silently drifts again.
def _plan(version: str):
    return [
        ("package.json",
         lambda: _read_json("package.json"),
         lambda: _sub_json("package.json", version)),
        ("pyproject.toml",
         lambda: _read("pyproject.toml", r'^version\s*=\s*"([^"]+)"'),
         lambda: _sub_regex("pyproject.toml", r'^version\s*=\s*"([^"]+)"',
                            'version = "{v}"', version)),
        ("__init__.py",
         lambda: _read("src/superlocalmemory/__init__.py",
                       r'^__version__\s*=\s*["\']([^"\']+)["\']'),
         lambda: _sub_regex("src/superlocalmemory/__init__.py",
                            r'^__version__\s*=\s*["\']([^"\']+)["\']',
                            '__version__ = "{v}"', version)),
        ("plugin-src/manifest.json",
         lambda: _read_json("plugin-src/manifest.json"),
         lambda: _sub_json("plugin-src/manifest.json", version)),
        ("vscode-extension/package.json",
         lambda: _read_json("vscode-extension/package.json"),
         lambda: _sub_json("vscode-extension/package.json", version)),
        # The official MCP Registry entry (4.2.6): both versions move with the release.
        ("server.json version",
         lambda: _read_json_at("server.json", ("version",)),
         lambda: _sub_json_at("server.json", ("version",), version)),
        ("server.json PyPI package",
         lambda: _read_json_at("server.json", ("packages", 0, "version")),
         lambda: _sub_json_at("server.json", ("packages", 0, "version"), version)),
        ("server.json npm package",
         lambda: _read_json_at("server.json", ("packages", 1, "version")),
         lambda: _sub_json_at("server.json", ("packages", 1, "version"), version)),
        ("plugin-src/requirements.txt",
         lambda: _read("plugin-src/requirements.txt", r"superlocalmemory==([^\s]+)"),
         lambda: _sub_regex("plugin-src/requirements.txt",
                            r"superlocalmemory==([^\s]+)",
                            "superlocalmemory=={v}", version)),
        ("package-lock.json",
         lambda: _read_json("package-lock.json"),
         lambda: _sub_json("package-lock.json", version)),
        ("package-lock.json root",
         lambda: _read_json("package-lock.json", package_root=True),
         lambda: _sub_json("package-lock.json", version, package_root=True)),
        ("plugin/.claude-plugin/plugin.json",
         lambda: _read_json("plugin/.claude-plugin/plugin.json"),
         lambda: _sub_json("plugin/.claude-plugin/plugin.json", version)),
        ("plugin/requirements.txt",
         lambda: _read("plugin/requirements.txt", r"superlocalmemory==([^\s]+)"),
         lambda: _sub_regex("plugin/requirements.txt",
                            r"superlocalmemory==([^\s]+)",
                            "superlocalmemory=={v}", version)),
        ("CITATION.cff",
         # Quotes are REQUIRED, not optional: the release contract test matches
         # ^version:\s*"..." exactly. An unquoted value must read as drift, or
         # this entry reports "ok" and the rewrite never runs.
         lambda: _read("CITATION.cff", r'^version:\s*"([^"]+)"'),
         lambda: _sub_regex("CITATION.cff", r'^version:\s*["\']?([^"\'\n]+?)["\']?$',
                            'version: "{v}"', version)),
        # date-released must be present, quoted, and not in the future:
        # tests/release/test_v37_rc_candidate_contract.py asserts all three.
        ("CITATION.cff date-released",
         lambda: _read("CITATION.cff", r'^date-released:\s*"([^"]+)"'),
         lambda: _sub_regex("CITATION.cff", r'^date-released:\s*"([^"]+)"',
                            'date-released: "%s"' % _today(), version,
                            reported=_today())),
        ("uv.lock",
         lambda: _read("uv.lock",
                       r'name = "superlocalmemory"\nversion = "([^"]+)"'),
         lambda: _sub_regex("uv.lock",
                            r'(?<=name = "superlocalmemory"\n)version = "([^"]+)"',
                            'version = "{v}"', version)),
        # Two copies of the agent rules carry a version footer. codex-plugin/ is
        # the generated artifact the test checks; plugin-src/rules/ is its source.
        # Both sat at 4.0.4 through two releases.
        ("codex-plugin/AGENTS.md",
         lambda: _read("codex-plugin/AGENTS.md", r"SuperLocalMemory v([0-9.]+)"),
         lambda: _sub_regex("codex-plugin/AGENTS.md", r"SuperLocalMemory v([0-9.]+)",
                            "SuperLocalMemory v{v}", version)),
        # Carries the version three times (BEGIN marker, END marker, footer),
        # so this one needs replace-all rather than the first match.
        ("plugin-src/rules/CLAUDE.md.fragment",
         lambda: _read("plugin-src/rules/CLAUDE.md.fragment",
                       r"SuperLocalMemory v([0-9.]+)"),
         lambda: _sub_all("plugin-src/rules/CLAUDE.md.fragment",
                          r"SuperLocalMemory v([0-9.]+)",
                          "SuperLocalMemory v{v}", version)),
        ("plugin-src/rules/AGENTS.md",
         lambda: _read("plugin-src/rules/AGENTS.md", r"SuperLocalMemory v([0-9.]+)"),
         lambda: _sub_regex("plugin-src/rules/AGENTS.md", r"SuperLocalMemory v([0-9.]+)",
                            "SuperLocalMemory v{v}", version)),
        # The version the marketplace advertises. Without it there is nothing
        # for a client to compare, so an installed plugin never looks out of
        # date -- which is exactly what "no upgrade in the plugins" meant. It is
        # a separate stamp from plugin.json's and has to be bumped with it.
        ("marketplace entry version",
         lambda: _read_json_at(".claude-plugin/marketplace.json",
                               ("plugins", 0, "version")),
         lambda: _sub_json_at(".claude-plugin/marketplace.json",
                              ("plugins", 0, "version"), version)),
        # Sixteen skill and agent files under plugin-src carry a version footer
        # and packaging copies them verbatim. tests/test_packaging walks exactly
        # this glob and fails on any that disagree with the package version.
        ("plugin-src/**/*.md footers",
         lambda: _read_glob("plugin-src/**/*.md",
                            r"SuperLocalMemory v([0-9]+\.[0-9]+\.[0-9]+)"),
         lambda: _sub_glob("plugin-src/**/*.md",
                           r"SuperLocalMemory v([0-9]+\.[0-9]+\.[0-9]+)",
                           "SuperLocalMemory v{v}", version)),
        # The Hermes plugin manifest source. scripts/build-hermes-plugin.mjs
        # re-stamps the built hermes-plugin/plugin.yaml from pyproject.toml, so
        # the built copy is always current; this source copy sat at 4.1.18 and
        # showed a stale label to anyone reading plugin-src.
        ("plugin-src/hermes/plugin.yaml",
         lambda: _read("plugin-src/hermes/plugin.yaml", r"^version:\s*(\S+)\s*$"),
         lambda: _sub_regex("plugin-src/hermes/plugin.yaml",
                            r"^version:\s*(\S+)\s*$",
                            "version: {v}", version)),
        # README.md no longer carries a release stamp (no title version, no
        # summary version, no badge), so it is not a source. Listing a source
        # that does not exist made --check red on every tree and made a real
        # bump fail after the other sources were already written.
    ]


def _date_ok(value: str) -> bool:
    """date-released is acceptable when it parses and is not in the future."""
    from datetime import date
    try:
        return date.fromisoformat(value) <= date.today()
    except ValueError:
        return False


_PREDICATES = {"CITATION.cff date-released": _date_ok}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("version", help="target version, e.g. 4.0.7")
    ap.add_argument("--check", action="store_true",
                    help="report drift and exit non-zero; change nothing")
    args = ap.parse_args(argv)

    if not re.fullmatch(r"\d+\.\d+\.\d+", args.version):
        raise SystemExit(f"not a release version: {args.version!r}")

    _STAGED.clear()
    drift: list[str] = []
    failed: list[str] = []
    for entry in _plan(args.version):
        label, read, write = entry[0], entry[1], entry[2]
        accepts = _PREDICATES.get(label, lambda v: v == args.version)
        try:
            current = read()
        except Exception as exc:
            print(f"  !! {label}: unreadable ({exc})")
            failed.append(label)
            continue

        if accepts(current):
            print(f"  ok {label:36s} {current}")
            continue

        drift.append(label)
        if args.check:
            print(f"  ✗  {label:36s} {current}  (want {args.version})")
            continue
        try:
            old, new = write()
        except (Exception, SystemExit) as exc:
            print(f"  !! {label}: cannot update ({exc})")
            failed.append(label)
            continue
        print(f"  ->  {label:36s} {old} -> {new}")

    if failed:
        _STAGED.clear()
        print(f"\n{len(failed)} source(s) could not be read or updated: "
              f"{', '.join(failed)}")
        if not args.check:
            print("nothing was written; fix the source list and run again")
        return 1
    if args.check and drift:
        print(f"\n{len(drift)} source(s) disagree with {args.version}")
        return 1
    if drift:
        _commit()
        print(f"\nupdated {len(drift)} source(s) to {args.version}")
    else:
        print(f"\nall sources already at {args.version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
