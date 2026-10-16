#!/usr/bin/env python3
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Pure helpers for the release fan-out (scripts/release/<channel>.sh).

Every function here is side-effect free except the CLI wrapper at the bottom,
so the channel scripts stay thin and the logic is unit-tested
(tests/test_ci_guards/test_release_distribution.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

STATUSES = ("ok", "skipped", "failed")
_TOP_LEVEL_VERSION = re.compile(r'^(  |\t)"version"(\s*:\s*)"([^"]*)"', re.MULTILINE)
_SAFE_CHANNEL = re.compile(r"^[a-z0-9-]+$")

# Human-gated listings: each needs a person (a review queue, a form or a PR to
# someone else's repository). The release opens one issue that lists them.
MANUAL_LISTINGS: tuple[tuple[str, str], ...] = (
    ("Claude plugin directory: approve the new version (auto-publish is off)",
     "https://claude.ai/directory/manage"),
    ("Cursor Marketplace: confirm the listing picked up the release",
     "https://cursor.com/marketplace/publish"),
    ("VS Code Marketplace: publish the .vsix attached to the GitHub release",
     "https://marketplace.visualstudio.com/manage"),
    ("Antigravity listing form: update the version if the form asks for one",
     "(form link not recorded in the repository; use the one from the original submission)"),
    ("Kimi curated tier: follow up on the curation request",
     "(request link not recorded in the repository)"),
    ("Kilo Code marketplace PR (mcps/superlocalmemory)",
     "https://github.com/Kilo-Org/kilo-marketplace"),
    ("Cline MCP marketplace issue",
     "https://github.com/cline/mcp-marketplace"),
    ("OpenCode: awesome-opencode PR",
     "https://github.com/awesome-opencode/awesome-opencode"),
    ("DeepSeek Harness: awesome-dsh-plugin PR",
     "https://github.com/awesome-dsh-plugin/awesome-dsh-plugin"),
    ("xAI plugin marketplace PR",
     "https://github.com/xai-org/plugin-marketplace/pull/1455"),
)


def changelog_section(text: str, version: str) -> str:
    """Return the body under ``## [version]`` up to the next ``## `` heading."""
    heading = re.compile(rf"^## \[{re.escape(version)}\][^\n]*\n", re.MULTILINE)
    match = heading.search(text)
    if match is None:
        raise ValueError(f"CHANGELOG.md has no '## [{version}]' section")
    rest = text[match.end():]
    following = re.search(r"^## ", rest, re.MULTILINE)
    body = (rest[: following.start()] if following else rest).strip()
    if not body:
        raise ValueError(f"CHANGELOG.md section {version} is empty")
    return body + "\n"


def set_json_version(text: str, version: str) -> tuple[str, bool]:
    """Set the top-level ``"version"`` of a JSON document, keeping its layout."""
    match = _TOP_LEVEL_VERSION.search(text)
    if match is None:
        raise ValueError("no top-level \"version\" key found")
    if match.group(3) == version:
        return text, False
    start, end = match.span(3)
    new_text = text[:start] + version + text[end:]
    if json.loads(new_text).get("version") != version:
        raise ValueError("version edit did not land on the top-level key")
    return new_text, True


def _numeric_parts(version: str) -> tuple[int, ...]:
    core = re.split(r"[-+]", version, maxsplit=1)[0]
    return tuple(int(part) for part in core.split("."))


def compare_versions(left: str, right: str) -> int:
    """Numeric comparison of dotted versions: -1, 0 or 1."""
    a, b = _numeric_parts(left), _numeric_parts(right)
    width = max(len(a), len(b))
    a, b = a + (0,) * (width - len(a)), b + (0,) * (width - len(b))
    return (a > b) - (a < b)


def yaml_scalar(text: str, key: str) -> str | None:
    """Read a top-level ``key: value`` scalar without a YAML dependency."""
    match = re.search(rf"^{re.escape(key)}:\s*['\"]?([^'\"\n#]+?)['\"]?\s*$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def tree_digest(root: Path) -> dict[str, str]:
    """sha256 of every regular file under ``root``, keyed by POSIX relative path."""
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def skill_changed(local: Mapping[str, str], published: Mapping[str, str]) -> bool:
    """True when any local file is missing upstream or differs from it.

    Files only the registry has (ClawHub adds a generated skill-card.md) are
    ignored: they are not ours to publish.
    """
    return any(published.get(path) != digest for path, digest in local.items())


def _read_status_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    pairs = (line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines() if "=" in line)
    return {key.strip(): value.strip() for key, value in pairs}


def record(channel: str, outcome: str, status_file: Path) -> dict[str, str]:
    """Map a GitHub step outcome plus the script's own status to one result."""
    if not _SAFE_CHANNEL.match(channel):
        raise ValueError(f"invalid channel name: {channel!r}")
    reported = _read_status_file(status_file)
    if outcome == "success":
        status = reported.get("status", "ok")
        status = status if status in STATUSES else "ok"
        detail = reported.get("detail", "")
    elif outcome == "failure":
        status, detail = "failed", reported.get("detail", "the channel script exited non-zero; see its log")
    else:
        status, detail = "skipped", f"step {outcome or 'did not run'}"
    return {"channel": channel, "status": status, "detail": detail}


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def summary_table(results: Iterable[Mapping[str, str]], channels: Sequence[str]) -> tuple[str, list[str]]:
    """Markdown channel table and the names of channels that failed."""
    by_channel = {item["channel"]: item for item in results}
    rows = ["| Channel | Result | Detail |", "| --- | --- | --- |"]
    failed: list[str] = []
    for channel in channels:
        item = by_channel.get(channel, {"status": "failed", "detail": "no result recorded (the job did not run)"})
        if item["status"] == "failed":
            failed.append(channel)
        rows.append(f"| {channel} | {item['status']} | {_cell(item.get('detail', ''))} |")
    return "\n".join(rows) + "\n", failed


def checklist_body(version: str) -> str:
    """Issue body listing every channel that still needs a person."""
    lines = [
        f"SuperLocalMemory v{version} is on PyPI and npm, and the release workflow has run",
        "every automatic channel (see the run's summary table). These listings need a person:",
        "",
    ]
    lines += [f"- [ ] {label}: {link}" for label, link in MANUAL_LISTINGS]
    lines += [
        "",
        "Automatic channels are described in docs/release-distribution.md. Tick every box",
        "or skip it consciously.",
    ]
    return "\n".join(lines) + "\n"


# --- CLI ---------------------------------------------------------------------


def _cmd_changelog(args: argparse.Namespace) -> int:
    sys.stdout.write(changelog_section(Path(args.changelog).read_text(encoding="utf-8"), args.version))
    return 0


def _cmd_set_json_version(args: argparse.Namespace) -> int:
    path = Path(args.file)
    new_text, changed = set_json_version(path.read_text(encoding="utf-8"), args.version)
    if changed and not args.check:
        path.write_text(new_text, encoding="utf-8")
    print("changed" if changed else "unchanged")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    print(compare_versions(args.left, args.right))
    return 0


def _cmd_yaml_scalar(args: argparse.Namespace) -> int:
    value = yaml_scalar(Path(args.file).read_text(encoding="utf-8"), args.key)
    if value is None:
        print(f"{args.file}: no top-level {args.key}", file=sys.stderr)
        return 1
    print(value)
    return 0


def _cmd_skill_changed(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.published_json).read_text(encoding="utf-8"))
    files = (payload.get("version") or {}).get("files") or []
    published = {item["path"]: item["sha256"] for item in files}
    print("changed" if skill_changed(tree_digest(Path(args.skill_dir)), published) else "unchanged")
    return 0


def _cmd_record(args: argparse.Namespace) -> int:
    result = record(args.channel, args.outcome, Path(args.status_file))
    Path(args.out).write_text(json.dumps(result) + "\n", encoding="utf-8")
    print(f"{result['channel']}: {result['status']}")
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(args.results).glob("*.json"))]
    table, failed = summary_table(results, args.channels.split(","))
    sys.stdout.write(table)
    if failed:
        print(f"\nFailed channels: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed and args.fail_on_failed else 0


def _cmd_checklist(args: argparse.Namespace) -> int:
    sys.stdout.write(checklist_body(args.version))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("changelog-section", help="print one version's CHANGELOG body")
    p.add_argument("version")
    p.add_argument("--changelog", default="CHANGELOG.md")
    p.set_defaults(fn=_cmd_changelog)

    p = sub.add_parser("set-json-version", help="set a JSON file's top-level version")
    p.add_argument("file")
    p.add_argument("version")
    p.add_argument("--check", action="store_true", help="report only, do not write")
    p.set_defaults(fn=_cmd_set_json_version)

    p = sub.add_parser("compare", help="print -1, 0 or 1")
    p.add_argument("left")
    p.add_argument("right")
    p.set_defaults(fn=_cmd_compare)

    p = sub.add_parser("yaml-scalar", help="print a top-level YAML scalar")
    p.add_argument("file")
    p.add_argument("key")
    p.set_defaults(fn=_cmd_yaml_scalar)

    p = sub.add_parser("skill-changed", help="compare a skill folder with clawhub inspect --json")
    p.add_argument("skill_dir")
    p.add_argument("published_json")
    p.set_defaults(fn=_cmd_skill_changed)

    p = sub.add_parser("record", help="write one channel's result JSON")
    p.add_argument("--channel", required=True)
    p.add_argument("--outcome", required=True)
    p.add_argument("--status-file", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=_cmd_record)

    p = sub.add_parser("summary", help="print the channel table")
    p.add_argument("--results", required=True)
    p.add_argument("--channels", required=True)
    p.add_argument("--fail-on-failed", action="store_true")
    p.set_defaults(fn=_cmd_summary)

    p = sub.add_parser("checklist-body", help="print the manual listing checklist")
    p.add_argument("version")
    p.set_defaults(fn=_cmd_checklist)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.fn(args)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"release_tools: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
