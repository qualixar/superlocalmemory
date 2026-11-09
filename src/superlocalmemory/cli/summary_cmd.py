# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory | https://qualixar.com

"""``slm summary`` — the readable layer over your memories (issue #113).

WHY THIS FILE EXISTS
--------------------
4.0.6 shipped the three generators in ``superlocalmemory/summaries/`` with no way
to call them: no command, no MCP tool, no route. The changelog listed the feature
as added, the issue reply said it had landed, and a user could do nothing with it.
This is that missing surface.

Three summaries, each bounded and traceable:

  ``slm summary session <id>``   what one session covered
  ``slm summary day [DATE]``     what a day's main topics were
  ``slm summary project <path>`` what was worked on in a project
  ``slm summary sessions``       which sessions there are to summarise

Every result states its coverage. Session data in particular is sparse — roughly
4% of facts carry a session id on a real store — so a session summary reports what
fraction it could actually see rather than presenting a slice as the whole.

No language model is required: the generators are extractive by default, so this
works in Local Guardian mode with nothing installed.
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from superlocalmemory.infra.data_root import state_path


def _coverage_is_complete(coverage: str) -> bool:
    """Whether *coverage* means "this really is the whole picture".

    Deliberately inverted. My first version listed the values that needed a
    caveat — ``("partial", "sparse", "none", "empty")`` — and three of those four
    are not values this system emits. The real vocabulary is COVERAGE_FULL /
    PARTIAL / INSUFFICIENT / NO_SESSION / UNAVAILABLE, so a session summary
    reporting "no_session" printed no caveat at all: the one honesty feature
    issue #113 asked for, silently inactive.

    Testing for completeness instead means any value that is not FULL — including
    one added later — gets the caveat. The failure mode becomes an unnecessary
    warning rather than a missing one.
    """
    try:
        from superlocalmemory.summaries.base import COVERAGE_FULL

        return coverage == COVERAGE_FULL
    except Exception:
        return coverage == "full"


def _db_path() -> Path:
    return state_path("memory.db")


def _active_profile() -> str:
    """Resolve the active profile without a daemon and without side effects.

    Reads the ``active`` pointer out of ``profiles.json`` directly.
    ``ProfileManager`` would give the same answer, but its constructor calls
    ``mkdir(parents=True)`` — creating directories is not something a read-only
    summary command should do. ``core.profiles`` also has no module-level
    accessor; ``get_active_profile`` there is a method on the manager, and the
    module-level one lives in ``server/routes/helpers.py``, which the CLI must
    not import.
    """
    try:
        from superlocalmemory.core.profiles import DEFAULT_PROFILES_FILE

        path = state_path(DEFAULT_PROFILES_FILE)
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            active = raw.get("active")
            if isinstance(active, str) and active:
                return active
    except Exception:
        pass
    return "default"


def _load_config() -> Any:
    """The active SLMConfig, so Mode B/C get LLM-written summaries.

    Without this the generators receive ``config=None``, ``get_mode_str`` returns
    ``"a"``, and the LLM branch is unreachable — every summary comes out
    extractive no matter which mode the user runs. That is not a graceful
    fallback, it is the enrichment path never being offered.

    Returns None on any failure, which lands on the extractive path. That is the
    right fallback: a deterministic summary beats an error.
    """
    try:
        from superlocalmemory.core.config import SLMConfig

        return SLMConfig.load()
    except Exception:
        return None


def _emit(result: Any, as_json: bool) -> None:
    """Print a SummaryResult as JSON or as prose."""
    if as_json:
        print(json.dumps({
            "kind": result.kind,
            "profile_id": result.profile_id,
            "content": result.content,
            "source_fact_ids": result.source_fact_ids,
            "coverage": result.coverage,
            "generated_by": result.generated_by,
            "metadata": result.metadata,
        }, indent=2, default=str))
        return

    print()
    print(result.content.rstrip() or "(nothing recorded)")
    print()

    # Coverage is not decoration. A summary built from a fraction of the data
    # that presents itself as the whole is the failure mode issue #113 called
    # out by name, so it is stated on every single result, not only bad ones.
    n = len(result.source_fact_ids)
    line = f"Built from {n} memor{'y' if n == 1 else 'ies'} · coverage: {result.coverage}"
    if not _coverage_is_complete(result.coverage):
        line += " — treat as a partial view, not a complete record"
    print(line)
    if result.generated_by:
        print(f"Method: {result.generated_by}")
    _print_source_ids(result.source_fact_ids)


#: Source ids printed in text mode; --json always carries every one.
_SHOWN_IDS = 20


def _print_source_ids(ids: list[str]) -> None:
    """The memories a summary came from, by the id ``slm delete`` and ``fetch`` take.

    Issue #113's binding constraint is that a summary traces back to its
    memories. Telling the reader to rerun with --json to find out which ones
    left the plain-text summary untraceable on the surface people actually read.
    """
    if not ids:
        return
    print("Memory ids:")
    for fact_id in ids[:_SHOWN_IDS]:
        print(f"  {fact_id}")
    if len(ids) > _SHOWN_IDS:
        print(f"  … and {len(ids) - _SHOWN_IDS} more (--json lists every one)")


def cmd_summary(args: Namespace) -> None:
    """Dispatch ``slm summary <subcommand>``."""
    sub = getattr(args, "summary_command", None)
    as_json = bool(getattr(args, "json", False))
    profile = getattr(args, "profile", None) or _active_profile()
    db = _db_path()
    cfg = _load_config()   # Mode B/C enrichment; None -> extractive

    if not db.exists():
        print(f"No memory database at {db}. Run `slm status` first.")
        return

    if sub == "session":
        from superlocalmemory.summaries import generate_session_summary

        _emit(generate_session_summary(db, args.session_id, profile, cfg), as_json)
        return

    if sub == "day":
        from superlocalmemory.summaries import generate_daily_reflection
        from superlocalmemory.summaries.base import local_offset_minutes

        target = getattr(args, "date", None) or date.today().isoformat()
        if target == "yesterday":
            target = (date.today() - timedelta(days=1)).isoformat()
        elif target == "today":
            target = date.today().isoformat()
        try:
            date.fromisoformat(target)
        except ValueError:
            print(f"Not a date: {target!r}. Use YYYY-MM-DD, 'today' or 'yesterday'.",
                  file=sys.stderr)
            sys.exit(2)
        # "today" here is this computer's today, so the day is bucketed in this
        # computer's time zone, not UTC (east of UTC, early-morning memories
        # used to land in the previous day's reflection).
        _emit(generate_daily_reflection(db, target, profile, cfg,
                                        tz_offset_minutes=local_offset_minutes(target)),
              as_json)
        return

    if sub == "sessions":
        _emit_sessions(db, profile, as_json)
        return

    if sub == "project":
        from superlocalmemory.summaries import generate_project_work_log

        path = getattr(args, "path", None) or str(Path.cwd())
        _emit(generate_project_work_log(db, path, profile, cfg), as_json)
        return

    print("Usage: slm summary {session <id> | day [DATE] | project [PATH]}")
    print()
    print("  slm summary day                 what you recorded today")
    print("  slm summary day yesterday       ...or yesterday")
    print("  slm summary day 2026-08-17      ...or a specific date")
    print("  slm summary project             work log for the current directory")
    print("  slm summary sessions            sessions you can summarise")
    print("  slm summary session <id>        what one session covered")
    print()
    print("Add --json to include the ids of the memories a summary came from.")


def _emit_sessions(db: Path, profile: str, as_json: bool) -> None:
    """Recent sessions with memories, so a session summary can be asked for."""
    from superlocalmemory.summaries.sessions import list_recent_sessions

    sessions = list_recent_sessions(db, profile)
    if as_json:
        print(json.dumps({"profile_id": profile, "sessions": sessions}, indent=2))
        return
    if not sessions:
        print("No sessions with saved memories yet. Most memories carry no session, "
              "so a day or project summary usually covers more.")
        return
    print(f"Recent sessions ({len(sessions)}), newest first:")
    for item in sessions:
        n = item["memory_count"]
        print(f"  {item['session_id']}  {n} memor{'y' if n == 1 else 'ies'}, "
              f"last {item['last_at']}")
    print("Summarise one with: slm summary session <id>")


def register_summary_parser(sub: Any) -> None:
    """Attach the ``summary`` parser. Called from cli/main.py."""
    p = sub.add_parser(
        "summary",
        help="Readable summaries of your memories (session, day, project)",
    )
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--profile", help="profile to summarise (default: active)")
    ssub = p.add_subparsers(dest="summary_command", title="summary subcommands")

    s = ssub.add_parser("session", help="what one session covered")
    s.add_argument("session_id", help="session id (see `slm status`)")
    s.add_argument("--json", action="store_true")
    s.add_argument("--profile")

    ls = ssub.add_parser("sessions", help="recent sessions you can summarise")
    ls.add_argument("--json", action="store_true")
    ls.add_argument("--profile")

    d = ssub.add_parser("day", help="what a day's main topics were")
    d.add_argument(
        "date", nargs="?",
        help="YYYY-MM-DD, 'today' or 'yesterday' (default: today)",
    )
    d.add_argument("--json", action="store_true")
    d.add_argument("--profile")

    pr = ssub.add_parser("project", help="what was worked on in a project")
    pr.add_argument(
        "path", nargs="?",
        help="project directory (default: current directory)",
    )
    pr.add_argument("--json", action="store_true")
    pr.add_argument("--profile")
