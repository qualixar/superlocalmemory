# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm flags rollback-retirements [--dry-run]``: bring back what 4.2.30 retired (D-23, D-49).

The kill switch does not undo retirements; this does. It always says first how many memories
would come back. Without ``--dry-run`` it then brings them back as one recorded decision (who,
channel ``cli``, when) that can be undone from the decision log. Any caller may run it.

With SuperLocalMemory running it goes through the daemon (the one writer); without, it writes the
store directly and briefly, like the other file-only commands.
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from typing import Any


def _words(n: int) -> str:
    return f"{n} memory" if n == 1 else f"{n} memories"


def _profile(args: Namespace) -> str:
    from superlocalmemory.cli.health_memory import _profile as active

    return active(args)


def _through_daemon(profile: str, dry_run: bool) -> dict[str, Any]:
    from superlocalmemory.cli.daemon import daemon_request
    from superlocalmemory.cli.trust_cmd import _who

    body: dict[str, Any] = {"dry_run": dry_run, "by": _who()}
    if profile:
        body["profile_id"] = profile
    got = daemon_request("POST", "/api/flags/rollback-retirements", body)
    if not isinstance(got, dict) or not got.get("success"):
        raise RuntimeError("SuperLocalMemory did not finish; nothing more was changed.")
    return got


def _directly(args: Namespace, profile: str, dry_run: bool) -> dict[str, Any]:
    from superlocalmemory.cli.doctor_truth import _db_path
    from superlocalmemory.cli.trust_cmd import _who
    from superlocalmemory.core import decisions
    from superlocalmemory.storage.memory_write import memory_write
    from superlocalmemory.truth import retirement_rollback

    path = _db_path(args)
    if not path.is_file():
        raise FileNotFoundError(f"No store at {path}")
    with memory_write(path) as conn:
        return retirement_rollback.run(conn, profile_id=profile or "default",
                                       decider=decisions.Decider(_who(), "cli"), dry_run=dry_run)


def _run(args: Namespace, profile: str, dry_run: bool) -> dict[str, Any]:
    from superlocalmemory.cli.daemon import is_daemon_running

    if is_daemon_running():
        return _through_daemon(profile, dry_run)
    return _directly(args, profile, dry_run)


def rollback_retirements(args: Namespace) -> int:
    use_json = bool(getattr(args, "json", False))
    dry_run = bool(getattr(args, "dry_run", False))
    profile = _profile(args) if not getattr(args, "profile", "") else str(args.profile).strip()
    try:
        first = _run(args, profile, True)
        count = int(first.get("would_come_back", 0))
        result = first if dry_run or not count else _run(args, profile, False)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - a plain refusal, never a traceback
        print(f"Nothing was changed: {exc}", file=sys.stderr)
        return 1
    if use_json:
        print(json.dumps({"profile": profile, "dry_run": dry_run, **{
            k: result.get(k) for k in ("would_come_back", "came_back", "refused", "decision_id")
        }}, indent=2))
        return 0
    if not count:
        print("No memory was retired by 4.2.30, so nothing needs to come back.")
        return 0
    print(f"{_words(count)} would come back.")
    if dry_run:
        print("Nothing was changed. Run it without --dry-run to bring them back.")
        return 0
    print(f"Brought back {_words(int(result.get('came_back', 0)))} that 4.2.30 had retired."
          + (f" {int(result['refused'])} could not be brought back and were left as they are."
             if result.get("refused") else ""))
    print("It is recorded under recent decisions, where Undo retires them again.")
    return 0


__all__ = ["rollback_retirements"]
