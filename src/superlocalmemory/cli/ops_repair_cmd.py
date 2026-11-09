# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""``slm ops repair-saves``: re-file saves stuck on a memory that is not theirs.

    slm ops repair-saves            # dry run: what would change, nothing written
    slm ops repair-saves --apply    # back up memory.db, then repair

See ``storage/save_repair`` for what is repaired and why. Reads and writes the
selected data root's memory.db directly, like ``slm doctor``, so it works with
the service stopped; a running service picks the requeued saves up on its next
pass. In a workspace that requires sign-in this is the computer owner's tool:
it needs file access to the data folder, nothing else.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from argparse import Namespace


def _db_path():
    from superlocalmemory.infra.data_root import state_path

    return state_path("memory.db")


def _print(args: Namespace, payload: dict, text: str) -> None:
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(text)


def _describe(summary: dict, review: tuple) -> str:
    by_action = summary["by_action"] or ""
    lines = [f"Saves to re-file: {summary['saves_to_refile']} {by_action}".rstrip(),
             f"Left alone for review: {summary['left_for_review']}"]
    lines += [f"  {count} x {why}" for why, count in sorted(
        summary["left_for_review_by_reason"].items(), key=lambda kv: -kv[1])]
    lines += [f"  {op_id}: {why}" for op_id, why in review[:20]]
    lines.append(f"Dead-letter entries of saves that later completed: "
                 f"{summary['stale_dead_letters']}")
    lines.append(f"Agent names to backfill from their save record: "
                 f"{summary['agent_names_to_backfill']}")
    unknown = summary["memories_still_without_agent"]
    if unknown:
        lines.append("Memories whose agent is not known anywhere (left as they are): "
                     + ", ".join(f"{k} {v}" for k, v in sorted(unknown.items())))
    return "\n".join(lines)


def cmd_ops_repair_saves(args: Namespace) -> None:
    from superlocalmemory.storage import save_repair

    db_path = _db_path()
    if not db_path.is_file():
        print(f"error: no memory.db at {db_path}", file=sys.stderr)
        sys.exit(1)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        plan = save_repair.plan(conn)
    finally:
        conn.close()
    summary = plan.summary()
    if not getattr(args, "apply", False):
        _print(args, {"dry_run": True, **summary, "review": list(plan.review)},
               _describe(summary, plan.review) + "\n\nNothing changed. Run again with "
               "--apply to repair (memory.db is backed up first).")
        return
    from superlocalmemory.storage.save_repair_apply import apply

    result = apply(db_path, plan)
    _print(args, {"dry_run": False, **summary, "refiled": result.refiled,
                  "skipped": result.skipped, "agent_names_backfilled":
                  result.agent_names_backfilled, "backup": result.backup,
                  "stale_dead_letters_cleared": result.stale_dead_letters_cleared},
           f"Re-filed {result.refiled} saves ({result.skipped} had changed and were skipped); "
           f"backfilled {result.agent_names_backfilled} agent names; cleared "
           f"{result.stale_dead_letters_cleared} dead-letter entries of saves that completed.\n"
           f"Backup: {result.backup}\nThe service enriches the re-filed saves on its next pass.")


__all__ = ["cmd_ops_repair_saves"]
