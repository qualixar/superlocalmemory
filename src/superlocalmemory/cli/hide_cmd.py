# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""``slm hide`` — keep a memory out of answers and session starts.

SLM flags machine bookkeeping on its own (``encoding/non_answer.py``): such a
row stays stored, listed and counted, but recall never returns it as an answer
and session start never shows it. A user had no way to say the same about a
row the rule cannot recognise ("Session ended with a quick test", a vague
note), and no way to overrule the rule when it is wrong.

    slm hide <fact_id> [<fact_id> ...]          # never an answer again
    slm hide --undo <fact_id> [<fact_id> ...]   # an answer again, for good

Both directions are stored as the user's own decision (``non_answer_user``),
which the automatic rule and every repair migration leave alone. Nothing is
deleted or edited; ``slm hide --undo`` restores the row exactly. Fact ids are
what ``slm recall --json`` and ``slm list`` show.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from typing import Any

__all__ = ["cmd_hide", "register_hide_parser", "set_hidden"]

_MAX_IDS = 500


def register_hide_parser(sub: Any) -> None:
    """Attach ``slm hide``. Called from cli/main.py."""
    p = sub.add_parser(
        "hide",
        help="Keep memories out of answers and session starts (--undo to show again)",
    )
    p.add_argument("fact_ids", nargs="+", help="Fact IDs (from `slm recall --json` or `slm list`)")
    p.add_argument("--undo", action="store_true",
                   help="Show these memories in answers again and stop hiding them automatically")
    p.add_argument("--json", action="store_true", help="machine-readable output")


def set_hidden(db_path: Any, profile_id: str, fact_ids: list[str], *, hidden: bool) -> dict:
    """Record the user's decision on each fact of ``profile_id``.

    Returns ``{"changed": [...], "unchanged": [...], "not_found": [...]}``.
    Raises ``RuntimeError`` when the store predates the flag (it is added by
    the schema step every SLM start runs).
    """
    from superlocalmemory.storage.memory_write import memory_write

    ids = list(dict.fromkeys(str(f).strip() for f in fact_ids if str(f).strip()))
    value = 1 if hidden else 0
    out: dict[str, list[str]] = {"changed": [], "unchanged": [], "not_found": []}
    with memory_write(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(atomic_facts)")}
        if not {"non_answer", "non_answer_user"} <= columns:
            raise RuntimeError("this store has not been upgraded yet: run `slm status` once, then retry")
        for fact_id in ids:
            row = conn.execute(
                "SELECT non_answer, non_answer_user FROM atomic_facts "
                "WHERE fact_id = ? AND profile_id = ?", (fact_id, profile_id)).fetchone()
            if row is None:
                out["not_found"].append(fact_id)
                continue
            if row[0] == value and row[1] == value:
                out["unchanged"].append(fact_id)
                continue
            conn.execute(
                "UPDATE atomic_facts SET non_answer = ?, non_answer_user = ? "
                "WHERE fact_id = ? AND profile_id = ?", (value, value, fact_id, profile_id))
            out["changed"].append(fact_id)
    return out


def cmd_hide(args: Namespace) -> int:
    """Run ``slm hide``. Returns a process exit code."""
    from superlocalmemory.core.config import SLMConfig

    use_json = bool(getattr(args, "json", False))
    hidden = not bool(getattr(args, "undo", False))
    ids = list(getattr(args, "fact_ids", []) or [])
    if len(ids) > _MAX_IDS:
        return _fail(f"at most {_MAX_IDS} ids per call", use_json)
    config = SLMConfig.load()
    db_path = config.base_dir / "memory.db"
    if not db_path.exists():
        return _fail("no memory store yet: nothing to hide", use_json)
    try:
        result = set_hidden(db_path, config.active_profile, ids, hidden=hidden)
    except RuntimeError as exc:
        return _fail(str(exc), use_json)
    if use_json:
        from superlocalmemory.cli.json_output import json_print

        json_print("hide", data={"hidden": hidden, "profile": config.active_profile, **result})
    else:
        _print_human(result, hidden)
    return 1 if result["not_found"] and not (result["changed"] or result["unchanged"]) else 0


def _print_human(result: dict, hidden: bool) -> None:
    verb = "Hidden from answers and session starts" if hidden else "Shown in answers again"
    for fact_id in result["changed"]:
        print(f"{verb}: {fact_id}")
    for fact_id in result["unchanged"]:
        print(f"Already {'hidden' if hidden else 'shown'}: {fact_id}")
    for fact_id in result["not_found"]:
        print(f"Not found in this profile: {fact_id}")
    if result["changed"]:
        print("Nothing was deleted or edited. "
              + ("Undo with: slm hide --undo <fact_id>" if hidden else
                 "SLM will not hide these automatically again."))


def _fail(message: str, use_json: bool) -> int:
    if use_json:
        from superlocalmemory.cli.json_output import json_print

        json_print("hide", error={"code": "HIDE_FAILED", "message": message})
    else:
        print(f"slm hide: {message}", file=sys.stderr)
    return 1
