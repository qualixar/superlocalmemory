# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm tools``: show or choose how many MCP tools every connected app sees."""

from __future__ import annotations

import json
import os
import sys
from argparse import Namespace

from superlocalmemory.mcp.profile_names import (
    UnknownProfileError,
    save_tool_set,
    saved_tool_set,
)
from superlocalmemory.mcp.profiles import _PROFILE_DEFINITIONS, PROFILE_DESCRIPTIONS

_DEFAULT_SET = "full"


def _counts() -> dict[str, int]:
    return {name: len(tools) for name, tools in _PROFILE_DEFINITIONS.items()}


def _current() -> tuple[str, str]:
    explicit = (os.environ.get("SLM_MCP_PROFILE") or "").strip().lower()
    if explicit:
        return explicit, "SLM_MCP_PROFILE"
    saved = saved_tool_set()
    return (saved, "saved") if saved else (_DEFAULT_SET, "default")


def cmd_tools(args: Namespace) -> None:
    if getattr(args, "action", None) == "set":
        try:
            chosen = save_tool_set(args.name or "")
        except UnknownProfileError as exc:
            print(f"slm tools: {exc} Use `default` to go back to {_DEFAULT_SET}.", file=sys.stderr)
            sys.exit(2)
        shown = chosen or f"{_DEFAULT_SET} (default)"
        print(f"Tool set saved: {shown}.")
        print("Restart your apps (or reconnect them in Dashboard → Connected apps) to see it.")
        print("An app whose own config sets SLM_MCP_PROFILE keeps that setting.")
        return
    name, source = _current()
    counts = _counts()
    if getattr(args, "json", False):
        print(json.dumps({"tool_set": name, "source": source, "counts": counts,
                          "descriptions": PROFILE_DESCRIPTIONS}))
        return
    print(f"Tool set: {name} ({source})")
    for profile, count in counts.items():
        mark = "*" if profile == name else " "
        print(f" {mark} {profile:<6} {count:>3} tools  {PROFILE_DESCRIPTIONS.get(profile, '')}")
    mark = "*" if name == "whole" else " "
    print(f" {mark} whole  every tool")
    print("Change it: slm tools set <name>   (slm tools set default goes back to full)")
