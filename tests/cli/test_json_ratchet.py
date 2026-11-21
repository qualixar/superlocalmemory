# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Every ``slm`` command answers in JSON: the ratchet.

The parser is the real one (``build_parser``), walked to every leaf command. A
leaf that does not take ``--json`` fails here by name. There is no exempt
list: a new command is not finished until an agent can drive it from a shell.
"""

from __future__ import annotations

import argparse

from superlocalmemory.cli.main import build_parser


def _subparsers(parser: argparse.ArgumentParser):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    return None


def leaf_commands(parser: argparse.ArgumentParser, path=()):
    """Every leaf command path with its parser (aliases counted once)."""
    sub = _subparsers(parser)
    if sub is None:
        yield path, parser
        return
    seen: set[int] = set()
    for name, child in sub.choices.items():
        if id(child) in seen:
            continue
        seen.add(id(child))
        yield from leaf_commands(child, (*path, name))


def _json_action(parser: argparse.ArgumentParser):
    return next((a for a in parser._actions if "--json" in a.option_strings), None)


def test_the_walk_finds_the_whole_command_tree():
    leaves = [" ".join(p) for p, _ in leaf_commands(build_parser("test"))]
    # A floor, not the exact count: the tree only grows.
    assert len(leaves) >= 200
    for known in ("recall", "db migrate", "remote keys add", "team status", "help"):
        assert known in leaves


def test_every_leaf_command_accepts_json():
    missing = [" ".join(path) for path, p in leaf_commands(build_parser("test"))
               if _json_action(p) is None]
    assert missing == [], f"{len(missing)} commands do not take --json: {missing}"


def test_every_json_flag_sets_args_json():
    """The global switch sets ``args.json``; a flag stored elsewhere would miss it."""
    wrong = [" ".join(path) for path, p in leaf_commands(build_parser("test"))
             if (a := _json_action(p)) is not None and a.dest != "json"]
    assert wrong == []


def test_every_json_flag_is_a_switch():
    """``--json`` takes no value on any command (``slm tool NAME --json '{...}'`` passes
    tool arguments after the tool name, which the leaf parser never sees)."""
    valued = [" ".join(path) for path, p in leaf_commands(build_parser("test"))
              if (a := _json_action(p)) is not None and a.nargs != 0]
    assert valued == []
