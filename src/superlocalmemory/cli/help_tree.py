# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""``slm help --json``: the whole command tree in one document.

Built from the real parser, so it cannot drift from what ``slm`` accepts. Each
command carries its path, one-line description, and every option with its
flags, type, whether it is required, its default and its choices. An agent
reads this once and can drive every command.
"""

from __future__ import annotations

import argparse
from typing import Any

_HELP_ACTIONS = (argparse._HelpAction, argparse._VersionAction)


def _subparsers(parser: argparse.ArgumentParser) -> argparse._SubParsersAction | None:
    return next((a for a in parser._actions if isinstance(a, argparse._SubParsersAction)), None)


def _jsonable(value: Any) -> Any:
    if value is argparse.SUPPRESS:
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _kind(action: argparse.Action) -> str:
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        return "flag"
    if isinstance(action, argparse._CountAction):
        return "count"
    if isinstance(action, argparse._AppendAction):
        return "list"
    if action.nargs in ("*", "+", argparse.REMAINDER) or isinstance(action.nargs, int) and action.nargs > 1:
        return "list"
    if action.type is int:
        return "integer"
    if action.type is float:
        return "number"
    return "string"


def describe_option(action: argparse.Action) -> dict[str, Any]:
    """One option or positional argument, as an agent needs it."""
    positional = not action.option_strings
    flag = isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction))
    required = bool(action.required) if not positional else action.nargs not in ("?", "*", argparse.REMAINDER)
    out: dict[str, Any] = {
        "name": action.dest if positional else action.option_strings[-1],
        "flags": list(action.option_strings),
        "dest": action.dest,
        "positional": positional,
        "type": _kind(action),
        "required": required,
        "default": None if flag and action.default is False else _jsonable(action.default),
        "help": (action.help or "") if action.help is not argparse.SUPPRESS else "",
    }
    if action.choices is not None:
        out["choices"] = [_jsonable(c) for c in action.choices]
    if action.nargs is not None and not flag:
        out["nargs"] = _jsonable(action.nargs)
    if action.metavar is not None:
        out["metavar"] = _jsonable(action.metavar)
    return out


def _help_of(parent: argparse._SubParsersAction | None, name: str) -> str:
    if parent is None:
        return ""
    for choice in parent._choices_actions:
        if choice.dest == name:
            return choice.help or ""
    return ""


def _aliases(parent: argparse._SubParsersAction, child: argparse.ArgumentParser) -> list[str]:
    return [n for n, p in parent.choices.items() if p is child]


def _primary_names(sub: argparse._SubParsersAction) -> list[str]:
    seen: dict[int, str] = {}
    for name, child in sub.choices.items():
        seen.setdefault(id(child), name)
    return list(seen.values())


def command_tree(parser: argparse.ArgumentParser) -> list[dict[str, Any]]:
    """Every command (groups and leaves) in depth-first order."""
    rows: list[dict[str, Any]] = []
    _walk(parser, (), None, rows)
    return rows


def _walk(parser: argparse.ArgumentParser, path: tuple[str, ...],
          parent: argparse._SubParsersAction | None, rows: list[dict[str, Any]]) -> None:
    sub = _subparsers(parser)
    if path:
        names = _aliases(parent, parser) if parent is not None else [path[-1]]
        rows.append({
            "path": " ".join(path),
            "summary": _help_of(parent, path[-1]),
            "aliases": [n for n in names if n != path[-1]],
            "has_subcommands": sub is not None,
            "subcommands": _primary_names(sub) if sub is not None else [],
            "options": [describe_option(a) for a in parser._actions
                        if not isinstance(a, (*_HELP_ACTIONS, argparse._SubParsersAction))],
        })
    if sub is None:
        return
    seen: set[int] = set()
    for name, child in sub.choices.items():
        if id(child) in seen:
            continue
        seen.add(id(child))
        _walk(child, (*path, name), sub, rows)


def subtree(parser: argparse.ArgumentParser, words: list[str]) -> list[dict[str, Any]]:
    """The commands at and under ``words`` (``["db"]`` -> db and its subcommands)."""
    prefix = " ".join(words)
    rows = command_tree(parser)
    if not prefix:
        return rows
    return [r for r in rows if r["path"] == prefix or r["path"].startswith(prefix + " ")]


def resolve_words(parser: argparse.ArgumentParser, argv: list[str]) -> list[str]:
    """The command path an argv names (``slm db migrate --help`` -> ``["db", "migrate"]``)."""
    words: list[str] = []
    node = parser
    for token in argv:
        if token.startswith("-"):
            continue
        sub = _subparsers(node)
        if sub is None or token not in sub.choices:
            break
        words.append(_aliases(sub, sub.choices[token])[0])
        node = sub.choices[token]
    return words


def leaf_parser(parser: argparse.ArgumentParser, argv: list[str]) -> argparse.ArgumentParser:
    """The parser of the command an argv names (the deepest one it reaches)."""
    node = parser
    for word in resolve_words(parser, argv):
        sub = _subparsers(node)
        if sub is None:
            break
        node = sub.choices[word]
    return node


def global_options() -> list[dict[str, Any]]:
    """What works before any command."""
    return [
        {"name": "--json", "flags": ["--json"], "type": "flag", "required": False,
         "help": "JSON for whatever command follows (same as SLM_JSON=1, or --json on it)"},
        {"name": "--version", "flags": ["-v", "--version"], "type": "flag", "required": False,
         "help": "Print the version"},
    ]


__all__ = ["command_tree", "describe_option", "global_options", "leaf_parser", "resolve_words",
           "subtree"]
