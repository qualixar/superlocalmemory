# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Which Claude Code hook commands are SLM's own (4.2.30, F-04).

A hook command is SLM's only when it **equals** a command SLM has written in some release
(after whitespace and temp-path normalisation). Before 4.2.30 any command that merely contained
``slm hook`` counted: a person's edited line (an extra variable, a script after it) or a
command like ``echo rslm hook finished`` was then replaced or deleted by the upgrade.

Three normalisations only, because they are the parts that differ between two correct copies of
the same SLM command: runs of whitespace, the temporary folder in front of SLM's two marker file
names (it differs by computer and user), and the 16-hex data-root suffix on those names.
Everything else is compared as written. A different timeout, an extra key or any other text is
the person's edit and is left alone.
"""

from __future__ import annotations

import re
from typing import Any

_NAME = r"slm-session-(?P<{g}>initialized|start-time)(?:-[0-9a-f]{{16}})?"
_ROOT = r"(?:[A-Za-z]:)?[/\\]"  # a path that starts at a drive or a root, never a word
#: A marker path as written by SLM: ``<tmp>/slm-session-initialized[-<16 hex>]`` (or the
#: ``slm-session-start-time`` twin), double-quoted, single-quoted or bare. The folder is dropped.
_MARKER = re.compile(
    '"' + _ROOT + r'[^"]*[/\\]' + _NAME.format(g="a") + '"'
    "|'" + _ROOT + r"[^']*[/\\]" + _NAME.format(g="b") + "'"
    "|" + _ROOT + r"[^\s'\";&|<>()]*[/\\]" + _NAME.format(g="c"))

#: Version 3.0's script hooks, written with the person's home folder in front.
_SCRIPT = re.compile(
    r"[^\s'\"]*[/\\]\.superlocalmemory[/\\]hooks[/\\]slm-(?:session-start|auto-capture)\.sh")


def _marker(match: re.Match[str]) -> str:
    return f"@{match.group('a') or match.group('b') or match.group('c')}@"


def normalise(command: Any) -> str:
    """The comparison form of a command: one-space whitespace and generic marker paths."""
    text = " ".join(str(command).split())
    if _SCRIPT.fullmatch(text):
        return "@legacy-script@"
    return _MARKER.sub(_marker, text)


#: Every command SLM has written into Claude Code's ``settings.json`` (normalised), with the
#: timeouts it was written with. Before 4.1.22 timeouts were written in milliseconds, so those
#: values are listed too. Built from the history of ``hooks/claude_code_hooks.py`` (the v3.0
#: script hooks up to 4.2.30); ``tests/test_hooks/test_hook_identity.py`` keeps the table in step
#: with what the current code writes. An empty tuple means any timeout (the v3.0 scripts).
# fmt: off
KNOWN: dict[str, tuple[int, ...]] = {
    'slm hook mandate 2>/dev/null || true': (5, 5000),
    'slm hook start 2>/dev/null || true; touch @start-time@': (15, 15000),
    'slm hook checkpoint 2>/dev/null || true': (5, 5000),
    'slm hook post_tool_outcome 2>/dev/null || true': (5, 5000),
    'slm hook cue 2>/dev/null || true': (5,),
    'slm hook user_prompt_rehash 2>/dev/null || true': (5, 5000),
    'slm hook topic_shift 2>/dev/null || true': (3, 3000),
    'slm hook auto_recall 2>/dev/null || true': (7,),
    'slm hook stop 2>/dev/null || true; rm -f @initialized@ @start-time@': (10, 10000),
    'slm hook stop_outcome 2>/dev/null || true': (10, 10000),
    'slm session close 2>/dev/null || true': (15, 15000),
    'slm hook before_web 2>/dev/null || true': (5, 5000),
    'touch @initialized@': (2, 500),
    ("test -f @initialized@ || test ! -f @start-time@ || "
     "{ echo '[SLM] Call mcp__superlocalmemory__session_init first'; exit 2; }"): (2, 500),
    'cmd /c "slm hook mandate 2>NUL || exit /b 0"': (5, 5000),
    'cmd /c "slm hook start 2>NUL & echo.>@start-time@ & exit /b 0"': (15, 15000),
    'cmd /c "slm hook checkpoint 2>NUL || exit /b 0"': (5, 5000),
    'cmd /c "slm hook post_tool_outcome 2>NUL || exit /b 0"': (5, 5000),
    'cmd /c "slm hook cue 2>NUL || exit /b 0"': (5,),
    'cmd /c "slm hook user_prompt_rehash 2>NUL || exit /b 0"': (5, 5000),
    'cmd /c "slm hook topic_shift 2>NUL || exit /b 0"': (3, 3000),
    'cmd /c "slm hook auto_recall 2>NUL || exit /b 0"': (7,),
    'cmd /c "slm hook stop 2>NUL & del /q @initialized@ @start-time@ 2>NUL & exit /b 0"': (10, 10000),
    'cmd /c "slm hook stop_outcome 2>NUL || exit /b 0"': (10, 10000),
    'cmd /c "slm session close 2>NUL || exit /b 0"': (15, 15000),
    'cmd /c "slm hook before_web 2>NUL || exit /b 0"': (5, 5000),
    'cmd /c "echo.>@initialized@"': (2, 500),
    ('cmd /c "if exist @initialized@ (exit /b 0) else if not exist @start-time@ (exit /b 0) '
     'else (echo [SLM] Call mcp__superlocalmemory__session_init first & exit /b 2)"'): (2, 500),
    'slm hook start 2>/dev/null || true': (15000,),
    'slm hook stop 2>/dev/null || true': (10000,),
    'cmd /c "slm hook start 2>NUL || exit /b 0"': (15000,),
    'cmd /c "slm hook stop 2>NUL || exit /b 0"': (10000,),
    '@legacy-script@': (),
}
# fmt: on


def is_known_command(command: Any) -> bool:
    """True when command equals a command some SLM release wrote (timeouts not looked at)."""
    return normalise(command) in KNOWN


def is_slm_hook(hook: Any) -> bool:
    """True when this hook object is exactly what an SLM release wrote.

    The command must be a known one, the object must hold nothing but type, command and
    timeout, and the timeout must be one SLM wrote for that command. Anything else (an extra
    variable, an extra key, a raised or lowered timeout) is the person's edit and is not owned.
    """
    if not isinstance(hook, dict) or set(hook) - {"type", "command", "timeout"}:
        return False
    if hook.get("type", "command") != "command":
        return False
    allowed = KNOWN.get(normalise(hook.get("command", "")))
    if allowed is None:
        return False
    if "timeout" not in hook or not allowed:
        return True  # no timeout written (an old or hand-trimmed copy): the host default applies
    return type(hook["timeout"]) is int and hook["timeout"] in allowed


_KEY = re.compile(r"(?<![\w-])slm\s+(?:hook\s+(\w+)|session\s+close)(?!\w)")


def subcommand(command: Any) -> str | None:
    """hook start / session close of an slm command, or None."""
    match = _KEY.search(str(command))
    if match is None:
        return None
    return f"hook {match.group(1)}" if match.group(1) else "session close"


def covered_by_edit(fresh_command: Any, person_commands: list[str]) -> bool:
    """True when a command the person owns already runs the same slm subcommand.

    SLM does not add its own copy then: the edited line is the person's version of that hook,
    and two of them would fire it twice.
    """
    wanted = subcommand(fresh_command)
    return wanted is not None and any(subcommand(c) == wanted for c in person_commands)
