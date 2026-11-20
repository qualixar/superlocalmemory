# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Read the two text formats launchd prints for one loaded job.

``launchctl list <label>`` prints an OpenStep-like dictionary that is *not*
valid OpenStep: array elements and the closing brace end in ``;``, so
``plutil`` and ``plistlib`` both reject it. It also never includes the job's
``EnvironmentVariables``. ``launchctl print <domain>/<label>`` is the only
place launchd reports the environment it will start the job with.

Both parsers are strict about structure and raise ``ValueError`` on anything
they do not understand, so callers fail closed.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(
    r'\s*(?:(?P<string>"(?:[^"\\]|\\.)*")|(?P<punct>[{}()=;,])|(?P<bare>[^\s{}()=;,"]+))'
)
_ESCAPES = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}
_INTEGER = re.compile(r"-?[0-9]+")


def _tokens(text: str) -> list[tuple[str, str]]:
    found, position, text = [], 0, text.rstrip()
    while position < len(text):
        match = _TOKEN.match(text, position)
        if match is None or match.end() == position:
            raise ValueError(f"unreadable launchctl output at offset {position}")
        position = match.end()
        kind = match.lastgroup
        found.append((kind, match[kind]))
    return found


def _unquote(raw: str) -> str:
    return re.sub(r"\\(.)", lambda m: _ESCAPES.get(m[1], m[1]), raw[1:-1])


def _scalar(kind: str, raw: str):
    if kind == "string":
        return _unquote(raw)
    if raw in {"true", "false"}:
        return raw == "true"
    return int(raw) if _INTEGER.fullmatch(raw) else raw


class _Parser:
    def __init__(self, text: str):
        self.items, self.at = _tokens(text), 0

    def _next(self) -> tuple[str, str]:
        if self.at >= len(self.items):
            raise ValueError("launchctl output ended early")
        self.at += 1
        return self.items[self.at - 1]

    def _peek(self) -> str | None:
        return self.items[self.at][1] if self.at < len(self.items) else None

    def value(self):
        kind, raw = self._next()
        if raw == "{" and kind == "punct":
            return self._dict()
        if raw == "(" and kind == "punct":
            return self._array()
        if kind == "punct":
            raise ValueError(f"unexpected {raw!r} in launchctl output")
        return _scalar(kind, raw)

    def _dict(self) -> dict:
        result = {}
        while self._peek() != "}":
            kind, key = self._next()
            if kind == "punct" or self._next()[1] != "=":
                raise ValueError("malformed launchctl dictionary")
            result[_scalar(kind, key) if kind == "string" else key] = self.value()
            if self._next()[1] != ";":
                raise ValueError("malformed launchctl dictionary entry")
        self._next()
        return result

    def _array(self) -> list:
        result = []
        while self._peek() != ")":
            result.append(self.value())
            if self._peek() in {";", ","}:
                self._next()
        self._next()
        return result


def parse_list_output(text: str) -> dict:
    """The dictionary ``launchctl list <label>`` printed."""
    parser = _Parser(text)
    record = parser.value()
    if parser._peek() == ";":
        parser._next()
    if not isinstance(record, dict) or parser.at != len(parser.items):
        raise ValueError("launchctl list did not print one dictionary")
    return record


_ENVIRONMENT_START = re.compile(r"\tenvironment = \{")
_ENVIRONMENT_ENTRY = re.compile(r"\t\t([^\s=]+) => (.*)")


def print_environment(text: str) -> dict[str, str]:
    """The ``environment`` block of ``launchctl print <domain>/<label>``.

    Only the job's own block (one tab deep), never ``inherited environment``
    or ``default environment``.
    """
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if _ENVIRONMENT_START.fullmatch(line)]
    if len(starts) != 1:
        raise ValueError("launchctl print has no single environment block")
    environment: dict[str, str] = {}
    for line in lines[starts[0] + 1:]:
        if line == "\t}":
            return environment
        entry = _ENVIRONMENT_ENTRY.fullmatch(line)
        if entry is None:
            raise ValueError("malformed launchctl print environment entry")
        environment[entry[1]] = entry[2]
    raise ValueError("launchctl print environment block is not closed")
