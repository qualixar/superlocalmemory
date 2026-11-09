# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

""""from X to Y" intervals, found in bounded time with the same result.

Two 4.2.13 patterns share one shape::

    OPENER \\s+ (.+?) \\s+ JOINER \\s+ (.+?) (?:[.,;]|$)

``fact_extractor._INTERVAL_RE`` (from/between ... to/and/until/through, on
every saved sentence) and ``temporal_parser._DURATION`` (from ... to, on a
fact's content and on a recall question). Each ``\\s+`` can give back any
part of a whitespace run and each lazy group can grow across one, so on a
miss the engine tries cubically many splits: "from" + 2000 spaces took 29.5 s.
A long run is not needed: ``("from" + 10 spaces) * 350`` took 0.8 s, so a
guard on run length alone would not bound the time.

The groups become stored interval dates, so the fix must return exactly the
old match: the FIRST one in the engine's backtracking order (leftmost start,
then the longest ``\\s+``, then the shortest group), not merely a match. Text
of at most ``SWITCH_CHARS`` characters still goes through the old pattern:
same code, same speed (worst crafted case measured at 64 characters: about
1 ms). Longer text goes to :func:`scan`, which runs the same search but solves
each (state, position) once, right to left, so it is linear in the text.
Character tests (keywords, ``\\b``, case folding, Unicode ``\\s``) are done by
regexes compiled with the old pattern's own flags, so they cannot drift.
Proven identical on 23,542,506 exhaustively generated strings (characters to
length 8, keyword tokens to length 6) and by the random differential tests in
tests/test_encoding/test_interval_scan.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Longest text the old pattern still searches (worst crafted case ~1 ms here).
SWITCH_CHARS = 64

_WS_RUN = re.compile(r"\s+")
_TERMINATORS = ".,;"
_NL = "\n"

Span = tuple[int, int]
#: (match span, group 1 span, group 2 span), as ``Match.span()`` gives them.
Spans = tuple[Span, Span, Span]
#: (group 2 start, group 2 end, match end); with group 1's end, then also its start.
_Tail = tuple[int, int, int]
_Grown = tuple[int, int, int, int]
_Head = tuple[int, int, int, int, int]


@dataclass(frozen=True)
class IntervalPattern:
    """An old interval pattern plus its opener and joiner, for :func:`scan`."""

    regex: re.Pattern[str]
    opener: re.Pattern[str]
    joiner: re.Pattern[str]

    def spans(self, text: str) -> Spans | None:
        """Exactly ``regex.search(text)``'s spans (or None), in bounded time."""
        if len(text) <= SWITCH_CHARS:
            found = self.regex.search(text)
            return None if found is None else (found.span(), found.span(1), found.span(2))
        return scan(self.opener, self.joiner, text)

    def groups(self, text: str) -> tuple[str, str] | None:
        """Exactly ``regex.search(text).groups()`` (or None), in bounded time."""
        found = self.spans(text)
        if found is None:
            return None
        (_, (a, b), (c, d)) = found
        return text[a:b], text[c:d]


def interval_pattern(regex: re.Pattern[str], opener: str, joiner: str) -> IntervalPattern:
    """Build the pair for ``regex``; raises if the pieces are not its shape."""
    expected = (opener + r"\s+(.+?)\s+" + joiner + r"\s+(.+?)")
    if regex.pattern not in (expected + r"(?:[.,;]|$)", expected + r"(?:\.|,|;|$)"):
        raise ValueError(f"not an interval pattern built from {opener!r} and {joiner!r}")
    return IntervalPattern(regex, re.compile(opener, regex.flags), re.compile(joiner, regex.flags))


def scan(opener: re.Pattern[str], joiner: re.Pattern[str], text: str) -> Spans | None:
    """The old pattern's first match, found in linear time."""
    first = opener.search(text)
    if first is None:
        return None
    # Every match needs a joiner after its opener; with none after the first
    # opener there is none after any later one either, so no match (exact).
    if joiner.search(text, first.end()) is None:
        return None
    lo, n = first.start(), len(text)
    ws = bytearray(n + 2)
    for run in _WS_RUN.finditer(text, lo):
        ws[run.start():run.end()] = b"\x01" * (run.end() - run.start())
    tail = _tail_table(text, ws, lo)
    head = _head_table(text, ws, _middle_table(joiner, text, ws, tail, lo), lo)
    for found in opener.finditer(text, lo):
        k = found.end()
        best = head[k + 1] if ws[k] else None
        if best is not None:
            p1, e1, p3, e2, end = best
            return (found.start(), end), (p1, e1), (p3, e2)
    return None


def _terminator(text: str, i: int, n: int) -> Span | None:
    """``(?:[.,;]|$)`` at ``i``: (group 2 end, match end), or None."""
    if i == n:
        return i, i
    if text[i] in _TERMINATORS:
        return i, i + 1
    if i == n - 1 and text[i] == _NL:
        return i, i
    return None


def _tail_table(text: str, ws: bytearray, lo: int) -> list[_Tail | None]:
    """``tail[p]``: first success of ``(.+?)TERM`` over the ends of an ``\\s+``
    that may stop anywhere from ``p`` to its run's end, longest first."""
    n = len(text)
    ends: list[Span | None] = [None] * (n + 2)
    tail: list[_Tail | None] = [None] * (n + 2)
    for i in range(n, lo - 1, -1):
        stop = _terminator(text, i, n)
        ends[i] = stop if stop is not None or text[i] == _NL else ends[i + 1]
        later, here = ends[i + 1], None
        if i < n and text[i] != _NL and later is not None:
            here = (i, later[0], later[1])
        tail[i] = tail[i + 1] if ws[i] and tail[i + 1] is not None else here
    return tail


def _middle_table(joiner: re.Pattern[str], text: str, ws: bytearray,
                  tail: list[_Tail | None], lo: int) -> list[_Tail | None]:
    """``middle[p]``: first success of ``JOINER\\s+(.+?)TERM`` over the ends of
    an ``\\s+`` that may stop anywhere from ``p`` to its run's end."""
    n = len(text)
    middle: list[_Tail | None] = [None] * (n + 2)
    for p in range(n, lo - 1, -1):
        here: _Tail | None = None
        if not ws[p]:
            word = joiner.match(text, p)
            if word is not None and ws[word.end()]:
                here = tail[word.end() + 1]
        middle[p] = middle[p + 1] if ws[p] and middle[p + 1] is not None else here
    return middle


def _head_table(text: str, ws: bytearray, middle: list[_Tail | None],
                lo: int) -> list[_Head | None]:
    """``head[p]``: first success of ``(.+?)\\s+JOINER\\s+(.+?)TERM`` over the
    ends of the opener's ``\\s+`` that may stop from ``p`` to its run's end."""
    n = len(text)
    grown: list[_Grown | None] = [None] * (n + 2)  # group 1 ends at the first e >= p
    head: list[_Head | None] = [None] * (n + 2)
    for p in range(n, lo - 1, -1):
        after = middle[p + 1] if ws[p] else None
        if after is not None:
            grown[p] = (p, after[0], after[1], after[2])
        elif p < n and text[p] != _NL:
            grown[p] = grown[p + 1]
        nxt, here = grown[p + 1], None
        if p < n and text[p] != _NL and nxt is not None:
            here = (p, nxt[0], nxt[1], nxt[2], nxt[3])
        head[p] = head[p + 1] if ws[p] and head[p + 1] is not None else here
    return head


__all__ = ["SWITCH_CHARS", "IntervalPattern", "Spans", "interval_pattern", "scan"]
