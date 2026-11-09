# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Lower-case technical words ("llama3.2", "snake_case"), found in bounded time (#310).

The pattern below is the one 4.1.25-4.2.13 used. Its loop
``(?:[_:.][a-z0-9]+|\\d[a-z0-9.]*)+`` can split a run of digits between loop
turns in exponentially many ways, and when no ending passes the lookahead
Python's engine tries every one: ``"a" + "1" * 30 + "-"`` ran for minutes, on a
save and on a recall question alike.

Its output is stored (entities, graph edges, keyword alt-keys), so the fix must
return exactly the same words, not merely similar ones. Ordinary text still
goes through the same pattern: same code, same speed. A text with a run of more
than ``_SWITCH_DIGITS`` digits inside one word goes to :func:`_scan`, which is
the same leftmost-first search the engine runs (same order of preference, so
the same match: the engine returns the FIRST ending it finds, not the longest),
but solves each (state, position) once, right to left. Proven identical on all
48,427,560 strings up to length 8 over ``a1._:-/ A`` and on every text of a
57,553-text real store; with at most 10 digits per run the old engine stays
under 5 ms even on crafted input.
"""

from __future__ import annotations

import re

LOWER_TECH = re.compile(r"(?<![\w.:/~\\-])([a-z][a-z0-9]*(?:[_:.][a-z0-9]+|\d[a-z0-9.]*)+"
                        r"|[a-z]{4,})(?![\w/\\-])")

#: Characters a match can span. One character class, so this scan is linear.
_RUN = re.compile(r"[a-z\d_:.]+")
#: Without digits the pattern cannot branch; with up to ten the engine is fast.
_SWITCH_DIGITS = 10
_WORD = re.compile(r"\w")
_FAIL = -1


def lower_tech_spans(text: str) -> list[tuple[int, int]]:
    """``(start, end)`` of every match, exactly as ``LOWER_TECH.finditer`` gives."""
    if _digit_heavy(text):
        return _scan(text)
    return [m.span(1) for m in LOWER_TECH.finditer(text)]


def _digit_heavy(text: str) -> bool:
    for m in _RUN.finditer(text):
        run = m.group()
        if len(run) > _SWITCH_DIGITS and sum(c.isdecimal() for c in run) > _SWITCH_DIGITS:
            return True
    return False


def _lower(c: str) -> bool:
    return "a" <= c <= "z"


def _alnum(c: str) -> bool:
    """``[a-z0-9]``: ASCII digits only, unlike ``\\d``."""
    return _lower(c) or "0" <= c <= "9"


def _behind_ok(text: str, i: int) -> bool:
    return i == 0 or not (_WORD.match(text[i - 1]) or text[i - 1] in ".:/~\\-")


def _ahead_ok(text: str, j: int) -> bool:
    return j >= len(text) or not (_WORD.match(text[j]) or text[j] in "/\\-")


def _scan(text: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    i, n = 0, len(text)
    while i < n:
        if _lower(text[i]) and _behind_ok(text, i):
            end = _technical_end(text, i)
            if end == _FAIL:
                end = _letters_end(text, i)
            if end != _FAIL:
                out.append((i, end))
                i = end
                continue
        i += 1
    return out


def _letters_end(text: str, start: int) -> int:
    """The ``[a-z]{4,}`` branch: greedy, giving back one letter at a time."""
    k = start
    while k < len(text) and _lower(text[k]):
        k += 1
    while k - start >= 4 and not _ahead_ok(text, k):
        k -= 1
    return k if k - start >= 4 else _FAIL


def _technical_end(text: str, start: int) -> int:
    """Where the engine's first match of the technical branch ends, or ``_FAIL``.

    States after the first letter, each the engine's own preference order:
    ``prefix`` = ``[a-z0-9]*`` (take one more, else start the loop),
    ``turn`` = one loop turn (``[_:.]`` then ``alnum_one``, or a digit then
    ``dotted``), ``alnum_one``/``alnum_more`` = ``[a-z0-9]+``, ``dotted`` =
    ``[a-z0-9.]*``, ``after`` = after a turn (another turn, else stop and
    test the lookahead). Each value is the end of the first success from there.
    """
    n, last = len(text), start + 1
    while last < n and (_alnum(text[last]) or text[last].isdecimal() or text[last] in "_:."):
        last += 1
    size = last - start + 2
    prefix, turn, alnum_one, alnum_more, dotted, after = ([_FAIL] * size for _ in range(6))
    for p in range(last, start, -1):
        k, c = p - start, text[p] if p < n else ""
        stop = p if _ahead_ok(text, p) else _FAIL
        alnum_one[k] = alnum_more[k + 1] if c and _alnum(c) else _FAIL
        turn[k] = (alnum_one[k + 1] if c and c in "_:." else
                   dotted[k + 1] if c and c.isdecimal() else _FAIL)
        after[k] = turn[k] if turn[k] != _FAIL else stop
        alnum_more[k] = _first(alnum_more[k + 1] if c and _alnum(c) else _FAIL, after[k])
        dotted[k] = _first(dotted[k + 1] if c and (_alnum(c) or c == ".") else _FAIL, after[k])
        prefix[k] = _first(prefix[k + 1] if c and _alnum(c) else _FAIL, turn[k])
    return prefix[1]


def _first(preferred: int, otherwise: int) -> int:
    return preferred if preferred != _FAIL else otherwise


__all__ = ["LOWER_TECH", "lower_tech_spans"]
