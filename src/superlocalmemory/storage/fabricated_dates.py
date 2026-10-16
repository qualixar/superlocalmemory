# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Recognise the date that builds before 4.1.6 made up, and the clause carrying it.

WHERE THE DATE CAME FROM
The fact-extraction prompt explained relative dates with a worked example,
"'Yesterday' with session date 2024-01-15 becomes '2024-01-14'". A local model
that did not know the date echoed that literal back as the fact's date: "The
file X was modified on 2024-01-15", "2024-01-15 File engine.py was modified".
4.1.6 (24 Aug 2026, commit 9d30aa4c) removed the example. 2026-01-15 appears the
same way (the year advanced, the day did not).

WHAT THIS MODULE DOES
Pure text functions, no database: does a text or a stored date carry the made-up
day, and what does the sentence read like with that clause taken out. A clause
goes only where the sentence still reads without it ("modified on D" ->
"modified"; "D File x was changed" -> "File x was changed"). Where the date is
the subject or the point ("D was a Tuesday", "the current date is D") nothing
can go cleanly, and :func:`remove_date_clauses` returns ``None``.
"""

from __future__ import annotations

import re

#: The made-up days, as stored in date columns.
FABRICATED_DAYS: tuple[str, ...] = ("2024-01-15", "2026-01-15")

_YEARS = r"(?:2024|2026)"
_MONTH = r"(?:January|Jan\.?)"
_CLOCK = r"\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?"
_ISO = (rf"{_YEARS}-01-15(?:[T ]{_CLOCK}(?:Z|[+-]\d{{2}}:?\d{{2}})?)?")
_SPELLED = (rf"(?:{_MONTH}\s+15(?:th)?,?\s+{_YEARS}|15\s+{_MONTH}\s+{_YEARS})"
            rf"(?:,?\s+at\s+{_CLOCK}(?:\s*[AaPp](?:\.[Mm]\.|[Mm]))?)?")
#: One fabricated date with its time of day, if any, never part of a longer token
#: (a clock right after the day belongs to it, so it is never left behind).
DATE = rf"(?<![\w-])(?:{_ISO}|{_SPELLED})(?![\w-]|:\d|[T ]\d{{1,2}}:\d)"
_DATE_RE = re.compile(DATE)
#: A date that opens a range ("from D to E") is not a clause on its own.
_NOT_A_RANGE = r"(?!\s*(?:to|until|through|and|-|–|—)\s*\d)"
_PREPOSITION = r"(?:on|as of|since|at|around|by|dated)"
_DATE_SUBJECT_VERB = (r"(?:is|was|are|were|has|had|will|would|can|could|should|"
                      r"marks?|became|becomes|remains?)\b")

_SUBJECT_FRAME = re.compile(
    rf"^\s*{DATE}\s+(?:is|was)\s+(?:the|a)\s+(?:date|day|time)\s+"
    rf"(?:when|on which|in which|at which)\s+(?P<rest>\S.*)$", re.DOTALL)
_LEADING = re.compile(
    rf"^\s*{DATE}\s*[,:;]?\s*(?:[-–—]\s*)?(?P<rest>(?!{_DATE_SUBJECT_VERB})"
    rf"[^\s,.;:].*)$", re.DOTALL)
_SENTENCE_OPENER = re.compile(
    rf"(?P<start>^|(?<=[.!?])\s+)(?:On|As of|Since|By|Around|Dated)\s+{DATE}"
    rf"{_NOT_A_RANGE}\s*,?\s+(?P<next>[^\s,])")
_INLINE = re.compile(
    rf",?\s+{_PREPOSITION}\s+{DATE}{_NOT_A_RANGE}", re.IGNORECASE)
_BRACKETED = re.compile(rf"\s*[(\[]\s*{DATE}\s*[)\]]")

#: The day as a person might write it, with or without a year, so a source that
#: names it in any form proves the date real. Broader than :data:`DATE` on
#: purpose: a false "the source names it" only keeps a fact as it is.
_JAN = r"jan(?:uary|\.)?"
_SOURCE_DAY = re.compile(
    r"(?<![\w.-])(?:"
    r"(?:2024|2026)-01-15"
    rf"|{_JAN}\s+15(?:st|nd|rd|th)?(?![\d])"
    rf"|15(?:st|nd|rd|th)?\s+(?:of\s+)?{_JAN}(?![a-z])"
    r"|0?1/15/(?:20)?(?:24|26)"
    r"|15[/.]0?1[/.](?:20)?(?:24|26)"
    r")(?![\w-]|\.\d)", re.IGNORECASE)

#: The first word is capitalised only if it is a plain short word, never an
#: identifier, path or user name ("file" -> "File"; "varunpratapbhardwaj" stays).
_PLAIN_WORD = re.compile(r"^[a-z]{1,12}\b(?![./_@-])")
#: Rewrites applied before giving up; each removes one clause.
_MAX_PASSES = 4


def contains_fabricated_date(text: str | None) -> bool:
    """Whether ``text`` mentions one of the made-up days, in any written form."""
    return bool(text) and _DATE_RE.search(str(text)) is not None


def source_names_the_day(text: str | None) -> bool:
    """Whether a source text names 15 January (any written form, any or no year)."""
    return bool(text) and _SOURCE_DAY.search(str(text)) is not None


def is_fabricated_date_value(value: object) -> bool:
    """Whether a stored date column holds one of the made-up days."""
    return isinstance(value, str) and value.strip()[:10] in FABRICATED_DAYS


def _capitalise(text: str) -> str:
    return text[0].upper() + text[1:] if _PLAIN_WORD.match(text) else text


def _tidy(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r",\s*([.;:!?])", r"\1", text)
    return text.strip()


def _one_pass(text: str) -> str:
    """Remove the first clause a rule recognises; return ``text`` when none does."""
    framed = _SUBJECT_FRAME.match(text)
    if framed:
        return _capitalise(framed.group("rest").strip())
    leading = _LEADING.match(text)
    if leading:
        return _capitalise(leading.group("rest").strip())
    opener = _SENTENCE_OPENER.search(text)
    if opener:
        joiner = "" if not opener.group("start") else " "
        return (text[:opener.start()] + joiner + opener.group("next").upper()
                + text[opener.end():])
    for rule in (_BRACKETED, _INLINE):
        removed = rule.sub("", text, count=1)
        if removed != text:
            return removed
    return text


def _reads(text: str) -> bool:
    """At least two words with a letter: a sentence, not a stub."""
    return len(re.findall(r"[A-Za-z]\S*", text)) >= 2


def remove_date_clauses(text: str | None) -> str | None:
    """``text`` without the clauses that carry a made-up date, or ``None``.

    ``None`` means nothing can be removed so that the sentence still reads: no
    made-up date, or one that is the subject or the point of the statement.
    """
    if not contains_fabricated_date(text):
        return None
    current = str(text)
    for _ in range(_MAX_PASSES):
        following = _one_pass(current)
        if following == current:
            break
        current = following
    current = _tidy(current)
    if current == _tidy(str(text)) or not _reads(current):
        return None
    return current


__all__ = ["DATE", "FABRICATED_DAYS", "contains_fabricated_date",
           "is_fabricated_date_value", "remove_date_clauses", "source_names_the_day"]
