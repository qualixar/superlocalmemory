# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Does a derived fact still say what its source memory said?

Facts written by an extractor (a local or cloud model, or the rule-based
rewriter) are checked against the memory they came from. The check is
mechanical and deliberately narrow, so a fact is only flagged for something a
reader would call a change of meaning:

* ``number_became_date``  — a date in the fact is a source number read as a date
  ("2004.6 ms" -> 2004-06);
* ``unsupported_date``     — a calendar date the source never states, does not
  reformat, and cannot have resolved from a relative time ("yesterday");
* ``unsupported_number``   — a measurement, version, amount or id the source
  does not contain;
* ``negation_lost``        — the source sentence says never/not/without, the
  fact does not;
* ``order_reversed``       — the source says A before B, the fact says B before A;
* ``status_lost``          — the source says something was *previously* true, the
  fact states it without that qualifier.

Relevance or a sufficiency score is not source support; this is. Pure
functions, no I/O, no model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from superlocalmemory.encoding.typed_values import (
    ISO_DATE_TOKEN_RE,
    bare_numbers,
    has_relative_time,
    typed_number_cores,
)

_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
_MONTH_WORD_RE = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b\.?(?:\s+(\d{1,2})"
    r"(?:st|nd|rd|th)?\b)?",
    re.IGNORECASE,
)
_DAY_MONTH_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b",
    re.IGNORECASE,
)
#: Dotted dates need a four-digit year, so a version such as 4.1.21 is not one.
_NUMERIC_DATE_RE = re.compile(
    r"(?<![\d.])(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}\.\d{1,2}\.\d{4})"
    r"(?![\d.]*\d)"
)
_NEGATION_RE = re.compile(
    r"\bno\b(?!\s*[,.!?])|"  # "no approval", not the discourse "No, we ..."
    r"\b(?:never|not|without|none|nobody|nothing|neither|nor|cannot|unless|"
    r"avoid|forbid(?:den)?|prohibit(?:ed)?|refuse[sd]?|disallow(?:ed)?)\b|\w+n['’]t\b",
    re.IGNORECASE,
)
_ORDER_RE = re.compile(r"\b(before|after|ahead of|prior to|followed by)\b", re.IGNORECASE)
_PAST_RE = re.compile(
    r"\b(?:previously|formerly|used to|no longer|earlier|originally|once was)\b", re.IGNORECASE,
)
_STATUS_MARKER_RE = re.compile(
    r"\b(?:previously|formerly|used to|no longer|earlier|originally|past|was|were|had|"
    r"old|former|replaced|until|before|deprecated|retired|dropped|switched|moved|changed)\b",
    re.IGNORECASE,
)
_CLAUSE_END_RE = re.compile(r"[,;:]|\b(?:but|now|currently|today|and now|however)\b", re.IGNORECASE)
_STOP = frozenset(
    "the a an and or of to in on at for with by from is are was were be been it this that "
    "these those as into over under than then so if its their our your his her they we you "
    "will would should can could may might must has have had do does did".split()
)


@dataclass(frozen=True)
class FidelityReport:
    """Result of checking one derived fact against its source memory."""

    ok: bool
    reasons: tuple[str, ...] = ()


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9][a-z0-9'’.+#-]*", text.lower()) if w not in _STOP]


def _stem(word: str) -> str:
    """Just enough stemming to match "uses" / "use" / "used" across a paraphrase."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    if len(word) > 3 and word.endswith("ed"):
        return word[:-1]
    return word


def _content_words(text: str) -> set[str]:
    return {_stem(w.strip(".'’")) for w in _words(text) if len(w) >= 2} - {""}


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _date_parser():
    """dateutil's parser, imported on first use.

    The CLI registers its ``db fidelity`` command (which imports this module)
    while building the parser for every command, so a module-level import made
    ``slm --help`` and ``slm --version`` need dateutil. The installer contract
    runs those from source before any dependency is installed.
    """
    from dateutil import parser

    return parser


def _source_dates(source: str) -> tuple[set[str], set[int], set[tuple[int, int]]]:
    """Dates the source states: full ISO forms, month numbers, and (month, day) pairs."""
    full: set[str] = set()
    months: set[int] = set()
    month_days: set[tuple[int, int]] = set()
    for match in _NUMERIC_DATE_RE.finditer(source):
        for dayfirst in (False, True):
            try:
                parsed = _date_parser().parse(match.group(1), dayfirst=dayfirst)
            except (ValueError, OverflowError):
                continue
            full.update({parsed.strftime("%Y-%m-%d"), parsed.strftime("%Y-%m")})
            months.add(parsed.month)
            month_days.add((parsed.month, parsed.day))
    for match in _MONTH_WORD_RE.finditer(source):
        month = _MONTHS.index(match.group(1).lower()[:3]) + 1
        months.add(month)
        if match.group(2):
            month_days.add((month, int(match.group(2))))
    for match in _DAY_MONTH_RE.finditer(source):
        month = _MONTHS.index(match.group(2).lower()[:3]) + 1
        months.add(month)
        month_days.add((month, int(match.group(1))))
    return full, months, month_days


def _number_components(source: str) -> list[set[int]]:
    """Each source number with a separator, as the set of integers it is made of."""
    groups: list[set[int]] = []
    for core in bare_numbers(source):
        parts = [p for p in re.split(r"[.,]", core) if p]
        if len(parts) >= 2:
            groups.append({int(p) for p in parts} | {int(p) % 100 for p in parts if len(p) == 4})
    return groups


#: Written dates a fact may carry: "June 1st, 2004", "1 June 2004", "June 2004".
_WRITTEN_DATE_RE = re.compile(
    r"\b(?:(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+"
    r"(?:\d{1,2}(?:st|nd|rd|th)?,?\s+)?\d{4}"
    r"|\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"[a-z]*\.?,?\s+\d{4})\b",
    re.IGNORECASE,
)


def _fact_dates(fact: str) -> list[tuple[str, int, int, int | None]]:
    """(as written, year, month, day) for every calendar date the fact states."""
    found: list[tuple[str, int, int, int | None]] = []
    for match in ISO_DATE_TOKEN_RE.finditer(fact):
        day = int(match.group(3)) if match.group(3) else None
        found.append((match.group(0), int(match.group(1)), int(match.group(2)), day))
    for match in _WRITTEN_DATE_RE.finditer(fact):
        text = match.group(0)
        has_day = bool(re.search(r"\d{1,2}(?:st|nd|rd|th)?\b(?!\d)", text[: -4]))
        try:
            parsed = _date_parser().parse(text, fuzzy=True)
        except (ValueError, OverflowError):
            continue
        found.append((text, parsed.year, parsed.month, parsed.day if has_day else None))
    return found


def _check_dates(fact: str, source: str) -> list[str]:
    reasons: list[str] = []
    full, months, month_days = _source_dates(source)
    days_named = {month for month, _ in month_days}
    relative = has_relative_time(source)
    components = _number_components(source)
    lowered = source.lower()
    for written, year, month, day in _fact_dates(fact):
        iso = f"{year:04d}-{month:02d}" + (f"-{day:02d}" if day else "")
        if written.lower() in lowered or iso in source or iso in full:
            continue
        # The year and month of the date are both pieces of one source number
        # (2004.6 -> 2004-06, 4.1.21 -> 2021-04), and the source names no such
        # month. A fuzzy parser fills the day in from today, so it is not used.
        if month not in months and any(
            (year in group or year % 100 in group) and month in group for group in components
        ):
            reasons.append("number_became_date")
            continue
        if relative:
            continue  # a resolved "yesterday" / "next Tuesday"
        if (day is not None and (month, day) in month_days) or (
            month in months and (day is None or month not in days_named)
        ):
            continue  # the source names this day, or the month without a day
        reasons.append("unsupported_date")
    return reasons + _check_years(fact, source, components, relative)


_YEAR_RE = re.compile(r"(?<![\d.,\-/])((?:19|20)\d{2})(?![\d\-/]|[.,]\d)")


def _check_years(
    fact: str, source: str, components: list[set[int]], relative: bool,
) -> list[str]:
    """A bare year the source never states ("in 2004" from "2004.6 ms")."""
    reasons: list[str] = []
    source_years = {m.group(1) for m in _YEAR_RE.finditer(source)}
    dated = {str(year) for _, year, _, _ in _fact_dates(fact)} | set(bare_numbers(source))
    for match in _YEAR_RE.finditer(fact):
        year = match.group(1)
        if year in source_years or year in dated:
            continue
        if any(int(year) in group for group in components):
            reasons.append("number_became_date")
        elif not relative and year not in source:
            reasons.append("unsupported_date")
    return reasons


def date_is_supported(iso_date: str | None, source_text: str) -> bool:
    """True when the source states, reformats or relatively implies this date.

    For a date a model put in a fact's date field: "2004.6 ms" does not
    support 2004-06-01, and the conversation's own date does not support a
    date for an event the source never dated.
    """
    if not iso_date:
        return True
    return not _check_dates(str(iso_date)[:10], source_text or "")


def _check_numbers(fact: str, source: str) -> list[str]:
    fact_dates = {m.group(0) for m in ISO_DATE_TOKEN_RE.finditer(fact)}
    support = bare_numbers(source)
    for core in typed_number_cores(fact):
        if core in support or any(core in d for d in fact_dates):
            continue
        return ["unsupported_number"]
    return []


def _best_match(fact: str, pieces: list[str]) -> str | None:
    """The piece of source text a fact was most plausibly derived from."""
    fact_words = _content_words(_NEGATION_RE.sub(" ", fact))
    if len(fact_words) < 2:
        return None
    best, best_score = None, 0.0
    for piece in pieces:
        overlap = len(fact_words & _content_words(piece))
        score = overlap / len(fact_words)
        if overlap >= 2 and score > best_score:
            best, best_score = piece, score
    return best if best_score >= 0.6 else None


def _best_sentence(fact: str, source: str) -> str | None:
    return _best_match(fact, _sentences(source))


def _clauses(sentence: str) -> list[str]:
    parts = re.split(r"[;:,]|\s[–—-]\s|\b(?:but|whereas|while|although|however)\b", sentence)
    return [p for p in parts if p and p.strip()]


def _negation_lost(fact: str, sentence: str) -> bool:
    """The clause the fact restates is negative and the fact is not.

    Clause level, so "we do not use A; we use B" does not flag a fact about B.
    """
    clause = _best_match(fact, _clauses(sentence)) or sentence
    return bool(_NEGATION_RE.search(clause)) and not _NEGATION_RE.search(fact)


def _order(text: str) -> tuple[str, str] | None:
    """(what comes first, what comes second) around an ordering word, or None.

    The nearest content word on each side names what is ordered, after words
    found on both sides are set aside: "the iOS app before the Android app"
    orders ios before android, not app before app.
    """
    match = _ORDER_RE.search(text)
    if not match:
        return None
    left = [w.strip(".'’") for w in _words(text[: match.start()])[-2:]]
    right = [w.strip(".'’") for w in _words(text[match.end():])[:2]]
    left_only = [w for w in left if w and w not in right]
    right_only = [w for w in right if w and w not in left]
    if not left_only or not right_only:
        return None
    first, second = left_only[-1], right_only[0]
    if match.group(1).lower() == "after":
        first, second = second, first
    return first, second


def _split_past_clause(sentence: str) -> tuple[set[str], set[str]]:
    """(words only in a "previously ..." clause, words only in the rest of the sentence)."""
    past_text, spans = "", []
    for match in _PAST_RE.finditer(sentence):
        rest = sentence[match.end():]
        end = _CLAUSE_END_RE.search(rest)
        stop = match.end() + (end.start() if end else len(rest))
        past_text += " " + sentence[match.end():stop]
        spans.append((match.start(), stop))
    if not spans:
        return set(), set()
    current_text = sentence
    for start, stop in reversed(spans):
        current_text = current_text[:start] + " " + current_text[stop:]
    past, current = _content_words(past_text), _content_words(current_text)
    return past - current, current - past


def _check_polarity(fact: str, source: str) -> list[str]:
    sentence = _best_sentence(fact, source)
    if sentence is None:
        return []
    reasons: list[str] = []
    if _negation_lost(fact, sentence):
        reasons.append("negation_lost")
    source_order, fact_order = _order(sentence), _order(fact)
    if source_order and fact_order and fact_order == (source_order[1], source_order[0]):
        reasons.append("order_reversed")
    past_only, current_only = _split_past_clause(sentence)
    fact_words = _content_words(fact)
    if (
        past_only
        and len(fact_words & past_only) > len(fact_words & current_only)
        and not _STATUS_MARKER_RE.search(fact)
    ):
        reasons.append("status_lost")
    return reasons


def check_fact_against_source(fact_text: str, source_text: str) -> FidelityReport:
    """Check one derived fact against the memory it was derived from."""
    fact = (fact_text or "").strip()
    source = (source_text or "").strip()
    if not fact or not source or fact in source:
        return FidelityReport(ok=True)
    reasons = _check_dates(fact, source) + _check_numbers(fact, source)
    reasons += _check_polarity(fact, source)
    unique = tuple(dict.fromkeys(reasons))
    return FidelityReport(ok=not unique, reasons=unique)


__all__ = ["FidelityReport", "check_fact_against_source", "date_is_supported"]
