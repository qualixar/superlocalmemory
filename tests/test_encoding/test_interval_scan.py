# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
""""from X to Y" intervals are found in bounded time, with the same result.

4.2.13 searched a save's sentences (fact_extractor) and a fact's content or a
recall question (temporal_parser) with two lazy "from X to Y" patterns. On
"from" followed by a long run of spaces, or by many short runs, they backtrack
cubically: "from" + 2000 spaces took 29.5 s on a save. Their groups become
stored interval dates, so the replacement must return exactly the old match
(same span, same groups, or None), not merely a match.
"""

from __future__ import annotations

import itertools
import random
import re
import time

from superlocalmemory.encoding import fact_extractor, temporal_parser
from superlocalmemory.encoding.interval_scan import SWITCH_CHARS, interval_pattern, scan

#: The 4.2.13 patterns, verbatim: the reference every result is compared with.
_OLD_A = re.compile(
    r"\b(?:from|between)\s+(.+?)\s+(?:to|and|until|through)\s+(.+?)(?:[.,;]|$)", re.IGNORECASE)
_OLD_B = re.compile(r"\bfrom\s+(.+?)\s+to\s+(.+?)(?:\.|,|;|$)", re.IGNORECASE)
_A = fact_extractor._INTERVAL
_B = temporal_parser._DURATION_SCAN
#: Single-letter keywords: same structure, so every character-level shape is reachable.
_OLD_C = re.compile(r"\b(?:f|bw)\s+(.+?)\s+(?:t|n)\s+(.+?)(?:[.,;]|$)", re.IGNORECASE)
_C = interval_pattern(_OLD_C, r"\b(?:f|bw)", r"(?:t|n)")


def _old(pattern: re.Pattern[str], text: str):
    found = pattern.search(text)
    return None if found is None else (found.span(), found.span(1), found.span(2))


def _same(pattern, old: re.Pattern[str], text: str) -> None:
    expected = _old(old, text)
    assert scan(pattern.opener, pattern.joiner, text) == expected, repr(text)
    assert pattern.spans(text) == expected, repr(text)


def test_the_call_sites_still_use_the_old_patterns_verbatim() -> None:
    assert fact_extractor._INTERVAL_RE.pattern == _OLD_A.pattern
    assert fact_extractor._INTERVAL_RE.flags == _OLD_A.flags
    assert temporal_parser._DURATION.pattern == _OLD_B.pattern
    assert temporal_parser._DURATION.flags == _OLD_B.flags
    assert _A.regex is fact_extractor._INTERVAL_RE and _B.regex is temporal_parser._DURATION


def test_every_short_character_string_matches_the_old_pattern_exactly() -> None:
    # 299,592 strings; the release check ran length 8 (19,173,960 strings).
    for length in range(1, 7):
        for chars in itertools.product("fbtn x\n.", repeat=length):
            _same(_C, _OLD_C, "".join(chars))


def test_every_short_token_sequence_matches_the_old_interval_pattern() -> None:
    # 299,592 sequences; the release check ran 12 tokens to length 6 (3,257,436).
    tokens = ("from", "between", "to", "and", " ", "\n", ".", "x")
    for length in range(1, 7):
        for parts in itertools.product(tokens, repeat=length):
            _same(_A, _OLD_A, "".join(parts))


def test_every_short_token_sequence_matches_the_old_duration_pattern() -> None:
    tokens = ("from", "to", "until", " ", "\n", ",", ";", "x")
    for length in range(1, 7):
        for parts in itertools.product(tokens, repeat=length):
            _same(_B, _OLD_B, "".join(parts))


_RANDOM_TOKENS = ("from", "FROM", "From", "between", "BeTwEeN", "to", "TO", "and", "until",
                  "UNTIL", "through", "x", "2020", "Jan", " ", " ", "  ", "\t", "\n", "\r",
                  "\u3000", "\x1c", "\u2028", ".", ",", ";", "-", "\u0130", "\u0131", "\u017f",
                  "\u212a", "\xe9")
#: Keyword-dense: words joined by whitespace, so most texts match and the group
#: spans are really compared.
_DENSE_WORDS = ("from", "fRoM", "between", "to", "AND", "until", "through", "unt\u0131l",
                "x", "x.", "x;", "\u0130", ".")
_DENSE_GAPS = (" ", " ", " ", "  ", "\t", "\n", "\u3000", " \n ")


def _dense_text(rnd: random.Random) -> str:
    words = [rnd.choice(_DENSE_WORDS) for _ in range(rnd.randint(2, 9))]
    return "".join(w + rnd.choice(_DENSE_GAPS) for w in words)[: rnd.randint(1, 80)]


def _random_text(rnd: random.Random, low: int, high: int) -> str:
    return "".join(rnd.choice(_RANDOM_TOKENS) for _ in range(rnd.randint(low, high)))


def test_random_mixed_case_and_unicode_text_matches_the_old_patterns() -> None:
    rnd = random.Random(2614)
    for _ in range(50_000):
        text = _random_text(rnd, 1, 14)
        _same(_A, _OLD_A, text)
        _same(_B, _OLD_B, text)


def test_random_keyword_dense_text_matches_the_old_patterns() -> None:
    rnd = random.Random(1430)
    for _ in range(50_000):
        text = _dense_text(rnd)
        _same(_A, _OLD_A, text)
        _same(_B, _OLD_B, text)


def test_long_ordinary_text_matches_the_old_patterns() -> None:
    # Over the switch length, on text the old engine still answers quickly.
    rnd = random.Random(42)
    words = ("from", "between", "to", "and", "until", "Monday", "2020", "the", "plan", "x.",
             "y,", "z;", "\n")
    for _ in range(2_000):
        text = " ".join(rnd.choice(words) for _ in range(rnd.randint(25, 80)))
        assert len(text) > SWITCH_CHARS
        _same(_A, _OLD_A, text)
        _same(_B, _OLD_B, text)


_ADVERSARIAL = (
    "from" + " " * 5000 + "x",
    "between" + " " * 5000 + "\nx",
    "from" + " " * 3000 + "x" + " " * 3000 + "to" + " " * 3000 + "\nx",
    ("from" + " " * 10) * 500 + "\nx",
    ("from    x    to    ") * 300 + "\nx",
    "from x" + " to" * 2000 + "\nx",
    "from " * 2000 + "\nx",
    " allowed to" + " " * 5000 + "0",
    "from to " * 8_000 + "\nx",
)


def test_adversarial_text_takes_milliseconds_not_minutes() -> None:
    for text in _ADVERSARIAL:
        for pattern in (_A, _B):
            start = time.perf_counter()
            pattern.spans(text)
            assert time.perf_counter() - start < 0.5, (repr(text[:30]), len(text))


def test_a_save_and_a_dated_fact_with_such_text_finish() -> None:
    parser = temporal_parser.TemporalParser("2026-11-09")
    for text in (_ADVERSARIAL[0], _ADVERSARIAL[3], "From " + " " * 5000 + "Jan 1 to Mar 1, 2026"):
        start = time.perf_counter()
        fact_extractor._extract_interval(text, "2026-11-09")
        parser.extract_dates_from_text(text)
        assert time.perf_counter() - start < 1.0, repr(text[:30])


def test_the_call_sites_give_the_same_dates_as_before() -> None:
    text = "We were in Lisbon from January 1, 2026 to March 1, 2026, then home."
    found = _OLD_A.search(text)
    assert fact_extractor._extract_interval(text, "2026-11-09") == (
        fact_extractor._try_parse_date(found.group(1).strip(), "2026-11-09"),
        fact_extractor._try_parse_date(found.group(2).strip(), "2026-11-09"))
    dated = temporal_parser.TemporalParser("2026-11-09").extract_dates_from_text(text)
    assert dated["interval_start"] is not None and dated["interval_end"] is not None
