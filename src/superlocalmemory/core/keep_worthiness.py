# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Whether a memory's content can justify a "keep" score at all.

A retention score above :data:`KEEP_THRESHOLD` means "worth keeping": the
forgetting curve files it ``active`` and ``slm doctor --refile-hidden`` offers
to un-hide it. Nothing that writes that score reads the memory. It is the
schema default 1.0 for a row created by a lifecycle move, and it is reset to
1.0 by every recall that touched the memory. So on 16 Oct 2026 the doctor
called "2024-01-15 File was modified" worth keeping, and four more like it.

This module is the content half of that decision. Three shapes are junk:

``placeholder_date``
    A short fact carrying a date the extractor invents when a turn has none.
    2024-01-15 appears in 173 of 28,331 facts on the owner's store, far above
    any real date before the store existed. 2026-01-15 is the same habit
    moved to the current year; both appeared in facts the owner deleted.
``test_string``
    A short fact that is a test or probe artifact: "concurrent stress test
    fact 2", "this is a test", "A controlled test string was used ...".
``content_free``
    Nothing left once dates, times and numbers are removed but generic words:
    "The file was modified", "The session ended at 16:07".

Junk is capped AT the threshold, never below it. A false positive therefore
loses only its "keep" status: it stays findable and decays as before. The
rules are length-bounded so a long note that merely quotes a placeholder, or
discusses a test, is never touched.

Pure functions, no I/O. :func:`register_sql` exposes the same predicate to
SQLite so a query and the Python caller cannot disagree.
"""

from __future__ import annotations

import re
import sqlite3

__all__ = [
    "KEEP_THRESHOLD", "PLACEHOLDER_DATES", "cap_retention", "fresh_retention",
    "is_keep_worthy", "junk_reason", "register_sql",
]

#: A score above this is "keep" (``math/ebbinghaus.py:lifecycle_zone`` files it
#: ``active``; ``core/memory_health.py`` calls a hidden one inconsistent).
KEEP_THRESHOLD = 0.8

#: Dates the extractor fabricates. Measured on the owner's store, 16 Oct 2026.
PLACEHOLDER_DATES = frozenset({"2024-01-15", "2026-01-15"})

#: Facts longer than this are notes, not extractions; the date rule skips them.
_PLACEHOLDER_MAX_WORDS = 40
#: A test artifact is one short line. Longer text that mentions a test is a note.
_TEST_MAX_WORDS = 25

_DATE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2})(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?"
)
_TIME = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AaPp][Mm])?\b")
#: A word starts with a letter in any script, so non-English memories count.
_WORD = re.compile(r"[^\W\d_][\w./-]*")
_TEST = re.compile(
    r"\b(?:controlled|dummy|probe|stress|benchmark|timing|placeholder)\s+test"
    r"\s+(?:string|fact|memory)\b"
    r"|\btest\s+fact\b"
    r"|^\W*test\s+memory\b"
    r"|^\W*this\s+is\s+(?:just\s+)?an?\s+(?:short\s+)?test\b"
    r"|\blorem\s+ipsum\b",
    re.IGNORECASE,
)
#: Words that say nothing on their own. A fact made only of these is empty.
_GENERIC = frozenset("""
a an the is was were be been being are am it its this that these those there
here of on in at to for by with as and or has have had do did does done
file files modified changed updated edited created saved written touched
opened closed started ended began finished happened occurred
date day time today yesterday tomorrow when which something thing data
session event
""".split())


def junk_reason(content: str | None) -> str | None:
    """Why ``content`` cannot justify a keep score, or None if it can."""
    text = (content or "").strip()
    if not text:
        return "content_free"
    words = len(text.split())
    if words <= _PLACEHOLDER_MAX_WORDS and any(
        d in PLACEHOLDER_DATES for d in _DATE.findall(text)
    ):
        return "placeholder_date"
    if words <= _TEST_MAX_WORDS and _TEST.search(text):
        return "test_string"
    remainder = _TIME.sub(" ", _DATE.sub(" ", text))
    if all(w.lower() in _GENERIC for w in _WORD.findall(remainder)):
        return "content_free"
    return None


def is_keep_worthy(content: str | None) -> bool:
    """Whether ``content`` may carry a score above :data:`KEEP_THRESHOLD`."""
    return junk_reason(content) is None


def cap_retention(score: float, content: str | None) -> float:
    """``score``, held at :data:`KEEP_THRESHOLD` when ``content`` is junk."""
    if score > KEEP_THRESHOLD and not is_keep_worthy(content):
        return KEEP_THRESHOLD
    return score


def fresh_retention(content: str | None) -> tuple[float, str]:
    """``(score, zone)`` for a memory just used: active, unless it is junk.

    A recall that returns junk is not evidence the junk is worth keeping; it
    is how the five memories the owner deleted on 16 Oct 2026 got their 1.0.
    """
    if is_keep_worthy(content):
        return 1.0, "active"
    return KEEP_THRESHOLD, "warm"


def register_sql(conn: sqlite3.Connection) -> None:
    """Expose ``slm_keep_worthy(content)`` (1 or 0) on ``conn``."""
    conn.create_function(
        "slm_keep_worthy", 1, lambda c: int(is_keep_worthy(c)), deterministic=True,
    )
