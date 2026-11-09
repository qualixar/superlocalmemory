# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Lower-case technical words are found in bounded time, with the same result (#310).

4.2.13's pattern backtracked exponentially on a lower-case word with a long run
of digits that a hyphen, slash, underscore or word character follows:
``"a" + "1" * 30 + "-"`` never finished, on a save and on a recall question.
The replacement must return exactly what the old pattern returned, because its
output is stored (entities, graph edges, keyword alt-keys): a single different
word would make old and new stores disagree.
"""

from __future__ import annotations

import itertools
import random
import re
import time

from superlocalmemory.encoding.entity_mentions import extract_mentions
from superlocalmemory.encoding.lower_tech_scan import lower_tech_spans
from superlocalmemory.retrieval.entity_channel import extract_query_entities

#: The 4.2.13 pattern, verbatim: the reference every result is compared with.
_OLD = re.compile(r"(?<![\w.:/~\\-])([a-z][a-z0-9]*(?:[_:.][a-z0-9]+|\d[a-z0-9.]*)+"
                  r"|[a-z]{4,})(?![\w/\\-])")


def _old(text: str) -> list[tuple[int, int]]:
    return [m.span(1) for m in _OLD.finditer(text)]


def test_every_short_string_matches_the_old_pattern_exactly() -> None:
    # 597,870 strings. The release check ran the same comparison up to length 8
    # (48,427,560 strings) and on every text of a 57k-text real store.
    for length in range(1, 7):
        for chars in itertools.product("a1._:-/ A", repeat=length):
            text = "".join(chars)
            assert lower_tech_spans(text) == _old(text), text


def test_random_text_with_unicode_matches_the_old_pattern() -> None:
    rnd = random.Random(310)
    alphabet = "abz019._:-/ ~\\AZé٣"  # ٣ is a Unicode digit: \d matches it, [0-9] does not
    for _ in range(50_000):
        text = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(8, 40)))
        if re.search(r"\d{11}", text):
            continue  # the old pattern itself is slow there
        assert lower_tech_spans(text) == _old(text), text


def test_long_digit_runs_take_milliseconds_not_hours() -> None:
    for text in ("a" + "1" * 30 + "-", "a" + "1" * 5000 + "-",
                 "x " + "a1" * 2000 + "/", "img20261004184612345678901234-edit"):
        start = time.perf_counter()
        lower_tech_spans(text)
        assert time.perf_counter() - start < 0.5, text[:40]


def test_a_save_and_a_question_with_such_a_word_finish() -> None:
    note = "Uploaded scan img2026100418461234567890123456789-edit to the archive"
    start = time.perf_counter()
    extract_mentions(note)
    extract_query_entities("where is img2026100418461234567890123456789-edit?")
    assert time.perf_counter() - start < 1.0


def test_digit_heavy_text_keeps_the_words_the_old_pattern_found() -> None:
    # Over the switch point (more than 10 digits in one run), on text the old
    # pattern still answers quickly: same spans.
    for text in ("build a1b2c3d4e5f6a7b8 ok", "llama3.2 and gemma3:27b on m5",
                 "ids a3fc66cf436c48ad, 745fe4772cae4528.", "v1.2.3.4.5.6.7.8.9.10.11 x"):
        assert lower_tech_spans(text) == _old(text), text
