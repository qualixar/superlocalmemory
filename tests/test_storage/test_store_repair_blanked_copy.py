# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""#308: ``is_blanked_copy`` stays exact but runs in linear time; a rejected original
is not restored.

The old matcher built one regex of N ``(?:.+?)`` pieces, which backtracks
polynomially on a long memory with many markers. ``_old_is_blanked_copy`` below
is that implementation, kept verbatim as the oracle for a differential test.
"""

from __future__ import annotations

import json
import random
import re
import sqlite3
import time

import pytest

from superlocalmemory.storage.store_repair import (
    apply_repair,
    is_blanked_copy,
)

_MARKER = re.compile(r"\[REDACTED:[A-Z_]+:[^\]]*\]")
MISMATCH = "queryable ingestion memory content mismatch"


def _old_is_blanked_copy(stored: str, original: str) -> bool:
    if not isinstance(stored, str) or not isinstance(original, str):
        return False
    if stored == original or not _MARKER.search(stored):
        return False
    parts = _MARKER.split(stored)
    pattern = "(?:.+?)".join(re.escape(p) for p in parts)
    return re.fullmatch(pattern, original, re.DOTALL) is not None


def _blank(rng: random.Random, original: str) -> str:
    """``original`` with up to four random non-empty spans replaced by markers."""
    if len(original) < 2:
        return original
    cuts = sorted(rng.sample(range(len(original) + 1), min(len(original) + 1, rng.randint(2, 8))))
    out, last = [], 0
    for start, end in zip(cuts[0::2], cuts[1::2]):
        if end > start and start >= last:
            out.append(original[last:start])
            out.append(f"[REDACTED:{rng.choice(['ENTROPY', 'PATH', 'KEY_X'])}:{rng.choice('ab.')}]")
            last = end
    out.append(original[last:])
    return "".join(out)


def _cases(seed: int, count: int):
    rng = random.Random(seed)
    alphabet = "ab c\n[]:."
    for _ in range(count):
        original = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        stored = _blank(rng, original)
        yield stored, original
        # Negatives: break the stored text in one small way.
        if stored:
            i = rng.randrange(len(stored))
            yield stored[:i] + rng.choice("abz") + stored[i + 1:], original
            yield stored[:i] + stored[i + 1:], original
        yield stored + rng.choice("ab"), original
        yield stored, original + rng.choice("ab")
        yield original, stored


def test_the_linear_matcher_gives_the_old_answer_on_generated_cases() -> None:
    seen_true = seen_false = 0
    for stored, original in _cases(seed=308, count=4000):
        expected = _old_is_blanked_copy(stored, original)
        assert is_blanked_copy(stored, original) is expected, (stored, original)
        seen_true += expected
        seen_false += not expected
    assert seen_true > 500 and seen_false > 500      # the sample exercises both answers


@pytest.mark.parametrize("stored,original", [
    ("[REDACTED:A:x]", ""),                      # a marker must stand for at least one char
    ("[REDACTED:A:x]", "a"),
    ("[REDACTED:A:x][REDACTED:B:y]", "a"),       # two markers need two chars
    ("[REDACTED:A:x][REDACTED:B:y]", "ab"),
    ("a[REDACTED:A:x]", "a"),
    ("a[REDACTED:A:x]", "ab"),
    ("[REDACTED:A:x]b", "ab"),
    ("a[REDACTED:A:x]a[REDACTED:A:x]a", "aaaaa"),
    ("a[REDACTED:A:x]a[REDACTED:A:x]a", "aaaa"),
    ("a[REDACTED:A:x]ab", "aXaab"),              # the suffix must not reuse the middle piece
    ("a[REDACTED:A:x]ab", "aab"),
    ("x[REDACTED:A:x]ab", "aXab"),
    ("[REDACTED:A:x]ab[REDACTED:B:y]", "1ab2ab3"),
    ("ab[REDACTED:A:x]ab", "abab"),
    ("ab[REDACTED:A:x]ab", "ab\nab"),             # a gap may be a newline
    (None, "a"),
    ("a", None),
])
def test_edge_cases_match_the_old_answer(stored, original) -> None:
    assert is_blanked_copy(stored, original) is _old_is_blanked_copy(stored, original)


def test_two_thousand_markers_take_linear_time() -> None:
    n = 2000
    original = "".join(f"segment{i} /very/long/path/{i}/file.pdf " for i in range(n))
    stored = "".join(f"segment{i} [REDACTED:PATH:{i}] " for i in range(n))
    start = time.perf_counter()
    assert is_blanked_copy(stored, original)
    assert not is_blanked_copy(stored + "tail", original)
    assert not is_blanked_copy(stored, original.replace("segment1999", "segmentXXXX"))
    assert time.perf_counter() - start < 1.0


# ---------------------------------------------------------------------------
# A rejected original is not restored (it would only fail again)
# ---------------------------------------------------------------------------

TEMPLATE_ORIGINAL = "You are summarizing a Claude Code session for /Users/me/work/project/x"
TEMPLATE_BLANKED = "You are summarizing a Claude Code session for [REDACTED:ENTROPY:x]"


@pytest.fixture()
def conn(tmp_path) -> sqlite3.Connection:
    c = sqlite3.connect(tmp_path / "m.db", isolation_level=None)
    c.executescript("""
        CREATE TABLE memories(memory_id TEXT PRIMARY KEY, content TEXT);
        CREATE TABLE atomic_facts(fact_id TEXT PRIMARY KEY, memory_id TEXT, content TEXT,
                                  canonical_entities_json TEXT);
        CREATE TABLE ingestion_operations(operation_id TEXT PRIMARY KEY, raw_content TEXT,
            queryable_fact_ids_json TEXT, state TEXT, last_error TEXT, attempt_count INT,
            next_retry_at REAL, lease_owner TEXT, lease_expires_at REAL, updated_at TEXT);
    """)
    c.execute("INSERT INTO memories VALUES ('m1', ?)", (TEMPLATE_BLANKED,))
    c.execute("INSERT INTO atomic_facts VALUES ('f1', 'm1', ?, '[]')", (TEMPLATE_BLANKED,))
    c.execute("INSERT INTO ingestion_operations VALUES ('op1', ?, ?, 'failed', ?, 9, 1e12, "
              "'', 0, '')", (TEMPLATE_ORIGINAL, json.dumps(["f1"]), MISMATCH))
    return c


def test_the_plan_leaves_a_rejected_original_alone(conn) -> None:
    from superlocalmemory.storage.store_repair import _plan_restores

    restorable, left = _plan_restores(conn)
    assert restorable == []
    assert left == ["op1"]


def test_apply_skips_a_rejected_original_even_from_a_stale_plan(conn) -> None:
    from superlocalmemory.storage import store_repair

    # A plan from before the gate refused this original (or a hand-built plan).
    item = store_repair.RestoreItem("op1", "m1", ("f1",), TEMPLATE_BLANKED, TEMPLATE_ORIGINAL)
    plan = store_repair.RepairPlan((item,), (), (), ())
    result = apply_repair(conn, plan)
    assert (result.memories_restored, result.memories_skipped) == (0, 1)
    assert conn.execute("SELECT content FROM memories WHERE memory_id='m1'"
                        ).fetchone()[0] == TEMPLATE_BLANKED
    assert conn.execute("SELECT state FROM ingestion_operations WHERE operation_id='op1'"
                        ).fetchone()[0] == "failed"
