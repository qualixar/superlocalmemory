# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Junk can never be scored as "worth keeping".

16 Oct 2026: ``slm doctor --refile-hidden`` offered six memories as "hidden even
though they are scored to keep". Five were junk and the owner deleted them by
hand. Their score was never a judgement about their content: retention is the
schema default 1.0, or reset to 1.0 by any recall that touched them. These are
the real strings, plus real-looking memories that must keep their score.
"""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.core.keep_worthiness import (
    KEEP_THRESHOLD,
    cap_retention,
    is_keep_worthy,
    junk_reason,
    register_sql,
)

#: Deleted by the owner on 16 Oct 2026, verbatim (the last one as the doctor
#: printed it, cut at 80 characters).
OWNER_DELETED = [
    ("2024-01-15 File was modified", "placeholder_date"),
    ("2024-01-15 was used as a controlled test string", "placeholder_date"),
    ("2024-01-15 is the date on which the super local memory website was established",
     "placeholder_date"),
    ("As of 2026-01-15 in production-v5.2 for videos/mcp-stateless, the system will di",
     "placeholder_date"),
]

#: Still on the owner's store, shaped like the above.
JUNK = [
    ("The file .claude.json was modified by Claude on 2024-01-15", "placeholder_date"),
    ("2024-01-15T00:00:00Z is when the session started", "placeholder_date"),
    ("A controlled test string was used for verifying pipeline behavior", "test_string"),
    ("concurrent stress test fact 2", "test_string"),
    ("this is a test", "test_string"),
    ("Test memory: Power restoration session completed on March 26, 2026.", "test_string"),
    ("lorem ipsum dolor sit amet", "test_string"),
    ("The file was modified", "content_free"),
    ("Files were updated", "content_free"),
    ("The session ended at 16:07", "content_free"),
    ("session ended 2026-10-14 15:26", "content_free"),
    ("2026-04-08", "content_free"),
    ("26 files were changed on 2026-08-08", "content_free"),
    ("22:18 is a time", "content_free"),
    ("Yesterday", "content_free"),
    ("", "content_free"),
    ("   ", "content_free"),
]

#: Real memories. None of them may lose its score.
REAL = [
    "Varun prefers dark mode",
    "Use uv",
    "m4 should be upgraded to 2.1.3 globally with super local memory",
    "SLM 4.2.7 shipped on 2026-10-15 to PyPI and npm",
    "The session ended because the MCP hub crashed",
    "2026-04-15 is the deadline for ASE 2026",
    "Incident: an aborted isolated registry test daemon on port 8765 made the "
    "dashboard show one temporary test memory from /private/tmp.",
    "Run the test suite module-wise, never the whole suite",
    "pytest -q cwd must agree between run.sh and the engine",
    "The test string fixture lives in tests/fixtures and is loaded by conftest "
    "for every storage test, which is why a broken fixture fails dozens of "
    "unrelated tests at once and must be fixed before any release goes out",
    "请只回复好的。我的项目代号是星河。",
    "Ich bevorzuge den dunklen Modus",
    "The camera-ready copy is due on 2026-01-16",
    "The release is 2.1.3",
]


class TestJunkReason:
    @pytest.mark.parametrize(("content", "reason"), OWNER_DELETED + JUNK)
    def test_junk_is_named(self, content: str, reason: str) -> None:
        assert junk_reason(content) == reason
        assert not is_keep_worthy(content)

    @pytest.mark.parametrize("content", REAL)
    def test_real_memories_are_keep_worthy(self, content: str) -> None:
        assert junk_reason(content) is None, content
        assert is_keep_worthy(content)

    def test_none_is_content_free(self) -> None:
        assert junk_reason(None) == "content_free"

    def test_a_long_note_quoting_the_placeholder_is_not_junk(self) -> None:
        note = ("Extractor bug: local models fabricate 2024-01-15 when a turn has "
                "no date, so 173 facts on the store carry it. The fix strips the "
                "date before scoring, measures the store read-only first, prints "
                "a sample for review, and leaves every real memory where it is, "
                "because a false positive here hides something the owner wrote.")
        assert junk_reason(note) is None


class TestCapRetention:
    def test_junk_cannot_score_above_the_keep_threshold(self) -> None:
        assert cap_retention(1.0, "2024-01-15 File was modified") == KEEP_THRESHOLD
        assert not cap_retention(1.0, "this is a test") > KEEP_THRESHOLD

    def test_junk_below_the_threshold_keeps_its_lower_score(self) -> None:
        assert cap_retention(0.03, "2024-01-15 File was modified") == 0.03

    def test_real_memory_keeps_its_score(self) -> None:
        assert cap_retention(1.0, "Varun prefers dark mode") == 1.0
        assert cap_retention(0.42, "Varun prefers dark mode") == 0.42

    def test_threshold_is_the_active_zone_boundary(self) -> None:
        from superlocalmemory.core.config import ForgettingConfig
        from superlocalmemory.math.ebbinghaus import EbbinghausCurve

        curve = EbbinghausCurve(ForgettingConfig())
        assert curve.lifecycle_zone(KEEP_THRESHOLD) != "active"
        assert curve.lifecycle_zone(KEEP_THRESHOLD + 1e-9) == "active"


class TestSqlFunction:
    def test_registered_function_agrees_with_python(self) -> None:
        conn = sqlite3.connect(":memory:")
        register_sql(conn)
        for content, _ in OWNER_DELETED + JUNK:
            assert conn.execute("SELECT slm_keep_worthy(?)", (content,)).fetchone()[0] == 0
        for content in REAL:
            assert conn.execute("SELECT slm_keep_worthy(?)", (content,)).fetchone()[0] == 1
        assert conn.execute("SELECT slm_keep_worthy(NULL)").fetchone()[0] == 0


class TestFreshRetention:
    def test_real_memory_is_refreshed_to_active(self) -> None:
        from superlocalmemory.core.keep_worthiness import fresh_retention

        assert fresh_retention("Varun prefers dark mode") == (1.0, "active")

    def test_junk_is_refreshed_only_to_warm(self) -> None:
        from superlocalmemory.core.keep_worthiness import fresh_retention

        assert fresh_retention("2024-01-15 File was modified") == (KEEP_THRESHOLD, "warm")
