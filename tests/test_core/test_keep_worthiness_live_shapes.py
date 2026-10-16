# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""More junk shapes, measured on the owner's store when M066 planned its restores.

M066 brings back memories the default-settings Langevin pass archived. Of 818
candidates on the owner's store, these shapes passed ``junk_reason`` and would
have come back to recall as noise: session/conversation boundary notes that
name only who or when, a file named with nothing said about it, and the
written-out form of a made-up date. ``junk_reason`` is the one predicate (decay
cap, refile, M066); these cases extend it and every REAL case still passes.
"""

from __future__ import annotations

import pytest

from superlocalmemory.core.keep_worthiness import PLACEHOLDER_DATES, junk_reason
from tests.test_core.test_junk_cannot_score_as_keep_worthy import REAL

#: Verbatim from the owner's store (16 Oct 2026).
LIVE_JUNK = [
    ("[Agantic_official]'s session ended at 00:57 on a Thursday", "content_free"),
    ("[Agantic_official]'s session ended at 11:47 yesterday", "content_free"),
    ("[Agantic_official]'s session ended at 23:41 on March 29, 2026", "content_free"),
    ("Agentic_official ended their session at 22:12 on August 3, 2026", "content_free"),
    ("Varun ended his session on July 17, 2026", "content_free"),
    ("Varun's session ended yesterday", "content_free"),
    ("2026-07-19 is the date when Varun's world session ended", "content_free"),
    ("The session ended at 20:52 on August 3, 2026", "content_free"),
    ("The session ended on 2026-08-02 at 18:25", "content_free"),
    ("2026-04-07 is the session end date", "content_free"),
    ("2026-05-28 is the end of the session", "content_free"),
    ("The conversation ended at 11:46", "content_free"),
    ("2026-05-10 is the current date", "content_free"),
    ("2026-04-23 04:27 IST", "content_free"),
    ("SESSION-HANDOFF-2026-07-20-v3.7.8-RELEASE.md was last modified on 2026-07-20",
     "content_free"),
    ("The file ISSUE-011-MCP-STATELESS-DRAFT.md was closed on 2026-07-30", "content_free"),
    ("The file AUDIT-PROMPT.md is located in the file system.", "content_free"),
    ("The file test_daemon_identity_contract.py is located in the same directory as the "
     "current script", "content_free"),
    ("The modification of sf-verify-pola.py occurred on January 15, 2024.", "placeholder_date"),
    ("The file slm-370-release-audit.md was last modified on 2024-01-14", "placeholder_date"),
]

#: Real memories of the same shapes: they must keep their score.
LIVE_REAL = [
    "The payment service will not use MongoDB",
    "The filesystem is read-only",
    "R3 achieved 70.4% performance in previous round",
    "Varun prefers using Varun's voice clone over Google TTS for the YouTube Shorts",
    "The session ended because the MCP hub crashed after the 4.2.7 upgrade",
    "In July 2026, Priya Shah approved personal as the default memory scope",
    "The file config.toml sets the Codex model to gpt-6.1-sol",
    "SLM 4.2.7 shipped on 2026-10-15 to PyPI and npm",
]


@pytest.mark.parametrize(("text", "reason"), LIVE_JUNK)
def test_live_junk_shapes(text: str, reason: str) -> None:
    assert junk_reason(text) == reason


@pytest.mark.parametrize("text", [*LIVE_REAL, *REAL])
def test_real_memories_stay_real(text: str) -> None:
    assert junk_reason(text) is None


def test_the_made_up_days_come_from_one_place() -> None:
    from superlocalmemory.storage.fabricated_dates import FABRICATED_DAYS

    assert PLACEHOLDER_DATES == frozenset(FABRICATED_DAYS)
