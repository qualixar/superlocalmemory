# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.21 — Stage 8 F5 (Mediums/Lows)

"""Stage 8 F5 regression — marker regex + LIMIT hardening.

S-L04: ``post_tool_outcome_hook._MARKER_RE`` now disallows colons in
the fact_id group, so a hostile tool response cannot emit
``slm:fact:evil:deadbeef:abcdef01`` and confuse the validator with a
wrong-grouped fact_id.

SEC-M2: the pending-window SELECT uses ``LIMIT 5`` (down from 20) to
reduce amplification surface on tool-response spam.
"""

from __future__ import annotations

from superlocalmemory.hooks import post_tool_outcome_hook as h


def test_s_l04_marker_regex_rejects_colon_in_fact_id() -> None:
    m = h._MARKER_RE.search(  # noqa: SLF001
        "before slm:fact:evil:deadbeef:abcdef01 after"
    )
    assert m is not None, "regex should find *some* match"
    # The fact_id group MUST NOT contain a colon — that was the bypass.
    assert ":" not in m.group(1), m.group(1)


def test_s_l04_marker_regex_accepts_hex_and_legacy_fact_ids() -> None:
    # Hex fact_id (production shape: uuid4().hex[:16]).
    m1 = h._MARKER_RE.search("slm:fact:abcdef0123456789:deadbeef")  # noqa: SLF001
    assert m1 is not None and m1.group(1) == "abcdef0123456789"
    # Legacy dash-style fact_id (matches existing test_outcome_hooks fixtures).
    m2 = h._MARKER_RE.search("x slm:fact:fact-42:deadbeef y")  # noqa: SLF001
    assert m2 is not None and m2.group(1) == "fact-42"


def test_s9_w3_pending_window_and_write_cap() -> None:
    """S9-W3 C6 / H-SKEP-03 / H-ARC-H4: SEC-M2's LIMIT 5 tightening
    silently dropped signals on heavy Claude Code sessions (30+ Reads
    + 10 recalls). The pending-row window is now 50 (so signals on the
    6th+ outcome still get recorded) and UPDATE amplification is
    capped at 10 via an outer PENDING_WRITE_CAP on the returned list.
    The hook no longer embeds the literal SQL — match the contract
    constants on the EngagementRewardModel class instead.
    """
    from superlocalmemory.learning.reward import EngagementRewardModel
    assert EngagementRewardModel.PENDING_MATCH_WINDOW == 50, (
        "post-Stage-9 contract: window must be 50 to cover heavy "
        "tool-use sessions without dropping signals"
    )
    assert EngagementRewardModel.PENDING_WRITE_CAP == 10, (
        "amplification cap must be 10 regardless of window size"
    )
