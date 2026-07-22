# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 — PII redaction on ingest (C4)

"""Opt-in PII redaction for ingested memory content.

For team / company deployments an operator may need memory to never persist
personal identifiers (email, phone, national ID, payment card, IP). This
module provides a pure, well-bounded scrubber that replaces detected PII with
``[PII:TYPE]`` markers. It is complementary to ``security_primitives.
redact_secrets`` (which handles API keys / tokens and always runs).

Design goals:
* **Low false-positive rate.** Card numbers are Luhn-validated; SSNs use the
  canonical grouping; phone matching requires a plausible separator shape.
* **Deterministic + pure.** No I/O, no config — the caller decides when to run
  it (gated by SLM_PII_REDACTION / config), so it is trivially testable.
* **Order matters.** Emails are redacted before phone/number sweeps so an
  email's local part is never mistaken for a number.
"""

from __future__ import annotations

import re

# Order-sensitive: earlier patterns win over later ones on overlapping spans.
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_IPV4 = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
)
# Phone: conservative to avoid eating ISO dates (4-2-2) / version strings.
# Only unambiguous shapes match:
#   * international  +CC then grouped digits
#   * parenthesized  (415) 555-0132
#   * strict US      415-555-0132 / 415.555.0132  (3-3-4, dot/dash only — a
#     space separator is excluded so "2026-07-22 12" style runs never match).
_PHONE = re.compile(
    r"(?<!\w)(?:"
    r"\+\d{1,3}[\s.\-]?\d{1,4}[\s.\-]?\d{2,4}[\s.\-]?\d{2,4}"
    r"|\(\d{3}\)[\s.\-]?\d{3}[\s.\-]?\d{4}"
    r"|\d{3}[.\-]\d{3}[.\-]\d{4}"
    r")(?!\w)"
)
# Candidate card: 13–19 digits, optionally grouped by space/dash. Luhn-checked.
_CARD_CANDIDATE = re.compile(r"(?<!\w)(?:\d[ -]?){13,19}(?!\w)")


def _luhn_ok(digits: str) -> bool:
    """Return True if ``digits`` (0-9 only) passes the Luhn checksum."""
    if not (13 <= len(digits) <= 19):
        return False
    total = 0
    parity = len(digits) % 2
    for i, ch in enumerate(digits):
        d = ord(ch) - 48
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _redact_cards(text: str, counter: list[int]) -> str:
    def _sub(m: re.Match[str]) -> str:
        raw = m.group(0)
        digits = re.sub(r"\D", "", raw)
        if _luhn_ok(digits):
            counter[0] += 1
            return "[PII:CARD]"
        return raw
    return _CARD_CANDIDATE.sub(_sub, text)


def redact_pii(text: str) -> tuple[str, int]:
    """Return ``(redacted_text, num_redactions)``.

    Never raises; a non-string or empty input is returned unchanged with 0.
    """
    if not isinstance(text, str) or not text:
        return text, 0

    counter = [0]

    def _count_sub(pattern: re.Pattern[str], label: str, s: str) -> str:
        def _sub(_m: re.Match[str]) -> str:
            counter[0] += 1
            return label
        return pattern.sub(_sub, s)

    out = text
    # Email first (protects local parts from the number sweeps).
    out = _count_sub(_EMAIL, "[PII:EMAIL]", out)
    # Payment cards before generic phone/number matching (Luhn-gated).
    out = _redact_cards(out, counter)
    out = _count_sub(_SSN, "[PII:SSN]", out)
    out = _count_sub(_IPV4, "[PII:IP]", out)
    out = _count_sub(_PHONE, "[PII:PHONE]", out)
    return out, counter[0]


def redact_pii_text(text: str) -> str:
    """Convenience wrapper returning only the redacted string."""
    return redact_pii(text)[0]
