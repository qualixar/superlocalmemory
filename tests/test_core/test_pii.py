# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 — PII redaction on ingest (C4)

"""PII scrubber: detection coverage + false-positive resistance."""

from __future__ import annotations

from superlocalmemory.core.pii import redact_pii


def test_email_redacted():
    out, n = redact_pii("contact alice@example.com for details")
    assert "[PII:EMAIL]" in out and "alice@example.com" not in out
    assert n == 1


def test_ssn_redacted():
    out, n = redact_pii("SSN 123-45-6789 on file")
    assert "[PII:SSN]" in out and "123-45-6789" not in out
    assert n == 1


def test_ipv4_redacted():
    out, n = redact_pii("client connected from 192.168.1.42 last night")
    assert "[PII:IP]" in out and "192.168.1.42" not in out


def test_valid_credit_card_redacted():
    # 4242 4242 4242 4242 is a well-known Luhn-valid test number.
    out, n = redact_pii("card 4242 4242 4242 4242 charged")
    assert "[PII:CARD]" in out and "4242" not in out
    assert n == 1


def test_invalid_card_not_redacted():
    # Fails Luhn — must NOT be redacted as a card (false-positive resistance).
    out, n = redact_pii("order id 1234 5678 9012 3456 shipped")
    assert "[PII:CARD]" not in out


def test_phone_redacted():
    out, n = redact_pii("call me at +1 415-555-0132 tomorrow")
    assert "[PII:PHONE]" in out


def test_multiple_and_count():
    out, n = redact_pii("email bob@x.io, ssn 111-22-3333, ip 10.0.0.1")
    assert n >= 3
    assert "bob@x.io" not in out and "111-22-3333" not in out


def test_clean_text_untouched():
    text = "The quarterly revenue grew 12 percent in Q4 across three regions."
    out, n = redact_pii(text)
    assert out == text and n == 0


def test_empty_and_nonstring_safe():
    assert redact_pii("") == ("", 0)
    assert redact_pii(None) == (None, 0)


def test_year_and_small_numbers_not_phone():
    # Bare 4-digit year / small counts should not trip the phone matcher.
    out, n = redact_pii("in 2026 we shipped 42 features")
    assert "[PII:PHONE]" not in out


def test_iso_dates_and_versions_not_phone():
    # Critical false-positive guard: ISO dates (4-2-2) and version/commit
    # strings must never be mistaken for phone numbers in memory content.
    for text in ["deployed on 2026-07-22 at noon",
                 "version 3.7.9 shipped 2026-07-22 12 builds",
                 "range 9-11 am then 2-4 pm"]:
        out, n = redact_pii(text)
        assert "[PII:PHONE]" not in out, text
        assert n == 0, text
