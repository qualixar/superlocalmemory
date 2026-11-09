# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Tests for PII filter (Phase F: The Learning Brain).

TDD RED phase: tests written before implementation.
3 tests per LLD Section 6.3.
"""

from __future__ import annotations

from superlocalmemory.parameterization.pii_filter import PIIFilter


# ---------------------------------------------------------------
# T14: Email redaction
# ---------------------------------------------------------------
def test_email_redaction():
    """Given text with an email, filter_text replaces it with [REDACTED:email]."""
    pii = PIIFilter()
    text = "Contact user@example.com for details"
    result = pii.filter_text(text)
    assert "[REDACTED:email]" in result
    assert "user@example.com" not in result


# ---------------------------------------------------------------
# T15: Phone redaction
# ---------------------------------------------------------------
def test_phone_redaction():
    """Given text with a phone number, filter_text replaces it with [REDACTED:phone]."""
    pii = PIIFilter()
    text = "Call +1-555-123-4567 for support"
    result = pii.filter_text(text)
    assert "[REDACTED:phone]" in result
    assert "555-123-4567" not in result


# ---------------------------------------------------------------
# T16: API key redaction
# ---------------------------------------------------------------
def test_api_key_redaction():
    """Given text with an API key, filter_text replaces it with [REDACTED:api_key]."""
    pii = PIIFilter()
    text = "Use sk-abc123def456ghi789jkl012mno for auth"
    result = pii.filter_text(text)
    assert "[REDACTED:api_key]" in result
    assert "sk-abc123def456ghi789jkl012mno" not in result


# ---------------------------------------------------------------
# Additional: has_pii and detect_pii_types
# ---------------------------------------------------------------
def test_has_pii_true():
    pii = PIIFilter()
    assert pii.has_pii("email is user@test.org") is True


def test_has_pii_false():
    pii = PIIFilter()
    assert pii.has_pii("no personal info here") is False


def test_detect_pii_types():
    pii = PIIFilter()
    types = pii.detect_pii_types("email user@test.org call 555-123-4567")
    assert "email" in types
    assert "phone" in types


def test_ssn_redaction():
    pii = PIIFilter()
    text = "SSN: 123-45-6789"
    result = pii.filter_text(text)
    assert "[REDACTED:ssn]" in result
    assert "123-45-6789" not in result


def test_credit_card_redaction():
    pii = PIIFilter()
    text = "Card: 4111 1111 1111 1111"
    result = pii.filter_text(text)
    assert "[REDACTED:credit_card]" in result


def test_clean_text_unchanged():
    pii = PIIFilter()
    text = "The user prefers TypeScript and React."
    assert pii.filter_text(text) == text
