# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3.3

"""PII Filter — Stateless PII detection and redaction for soft prompts.

Ensures no personally identifiable information leaks into generated prompts.
Patterns cover: email, phone, SSN, credit card, IP address, API keys.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PII regex patterns
# ---------------------------------------------------------------------------

# Order matters: longer/more-specific patterns first to prevent partial matches.
# Credit card (16 digits) must come before phone (7-12 digits).
# SSN (XXX-XX-XXXX) must come before phone to avoid partial match.
PII_PATTERNS: dict[str, re.Pattern] = {
    "api_key": re.compile(
        r"\b(?:sk-|pk-|api[_-]?key[_-]?)[A-Za-z0-9_-]{20,}\b", re.IGNORECASE
    ),
    "email": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"
    ),
    "credit_card": re.compile(
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
    ),
    "ssn": re.compile(
        r"\b\d{3}-\d{2}-\d{4}\b"
    ),
    "ip_address": re.compile(
        r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
    ),
    "phone": re.compile(
        r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"
    ),
}


# ---------------------------------------------------------------------------
# PIIFilter class
# ---------------------------------------------------------------------------

class PIIFilter:
    """Stateless PII detection and redaction.

    Usage:
        pii = PIIFilter()
        clean = pii.filter_text("email user@test.com")
        # -> "email [REDACTED:email]"
    """

    def __init__(self) -> None:
        """Stateless — no initialization needed."""

    def filter_text(self, text: str) -> str:
        """Replace all PII matches with [REDACTED:type] placeholders.

        Args:
            text: Input text potentially containing PII.

        Returns:
            Text with all detected PII replaced by redaction markers.
        """
        result = text
        for pii_type, pattern in PII_PATTERNS.items():
            result = pattern.sub(f"[REDACTED:{pii_type}]", result)
        return result

    def has_pii(self, text: str) -> bool:
        """Check whether text contains any PII.

        Args:
            text: Input text to scan.

        Returns:
            True if any PII pattern matches.
        """
        for pattern in PII_PATTERNS.values():
            if pattern.search(text):
                return True
        return False

    def detect_pii_types(self, text: str) -> list[str]:
        """Detect which PII types are present in text.

        Args:
            text: Input text to scan.

        Returns:
            List of PII type names found (e.g., ["email", "phone"]).
        """
        found: list[str] = []
        for pii_type, pattern in PII_PATTERNS.items():
            if pattern.search(text):
                found.append(pii_type)
        return found
