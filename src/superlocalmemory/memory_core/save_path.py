# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Prepare text for a durable write.

Two kinds of text reach a write door:

* ``USER_TEXT`` is what a person typed or imported. It is stored as written;
  the only change is personal-data redaction, and only when the operator turned
  it on. With redaction off the bytes pass through untouched.
* ``DERIVED_TEXT`` is text SLM produced (extracted, summarised, transcribed).
  It is Unicode-normalised and always has credentials stripped.

Pure: no I/O. Content is never logged, only counts, at DEBUG.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum

from superlocalmemory.core.pii import redact_pii
from superlocalmemory.core.security_primitives import redact_secrets

logger = logging.getLogger(__name__)

_ENV_ON = ("1", "on", "true", "yes")
_SECRET_MARKER = re.compile(r"\[REDACTED:")


class ContentOrigin(str, Enum):
    USER_TEXT = "user_text"
    DERIVED_TEXT = "derived_text"


@dataclass(frozen=True)
class PreparedContent:
    text: str
    pii_count: int
    secret_count: int
    content_sha256: str


def pii_redaction_enabled(config: object | None) -> bool:
    """On when the config sets ``pii_redaction`` or ``SLM_PII_REDACTION`` is set.

    Default off: personal use is unchanged; team operators opt in.
    """
    if config is not None and getattr(config, "pii_redaction", False):
        return True
    return os.environ.get("SLM_PII_REDACTION", "").strip().lower() in _ENV_ON


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare_for_save(
    text: str, *, origin: ContentOrigin, pii_redaction: bool,
) -> PreparedContent:
    """Return the text to store, with how much was redacted."""
    if not isinstance(text, str):
        raise TypeError("prepare_for_save expects str")
    pii_count = 0
    secret_count = 0
    if text:
        if origin is ContentOrigin.DERIVED_TEXT:
            text = unicodedata.normalize("NFC", text)
            before = len(_SECRET_MARKER.findall(text))
            text = redact_secrets(text, aggression="high")
            secret_count = max(0, len(_SECRET_MARKER.findall(text)) - before)
        if pii_redaction:
            text, pii_count = redact_pii(text)
    logger.debug("prepared content: pii=%d secrets=%d", pii_count, secret_count)
    return PreparedContent(text, pii_count, secret_count, _digest(text))
