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
from typing import Any

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


def effective_pii_redaction(config: object | None) -> bool:
    """Like ``pii_redaction_enabled``, plus the deployment policy.

    The daemon upgrades its engine config from the deployment file at start-up;
    a process that loads a plain config (an MCP server, a CLI) has to ask the
    deployment file itself. Unreadable policy counts as not set.
    """
    if pii_redaction_enabled(config):
        return True
    try:
        from superlocalmemory.core.config import load_deployment_config

        return bool(load_deployment_config().pii_redaction)
    except Exception:  # noqa: BLE001 - an unreadable policy never blocks a read
        return False


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


def prepare_user_text(config: object | None, text: str) -> PreparedContent:
    """Prepare user text for the engine ``config``; logs only the count."""
    prepared = prepare_for_save(
        text, origin=ContentOrigin.USER_TEXT,
        pii_redaction=pii_redaction_enabled(config),
    )
    if prepared.pii_count:
        logger.info(
            "PII redaction: scrubbed %d identifier(s) before save", prepared.pii_count,
        )
    return prepared


def prepare_metadata(value: Any, *, pii_redaction: bool) -> tuple[Any, int]:
    """Redact every string in dicts, lists and tuples (keys too); never mutates.

    Non-string leaves are returned as they are. Off, the input comes back as is.
    """
    if not pii_redaction:
        return value, 0
    if isinstance(value, str):
        return redact_pii(value)
    if isinstance(value, dict):
        out: dict = {}
        total = 0
        for key, item in value.items():
            new_key, k_count = prepare_metadata(key, pii_redaction=True)
            new_item, v_count = prepare_metadata(item, pii_redaction=True)
            out[new_key] = new_item
            total += k_count + v_count
        return out, total
    if isinstance(value, (list, tuple)):
        pairs = [prepare_metadata(item, pii_redaction=True) for item in value]
        items = [p[0] for p in pairs]
        return (tuple(items) if isinstance(value, tuple) else items), sum(p[1] for p in pairs)
    return value, 0


def prepare_key(key: str, *, pii_redaction: bool) -> str:
    """Return a dedup key that holds no personal data.

    A key that redaction would change is replaced by a hash of itself, so the
    same raw key still maps to the same stored key; other keys are kept.
    """
    if pii_redaction and isinstance(key, str) and redact_pii(key)[1]:
        return "redacted:" + _digest(key)
    return key


def same_after_redaction(existing_raw: str, prepared_text: str, pii_redaction: bool) -> bool:
    """True when stored raw text, once prepared, equals ``prepared_text``."""
    if not pii_redaction or not isinstance(existing_raw, str):
        return False
    return prepare_for_save(
        existing_raw, origin=ContentOrigin.USER_TEXT, pii_redaction=True,
    ).text == prepared_text
