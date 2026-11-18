# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Prepare text for a durable write.

Two kinds of text reach a write door:

* ``USER_TEXT`` is what a person typed or imported. It is stored as written;
  the only changes are personal-data redaction, when the operator turned it on,
  and the secret screen (``trust.secret_screen``, WS-5), which replaces a
  credential with a typed placeholder. With both off the bytes pass through
  untouched.
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
    #: Credentials the secret screen replaced, by type (counts only; ``()`` when it did not run).
    secret_types: tuple[tuple[str, int], ...] = ()


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


#: The longest text ``scan_sensitive`` will read. Past it the scan refuses, and
#: the caller treats the text as not scanned (never as clean).
MAX_SCAN_CHARS = 1_000_000


@dataclass(frozen=True)
class SensitiveScan:
    """How much personal data and how many credentials a text holds."""

    pii: int
    secrets: int

    @property
    def clean(self) -> bool:
        return self.pii == 0 and self.secrets == 0


def scan_sensitive(text: str) -> SensitiveScan:
    """Count personal identifiers and credentials in ``text``; change nothing.

    For vetting only: the count does not depend on the redaction setting, and
    it reads the whole text, not the part that will be stored. Raises
    ``ValueError`` when the text cannot be scanned in full (not a string, or
    longer than ``MAX_SCAN_CHARS``): "not scanned" must never read as "clean".
    """
    if not isinstance(text, str):
        raise ValueError("scan_sensitive expects str")
    if len(text) > MAX_SCAN_CHARS:
        raise ValueError("text is too long to scan in full")
    if not text:
        return SensitiveScan(0, 0)
    pii = redact_pii(text)[1]
    before = len(_SECRET_MARKER.findall(text))
    redacted = redact_secrets(text, aggression="high")
    return SensitiveScan(pii, max(0, len(_SECRET_MARKER.findall(redacted)) - before))


def _screen_user_text(text: str) -> tuple[str, tuple[tuple[str, int], ...]]:
    """The secret screen for a person's or an agent's own text (``trust.secret_screen``)."""
    from superlocalmemory.core.feature_flags import FlagState
    from superlocalmemory.trust import secret_screen

    mode = secret_screen.state()
    if mode is FlagState.OFF:
        return text, ()
    got = secret_screen.screen(text)
    secret_screen.note(got.counts, acting=mode is FlagState.ON)
    if mode is FlagState.ON:
        return got.text, tuple(sorted(got.counts.items()))
    return text, ()


def prepare_for_save(
    text: str, *, origin: ContentOrigin, pii_redaction: bool,
) -> PreparedContent:
    """Return the text to store, with how much was redacted."""
    if not isinstance(text, str):
        raise TypeError("prepare_for_save expects str")
    pii_count = 0
    secret_count = 0
    secret_types: tuple[tuple[str, int], ...] = ()
    if text:
        if origin is ContentOrigin.DERIVED_TEXT:
            text = unicodedata.normalize("NFC", text)
            before = len(_SECRET_MARKER.findall(text))
            text = redact_secrets(text, aggression="high")
            secret_count = max(0, len(_SECRET_MARKER.findall(text)) - before)
        else:
            text, secret_types = _screen_user_text(text)
            secret_count = sum(n for _kind, n in secret_types)
        if pii_redaction:
            text, pii_count = redact_pii(text)
    logger.debug("prepared content: pii=%d secrets=%d", pii_count, secret_count)
    return PreparedContent(text, pii_count, secret_count, _digest(text), secret_types)


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

    Non-string leaves are returned as they are. With personal-data redaction off and the
    secret screen off, the input comes back as is; with the screen on, credentials in any
    string become typed placeholders first (``trust.secret_screen``).
    """
    value = _screen_metadata(value)
    if not pii_redaction:
        return value, 0
    return _redact_tree(value)


def _redact_tree(value: Any) -> tuple[Any, int]:
    if isinstance(value, str):
        return redact_pii(value)
    if isinstance(value, dict):
        out: dict = {}
        total = 0
        for key, item in value.items():
            new_key, k_count = _redact_tree(key)
            new_item, v_count = _redact_tree(item)
            out[new_key] = new_item
            total += k_count + v_count
        return out, total
    if isinstance(value, (list, tuple)):
        pairs = [_redact_tree(item) for item in value]
        items = [p[0] for p in pairs]
        return (tuple(items) if isinstance(value, tuple) else items), sum(p[1] for p in pairs)
    return value, 0


def _screen_metadata(value: Any) -> Any:
    from superlocalmemory.core.feature_flags import FlagState
    from superlocalmemory.trust import secret_screen

    if secret_screen.state() is not FlagState.ON:
        return value
    screened, counts = secret_screen.screen_tree(value)
    secret_screen.note(counts, acting=True)
    return screened


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
