# compress/align.py
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""CacheAligner — flags volatile tokens in a system prompt.

A "volatile" token is one that differs between otherwise-identical requests —
a UUID, an ISO-8601 timestamp, a JWT, or a hex digest. When such tokens sit in
the stable prefix of a prompt they break provider prefix caching, so the router
surfaces them. This module only *detects*; it never rewrites the prompt.
"""

from __future__ import annotations

import base64
import binascii
import logging
import uuid as _uuid
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("slm.optimize.compress.align")

_HEX_HASH_LENGTHS = frozenset({32, 40, 64})   # md5 / sha1 / sha256 hex widths
_UUID_CANONICAL_LEN = 36                       # 8-4-4-4-12, dashes included
_JWT_SEGMENTS = 3                              # header.payload.signature
_JWT_MIN_SEGMENT_BYTES = 4
_MAX_FINDINGS = 20                             # cap the sample list we retain
_SAMPLE_LEN = 20                               # chars kept per finding sample
_TOKEN_STRIP = ".,;:!?\"'()[]{}<>`|\\"         # trimmed off each raw token

_LABEL_UUID = "uuid"
_LABEL_ISO8601 = "iso8601"
_LABEL_JWT = "jwt"
_LABEL_HEX_HASH = "hex_hash"


@dataclass(frozen=True)
class VolatileFinding:
    label: str
    sample: str


@dataclass
class AlignResult:
    prefix_stable: bool = True
    stability_score: float = 1.0
    findings: list[VolatileFinding] = field(default_factory=list)
    total_tokens_scanned: int = 0


class CacheAligner:
    """Detects volatile tokens in system prompts. No mutation. Thread-safe."""

    def detect(self, system_prompt: str) -> AlignResult:
        try:
            return _detect(system_prompt)
        except Exception as exc:
            logger.debug("CacheAligner.detect failed (non-fatal): %s", exc)
            return AlignResult()


def _detect(text: str) -> AlignResult:
    tokens = _split_tokens(text)
    findings: list[VolatileFinding] = []
    volatile = 0
    for token in tokens:
        label = _classify_token(token)
        if label is None:
            continue
        volatile += 1
        if len(findings) < _MAX_FINDINGS:
            findings.append(VolatileFinding(label=label, sample=token[:_SAMPLE_LEN]))

    total = len(tokens)
    score = round(1.0 - volatile / total, 4) if total else 1.0
    return AlignResult(
        prefix_stable=(volatile == 0),
        stability_score=score,
        findings=findings,
        total_tokens_scanned=total,
    )


def _split_tokens(content: str) -> list[str]:
    """Whitespace-split, then strip surrounding punctuation/quoting from each
    token so a value wrapped in backticks or brackets is still recognised."""
    if not content:
        return []
    tokens: list[str] = []
    for raw in content.split():
        cleaned = raw.strip(_TOKEN_STRIP)
        if cleaned:
            tokens.append(cleaned)
    return tokens


def _classify_token(token: str) -> str | None:
    """Return the volatile-kind label for a token, or None when it is stable.

    Ordered most-specific first: a canonical-length UUID before a dotted JWT,
    then timestamps, then bare hex digests.
    """
    if _is_uuid(token):
        return _LABEL_UUID
    if "." in token and _is_jwt_shape(token):
        return _LABEL_JWT
    if _is_iso8601(token):
        return _LABEL_ISO8601
    if _is_hex_hash(token):
        return _LABEL_HEX_HASH
    return None


def _is_uuid(token: str) -> bool:
    if len(token) != _UUID_CANONICAL_LEN or token.count("-") != 4:
        return False
    try:
        _uuid.UUID(token)
    except (ValueError, AttributeError):
        return False
    return True


def _is_iso8601(token: str) -> bool:
    if len(token) < 8 or ("T" not in token and "-" not in token):
        return False
    # datetime.fromisoformat pre-3.11 rejects a trailing 'Z'; normalise it.
    candidate = token[:-1] + "+00:00" if token.endswith("Z") else token
    try:
        datetime.fromisoformat(candidate)
    except (ValueError, TypeError):
        return False
    return True


def _is_jwt_shape(token: str) -> bool:
    segments = token.split(".")
    if len(segments) != _JWT_SEGMENTS:
        return False
    for seg in segments:
        if len(seg) < _JWT_MIN_SEGMENT_BYTES:
            return False
        padded = seg + "=" * (-len(seg) % 4)  # restore base64url padding
        try:
            base64.urlsafe_b64decode(padded.encode("ascii"))
        except (binascii.Error, ValueError, UnicodeEncodeError):
            return False
    return True


def _is_hex_hash(token: str) -> bool:
    if len(token) not in _HEX_HASH_LENGTHS:
        return False
    try:
        int(token, 16)
    except ValueError:
        return False
    return True
