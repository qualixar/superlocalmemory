# compress/align.py
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
#
# Volatile-detection algorithm adapted from:
#   headroom/transforms/cache_aligner.py (Apache-2.0, Headroom contributors)
#   Specifically: _is_uuid(), _is_iso8601(), _is_jwt_shape(), _is_hex_hash(),
#   _classify_token(), _split_tokens(), detect_volatile_content()
#   Lines: cache_aligner.py:76-200
#   Attribution: See ATTRIBUTION.md.

"""CacheAligner — volatile-token detector for system prompt prefix stability.

Phase 2: Detection only. No mutation of the system prompt.
"""

from __future__ import annotations

import base64
import binascii
import logging
import uuid as _uuid
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("slm.optimize.compress.align")

_HEX_HASH_LENGTHS = frozenset({32, 40, 64})
_UUID_CANONICAL_LEN = 36  # L-02: 36 chars INCLUDING 4 dashes (RFC 4122 canonical form: 8-4-4-4-12)
_JWT_SEGMENT_COUNT = 3
_JWT_MIN_SEGMENT_BYTES = 4

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
    volatile_count = 0

    for token in tokens:
        label = _classify_token(token)
        if label is not None:
            volatile_count += 1
            if len(findings) < 20:
                findings.append(VolatileFinding(label=label, sample=token[:20]))

    total = len(tokens)
    score = 1.0 - (volatile_count / total) if total > 0 else 1.0

    return AlignResult(
        prefix_stable=(volatile_count == 0),
        stability_score=round(score, 4),
        findings=findings,
        total_tokens_scanned=total,
    )


def _split_tokens(content: str) -> list[str]:
    if not content:
        return []
    tokens: list[str] = []
    for raw in content.split():
        cleaned = raw.strip(".,;:!?\"'()[]{}<>`|\\")
        if cleaned:
            tokens.append(cleaned)
    return tokens


def _classify_token(token: str) -> str | None:
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
    candidate = token[:-1] + "+00:00" if token.endswith("Z") else token
    try:
        datetime.fromisoformat(candidate)
    except (ValueError, TypeError):
        return False
    return True


def _is_jwt_shape(token: str) -> bool:
    if token.count(".") != _JWT_SEGMENT_COUNT - 1:
        return False
    segments = token.split(".")
    for seg in segments:
        if len(seg) < _JWT_MIN_SEGMENT_BYTES:
            return False
        padded = seg + "=" * (-len(seg) % 4)
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
