# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""What a cached derivation is keyed by: the content, who derived it, and how."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

_SHA = re.compile(r"[0-9a-f]{64}")
_DERIVER = re.compile(r"[a-z0-9._-]{1,64}")
_MAX_FIELD = 64


@dataclass(frozen=True)
class CacheKey:
    """The full identity of one derived payload."""

    content_sha256: str
    deriver_id: str
    deriver_version: str
    model_id: str = ""
    params_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.content_sha256, str) or not _SHA.fullmatch(self.content_sha256):
            raise ValueError("content_sha256 must be 64 lowercase hex characters")
        if not isinstance(self.deriver_id, str) or not _DERIVER.fullmatch(self.deriver_id):
            raise ValueError("deriver_id must match [a-z0-9._-]{1,64}")
        if not isinstance(self.deriver_version, str) or not (
                0 < len(self.deriver_version) <= _MAX_FIELD):
            raise ValueError("deriver_version must be 1 to 64 characters")
        for name in ("model_id", "params_hash"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string")

    def as_tuple(self) -> tuple[str, str, str, str, str]:
        return (self.content_sha256, self.deriver_id, self.deriver_version,
                self.model_id, self.params_hash)


def params_hash(params: Mapping[str, Any]) -> str:
    """A stable hash of JSON-safe parameters, independent of key order.

    A deriver that caches derived text must include the personal-data redaction
    policy in its parameters (for example ``{"redaction": "on"}`` or
    ``{"redaction": "off"}``), so text derived without redaction is never served
    once redaction is on.
    """
    try:
        text = json.dumps(dict(params), sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cache parameters must be JSON-safe: {exc}") from exc
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
