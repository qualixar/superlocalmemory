# optimize/cache/context_key.py
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
#
# LLD-03 §4.1 — Multi-turn context-aware key builder for the semantic cache.
# Source: ContextCache / SmartCache (arXiv:2506.22791 §3) — context-aware
# cache keys prevent false reuse across semantically overlapping but
# conversationally distinct turns.
#
# The fingerprint is NOT the full cache key (that is KeyBuilder's job).
# It is an auxiliary scope guard: a semantic hit is accepted ONLY IF
# the stored entry's context fingerprint matches the query's context
# fingerprint (or is absent — single-turn entries).
#
# A-22 fix: 16-hex-char fingerprint (64 bits → birthday at 2^32 entries,
# safe for any realistic tenant). 8 hex chars (32 bits) birthday at 2^16.

from __future__ import annotations

import hashlib
import json
from typing import Any

_CONTEXT_SCHEMA_VERSION: int = 1


class ContextKeyBuilder:
    """Builds a context fingerprint for multi-turn semantic cache lookup.

    Args:
        window_turns: Number of prior assistant+user turns to include.
                      Default: 3 (matches LLD-03 §3.2 default).
    """

    def __init__(self, window_turns: int = 3) -> None:
        if window_turns < 1:
            raise ValueError(f"window_turns must be >= 1, got {window_turns}")
        self._window = window_turns

    def build(self, messages: list[dict[str, Any]], tenant_id: str) -> str:
        """Build a 16-hex-char (64-bit) context fingerprint.

        Takes the last `window_turns * 2` messages (user + assistant pairs),
        canonicalizes them, SHA-256 hashes them, returns first 16 hex chars.

        Returns:
            16-hex context fingerprint, e.g. "a3f2b1c0d4e5f601". Never empty.
            Single-turn (no prior context) returns the SHA-256 of an empty
            canonical context array — a stable sentinel.
        """
        prior = messages[:-1] if len(messages) > 1 else []
        window = prior[-(self._window * 2):]

        payload = {
            "v": _CONTEXT_SCHEMA_VERSION,
            "tenant": tenant_id,
            "ctx": window,
        }
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def turn_count(self, messages: list[dict[str, Any]]) -> int:
        """Return the number of completed conversation turns (assistant messages)."""
        return sum(1 for m in messages if m.get("role") == "assistant")
