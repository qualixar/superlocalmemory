# compress/extractive_json.py
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
#
# Structural masking pattern adapted from:
#   headroom/compression/handlers/json_handler.py (Apache-2.0, Headroom contributors)
#   Specifically: JSONStructureHandler._extract_mask(), JSONToken, JSONTokenType
#   Attribution: See ATTRIBUTION.md.
#
# HARD RULE: This compressor MUST NEVER prune or reorder JSON keys.
# Keys pruned = structured semantics corrupted non-recoverably. Not configurable.

"""JSONCompressor — lossless-ish extractive JSON compressor."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("slm.optimize.compress.json")

VALUE_TRUNCATE_CHARS: int = 120
VALUE_TRUNCATE_SUFFIX: str = "\u2026"  # ellipsis
MAX_ARRAY_ITEMS_SHOWN: int = 5
ARRAY_REMAINDER_KEY: str = "__slm_omitted__"
MIN_VALUE_LEN_TO_TRUNCATE: int = 40


class JSONCompressor:
    """Lossless-ish JSON compressor. Thread-safe (no mutable state)."""

    def compress(self, parsed: Any) -> str:
        try:
            masked = self._mask(parsed, depth=0)
            return json.dumps(masked, ensure_ascii=False, separators=(",", ":"))
        except Exception as exc:
            logger.warning("JSONCompressor.compress failed — returning original: %s", exc)
            return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))

    def _mask(self, obj: Any, depth: int) -> Any:
        if isinstance(obj, dict):
            return {k: self._mask(v, depth + 1) for k, v in obj.items()}
        if isinstance(obj, list):
            return self._mask_array(obj, depth)
        if isinstance(obj, str):
            return self._mask_string(obj)
        return obj

    def _mask_array(self, arr: list, depth: int) -> list:
        if len(arr) <= MAX_ARRAY_ITEMS_SHOWN:
            return [self._mask(item, depth + 1) for item in arr]
        shown = [self._mask(item, depth + 1) for item in arr[:MAX_ARRAY_ITEMS_SHOWN]]
        collision = any(
            isinstance(item, dict) and ARRAY_REMAINDER_KEY in item
            for item in arr
        )
        if collision:
            logger.warning(
                "JSONCompressor: input contains reserved key %r — skipping array sentinel",
                ARRAY_REMAINDER_KEY,
            )
        else:
            shown.append({ARRAY_REMAINDER_KEY: len(arr) - MAX_ARRAY_ITEMS_SHOWN})
        return shown

    def _mask_string(self, s: str) -> str:
        if len(s) < MIN_VALUE_LEN_TO_TRUNCATE:
            return s
        if len(s) <= VALUE_TRUNCATE_CHARS:
            return s
        return s[:VALUE_TRUNCATE_CHARS] + VALUE_TRUNCATE_SUFFIX
