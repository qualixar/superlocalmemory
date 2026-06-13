# compress/prose_llmlingua.py
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
#
# LLMLingua-2 paper: Microsoft Research + MIT (2024).
# Model: XLM-RoBERTa fine-tuned as token binary classifier.
# License: MIT (github.com/microsoft/LLMLingua)

"""LLMLinguaCompressor — opt-in LLMLingua-2 prose compressor.

SAFETY RULES (NON-NEGOTIABLE):
  1. ONLY called for prose/narrative content — NEVER JSON or code.
  2. compress_mode MUST be "aggressive" in optimize.json.
  3. Lossy — CCR stores original before this runs.
  4. Import errors caught — returns original on failure.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("slm.optimize.compress.llmlingua")

_DEFAULT_RATE: float = 0.5
_MODEL_BERT: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
_MODEL_XLM: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"


class LLMLinguaCompressor:
    """Opt-in LLMLingua-2 prose compressor."""

    def __init__(
        self,
        model_name: str = _MODEL_XLM,
        device_map: str = "cpu",
        rate: float = _DEFAULT_RATE,
    ) -> None:
        try:
            from llmlingua import PromptCompressor  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "llmlingua package not installed. Install: pip install llmlingua."
            ) from e

        logger.info("Loading LLMLingua-2 model=%s device=%s", model_name, device_map)
        self._compressor: Any = PromptCompressor(
            model_name=model_name,
            use_llmlingua2=True,
            device_map=device_map,
        )
        self._rate = rate
        logger.info("LLMLingua-2 loaded successfully model=%s", model_name)

    def compress(self, text: str, rate: float | None = None) -> str:
        """Compress prose text using LLMLingua-2. Fail-open: returns original on error."""
        effective_rate = rate if rate is not None else self._rate
        try:
            result = self._compressor.compress_prompt(
                [text],
                rate=effective_rate,
                use_token_level_filter=True,
            )
            compressed = result.get("compressed_prompt", text)
            if not isinstance(compressed, str) or not compressed:
                logger.warning("LLMLingua-2 returned unexpected output — passthrough")
                return text
            return compressed
        except Exception as exc:
            logger.warning("LLMLingua-2 compress failed — passthrough: %s", exc)
            return text
