# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
#
# ATTRIBUTION: PricingRegistry + stale-detection pattern adapted from:
#   headroom/pricing/registry.py (Apache-2.0)

"""Savings estimator — converts MetricsSnapshot to dollar/rupee savings."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

from superlocalmemory.optimize.storage.db import MetricsSnapshot

logger = logging.getLogger("superlocalmemory.optimize.metrics.estimator")

# DATED: 2026-06-07 — prices verified from official pages on this date.
BUILTIN_PRICING_TABLE: dict[str, dict[str, float]] = {
    "anthropic": {"input_per_1m_usd": 3.00, "output_per_1m_usd": 15.00},
    "openai":    {"input_per_1m_usd": 2.50, "output_per_1m_usd": 10.00},
    "gemini":    {"input_per_1m_usd": 1.25, "output_per_1m_usd": 10.00},
}

_PRICING_STALE_DAYS: int = 90


class SavingsEstimator:
    """Converts MetricsSnapshot to dollar/rupee savings. Daemon-scoped singleton.

    Pricing (June 2026 — dated defaults, user-configurable):
        anthropic: $3.00/M input tokens
        openai:    $2.50/M input tokens
        gemini:    $1.25/M input tokens

    CRITICAL pricing rule (INTERFACE-CONTRACT §7):
        Cache SKIP saves BOTH input+output tokens (whole call avoided).
        Compression saves INPUT tokens ONLY (output tokens not compressed).
        NEVER apply output-token pricing to compression savings.

    INR conversion: 83.5 (hardcoded, configurable).
    Stale detection: if pricing data is > 90 days old → include is_stale=True in output.
    """

    INR_RATE: float = 83.5
    _PRICING_DATE: str = "2026-06-07"

    def estimate(self, snap: MetricsSnapshot, provider: str = "anthropic") -> dict[str, Any]:
        """Compute savings from a MetricsSnapshot.

        Returns:
            dict with keys: usd, inr, tokens_saved_total, cache_tokens,
            compress_tokens, is_stale, pricing_date
        """
        provider_key = provider if provider in BUILTIN_PRICING_TABLE else "anthropic"
        rates = BUILTIN_PRICING_TABLE[provider_key]
        input_rate = rates["input_per_1m_usd"]
        output_rate = rates["output_per_1m_usd"]

        cache_tokens = snap.tokens_saved_input + snap.tokens_saved_output
        compress_tokens = snap.tokens_saved_compress
        tokens_saved_total = cache_tokens + compress_tokens

        # Cache skip: input + output tokens saved (whole call avoided)
        savings_cache_usd = (snap.tokens_saved_input / 1_000_000) * input_rate + (snap.tokens_saved_output / 1_000_000) * output_rate
        # Compression: input tokens only
        savings_compress_usd = (compress_tokens / 1_000_000) * input_rate
        total_usd = savings_cache_usd + savings_compress_usd

        is_stale = self._is_stale()

        return {
            "usd": round(total_usd, 6),
            "inr": round(total_usd * self.INR_RATE, 4),
            "tokens_saved_total": tokens_saved_total,
            "cache_tokens": cache_tokens,
            "compress_tokens": compress_tokens,
            "is_stale": is_stale,
            "pricing_date": self._PRICING_DATE,
        }

    def _is_stale(self) -> bool:
        """Return True if pricing data is older than _PRICING_STALE_DAYS."""
        try:
            table_date = date.fromisoformat(self._PRICING_DATE)
            return (date.today() - table_date) > timedelta(days=_PRICING_STALE_DAYS)
        except ValueError:
            return True
