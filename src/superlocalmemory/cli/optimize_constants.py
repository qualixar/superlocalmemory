# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Constants shared by Optimize CLI handlers.

No imports from optimize.* at module load time (lazy import in functions only).
"""

from __future__ import annotations

from typing import Final

OPTIMIZE_DEFAULT_PORT: Final[int] = 8765

# Cache skip saves BOTH input+output tokens (whole call avoided).
# For CLI display we show the conservative input-price lower bound.
DEFAULT_COST_PER_MILLION_INPUT_TOKENS: Final[dict[str, float]] = {
    "anthropic": 3.00,
    "openai":    2.50,
    "gemini":    1.25,
    "default":   3.00,
}
_PRICING_DATE: Final[str] = "2026-06-07"

AGGRESSIVE_MODE_WARNING: Final[str] = (
    "WARNING: Aggressive mode may reduce output fidelity.\n"
    "  Do NOT use for: code generation, legal text, exact-output tasks, math.\n"
    "  Safe for: summarization, brainstorming, open-ended chat.\n"
    "  To revert: slm compress mode safe"
)
