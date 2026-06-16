# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""_savings_math.py — PURE math module for WP-14 metered-pipeline $-savings dogfood.

Zero I/O. No network. No side effects.

Prices used here come from R6 (2026-06-16):
  Anthropic Sonnet 4.6: $3.00/MTok input, $15.00/MTok output — VERIFIED
  Anthropic Opus 4.8: $5.00/MTok input, $25.00/MTok output — VERIFIED
  Anthropic Haiku 4.5: $1.00/MTok input, $5.00/MTok output — VERIFIED
  Cache-read multiplier: 0.10x (90% off cached portion) — VERIFIED
  Source: https://platform.claude.com/docs/en/about-claude/pricing (fetched 2026-06-16)

  OpenAI prices: UNVERIFIED-OFFICIAL (openai.com/api/pricing unreachable at fetch time;
  data from aipricing.guru aggregator). Do not cite as authoritative.

CRIT notes (from LLD §9):
  1. Blended input-price is a conservative lower-bound (output tokens cost more).
  2. tokens_saved_compress is a word-count proxy (not a real tokenizer).
  3. chars//4 is used in sim for input tokens (char-exact by construction).
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=False)
class MetricsDelta:
    """Difference between two MetricsSnapshots, clamped to ≥0.

    Negative values indicate the DB was reset or shared between runs —
    clamped to 0 (conservative, never overclaim).
    """

    hits: int
    misses: int
    tokens_saved_input: int
    tokens_saved_output: int
    tokens_saved_compress: int

    def __post_init__(self) -> None:
        # Clamp all fields to ≥ 0 (immutable-safe: assign back to self)
        self.hits = max(0, self.hits)
        self.misses = max(0, self.misses)
        self.tokens_saved_input = max(0, self.tokens_saved_input)
        self.tokens_saved_output = max(0, self.tokens_saved_output)
        self.tokens_saved_compress = max(0, self.tokens_saved_compress)

    @property
    def total_tokens_saved(self) -> int:
        return (
            self.tokens_saved_input
            + self.tokens_saved_output
            + self.tokens_saved_compress
        )

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


@dataclasses.dataclass(frozen=True)
class Report:
    """Immutable dogfood savings report."""

    mode: str                    # "sim" or "live"
    provider: str
    prompts: int
    repeat: int
    delta: MetricsDelta
    price_per_1m: float
    price_source: str            # e.g. "DEFAULT_COST_PER_MILLION_INPUT_TOKENS[anthropic], 2026-06-07 (R6)"
    savings_usd: float
    savings_inr: float
    inr_rate: float

    def to_json(self) -> str:
        """Serialize to raw JSON ledger format per LLD §5."""
        d = self.delta
        formula_str = (
            f"({d.tokens_saved_input} + {d.tokens_saved_output} + "
            f"{d.tokens_saved_compress}) / 1_000_000 * {self.price_per_1m}"
        )
        payload: dict[str, Any] = {
            "mode": self.mode,
            "provider": self.provider,
            "workload": {
                "prompts": self.prompts,
                "repeat": self.repeat,
                "total_requests": self.prompts * (1 + self.repeat),
            },
            "delta": {
                "hits": d.hits,
                "misses": d.misses,
                "tokens_saved_input": d.tokens_saved_input,
                "tokens_saved_output": d.tokens_saved_output,
                "tokens_saved_compress": d.tokens_saved_compress,
                "total_tokens_saved": d.total_tokens_saved,
                "hit_rate": round(d.hit_rate, 4),
            },
            "pricing": {
                "price_per_1m": self.price_per_1m,
                "source": self.price_source,
                "date": "2026-06-07",
                "inr_rate": self.inr_rate,
            },
            "computed": {
                "savings_usd": round(self.savings_usd, 6),
                "savings_inr": round(self.savings_inr, 4),
                "formula": formula_str,
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        return json.dumps(payload, indent=2)

    def to_markdown(self) -> str:
        """Render human-readable Markdown report per LLD §5 / AC3."""
        d = self.delta
        mode_label = "simulated" if self.mode == "sim" else "live"
        openai_note = (
            "\n> **UNVERIFIED-OFFICIAL:** OpenAI prices sourced from third-party "
            "aggregator (openai.com/api/pricing was unreachable). Do not cite externally."
            if self.provider.lower() == "openai"
            else ""
        )

        return f"""# SLM Optimize — Metered-Pipeline $-Savings Report ({mode_label.upper()})

> Generated by `scripts/dogfood_savings.py` — WP-14 evidence artifact.
> Mode: **{self.mode}** | Provider: **{self.provider}** | Date: {datetime.now(timezone.utc).strftime("%Y-%m-%d")}

## Workload

| Parameter | Value |
|-----------|-------|
| Prompts (N) | {self.prompts} |
| Repeats per prompt (R) | {self.repeat} |
| Total requests | {self.prompts * (1 + self.repeat)} |
| Hit rate | {d.hit_rate * 100:.1f}% (R/{1 + self.repeat} = {self.repeat}/{1 + self.repeat}) |

## Token Savings (delta from MetricsSnapshot)

| Component | Tokens saved |
|-----------|-------------|
| Input (cache hit) | {d.tokens_saved_input:,} |
| Output (cache hit) | {d.tokens_saved_output:,} |
| Compress (word-count proxy) | {d.tokens_saved_compress:,} |
| **Total** | **{d.total_tokens_saved:,}** |

> Tokens saved: produced by SHIPPED CacheManager.on_hit recovery (M-01/M-02), not invented.

## Pricing

| Field | Value |
|-------|-------|
| Price per 1M tokens | ${self.price_per_1m:.2f} (blended input, conservative lower bound) |
| Source | {self.price_source} |
| USD→INR rate | {self.inr_rate} |
{openai_note}

## Computed Savings

**Formula:** `total_tokens_saved / 1_000_000 × price_per_1m`

| Metric | Value |
|--------|-------|
| Savings (USD) | **${self.savings_usd:.4f}** |
| Savings (INR) | **₹{self.savings_inr:.2f}** |

> Formula: `({d.tokens_saved_input} + {d.tokens_saved_output} + {d.tokens_saved_compress}) / 1_000_000 × {self.price_per_1m} = {self.savings_usd:.6f} USD`

## Methodology & Caveats

1. **Simulated workload (this run):** synthetic prompts with char-exact content (chars//4 = exact tokens by construction). No API key or network required.
2. **Savings metric is REAL:** tokens_saved_* fields come from `CacheManager.on_hit` M-01/M-02 recovery in the SHIPPED proxy — not a hardcoded percentage.
3. **Blended input price is a conservative lower bound:** output tokens cost 5× more ($15/MTok for Sonnet 4.6 output vs $3 input). This report uses input price only → understates true savings.
4. **Hit rate is workload-dependent:** this run simulates R={self.repeat} repeats per prompt → hit_rate = R/(1+R) = {self.repeat/(1+self.repeat)*100:.1f}%. Real workloads vary.
5. **tokens_saved_compress is a word-count proxy** (not a real tokenizer). Field name "bytes" in DB schema is legacy; unit is word count.
6. **Pricing is a point-in-time snapshot** (source date: 2026-06-07). Verify before citing externally.
7. **Local/self-hosted usage = $0 savings** (no metered API). Savings apply only to metered API workloads (Anthropic, OpenAI, Gemini).
8. **Live mode:** `--mode live --i-will-spend-money + SLM_DOGFOOD_API_KEY` routes real HTTP through proxy at 127.0.0.1:8765. ~$0.05–$0.20 per run. Never run in CI.

## Verification

Cross-check with: `slm optimize savings --provider {self.provider} --json`
(same formula as `optimize_cmd.py:145-155`; numbers should agree within rounding).
"""


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def compute_savings_usd(
    tokens_saved_input: int,
    tokens_saved_output: int,
    tokens_saved_compress: int,
    *,
    price_per_1m: float,
) -> float:
    """Compute estimated USD savings from token counts.

    Contract-twin of optimize_cmd.py:145-155:
        tokens_saved = ti + to + tc
        estimated_savings_usd = tokens_saved / 1_000_000 * rate

    CRIT-1: This uses the blended input price — a conservative lower bound.
    Output tokens cost 5× more than input (e.g. $15 vs $3/MTok for Sonnet 4.6).
    """
    total = tokens_saved_input + tokens_saved_output + tokens_saved_compress
    return total / 1_000_000 * price_per_1m


def resolve_price(
    provider: str,
    default_table: dict[str, float],
    overrides: dict[str, Any],
) -> tuple[float, str]:
    """Resolve price per 1M tokens for a provider.

    Priority: overrides[provider].input_per_1m_usd > default_table[provider] > default_table["default"].

    Returns (price, source_description).
    """
    # 1. Check config overrides
    if provider in overrides:
        override_val = overrides[provider]
        if isinstance(override_val, dict) and "input_per_1m_usd" in override_val:
            price = float(override_val["input_per_1m_usd"])
            return price, f"pricing_override[{provider}].input_per_1m_usd"

    # 2. Default table lookup
    if provider in default_table:
        price = float(default_table[provider])
        return price, f"default_table[{provider}]"

    # 3. Fallback to "default"
    price = float(default_table.get("default", 3.00))
    return price, "default_table[default]"


def to_inr(usd: float, rate: float) -> float:
    """Convert USD to INR at the given exchange rate."""
    return usd * rate


def metrics_delta(before: Any, after: Any) -> MetricsDelta:
    """Compute the delta between two MetricsSnapshots, clamping to ≥0.

    Uses attribute access (compatible with MetricsSnapshot dataclass).
    """
    return MetricsDelta(
        hits=after.hits - before.hits,
        misses=after.misses - before.misses,
        tokens_saved_input=after.tokens_saved_input - before.tokens_saved_input,
        tokens_saved_output=after.tokens_saved_output - before.tokens_saved_output,
        tokens_saved_compress=after.tokens_saved_compress - before.tokens_saved_compress,
    )
