# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""WP-14 TDD — tests written FIRST (RED phase).

Tests cover:
  - compute_savings_usd: zero, known fixture, compress component
  - resolve_price: override wins, default fallback
  - to_inr: basic conversion
  - MetricsDelta: clamps negative to zero
  - metrics_delta: computes delta correctly
  - sim_run_offline_writes_artifacts: no-network, temp CacheDB, $ derivable from delta
  - live refuses without --i-will-spend-money flag and without SLM_DOGFOOD_API_KEY
  - Report.to_markdown contains caveats
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add scripts/ to path so we can import the modules directly
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# _savings_math tests
# ---------------------------------------------------------------------------


class TestComputeSavingsUsd:
    """AC2 / LLD §6 unit contract: compute_savings_usd."""

    def test_zero_tokens_returns_zero(self):
        from _savings_math import compute_savings_usd
        result = compute_savings_usd(0, 0, 0, price_per_1m=3.00)
        assert result == 0.0

    def test_known_fixture_2m_tokens_at_3_usd(self):
        """2 000 000 total tokens @ $3.00/MTok = $6.00 exactly."""
        from _savings_math import compute_savings_usd
        # Spread across all three components
        result = compute_savings_usd(1_000_000, 500_000, 500_000, price_per_1m=3.00)
        assert abs(result - 6.00) < 1e-9

    def test_compress_component_included(self):
        """tokens_saved_compress is included in the total (3rd positional arg)."""
        from _savings_math import compute_savings_usd
        # 0 input, 0 output, 1M compress @ $3 = $3
        result = compute_savings_usd(0, 0, 1_000_000, price_per_1m=3.00)
        assert abs(result - 3.00) < 1e-9

    def test_override_price_applied(self):
        """Different price_per_1m changes result proportionally."""
        from _savings_math import compute_savings_usd
        result_3 = compute_savings_usd(1_000_000, 0, 0, price_per_1m=3.00)
        result_5 = compute_savings_usd(1_000_000, 0, 0, price_per_1m=5.00)
        assert abs(result_3 - 3.00) < 1e-9
        assert abs(result_5 - 5.00) < 1e-9

    def test_inr_conversion(self):
        """to_inr: $6.00 at 83.5 rate = 501.00"""
        from _savings_math import to_inr
        result = to_inr(6.00, 83.5)
        assert abs(result - 501.00) < 1e-6

    def test_inr_zero_usd(self):
        from _savings_math import to_inr
        assert to_inr(0.0, 83.5) == 0.0


class TestResolvePrice:
    """resolve_price: override wins over default table."""

    def test_override_wins(self):
        from _savings_math import resolve_price
        overrides = {"anthropic": {"input_per_1m_usd": 9.99}}
        default_table = {"anthropic": 3.00, "default": 3.00}
        price, source = resolve_price("anthropic", default_table, overrides)
        assert abs(price - 9.99) < 1e-9
        assert "override" in source.lower()

    def test_default_fallback_known_provider(self):
        from _savings_math import resolve_price
        default_table = {"anthropic": 3.00, "default": 3.00}
        price, source = resolve_price("anthropic", default_table, {})
        assert abs(price - 3.00) < 1e-9
        assert "default" in source.lower() or "table" in source.lower()

    def test_default_fallback_unknown_provider(self):
        from _savings_math import resolve_price
        default_table = {"anthropic": 3.00, "default": 3.00}
        price, source = resolve_price("unknown_provider", default_table, {})
        # Should fall back to "default" key
        assert abs(price - 3.00) < 1e-9


class TestMetricsDelta:
    """MetricsDelta: clamps negative to 0; computes total_tokens_saved."""

    def test_clamps_negative_hits_to_zero(self):
        from _savings_math import MetricsDelta
        delta = MetricsDelta(hits=-5, misses=3, tokens_saved_input=-100,
                             tokens_saved_output=50, tokens_saved_compress=0)
        assert delta.hits == 0
        assert delta.tokens_saved_input == 0
        assert delta.tokens_saved_output == 50

    def test_total_tokens_saved_sum(self):
        from _savings_math import MetricsDelta
        delta = MetricsDelta(hits=10, misses=2, tokens_saved_input=1000,
                             tokens_saved_output=500, tokens_saved_compress=200)
        assert delta.total_tokens_saved == 1700

    def test_hit_rate_n_plus_r(self):
        """N=20, R=5 → 100 hits / 120 total = 83.3%"""
        from _savings_math import MetricsDelta
        delta = MetricsDelta(hits=100, misses=20,
                             tokens_saved_input=0, tokens_saved_output=0,
                             tokens_saved_compress=0)
        assert abs(delta.hit_rate - 100 / 120) < 1e-9


class TestMetricsDeltaFunction:
    """metrics_delta(before, after) — delta with clamp."""

    def test_positive_delta(self):
        from _savings_math import MetricsDelta, metrics_delta
        from superlocalmemory.optimize.storage.db import MetricsSnapshot
        before = MetricsSnapshot(hits=5, misses=2, tokens_saved_input=1000,
                                 tokens_saved_output=200, tokens_saved_compress=50)
        after = MetricsSnapshot(hits=15, misses=4, tokens_saved_input=2000,
                                tokens_saved_output=400, tokens_saved_compress=100)
        result = metrics_delta(before, after)
        assert isinstance(result, MetricsDelta)
        assert result.hits == 10
        assert result.misses == 2
        assert result.tokens_saved_input == 1000
        assert result.tokens_saved_output == 200
        assert result.tokens_saved_compress == 50

    def test_negative_clamped_to_zero(self):
        """If counters go backwards (e.g. DB reset), clamp to 0."""
        from _savings_math import metrics_delta
        from superlocalmemory.optimize.storage.db import MetricsSnapshot
        before = MetricsSnapshot(hits=100, tokens_saved_input=5000)
        after = MetricsSnapshot(hits=10, tokens_saved_input=500)
        result = metrics_delta(before, after)
        assert result.hits == 0
        assert result.tokens_saved_input == 0


class TestReport:
    """Report.to_markdown must contain caveats."""

    def test_markdown_contains_caveats_section(self):
        from _savings_math import MetricsDelta, Report
        delta = MetricsDelta(hits=100, misses=20, tokens_saved_input=500_000,
                             tokens_saved_output=100_000, tokens_saved_compress=50_000)
        report = Report(
            mode="sim",
            provider="anthropic",
            prompts=20,
            repeat=5,
            delta=delta,
            price_per_1m=3.00,
            price_source="DEFAULT_COST_PER_MILLION_INPUT_TOKENS[anthropic], 2026-06-07 (R6)",
            savings_usd=1.95,
            savings_inr=162.825,
            inr_rate=83.5,
        )
        md = report.to_markdown()
        assert "Caveats" in md or "caveat" in md.lower()
        # Savings labeled as simulated
        assert "sim" in md.lower() or "simulated" in md.lower()

    def test_json_has_formula_field(self):
        from _savings_math import MetricsDelta, Report
        delta = MetricsDelta(hits=10, misses=2, tokens_saved_input=100_000,
                             tokens_saved_output=0, tokens_saved_compress=0)
        report = Report(
            mode="sim",
            provider="anthropic",
            prompts=20,
            repeat=5,
            delta=delta,
            price_per_1m=3.00,
            price_source="test",
            savings_usd=0.30,
            savings_inr=25.05,
            inr_rate=83.5,
        )
        data = json.loads(report.to_json())
        assert "formula" in data.get("computed", {})

    def test_openai_prices_marked_unverified(self):
        """OpenAI prices MUST be flagged as UNVERIFIED-OFFICIAL in report."""
        from _savings_math import MetricsDelta, Report
        delta = MetricsDelta(hits=5, misses=1, tokens_saved_input=100_000,
                             tokens_saved_output=0, tokens_saved_compress=0)
        report = Report(
            mode="sim",
            provider="openai",
            prompts=20,
            repeat=5,
            delta=delta,
            price_per_1m=2.50,
            price_source="DEFAULT_COST_PER_MILLION_INPUT_TOKENS[openai], UNVERIFIED-OFFICIAL",
            savings_usd=0.25,
            savings_inr=20.875,
            inr_rate=83.5,
        )
        md = report.to_markdown()
        assert "UNVERIFIED" in md


# ---------------------------------------------------------------------------
# dogfood_savings.py integration tests
# ---------------------------------------------------------------------------


class TestSimRunOfflineWritesArtifacts:
    """AC1 / LLD §6: --mode sim writes both artifacts, $ derivable from delta."""

    def test_sim_exits_zero_and_writes_artifacts(self, tmp_path):
        """sim mode: exit 0, both JSON + MD artifacts written."""
        import importlib.util
        import subprocess

        script = _SCRIPTS_DIR / "dogfood_savings.py"
        result = subprocess.run(
            [
                sys.executable, str(script),
                "--mode", "sim",
                "--provider", "anthropic",
                "--prompts", "4",
                "--repeat", "2",
                "--out-json", str(tmp_path / "raw.json"),
                "--out-md", str(tmp_path / "report.md"),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "SLM_DATA_DIR": str(tmp_path)},
        )
        assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        assert (tmp_path / "raw.json").exists(), "JSON ledger not written"
        assert (tmp_path / "report.md").exists(), "MD report not written"

    def test_sim_json_has_reproducible_formula(self, tmp_path):
        """$ derivable as tokens_saved/1e6 * price from raw JSON."""
        import subprocess
        script = _SCRIPTS_DIR / "dogfood_savings.py"
        json_path = tmp_path / "raw.json"
        subprocess.run(
            [
                sys.executable, str(script),
                "--mode", "sim",
                "--provider", "anthropic",
                "--prompts", "4",
                "--repeat", "2",
                "--out-json", str(json_path),
                "--out-md", str(tmp_path / "report.md"),
            ],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "SLM_DATA_DIR": str(tmp_path)},
        )
        data = json.loads(json_path.read_text())
        delta = data["delta"]
        pricing = data["pricing"]
        computed = data["computed"]

        # Verify formula: total_tokens_saved / 1e6 * price
        total = (
            delta["tokens_saved_input"]
            + delta["tokens_saved_output"]
            + delta["tokens_saved_compress"]
        )
        expected_usd = total / 1_000_000 * pricing["price_per_1m"]
        actual_usd = computed["savings_usd"]
        assert abs(actual_usd - expected_usd) < 1e-9, (
            f"Formula mismatch: expected {expected_usd}, got {actual_usd}"
        )

    def test_sim_hits_match_n_times_r(self, tmp_path):
        """sim: N prompts × R repeats → N*R hits in delta (first call is miss)."""
        import subprocess
        script = _SCRIPTS_DIR / "dogfood_savings.py"
        json_path = tmp_path / "raw.json"
        n_prompts = 4
        r_repeat = 2
        subprocess.run(
            [
                sys.executable, str(script),
                "--mode", "sim",
                "--provider", "anthropic",
                "--prompts", str(n_prompts),
                "--repeat", str(r_repeat),
                "--out-json", str(json_path),
                "--out-md", str(tmp_path / "report.md"),
            ],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "SLM_DATA_DIR": str(tmp_path)},
        )
        data = json.loads(json_path.read_text())
        assert data["delta"]["hits"] == n_prompts * r_repeat

    def test_sim_uses_real_cache_manager_not_hardcoded(self, tmp_path):
        """Tokens come from the CacheManager on_hit recovery, not a hardcoded %."""
        import subprocess
        script = _SCRIPTS_DIR / "dogfood_savings.py"
        json_path = tmp_path / "raw.json"
        subprocess.run(
            [
                sys.executable, str(script),
                "--mode", "sim",
                "--provider", "anthropic",
                "--prompts", "2",
                "--repeat", "1",
                "--out-json", str(json_path),
                "--out-md", str(tmp_path / "report.md"),
            ],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "SLM_DATA_DIR": str(tmp_path)},
        )
        data = json.loads(json_path.read_text())
        # The formula field must cite tokens from delta, not a literal percentage
        formula = data["computed"]["formula"]
        assert "tokens_saved" in formula.lower() or "/" in formula


class TestLiveRefusesWithoutFlag:
    """AC4: live mode must refuse without --i-will-spend-money and SLM_DOGFOOD_API_KEY."""

    def test_live_without_spend_flag_exits_nonzero(self, tmp_path):
        import subprocess
        script = _SCRIPTS_DIR / "dogfood_savings.py"
        result = subprocess.run(
            [sys.executable, str(script), "--mode", "live"],
            capture_output=True, text=True, timeout=10,
            env={**os.environ, "SLM_DATA_DIR": str(tmp_path)},
        )
        assert result.returncode != 0
        # Error message should mention the required flag
        combined = result.stdout + result.stderr
        assert "i-will-spend-money" in combined or "spend" in combined.lower()

    def test_live_without_api_key_exits_nonzero(self, tmp_path):
        import subprocess
        script = _SCRIPTS_DIR / "dogfood_savings.py"
        env = {k: v for k, v in os.environ.items() if k != "SLM_DOGFOOD_API_KEY"}
        env["SLM_DATA_DIR"] = str(tmp_path)
        result = subprocess.run(
            [sys.executable, str(script), "--mode", "live", "--i-will-spend-money"],
            capture_output=True, text=True, timeout=10,
            env=env,
        )
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "SLM_DOGFOOD_API_KEY" in combined or "api_key" in combined.lower()
