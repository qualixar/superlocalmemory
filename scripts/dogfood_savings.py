#!/usr/bin/env python3
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""dogfood_savings.py — WP-14 metered-pipeline $-savings evidence script.

Drives a repeated-prompt workload through the production CacheManager +
MetricsCollector, reads the MetricsSnapshot delta, and emits a Report
(JSON ledger + Markdown) with reproducible $ savings.

RESEARCH INTEGRITY: Every $ number traces to a real (tokens_saved × price)
computation from actual cache/compress stats. No hardcoded percentages.
Prices cite R6 (2026-06-16). OpenAI prices are UNVERIFIED-OFFICIAL.

Usage:
    # Sim (no API key, no network, CI-safe):
    python3 scripts/dogfood_savings.py --mode sim --provider anthropic

    # Live (ONE manual run by Varun, ~$0.05-0.20):
    SLM_DOGFOOD_API_KEY=sk-... python3 scripts/dogfood_savings.py \\
        --mode live --i-will-spend-money --provider anthropic

AC6: proxy state is restored in try/finally — including on exception.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Path setup — allow running as scripts/dogfood_savings.py from repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
_SCRIPTS = _REPO_ROOT / "scripts"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _savings_math import (  # noqa: E402 — import after path setup
    MetricsDelta,
    Report,
    compute_savings_usd,
    metrics_delta,
    resolve_price,
    to_inr,
)

# ---------------------------------------------------------------------------
# Constants (from optimize_constants.py — read-only)
# ---------------------------------------------------------------------------
# We import here to maintain contract-twin parity with optimize_cmd.py:145-155.
from superlocalmemory.cli.optimize_constants import (  # noqa: E402
    DEFAULT_COST_PER_MILLION_INPUT_TOKENS,
    _PRICING_DATE,
)

_DEFAULT_PROMPTS = 20
_DEFAULT_REPEAT = 5
_DEFAULT_PROVIDER = "anthropic"
_DEFAULT_INR_RATE = 83.5

# Openai prices are from R6 but UNVERIFIED-OFFICIAL.
_OPENAI_UNVERIFIED_NOTE = "UNVERIFIED-OFFICIAL"

# Output paths per LLD §2 / D-B LOCKED
_EVIDENCE_DIR = _REPO_ROOT / ".backup" / "v3.6.14" / "evidence" / "wp14"
_DOCS_EVIDENCE_MD = _REPO_ROOT / "docs" / "evidence" / "optimize-savings.md"


# ---------------------------------------------------------------------------
# Sim workload
# ---------------------------------------------------------------------------

def _build_sim_prompt(index: int) -> str:
    """Build a char-exact prompt so chars//4 = integer tokens exactly.

    Pattern: 40 chars (10 tokens) per prompt, varied by index.
    This ensures M-01 (chars//4) produces predictable, non-zero values.
    """
    # 40 chars = 10 tokens exactly via chars // 4 = 40 // 4 = 10
    base = f"Prompt{index:04d}: What is the capital of France? "
    # Pad/truncate to exactly 40 chars
    padded = base[:40].ljust(40)
    return padded


def _build_sim_response(output_tokens: int = 8) -> dict:
    """Build a synthetic provider response with a usage block.

    The usage block is required for M-02 (output token recovery) in on_hit.
    """
    return {
        "id": "sim-resp-001",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Paris is the capital of France."}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10,
            "output_tokens": output_tokens,
        },
    }


def run_workload_sim(
    n_prompts: int,
    n_repeat: int,
    tmp_db_path: Path,
) -> None:
    """Drive N prompts × (1 miss + R hits) through production CacheManager.

    Uses a temp CacheDB — never touches user's llmcache.db (AC8 / LLD §8).
    Synthetic responses include a `usage` block so M-02 recovers output tokens.
    """
    import superlocalmemory.optimize.storage.db as _db_mod
    from superlocalmemory.optimize.cache.manager import CacheManager
    from superlocalmemory.optimize.metrics.counters import MetricsCollector
    from superlocalmemory.optimize.proxy.lifecycle import ProxyRequest
    from superlocalmemory.optimize.storage.db import CacheDB

    # Isolate AES key to tmp dir (mirrors tests/optimize/conftest.py pattern)
    original_key_file = getattr(_db_mod, "_KEY_FILE", None)
    _db_mod._KEY_FILE = tmp_db_path.parent / "opt-key.bin"

    try:
        cache_db = CacheDB(tmp_db_path)
        cm = CacheManager(cache_db)
        # Reset MetricsCollector singleton for isolation
        MetricsCollector._instance = None
        collector = MetricsCollector.get_instance()

        for i in range(n_prompts):
            prompt_text = _build_sim_prompt(i)
            sim_response = _build_sim_response()
            resp_bytes = json.dumps(sim_response).encode()

            # Build a ProxyRequest (immutable)
            req = ProxyRequest(
                provider="anthropic",
                method="POST",
                path="/v1/messages",
                headers={},  # redacted per CWE-532
                body={
                    "model": "claude-sonnet-4-6",
                    "messages": [{"role": "user", "content": prompt_text}],
                    "max_tokens": 50,
                },
                body_bytes=b"",
                request_id=f"sim-{i:04d}",
                stream=False,
                has_tools=False,
            )

            # First call: miss (store response)
            hit = cm.check(req)
            if hit is None or not hit.hit:
                from superlocalmemory.optimize.proxy.lifecycle import ProviderResponse
                prov_resp = ProviderResponse(
                    modified=False,
                    body=sim_response,
                    body_bytes=resp_bytes,
                    tokens_before=0,
                    tokens_after=0,
                    strategy="none",
                )
                cm.store(req, prov_resp)

            # Repeat calls: hits (tokens recovered via on_hit M-01/M-02)
            for _ in range(n_repeat):
                hit = cm.check(req)
                if hit is not None and hit.hit:
                    cm.on_hit(req, hit.data or resp_bytes, tokens_saved=0)
                else:
                    # Should not happen in sim; record miss for safety
                    collector.on_miss()

    finally:
        if original_key_file is not None:
            _db_mod._KEY_FILE = original_key_file


# ---------------------------------------------------------------------------
# Proxy context manager (for live mode, AC6)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def with_proxy_enabled() -> Iterator[None]:
    """Enable proxy for the duration of the block; restore in finally.

    Reads optimize.json, sets enable_cache=True, restores on exit (AC6).
    Only safe for ONE concurrent user — live docs "no other proxy traffic".
    """
    from superlocalmemory.optimize.config.store import ConfigStore

    store = ConfigStore()
    cfg = store.get()
    original_enable_cache = cfg.enable_cache
    original_enable_compress = cfg.enable_compress

    try:
        # Enable cache for the live run
        new_cfg = store.get()
        new_cfg.enable_cache = True
        store.save(new_cfg)
        yield
    finally:
        # Restore exactly (AC6 — even on exception)
        restore_cfg = store.get()
        restore_cfg.enable_cache = original_enable_cache
        restore_cfg.enable_compress = original_enable_compress
        store.save(restore_cfg)


# ---------------------------------------------------------------------------
# Live workload (AC4 gated)
# ---------------------------------------------------------------------------

def run_workload_live(
    n_prompts: int,
    n_repeat: int,
    api_key: str,
    provider: str,
) -> None:
    """Route real HTTP through proxy at 127.0.0.1:8765.

    SECURITY: api_key comes from env only, NEVER logged or printed.
    This function is excluded from CI coverage (live path).
    """
    import urllib.request  # stdlib only — no new deps

    port = 8765
    base_url = f"http://127.0.0.1:{port}"

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,  # never logged
    }

    prompt = "What is 2 + 2? Answer in one word."

    for i in range(n_prompts):
        body = json.dumps({
            "model": "claude-haiku-4-5",  # cheapest Anthropic model
            "messages": [{"role": "user", "content": f"[{i}] {prompt}"}],
            "max_tokens": 16,
        }).encode()

        # First call: miss
        req = urllib.request.Request(
            f"{base_url}/anthropic/v1/messages",
            data=body,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            resp.read()

        # Repeat calls: hits
        for _ in range(n_repeat):
            req2 = urllib.request.Request(
                f"{base_url}/anthropic/v1/messages",
                data=body,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req2, timeout=30) as resp2:  # noqa: S310
                resp2.read()


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

def _snapshot_from_collector() -> object:
    """Read current MetricsCollector state as a MetricsSnapshot-compatible object."""
    from superlocalmemory.optimize.metrics.counters import MetricsCollector
    from superlocalmemory.optimize.storage.db import MetricsSnapshot

    c = MetricsCollector.get_instance()
    snap = c.snapshot() if hasattr(c, "snapshot") else None
    if snap is not None:
        return snap
    # Fallback: read fields directly
    return MetricsSnapshot(
        hits=c._hits,
        misses=c._misses,
        tokens_saved_input=c._tokens_saved_input,
        tokens_saved_output=c._tokens_saved_output,
        tokens_saved_compress=c._tokens_saved_compress,
    )


def _zero_snapshot() -> object:
    """Return a zeroed MetricsSnapshot."""
    from superlocalmemory.optimize.storage.db import MetricsSnapshot
    return MetricsSnapshot()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_price_source(provider: str, price: float, source_key: str) -> str:
    """Build a human-readable price source string for the report."""
    date = _PRICING_DATE
    r6 = "R6 (2026-06-16)"
    if provider.lower() == "openai":
        return (
            f"DEFAULT_COST_PER_MILLION_INPUT_TOKENS[{provider}]={price}, "
            f"{date}, {r6}, {_OPENAI_UNVERIFIED_NOTE}"
        )
    return f"{source_key}={price}, {date}, {r6}, VERIFIED (Anthropic official)"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="WP-14: metered-pipeline $-savings dogfood evidence script.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "RESEARCH INTEGRITY: Every $ number traces to tokens_saved × price.\n"
            "Sim mode is CI-safe (no key, no network). Live mode costs real money.\n"
            "OpenAI prices: UNVERIFIED-OFFICIAL (see R6 2026-06-16).\n"
        ),
    )
    parser.add_argument(
        "--mode", choices=["sim", "live"], default="sim",
        help="sim = offline/CI-safe; live = real HTTP via proxy (costs money)",
    )
    parser.add_argument(
        "--provider", default=_DEFAULT_PROVIDER,
        choices=["anthropic", "openai", "gemini"],
        help="Provider key for pricing lookup (default: anthropic)",
    )
    parser.add_argument(
        "--prompts", type=int, default=_DEFAULT_PROMPTS,
        help="Number of distinct prompts (N). Default: 20",
    )
    parser.add_argument(
        "--repeat", type=int, default=_DEFAULT_REPEAT,
        help="Number of cache-hit repeats per prompt (R). Default: 5",
    )
    parser.add_argument(
        "--i-will-spend-money", action="store_true",
        help="Required for --mode live. Confirms you accept real API spend.",
    )
    parser.add_argument(
        "--out-md", type=Path, default=None,
        help="Output path for Markdown report (default: docs/evidence/optimize-savings.md)",
    )
    parser.add_argument(
        "--out-json", type=Path, default=None,
        help="Output path for JSON ledger (default: .backup/v3.6.14/evidence/wp14/raw-metrics-<mode>-<utc>.json)",
    )

    args = parser.parse_args()

    # AC4: live mode guards
    if args.mode == "live":
        if not getattr(args, "i_will_spend_money", False):
            print(
                "ERROR: --mode live requires --i-will-spend-money flag.\n"
                "This mode routes real requests through the proxy and costs real money.",
                file=sys.stderr,
            )
            return 2

        api_key = os.environ.get("SLM_DOGFOOD_API_KEY", "")
        if not api_key:
            print(
                "ERROR: SLM_DOGFOOD_API_KEY environment variable not set.\n"
                "Set it to your Anthropic API key before running --mode live.",
                file=sys.stderr,
            )
            return 2

    # Resolve price (parity with optimize_cmd.py:146-153)
    price, price_source_key = resolve_price(
        args.provider,
        DEFAULT_COST_PER_MILLION_INPUT_TOKENS,
        {},  # no config overrides in dogfood — uses defaults for reproducibility
    )
    price_source = _build_price_source(args.provider, price, price_source_key)

    # ---------------------------------------------------------------------------
    # Run workload and collect delta
    # ---------------------------------------------------------------------------

    if args.mode == "sim":
        with tempfile.TemporaryDirectory() as _tmp:
            tmp_dir = Path(_tmp)
            tmp_db = tmp_dir / "llmcache.db"

            # Capture before (zeroed — fresh temp DB)
            before = _zero_snapshot()

            run_workload_sim(args.prompts, args.repeat, tmp_db)

            # Read after from the MetricsCollector (which on_hit updated)
            after = _snapshot_from_collector()

    else:
        # Live mode: use real CacheDB via proxy
        before_db = _read_live_snapshot()
        with with_proxy_enabled():
            api_key = os.environ["SLM_DOGFOOD_API_KEY"]  # already validated above
            run_workload_live(args.prompts, args.repeat, api_key, args.provider)
        after_db = _read_live_snapshot()
        before = before_db
        after = after_db

    # ---------------------------------------------------------------------------
    # Compute savings
    # ---------------------------------------------------------------------------

    delta = metrics_delta(before, after)
    savings_usd = compute_savings_usd(
        delta.tokens_saved_input,
        delta.tokens_saved_output,
        delta.tokens_saved_compress,
        price_per_1m=price,
    )
    savings_inr = to_inr(savings_usd, _DEFAULT_INR_RATE)

    report = Report(
        mode=args.mode,
        provider=args.provider,
        prompts=args.prompts,
        repeat=args.repeat,
        delta=delta,
        price_per_1m=price,
        price_source=price_source,
        savings_usd=savings_usd,
        savings_inr=savings_inr,
        inr_rate=_DEFAULT_INR_RATE,
    )

    # ---------------------------------------------------------------------------
    # Write artifacts
    # ---------------------------------------------------------------------------

    import datetime as _dt

    utc_tag = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # JSON ledger path
    if args.out_json is not None:
        json_path = Path(args.out_json)
    else:
        _EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        json_path = _EVIDENCE_DIR / f"raw-metrics-{args.mode}-{utc_tag}.json"

    # MD report path
    if args.out_md is not None:
        md_path = Path(args.out_md)
    else:
        md_path = _DOCS_EVIDENCE_MD

    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(report.to_json(), encoding="utf-8")
    md_path.write_text(report.to_markdown(), encoding="utf-8")

    print(report.to_markdown())
    print(f"\nJSON ledger: {json_path}")
    print(f"MD report:   {md_path}")

    return 0


def _read_live_snapshot() -> object:
    """Read snapshot from the live CacheDB (for --mode live)."""
    from superlocalmemory.optimize.storage.db import CacheDB
    db = CacheDB.get_default()
    return db.metrics_load()


if __name__ == "__main__":
    raise SystemExit(main())
