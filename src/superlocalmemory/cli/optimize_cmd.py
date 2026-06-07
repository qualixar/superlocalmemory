# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Handlers for ``slm optimize status|on|off|savings``."""

from __future__ import annotations

import dataclasses
import json
import sys
from argparse import Namespace

from superlocalmemory.cli.optimize_constants import (
    DEFAULT_COST_PER_MILLION_INPUT_TOKENS,
    OPTIMIZE_DEFAULT_PORT,
    _PRICING_DATE,
)


def _get_store():
    from superlocalmemory.optimize.config.store import ConfigStore
    return ConfigStore()


def _get_cache_db():
    from superlocalmemory.optimize.storage.db import CacheDB
    return CacheDB()


def _write_config(**fields) -> None:
    """5-step immutable config-write. Call from any handler."""
    store = _get_store()
    cfg = store.get()
    try:
        cfg = dataclasses.replace(cfg, **fields)
        store.save(cfg)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"Error writing config: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_optimize(args: Namespace) -> None:
    """Top-level dispatcher for ``slm optimize <subcommand>``."""
    sub = getattr(args, "opt_command", None)
    _dispatch = {
        "status": cmd_optimize_status,
        "on": cmd_optimize_on,
        "off": cmd_optimize_off,
        "savings": cmd_optimize_savings,
    }
    handler = _dispatch.get(sub or "")
    if handler:
        handler(args)
    else:
        print("Usage: slm optimize status|on|off|savings [--json] [--since N]")
        sys.exit(0)


def cmd_optimize_status(args: Namespace) -> None:
    """Print current Optimize module status."""
    cfg = _get_store().get()
    use_json = getattr(args, "json", False)

    proxy_running = False
    try:
        import urllib.request
        _url = f"http://127.0.0.1:{OPTIMIZE_DEFAULT_PORT}/health"
        _req = urllib.request.Request(_url, method="GET")
        with urllib.request.urlopen(_req, timeout=1) as _resp:
            proxy_running = _resp.status == 200
    except Exception:
        pass

    if use_json:
        data = {
            "status": "ok",
            "optimize_enabled": cfg.enabled,
            "cache_enabled": cfg.cache_enabled,
            "semantic_enabled": cfg.semantic_enabled,
            "compress_enabled": cfg.compress_enabled,
            "compress_mode": cfg.compress_mode,
            "proxy_running": proxy_running,
            "proxy_port": OPTIMIZE_DEFAULT_PORT,
            "config_version": cfg.config_version,
        }
        print(json.dumps(data, indent=2))
        return

    state = "ON" if cfg.enabled else "OFF"
    print(f"Optimize: {state}")
    print(f"  Cache:     {'enabled' if cfg.cache_enabled else 'disabled'}"
          f"  (exact: {cfg.ttl.exact_seconds}s TTL,"
          f" semantic: {'OFF' if not cfg.semantic_enabled else f'{cfg.ttl.semantic_seconds}s'})")
    print(f"  Compress:  {'enabled' if cfg.compress_enabled else 'disabled'}"
          f"  (mode: {cfg.compress_mode},"
          f" code: {'ON' if cfg.compress_code else 'OFF'},"
          f" prose: {'ON' if cfg.compress_prose else 'OFF'},"
          f" CCR: {'ON' if cfg.compress_ccr else 'OFF'})")
    proxy_status = f"running on :{OPTIMIZE_DEFAULT_PORT}" if proxy_running else "not running"
    print(f"  Proxy:     {proxy_status}")
    print(f"  Config:    ~/.superlocalmemory/optimize.json  (version {cfg.config_version})")


def cmd_optimize_on(args: Namespace) -> None:
    """Enable all Optimize features (cache + compress)."""
    _write_config(enabled=True, cache_enabled=True, compress_enabled=True)
    use_json = getattr(args, "json", False)
    if use_json:
        print(json.dumps({"status": "ok", "optimize_enabled": True}))
    else:
        print("Optimize enabled. Run 'slm proxy' to start the proxy."
              " Daemon hot-reload: active within 2s.")


def cmd_optimize_off(args: Namespace) -> None:
    """Disable all Optimize features. Does NOT stop proxy."""
    _write_config(
        enabled=False,
        cache_enabled=False,
        semantic_enabled=False,
        compress_enabled=False,
    )
    use_json = getattr(args, "json", False)
    if use_json:
        print(json.dumps({"status": "ok", "optimize_enabled": False}))
    else:
        print("Optimize disabled. Proxy (if running) will pass through calls unchanged.")


def cmd_optimize_savings(args: Namespace) -> None:
    """Print token/cost savings from CacheDB.metrics_load()."""
    since = getattr(args, "since", 7)
    provider = getattr(args, "provider", None)
    use_json = getattr(args, "json", False)

    if since <= 0:
        print("Error: --since must be a positive integer.", file=sys.stderr)
        sys.exit(1)

    snap = _get_cache_db().metrics_load()
    cfg = _get_store().get()

    tokens_saved = snap.tokens_saved_input + snap.tokens_saved_output + snap.tokens_saved_compress
    provider_key = provider or "default"
    rate = DEFAULT_COST_PER_MILLION_INPUT_TOKENS.get(
        provider_key,
        DEFAULT_COST_PER_MILLION_INPUT_TOKENS["default"],
    )
    # Allow config overrides
    if cfg.pricing_overrides and provider_key in cfg.pricing_overrides:
        rate = cfg.pricing_overrides[provider_key].get("input_per_1m_usd", rate)

    estimated_savings_usd = tokens_saved / 1_000_000 * rate

    if use_json:
        data = {
            "exact_hits": snap.hits,
            "semantic_hits": snap.misses,
            "tokens_saved": tokens_saved,
            "estimated_savings_usd": round(estimated_savings_usd, 6),
            "pricing_date": _PRICING_DATE,
            "tokens_saved_input": snap.tokens_saved_input,
            "tokens_saved_output": snap.tokens_saved_output,
            "tokens_saved_compress": snap.tokens_saved_compress,
        }
        print(json.dumps(data, indent=2))
        return

    print(f"Savings (last {since} days):")
    print(f"  Exact cache hits:    {snap.hits:>5}   ({snap.tokens_saved_input:,} input tokens saved)")
    print(f"  Semantic cache hits: {snap.misses:>5}")
    print(f"  Tokens saved (total): {tokens_saved:,}")
    print(f"  Estimated savings:  ~${estimated_savings_usd:.4f} (at ${rate:.2f}/M tokens)")
    print(f"  Pricing date: {_PRICING_DATE}")
