# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Handlers for ``slm compress status|mode|code|prose|ccr``."""

from __future__ import annotations

import dataclasses
import json
import sys
from argparse import Namespace

from superlocalmemory.cli.optimize_constants import AGGRESSIVE_MODE_WARNING


def _get_store():
    from superlocalmemory.optimize.config.store import ConfigStore
    return ConfigStore()


def _get_cache_db():
    from superlocalmemory.optimize.storage.db import CacheDB
    return CacheDB()


def _write_config(**fields) -> None:
    """5-step immutable config-write."""
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


def cmd_compress(args: Namespace) -> None:
    """Top-level dispatcher for ``slm compress <subcommand>``."""
    sub = getattr(args, "compress_command", None)
    _dispatch = {
        "status": cmd_compress_status,
        "mode": cmd_compress_mode,
        "code": cmd_compress_code,
        "prose": cmd_compress_prose,
        "ccr": cmd_compress_ccr,
        "align": cmd_compress_align,
    }
    handler = _dispatch.get(sub or "")
    if handler:
        handler(args)
    else:
        print("Usage: slm compress status|mode|code|prose|ccr [options]")
        sys.exit(0)


def cmd_compress_status(args: Namespace) -> None:
    """Print compression status."""
    use_json = getattr(args, "json", False)
    cfg = _get_store().get()

    if use_json:
        data = {
            "status": "ok",
            "compress_enabled": cfg.compress_enabled,
            "compress_mode": cfg.compress_mode,
            "compress_code": cfg.compress_code,
            "compress_prose": cfg.compress_prose,
            "compress_ccr": cfg.compress_ccr,
        }
        print(json.dumps(data, indent=2))
        return

    print("Compression status:")
    print(f"  Enabled:  {'yes' if cfg.compress_enabled else 'no'}")
    print(f"  Mode:     {cfg.compress_mode}")
    print(f"  Code:     {'ON' if cfg.compress_code else 'OFF'}"
          "  (extractive JSON/code — lossless structure)")
    print(f"  Prose:    {'ON' if cfg.compress_prose else 'OFF'}")
    print(f"  CCR:      {'ON' if cfg.compress_ccr else 'OFF'}"
          " (reversible context retrieval)")


def cmd_compress_mode(args: Namespace) -> None:
    """Set compression mode to safe or aggressive."""
    use_json = getattr(args, "json", False)
    mode_value = getattr(args, "mode_value", "safe")

    if mode_value == "aggressive":
        print(AGGRESSIVE_MODE_WARNING)

    _write_config(compress_mode=mode_value)

    if use_json:
        print(json.dumps({"status": "ok", "compress_mode": mode_value}))
        return

    print(f"Compression mode set to: {mode_value}.")
    print("Daemon hot-reload: active within 2s. No restart required.")


def cmd_compress_code(args: Namespace) -> None:
    """Enable or disable code/JSON compression."""
    use_json = getattr(args, "json", False)
    value = getattr(args, "code_value", "on")

    _write_config(compress_code=(value == "on"))

    if use_json:
        print(json.dumps({"status": "ok", "compress_code": value == "on"}))
        return

    print(f"Code compression: {'ENABLED' if value == 'on' else 'DISABLED'}.")
    print("Daemon hot-reload: active within 2s.")


def cmd_compress_prose(args: Namespace) -> None:
    """Enable or disable prose compression."""
    use_json = getattr(args, "json", False)
    value = getattr(args, "prose_value", "off")

    store = _get_store()
    cfg = store.get()

    fields: dict = {"compress_prose": (value == "on")}
    if value == "on" and not cfg.compress_enabled:
        fields["compress_enabled"] = True

    try:
        cfg = dataclasses.replace(cfg, **fields)
        store.save(cfg)
    except (ValueError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if use_json:
        print(json.dumps({"status": "ok", "compress_prose": value == "on"}))
        return

    print(f"Prose compression: {'ENABLED' if value == 'on' else 'DISABLED'}.")
    if value == "on" and "compress_enabled" in fields:
        print("  (also enabled global compress)")
    print("Daemon hot-reload: active within 2s.")


def cmd_compress_align(args: Namespace) -> None:
    """Enable or disable alignment compression."""
    use_json = getattr(args, "json", False)
    value = getattr(args, "align_value", "on")

    _write_config(compress_align=(value == "on"))

    if use_json:
        print(json.dumps({"status": "ok", "compress_align": value == "on"}))
        return

    print(f"Alignment compression: {'ENABLED' if value == 'on' else 'DISABLED'}.")
    print("Daemon hot-reload: active within 2s.")


def cmd_compress_ccr(args: Namespace) -> None:
    """Enable or disable CCR (Compressed Context Retrieval)."""
    use_json = getattr(args, "json", False)
    value = getattr(args, "ccr_value", "off")

    _write_config(compress_ccr=(value == "on"))

    if use_json:
        print(json.dumps({"status": "ok", "compress_ccr": value == "on"}))
        return

    print(f"CCR (Compressed Context Retrieval): {'ENABLED' if value == 'on' else 'DISABLED'}.")
    if value == "on":
        print("Originals stored in llmcache.db for reversible retrieval.")
    print("Daemon hot-reload: active within 2s.")
