# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Handlers for ``slm compress status|mode|prose``."""

from __future__ import annotations

import dataclasses
import json
import sys
from argparse import Namespace

from superlocalmemory.cli.optimize_constants import AGGRESSIVE_MODE_WARNING


def _get_store():
    from superlocalmemory.optimize.config.store import ConfigStore
    return ConfigStore()


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
        "mode":   cmd_compress_mode,
        "prose":  cmd_compress_prose,
        # removed in v3.6.10: code, ccr, align (extractive compressors removed)
        "code":   _cmd_compress_removed("code"),
        "ccr":    _cmd_compress_removed("ccr"),
        "align":  _cmd_compress_removed("align"),
    }
    handler = _dispatch.get(sub or "")
    if handler:
        handler(args)
    else:
        print("Usage: slm compress status|mode|prose [options]")
        sys.exit(0)


def _cmd_compress_removed(name: str):
    def handler(args: Namespace) -> None:
        print(
            f"slm compress {name}: removed in SLM v3.6.10. "
            f"Extractive {name} compression has been replaced by "
            f"Layer 1 (lossless whitespace) + Layer 2 (LLMLingua-2 prose). "
            f"Use 'slm compress prose on' to enable prose compression."
        )
        sys.exit(0)
    return handler


def cmd_compress_status(args: Namespace) -> None:
    """Print compression status."""
    use_json = getattr(args, "json", False)
    cfg = _get_store().get()

    if use_json:
        data = {
            "status": "ok",
            "compress_enabled": cfg.compress_enabled,
            "compress_mode": cfg.compress_mode,
            "compress_prose": cfg.compress_prose,
            "compress_protect_recent": cfg.compress_protect_recent,
        }
        print(json.dumps(data, indent=2))
        return

    print("Compression status:")
    print(f"  Enabled:         {'yes' if cfg.compress_enabled else 'no'}")
    print(f"  Mode:            {cfg.compress_mode}")
    print(f"  Prose (Layer 2): {'ON' if cfg.compress_prose else 'OFF'}"
          "  (LLMLingua-2, aggressive mode only)")
    print(f"  Protect recent:  {cfg.compress_protect_recent} user turns")
    print("  Layer 1 (lossless whitespace normalization) is always ON when enabled.")


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


def cmd_compress_prose(args: Namespace) -> None:
    """Enable or disable prose compression (Layer 2, LLMLingua-2)."""
    use_json = getattr(args, "json", False)
    value = getattr(args, "prose_value", "off")

    store = _get_store()
    cfg = store.get()

    fields: dict = {"compress_prose": (value == "on")}
    if value == "on":
        # Prose (Layer 2) only fires in aggressive mode, so turning it on sets a
        # COHERENT state — it can never be left enabled-but-inert in safe mode.
        if not cfg.compress_enabled:
            fields["compress_enabled"] = True
        if cfg.compress_mode != "aggressive":
            fields["compress_mode"] = "aggressive"

    try:
        cfg = dataclasses.replace(cfg, **fields)
        store.save(cfg)
    except (ValueError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if use_json:
        print(json.dumps({
            "status": "ok",
            "compress_prose": value == "on",
            "compress_mode": cfg.compress_mode,
        }))
        return

    print(f"Prose compression (LLMLingua-2): {'ENABLED' if value == 'on' else 'DISABLED'}.")
    if value == "on":
        if "compress_mode" in fields:
            print("  Compression mode set to 'aggressive' (required for Layer 2).")
        print("  Requires the llmlingua package for lossy prose compression.")
        if "compress_enabled" in fields:
            print("  (also enabled global compression)")
    print("Daemon hot-reload: active within 2s.")
