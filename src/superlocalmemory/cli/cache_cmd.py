# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Handlers for ``slm cache status|clear|invalidate|ttl|semantic``."""

from __future__ import annotations

import dataclasses
import json
import sys
from argparse import Namespace


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


def cmd_cache(args: Namespace) -> None:
    """Top-level dispatcher for ``slm cache <subcommand>``."""
    sub = getattr(args, "cache_command", None)
    _dispatch = {
        "status": cmd_cache_status,
        "clear": cmd_cache_clear,
        "invalidate": cmd_cache_invalidate,
        "ttl": cmd_cache_ttl,
        "semantic": cmd_cache_semantic,
    }
    handler = _dispatch.get(sub or "")
    if handler:
        handler(args)
    else:
        print("Usage: slm cache status|clear|invalidate|ttl|semantic [options]")
        sys.exit(0)


def cmd_cache_status(args: Namespace) -> None:
    """Print cache status from CacheDB + ConfigStore."""
    use_json = getattr(args, "json", False)
    tenant = getattr(args, "tenant", "default")

    cfg = _get_store().get()
    db = _get_cache_db()
    entry_count = db.entry_count(tenant)
    db_size = db.db_size_bytes()
    snap = db.metrics_load()

    if use_json:
        data = {
            "status": "ok",
            "entries_exact": entry_count,
            "db_size_bytes": db_size,
            "ttl_exact": cfg.ttl.exact_seconds,
            "ttl_semantic": cfg.ttl.semantic_seconds,
            "hits": snap.hits,
            "misses": snap.misses,
            "hit_rate": snap.hit_rate,
        }
        print(json.dumps(data, indent=2))
        return

    print("Cache status:")
    print(f"  Entries (exact):   {entry_count}  (not expired)")
    semantic_state = "OFF" if not cfg.semantic_enabled else "ON"
    print(f"  Semantic index:    {semantic_state}")
    size_mb = db_size / (1024 * 1024)
    print(f"  DB size:           {size_mb:.1f} MB  (~/.superlocalmemory/llmcache.db)")
    print(f"  TTL (exact):       {cfg.ttl.exact_seconds}s")
    print(f"  TTL (semantic):    {cfg.ttl.semantic_seconds}s")
    print(f"  Hits:              {snap.hits}")
    print(f"  Misses:            {snap.misses}")
    print(f"  Hit rate:          {snap.hit_rate:.1%}")


def cmd_cache_clear(args: Namespace) -> None:
    """Delete all cache entries for a tenant."""
    use_json = getattr(args, "json", False)
    tenant = getattr(args, "tenant", "default")

    db = _get_cache_db()
    deleted = db.clear_tenant(tenant)

    if use_json:
        print(json.dumps({"status": "ok", "deleted": deleted}))
        return

    print(f"Cache cleared: {deleted} entries deleted.")
    print("Hot-reload: daemon will pick up config change within 2s.")


def cmd_cache_invalidate(args: Namespace) -> None:
    """Delete entries matching a tag."""
    use_json = getattr(args, "json", False)
    tag = getattr(args, "tag", None)

    if not tag:
        print("Error: --tag is required.", file=sys.stderr)
        sys.exit(1)

    db = _get_cache_db()
    deleted = db.invalidate_by_tag(tag)

    if use_json:
        print(json.dumps({"status": "ok", "deleted": deleted, "tag": tag}))
        return

    print(f"Invalidated: {deleted} entries with tag \"{tag}\".")


def cmd_cache_ttl(args: Namespace) -> None:
    """Set exact-cache and/or semantic-cache TTL."""
    use_json = getattr(args, "json", False)
    ttl_set = getattr(args, "ttl_set", None)
    ttl_semantic = getattr(args, "ttl_semantic", None)

    if ttl_set is not None and ttl_set <= 0:
        print("Error: --set must be a positive integer.", file=sys.stderr)
        sys.exit(1)
    if ttl_semantic is not None and ttl_semantic <= 0:
        print("Error: --semantic must be a positive integer.", file=sys.stderr)
        sys.exit(1)

    store = _get_store()
    cfg = store.get()

    fields: dict = {}
    if ttl_set is not None:
        new_ttl = dataclasses.replace(cfg.ttl, exact_seconds=ttl_set)
        fields["ttl"] = new_ttl
    if ttl_semantic is not None:
        new_ttl = dataclasses.replace(cfg.ttl, semantic_seconds=ttl_semantic)
        fields["ttl"] = new_ttl

    if not fields:
        print("Error: specify --set and/or --semantic.", file=sys.stderr)
        sys.exit(1)

    try:
        cfg = dataclasses.replace(cfg, **fields)
        store.save(cfg)
    except (ValueError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if use_json:
        result: dict = {"status": "ok"}
        if ttl_set is not None:
            result["ttl_exact"] = ttl_set
        if ttl_semantic is not None:
            result["ttl_semantic"] = ttl_semantic
        print(json.dumps(result, indent=2))
        return

    if ttl_set is not None:
        print(f"TTL (exact) set to {ttl_set}s.")
    if ttl_semantic is not None:
        print(f"TTL (semantic) set to {ttl_semantic}s.")
    print("Daemon hot-reload: active within 2s. No restart required.")


def cmd_cache_semantic(args: Namespace) -> None:
    """Enable or disable semantic cache."""
    use_json = getattr(args, "json", False)
    value = getattr(args, "semantic_value", "off")

    _write_config(semantic_enabled=(value == "on"))

    if use_json:
        print(json.dumps({"status": "ok", "semantic_enabled": value == "on"}))
        return

    if value == "on":
        print("Semantic cache: ENABLED.")
        print("Note: requires embedding model (~500MB). Run `slm warmup` if not already done.")
    else:
        print("Semantic cache: DISABLED.")
    print("Daemon hot-reload: active within 2s.")
