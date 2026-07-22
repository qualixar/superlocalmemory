# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Handler for ``slm proxy``."""

from __future__ import annotations

import dataclasses
import json
import sys
from argparse import Namespace

_DEFAULT_PORT: int = 8765


def _get_store():
    from superlocalmemory.optimize.config.store import ConfigStore
    return ConfigStore()


def _ensure_running(port: int) -> bool:
    """Liveness probe: return True if the SLM daemon is responding at *port*.

    Calls GET /health with a 2-second timeout.  The SLM daemon (which also acts
    as the optimize proxy) is the process that answers on :8765; if it responds
    the proxy layer is alive. We do NOT call lifecycle.ensure_proxy_running()
    here because that function reads config via the daemon-internal store
    (get_optimize_config/_store) which is None in a CLI subprocess context.
    """
    try:
        import urllib.request
        url = f"http://127.0.0.1:{port}/health"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def cmd_proxy(args: Namespace) -> None:
    """Start the SLM optimization proxy (or report if already running)."""
    use_json = getattr(args, "json", False)
    port = getattr(args, "port", _DEFAULT_PORT)
    provider = getattr(args, "provider", "anthropic")
    no_compress = getattr(args, "no_compress", False)
    semantic = getattr(args, "semantic", False)

    # CRIT-3 FIX: validate port before any config write
    if not (1024 <= port <= 65535):
        print(f"Error: --port must be 1024–65535, got {port}.", file=sys.stderr)
        sys.exit(1)

    store = _get_store()
    cfg = store.get()

    # Build config updates
    providers = dict(cfg.providers) if cfg.providers else {}
    from superlocalmemory.optimize.config.schema import ProviderConfig
    existing = providers.get(provider, ProviderConfig())
    providers[provider] = ProviderConfig(
        enabled=existing.enabled,
        base_url=f"http://localhost:{port}",
    )

    fields: dict = {"providers": providers, "proxy_enabled": True}
    if no_compress:
        fields["compress_enabled"] = False
    if semantic:
        fields["semantic_enabled"] = True

    try:
        cfg = dataclasses.replace(cfg, **fields)
        store.save(cfg)
    except (ValueError, OSError) as e:
        print(f"Error writing config: {e}", file=sys.stderr)
        sys.exit(1)

    running = _ensure_running(port)

    if use_json:
        data = {
            "status": "running" if running else "failed",
            "port": port,
            "anthropic_url": f"http://localhost:{port}",
            "openai_url": f"http://localhost:{port}/v1",
            "provider": provider,
            "compress": not no_compress,
            "semantic": semantic,
        }
        print(json.dumps(data, indent=2))
    else:
        if running:
            print(f"SLM proxy starting on :{port}...")
            print(f"  Anthropic surface:  http://localhost:{port}")
            print(f"  OpenAI surface:     http://localhost:{port}/v1")
            print()
            print(f"Set ANTHROPIC_BASE_URL=http://localhost:{port} before running Claude Code.")
            print("Or run: slm wrap claude")
            print()
            print("Proxy ready.")
            print()
            print("Note: 'slm proxy' enables the proxy independently — it does not")
            print("flip the master optimize switch, so 'slm optimize status' may")
            print("show OFF while the proxy is running.")
        else:
            print("Error: proxy failed to start. Check logs.", file=sys.stderr)
            sys.exit(1)
