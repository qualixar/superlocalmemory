# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""HTTP proxy that lets MCP processes use the daemon as their worker.

Without this, every MCP process (one per IDE) would spawn its own
``recall_worker`` subprocess through ``WorkerPool.shared()`` and load
the ONNX embedder into that subprocess. With N IDEs open the total
RSS was approximately N x 1.6 GB — the exact failure Path B was built
to avoid.

With this proxy, the MCP process opens an HTTP connection to the
single long-lived daemon (already running for dashboard / mesh /
health) and forwards ``recall`` and ``store`` calls there. Heavy
engine state exists in exactly one process: the daemon.
"""
from __future__ import annotations

import logging
import urllib.parse
from typing import Any

logger = logging.getLogger(__name__)


class DaemonPoolProxy:
    """:class:`WorkerPool`-shaped facade that talks to the daemon over HTTP.

    The shape matches ``WorkerPool.recall`` / ``WorkerPool.store`` so that
    the existing pool adapter in ``mcp/_pool_adapter.py`` can swap between
    a local subprocess pool and the daemon proxy without any adapter
    change. Errors are returned as ``{"ok": False, "error": "..."}``
    envelopes — the adapter is responsible for surfacing those.
    """

    def __init__(self, port: int, *, timeout_s: float = 30.0) -> None:  # v3.4.59: 8s→30s — observed recall takes 13.4s on dense graph (2.1M edges); 8s always timed out → degraded mode
        self._port = port
        self._timeout = timeout_s

    def recall(
        self, query: str, limit: int = 10, session_id: str = "",
        fast: bool | None = None,
        include_global: bool | None = None,
        include_shared: bool | None = None,
        window: str | None = None,
    ) -> dict[str, Any]:
        _params: dict[str, Any] = {
            "q": query,
            "limit": limit,
            "session_id": session_id or "",
        }
        # v3.8.2 client-driven agentic: only send ``fast`` when the caller set it
        # explicitly. Unset (None) lets the daemon resolve the configured
        # client-driven-agentic default — the same way scope flags are handled.
        if fast is not None:
            _params["fast"] = "true" if fast else "false"
        # v3.6.15 multi-scope: only send the scope flags when explicitly set, so
        # an unset value lets the daemon resolve the configured default (shared
        # is opt-in). "None" must NOT become the string "none" on the wire.
        if include_global is not None:
            _params["include_global"] = "true" if include_global else "false"
        if include_shared is not None:
            _params["include_shared"] = "true" if include_shared else "false"
        if window:
            _params["window"] = window
        params = urllib.parse.urlencode(_params)
        try:
            from superlocalmemory.cli.daemon import daemon_request

            data = daemon_request(
                "GET",
                f"/recall?{params}",
                timeout_seconds=self._timeout,
            )
        except Exception as exc:
            logger.warning("daemon /recall failed: %s", exc)
            return {"ok": False, "error": str(exc)}
        if not isinstance(data, dict):
            return {"ok": False, "error": "owned daemon unavailable"}
        data.setdefault("ok", True)
        return data

    def store(
        self, content: str, metadata: dict | None = None,
    ) -> dict[str, Any]:
        body = {
            "content": content,
            "tags": (metadata or {}).get("tags", ""),
            "metadata": metadata or {},
            "session_id": (metadata or {}).get("session_id", ""),
            "idempotency_key": (metadata or {}).get("idempotency_key") or None,
        }
        try:
            # One identity-aware daemon client owns descriptor validation,
            # capability delivery, and exact-instance targeting.  A raw urllib
            # POST here previously became unauthenticated when /remember was
            # hardened and could also attach to a stale/foreign port.
            from superlocalmemory.cli.daemon import daemon_request

            data = daemon_request("POST", "/remember", body)
        except Exception as exc:
            logger.warning("daemon /remember failed: %s", exc)
            return {"ok": False, "error": str(exc)}
        if not isinstance(data, dict):
            return {"ok": False, "error": "owned daemon unavailable"}
        data.setdefault("ok", True)
        return data


def choose_pool() -> Any:
    """Return the best available pool for this MCP process.

    Preference order:
      1. Running daemon — use HTTP proxy (keeps ONNX in ONE process)
      2. No daemon — fall back to ``WorkerPool.shared()`` (spawns a
         local subprocess with a FULL engine). This keeps single-user
         / first-launch scenarios working.
    """
    try:
        from superlocalmemory.cli.daemon import _get_port, is_daemon_running
        if is_daemon_running():
            return DaemonPoolProxy(port=_get_port())
    except Exception as exc:
        logger.warning("daemon probe failed — falling back to subprocess pool: %s", exc)
    from superlocalmemory.core.worker_pool import WorkerPool
    return WorkerPool.shared()
