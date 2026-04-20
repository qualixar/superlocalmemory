# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-01 §4.4

"""PostToolUse async:true hook — fire-and-forget prewarm via stdlib urllib.

LLD reference: `.backup/active-brain/lld/LLD-01-context-cache-and-hot-path-hooks.md`
Section 4.4.

HARD RULES (enforced by tests):
  - stdlib only — ``urllib.request``, NOT ``httpx`` / ``requests``.
  - Always emits ``{"async": true}`` and exits 0 — even on daemon-down,
    missing token, or broken payload.
  - Includes ``X-SLM-Hook-Token`` header from ``~/.superlocalmemory/.install_token``
    when available. Without a token, the POST is skipped (daemon would
    reject anyway) but we still emit ``{"async": true}``.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


_ALLOWED_DAEMON_HOSTS: frozenset[str] = frozenset({
    "127.0.0.1", "localhost", "::1", "[::1]",
})


def _sanitised_daemon_url() -> str:
    """Return the configured daemon URL only if it's loopback-scoped.

    S8-SEC-02: without this guard, a hostile env (e.g. a compromised
    shell profile) could set ``SLM_HOOK_DAEMON_URL`` to a remote host
    and exfiltrate the install token via the ``X-SLM-Hook-Token``
    header. We refuse any non-loopback URL and fall back to the local
    daemon.
    """
    raw = os.environ.get("SLM_HOOK_DAEMON_URL", "").strip()
    if not raw:
        return "http://127.0.0.1:8765"
    try:
        from urllib.parse import urlparse
        parsed = urlparse(raw)
    except Exception:  # pragma: no cover — urllib always importable
        return "http://127.0.0.1:8765"
    if parsed.scheme not in ("http", "https"):
        return "http://127.0.0.1:8765"
    host = (parsed.hostname or "").lower()
    if host not in _ALLOWED_DAEMON_HOSTS:
        return "http://127.0.0.1:8765"
    # Preserve the scheme + port (user may bind daemon on a non-default port).
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{host}{port}"


DAEMON_URL: str = _sanitised_daemon_url()
DAEMON_TIMEOUT: float = float(os.environ.get("SLM_HOOK_DAEMON_TIMEOUT", "0.5"))
PREWARM_PATH: str = "/internal/prewarm"

_INPUT_CAP: int = 2000
_OUTPUT_CAP: int = 4000


def _install_token() -> str:
    """Read the install token. Returns '' on any problem."""
    path = Path.home() / ".superlocalmemory" / ".install_token"
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:  # pragma: no cover — FS transient failure
        return ""


def _summarize(obj, cap: int) -> str:
    """Stringify with size cap."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj[:cap]
    try:
        return json.dumps(obj, default=str)[:cap]
    except Exception:  # pragma: no cover — exotic non-serializable object
        try:
            return str(obj)[:cap]
        except Exception:
            return ""


def _post(body: dict, token: str) -> None:
    """Fire the prewarm POST. Silently swallows all failures."""
    try:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{DAEMON_URL}{PREWARM_PATH}",
            data=data,
            headers={
                "Content-Type": "application/json",
                "X-SLM-Hook-Token": token,
            },
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=DAEMON_TIMEOUT)
        try:
            resp.read()
        except Exception:  # pragma: no cover — partial response flush
            pass
        try:
            resp.close()
        except Exception:  # pragma: no cover — already closed
            pass
    except Exception:
        # Daemon unreachable / timeout / auth rejected — by spec this is
        # best-effort. Hook still returns {"async": true} upstream.
        return


def main() -> int:
    """Entry point. Reads stdin JSON, posts to daemon, always prints
    ``{"async": true}`` and returns 0."""
    try:
        raw = sys.stdin.read()
    except Exception:  # pragma: no cover — stdin unreadable in container
        sys.stdout.write('{"async": true}')
        return 0

    if not raw:
        sys.stdout.write('{"async": true}')
        return 0

    try:
        payload = json.loads(raw)
    except Exception:
        sys.stdout.write('{"async": true}')
        return 0

    if not isinstance(payload, dict):
        sys.stdout.write('{"async": true}')
        return 0

    try:
        token = _install_token()
        if token:
            body = {
                "session_id": str(payload.get("session_id", "")),
                "tool_name": str(payload.get("tool_name", "")),
                "input_summary": _summarize(payload.get("tool_input"), _INPUT_CAP),
                "output_summary": _summarize(payload.get("tool_response"), _OUTPUT_CAP),
            }
            _post(body, token)
    except Exception:  # pragma: no cover — defense in depth
        # Never propagate.
        pass

    sys.stdout.write('{"async": true}')
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI entry only
    sys.exit(main())
