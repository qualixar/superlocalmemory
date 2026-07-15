# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""Opt-in API-key authentication middleware.

When ``~/.superlocalmemory/api_key`` exists, write endpoints require the
``X-SLM-API-Key`` header.  Read endpoints remain open for backward
compatibility.  If the key file is absent auth is completely disabled --
all requests pass.

V3 change: base directory moved from ``~/.claude-memory/`` to
``~/.superlocalmemory/``.
"""

import hashlib
import hmac
import logging
from pathlib import Path
from typing import Optional

from superlocalmemory.infra.data_root import DynamicStatePath

logger = logging.getLogger("superlocalmemory.auth")

# V3 base directory
MEMORY_DIR = DynamicStatePath()
API_KEY_FILE = DynamicStatePath("api_key")


def _load_api_key_hash(key_file: Optional[Path] = None) -> Optional[str]:
    """Load and hash the API key from disk.

    Args:
        key_file: Override path (useful for testing).

    Returns:
        SHA-256 hex digest of the stored key, or ``None`` when auth is
        not configured.
    """
    path = Path(key_file if key_file is not None else API_KEY_FILE)
    if not path.exists():
        return None
    try:
        key = path.read_text().strip()
        if not key:
            return None
        return hashlib.sha256(key.encode()).hexdigest()
    except Exception as exc:
        logger.warning("Failed to load API key: %s", exc)
        return None


def check_api_key(
    request_headers: dict,
    is_write: bool = False,
    key_file: Optional[Path] = None,
) -> bool:
    """Authorize a request against the stored API key.

    Returns ``True`` when:
    * No key file exists (auth not configured -- backward compatible).
    * The request is a read operation (reads always allowed).
    * The ``X-SLM-API-Key`` header matches the stored key.

    Args:
        request_headers: Mapping of HTTP header names to values.
        is_write: ``True`` for mutating operations that require auth.
        key_file: Override key-file path (testing).
    """
    key_hash = _load_api_key_hash(key_file)

    # No key file = auth disabled
    if key_hash is None:
        return True

    # Reads are always permitted
    if not is_write:
        return True

    # Writes require a matching key
    provided = request_headers.get("x-slm-api-key", "")
    return verify_api_key(provided, key_file=key_file)


def verify_api_key(
    presented: str,
    key_file: Optional[Path] = None,
) -> bool:
    """Verify a configured API key; unconfigured auth is never identity."""
    if not isinstance(presented, str) or not presented:
        return False
    expected = _load_api_key_hash(key_file)
    if expected is None:
        return False
    actual = hashlib.sha256(presented.encode()).hexdigest()
    return hmac.compare_digest(actual, expected)


def authorize_http_mcp_request(
    request_headers: dict,
    *,
    client_host: str,
    key_file: Optional[Path] = None,
) -> bool:
    """Authorize the Streamable-HTTP MCP transport boundary.

    An MCP session identifier is routing state, not authentication.  Loopback
    keeps the local-first compatibility contract, while every non-loopback
    peer must present the configured SLM API key.  The LAN allowlist limits
    reachability but deliberately does not grant a write identity.
    """
    if client_host in ("127.0.0.1", "::1", "localhost"):
        return True
    provided = request_headers.get("x-slm-api-key", "")
    return verify_api_key(provided, key_file=key_file)
