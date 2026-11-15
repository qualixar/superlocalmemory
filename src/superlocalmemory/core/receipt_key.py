# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The local key judge receipts are hashed under (HMAC-SHA256).

A plain SHA-256 of two short memories survives the memories' erasure and can
confirm a guess of the erased text from the surviving one. Under a key that
never leaves ``<SLM_HOME>/keys`` (owner-only), the hash still identifies what
was judged -- the same pair gives the same value, so a pair is never sent to a
judge twice -- without being checkable by anyone holding only the database.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from pathlib import Path

KEY_FILE = "judge-receipts.key"


def load_or_create(slm_home: Path) -> bytes:
    folder = Path(slm_home) / "keys"
    path = folder / KEY_FILE
    try:
        key = path.read_bytes()
        if len(key) >= 32:
            return key
    except FileNotFoundError:
        pass
    folder.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(folder, 0o700)
    except OSError:  # Windows: inherits the profile folder's ACL
        pass
    key = secrets.token_bytes(32)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        # Another process (the daemon, an MCP server, a hook) created it first.
        return _read_existing(path)
    with os.fdopen(fd, "wb") as handle:
        handle.write(key)
    return key


def _read_existing(path: Path) -> bytes:
    """The key another process just wrote; it may still be mid-write."""
    for _ in range(50):
        key = path.read_bytes()
        if len(key) >= 32:
            return key
        time.sleep(0.01)
    raise OSError(f"receipt key at {path} is incomplete")


def digest(key: bytes, payload: bytes) -> str:
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


__all__ = ["KEY_FILE", "digest", "load_or_create"]
